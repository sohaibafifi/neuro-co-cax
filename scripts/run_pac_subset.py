"""PAC sufficient-subset per method.

For each attribution method, build an AttributionTrace whose
`top_k_nodes` is the full per-step node ranking, then drive
`neuro_co.cax.cp_minimal_subset` with that trace. Record how many
(b, t) cells succeed (subset found within `max_k`) and the mean
sufficient-subset size among successes.

Defaults (matching paper text):
    pac_epsilon = 0.2
    pac_delta   = 0.2
    sigma       = 0.05
    max_k       = 25
    bonferroni  = True  -> M_bonf = 70

CVRPTW seed-0, B = 8, T = 8 -> 64 cells per method.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from neuro_co.xai.attribution import (
    AttributionTrace,
    contrastive_attribution,
    deeplift_attribution,
    gradient_attribution,
    integrated_gradients,
)
from neuro_co.cax.benchmark import (
    _instantiate,
    _load_ckpt,
    _load_hydra_cfg,
    _pick_device,
    _resolve_ckpt,
)
from neuro_co.cax.cp_minimal_subset import cp_minimal_subset
from neuro_co.cax.lambda_attribution import lambda_attribution


PROBLEM_FEATURE_UNION: dict[str, tuple[str, ...]] = {
    "vrptw": ("locs", "demand", "demand_linehaul", "time_windows", "durations"),
    "fjsp": ("proc_times", "num_eligible"),
    "op": ("locs", "prize"),
}


def _present(td, keys):
    return tuple(
        k for k in keys
        if k in td.keys()
        and hasattr(td[k], "dtype")
        and td[k].dtype != torch.bool
    )


def _num_nodes(td) -> int:
    if "locs" in td.keys():
        return int(td["locs"].shape[-2])
    if "proc_times" in td.keys():
        return int(td["proc_times"].shape[-1])
    if "action_mask" in td.keys():
        return int(td["action_mask"].shape[-1])
    return 0


def _cax_trace(model, env, td, *, problem, max_steps, top_k, multipliers="lp"):
    """Flatten LP-CAX per-family node scores into an AttributionTrace."""
    policy = model.policy if hasattr(model, "policy") else model
    la = lambda_attribution(
        policy, env, td,
        problem=problem, top_k=top_k, max_steps=max_steps,
        multipliers=multipliers,
    )
    pfn = la.per_family_node_scores  # [K, B, T, N]
    if la.multipliers is not None:
        w = la.multipliers.abs().view(-1, 1, 1, 1).to(pfn.device).to(pfn.dtype)
        node = (pfn * w).sum(dim=0)
    else:
        node = pfn.sum(dim=0)
    k_eff = min(top_k, node.shape[-1])
    top_scores, top_idx = node.topk(k_eff, dim=-1)
    B, T, _ = node.shape
    feature_keys = sorted({k for keys in la.feature_keys_per_family for k in keys})
    return AttributionTrace(
        actions=torch.zeros(B, T, dtype=torch.long),
        log_probs=torch.zeros(B, T),
        node_scores=node.detach().cpu(),
        top_k_nodes=top_idx.detach().cpu(),
        top_k_scores=top_scores.detach().cpu(),
        feature_keys=feature_keys,
    )


def run_problem(
    problem: str,
    run_dir: Path,
    *,
    batch: int = 8,
    max_steps: int = 8,
    pac_epsilon: float = 0.2,
    pac_delta: float = 0.2,
    sigma: float = 0.05,
    max_k: int = 25,
    seed: int = 0,
):
    cfg = _load_hydra_cfg(run_dir)
    env, model = _instantiate(cfg)
    _load_ckpt(model, _resolve_ckpt(run_dir))
    device = _pick_device()
    model = model.to(device).eval()

    torch.manual_seed(seed)
    td = env.reset(batch_size=batch).to(device)
    policy = model.policy if hasattr(model, "policy") else model

    n_nodes = _num_nodes(td)
    top_k_full = n_nodes  # full ranking so cp_minimal_subset can grow up to max_k
    union = _present(td, PROBLEM_FEATURE_UNION[problem])
    print(
        f"[{problem}] B={batch} T={max_steps} N={n_nodes} max_k={max_k} "
        f"eps={pac_epsilon} delta={pac_delta} sigma={sigma} "
        f"feature_keys={union}",
        flush=True,
    )

    method_factories = {
        "gradient": lambda: gradient_attribution(
            policy, env, td,
            feature_keys=union, top_k=top_k_full, max_steps=max_steps,
        ),
        "ig": lambda: integrated_gradients(
            policy, env, td,
            feature_keys=union, top_k=top_k_full, max_steps=max_steps,
            ig_steps=8, problem=problem,
        ),
        "deeplift": lambda: deeplift_attribution(
            policy, env, td,
            feature_keys=union, top_k=top_k_full, max_steps=max_steps,
            problem=problem,
        ),
        "contrastive": lambda: contrastive_attribution(
            policy, env, td,
            feature_keys=union, top_k=top_k_full, max_steps=max_steps,
        ),
        "cax": lambda: _cax_trace(
            model, env, td,
            problem=problem, max_steps=max_steps, top_k=top_k_full,
        ),
    }

    results = {}
    for name, factory in method_factories.items():
        print(f"[{problem}] running {name}", flush=True)
        t0 = time.time()
        try:
            trace = factory()
        except Exception as exc:  # noqa: BLE001
            print(f"[{problem}] {name} trace FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            report = cp_minimal_subset(
                policy, env, td,
                pac_epsilon=pac_epsilon,
                pac_delta=pac_delta,
                sigma=sigma,
                max_k=max_k,
                feature_keys=union,
                max_steps=max_steps,
                trace=trace,
                bonferroni=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{problem}] {name} subset FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        sz = report.subset_size
        succeeded = (sz > 0)
        n_succ = int(succeeded.sum().item())
        n_total = int(sz.numel())
        mean_size = float(sz[succeeded].float().mean().item()) if n_succ else float("nan")
        dt = time.time() - t0
        results[name] = {
            "succeeded": n_succ,
            "total": n_total,
            "mean_subset_size": mean_size,
            "samples_drawn_M": int(report.samples_drawn),
            "elapsed_s": dt,
        }
        print(
            f"[{problem}] {name}: {n_succ}/{n_total} succeeded, "
            f"mean |S*|={mean_size:.2f}, M={report.samples_drawn} "
            f"({dt:.1f}s)",
            flush=True,
        )

    out = {
        "problem": problem,
        "run_dir": str(run_dir),
        "batch": batch,
        "max_steps": max_steps,
        "num_nodes": n_nodes,
        "pac_epsilon": pac_epsilon,
        "pac_delta": pac_delta,
        "sigma": sigma,
        "max_k": max_k,
        "feature_keys_used": list(union),
        "methods": results,
    }
    out["seed"] = int(seed)
    out_dir = Path("experiments/pac_subset")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{problem}_seed{int(seed)}.json"
    (out_dir / fname).write_text(json.dumps(out, indent=2))
    print(f"[{problem}] wrote {out_dir / fname}", flush=True)
    return out


if __name__ == "__main__":
    prob = sys.argv[1] if len(sys.argv) > 1 else "vrptw"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        f"outputs/{prob}/train_seed{seed}"
    )
    run_problem(prob, run_dir, seed=seed)
