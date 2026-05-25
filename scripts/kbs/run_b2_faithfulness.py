"""Deletion-curve faithfulness sweep (KBS short communication, B2).

For each (problem, method) pair, run attribution once with top_k=10
and then call `deletion_flip_rate` at k in {1, 3, 5, 10}. Report the
mean flip rate per k and the across-k AUC.

A high flip rate at k means that masking the top-k attributed nodes
changes the policy's argmax — i.e. the attribution actually
pinpointed inputs the policy relied on. Higher AUC = more faithful.

Output: one JSON per problem under ``experiments/kbs_faithfulness/``.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import torch

from neuro_co.xai.attribution import (
    AttributionTrace,
    contrastive_attribution,
    deeplift_attribution,
    gradient_attribution,
    integrated_gradients,
)
from neuro_co.xai.faithfulness import deletion_flip_rate
from neuro_co.cax.benchmark import (
    _instantiate,
    _load_ckpt,
    _load_hydra_cfg,
    _pick_device,
    _resolve_ckpt,
)
from neuro_co.cax.lambda_attribution import lambda_attribution


# Union of every feature key consumed by any constraint family per
# problem. The attribution code drops keys that aren't in the
# TensorDict, so listing extras is safe.
PROBLEM_FEATURE_UNION: dict[str, tuple[str, ...]] = {
    "vrptw": ("locs", "demand", "demand_linehaul", "time_windows", "durations"),
    "fjsp": ("proc_times", "num_eligible"),
    "op": ("locs", "prize"),
}


def _present(td, keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        k
        for k in keys
        if k in td.keys() and hasattr(td[k], "dtype") and td[k].dtype != torch.bool
    )


def _num_nodes(td) -> int:
    if "locs" in td.keys():
        return int(td["locs"].shape[-2])
    if "proc_times" in td.keys():
        return int(td["proc_times"].shape[-1])
    if "action_mask" in td.keys():
        return int(td["action_mask"].shape[-1])
    return 0


def _cax_trace(
    model, env, td, *, problem: str, max_steps: int, top_k: int, multipliers: str
) -> AttributionTrace:
    """Run lambda_attribution and flatten to AttributionTrace.

    Per-node CAX score = sum_k |mu_k| * per_family_node_scores[k].
    With ``multipliers='lp'`` the weighting follows the LP duals
    (the headline CAX variant in the paper). With ``multipliers=None``
    we get the equal-weight proxy.
    """
    policy = model.policy if hasattr(model, "policy") else model
    la = lambda_attribution(
        policy,
        env,
        td,
        problem=problem,
        top_k=top_k,
        max_steps=max_steps,
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
    batch: int = 16,
    max_steps: int = 8,
    top_k_max: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    seed: int = 0,
) -> dict:
    cfg = _load_hydra_cfg(run_dir)
    env, model = _instantiate(cfg)
    ckpt = _resolve_ckpt(run_dir)
    _load_ckpt(model, ckpt)
    device = _pick_device()
    model = model.to(device).eval()

    torch.manual_seed(int(seed))
    td = env.reset(batch_size=batch).to(device)
    policy = model.policy if hasattr(model, "policy") else model

    n_nodes = _num_nodes(td)
    top_k = min(top_k_max, n_nodes) if n_nodes else top_k_max
    ks = tuple(k for k in ks if k <= top_k)

    union = _present(td, PROBLEM_FEATURE_UNION[problem])
    print(
        f"[{problem}] B={batch} T={max_steps} N={n_nodes} top_k={top_k} "
        f"ks={ks} feature_keys={union}",
        flush=True,
    )

    methods = {
        "gradient": lambda: gradient_attribution(
            policy, env, td,
            feature_keys=union, top_k=top_k, max_steps=max_steps,
        ),
        "ig": lambda: integrated_gradients(
            policy, env, td,
            feature_keys=union, top_k=top_k, max_steps=max_steps,
            ig_steps=8, problem=problem,
        ),
        "deeplift": lambda: deeplift_attribution(
            policy, env, td,
            feature_keys=union, top_k=top_k, max_steps=max_steps,
            problem=problem,
        ),
        "contrastive": lambda: contrastive_attribution(
            policy, env, td,
            feature_keys=union, top_k=top_k, max_steps=max_steps,
        ),
        "cax": lambda: _cax_trace(
            model, env, td,
            problem=problem, max_steps=max_steps, top_k=top_k,
            multipliers="lp",
        ),
    }

    results: dict[str, dict] = {}
    for name, factory in methods.items():
        t0 = time.time()
        print(f"[{problem}] running {name}", flush=True)
        try:
            trace = factory()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{problem}] {name} FAILED at attribution: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        per_k: dict[int, float] = {}
        for k in ks:
            try:
                report = deletion_flip_rate(
                    trace, policy, env, td, top_k=k, baseline="mean"
                )
                per_k[int(k)] = float(report.mean_flip_rate)
                print(
                    f"  k={k}: flip_rate={report.mean_flip_rate:.4f} "
                    f"steps={report.num_steps}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  k={k}: FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )
                per_k[int(k)] = float("nan")
        finite = [v for v in per_k.values() if v == v]
        auc = float(sum(finite) / len(finite)) if finite else float("nan")
        dt = time.time() - t0
        results[name] = {"per_k": per_k, "auc": auc, "elapsed_s": dt}
        print(f"[{problem}] {name} AUC={auc:.4f} ({dt:.1f}s)", flush=True)

    out = {
        "problem": problem,
        "run_dir": str(run_dir),
        "batch": batch,
        "max_steps": max_steps,
        "num_nodes": n_nodes,
        "ks": list(ks),
        "feature_keys_used": list(union),
        "methods": results,
    }
    out["seed"] = int(seed)
    out_dir = Path("experiments/kbs_faithfulness")
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
