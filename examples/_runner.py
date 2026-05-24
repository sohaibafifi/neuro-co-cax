"""Shared runner: train (or load) -> adjudicate -> certify -> stats.

Used by `adjudicate_{vrptw,op,fjsp}.py`. The CLI in each problem stub
just sets the problem name and a few defaults.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from neuro_co.cax.adjudicate import bootstrap_ci_diff, mcnemar_pvalue
from neuro_co.cax.constraint_map import get_constraints
from neuro_co.cax.cp_counterfactual import cp_counterfactual
from neuro_co.cax.feasibility import is_feasible
from neuro_co.cax.lambda_attribution import lambda_attribution


# Per-problem perturbation budgets matching paper-cax §3.
DEFAULT_EPS: dict[str, dict[str, float]] = {
    "vrptw": {"locs": 30.0, "demand": 0.02, "time_windows": 30.0, "durations": 0.0},
    "op":    {"locs": 0.3, "prize": 0.5, "max_length": 0.0},
    "fjsp":  {"proc_times": 20.0, "num_eligible": 1.0},
}


@dataclass
class Result:
    problem: str
    seed: int
    n_arith: int
    n_cert: int
    proxy_acc: float
    lp_acc: float


def _instantiate_env_and_policy(problem: str, num_loc: int, ckpt: Path | None):
    """Construct an rl4co env + AttentionModel policy.

    If `ckpt` exists, load weights; otherwise return a fresh policy
    (useful for sanity-checking the pipeline without a trained model).
    """
    from rl4co.models import AttentionModel
    from rl4co.models.zoo.am.policy import AttentionModelPolicy

    env_map = {
        "vrptw": ("rl4co.envs.routing.cvrptw.env", "CVRPTWEnv"),
        "op":    ("rl4co.envs.routing.op.env", "OPEnv"),
        "fjsp":  ("rl4co.envs.scheduling.fjsp.env", "FJSPEnv"),
    }
    mod, cls = env_map[problem]
    import importlib

    env_cls = getattr(importlib.import_module(mod), cls)
    env = env_cls(generator_params={"num_loc": num_loc} if problem != "fjsp" else {})

    policy = AttentionModelPolicy(
        env_name=problem, embed_dim=128, num_encoder_layers=3, num_heads=8
    )
    model = AttentionModel(env=env, policy=policy)
    if ckpt is not None and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[runner] loaded checkpoint {ckpt}")
    else:
        print("[runner] no checkpoint provided -- using untrained policy "
              "(numbers will be noisy)")
    model.eval()
    return env, model


def _compute_cf_top(cf_delta, flipped, names, fam_to_keys):
    """Mass-weighted top family per (b, t) flipped cell."""
    B, T = flipped.shape
    top = torch.full((B, T), -1, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            if not bool(flipped[b, t]):
                continue
            best_idx, best_mass = -1, -1.0
            for k_idx, fam in enumerate(names):
                mass = 0.0
                for fk in fam_to_keys.get(fam, ()):
                    if fk in cf_delta:
                        mass += float(cf_delta[fk][t, b].abs().sum().item())
                if mass > best_mass:
                    best_mass, best_idx = mass, k_idx
            top[b, t] = best_idx
    return top


def adjudicate(
    problem: str,
    seeds: Iterable[int] = (0, 1, 2),
    num_loc: int = 50,
    batch_size: int = 16,
    max_steps: int = 8,
    cf_shots: int = 128,
    ckpt_template: str | None = None,
    out: Path | None = None,
) -> list[Result]:
    """Run the paper-cax adjudication on `problem` across `seeds`.

    `ckpt_template` is a `str.format`-style path with `{seed}` placeholder,
    e.g. `"outputs/vrptw/train_seed{seed}/checkpoints/last.ckpt"`.
    """
    eps = DEFAULT_EPS[problem]
    feature_keys = tuple(eps.keys())
    families = get_constraints(problem)
    names = [n for n, _ in families]
    fam_to_keys = {n: tuple(keys) for n, keys in families}

    results: list[Result] = []
    pooled = {"proxy": [], "lp": []}
    for s in seeds:
        ckpt = Path(ckpt_template.format(seed=s)) if ckpt_template else None
        env, model = _instantiate_env_and_policy(problem, num_loc, ckpt)
        torch.manual_seed(0)
        td = env.reset(batch_size=batch_size)

        # Λ-attribution per backend.
        lambda_top: dict[str, torch.Tensor] = {}
        for mode in ("proxy", "lp"):
            mult = None if mode == "proxy" else mode
            attr = lambda_attribution(
                model, env, td, problem=problem,
                max_steps=max_steps, multipliers=mult,
            )
            lambda_top[mode] = attr.top_family_per_step()

        # CF search (arithmetic stage-1 inside loop).
        cf = cp_counterfactual(
            model, env, td, problem=problem,
            feature_keys=feature_keys, epsilon=eps,
            sigma={k: v / 3 for k, v in eps.items()},
            max_shots=cf_shots, max_steps=max_steps,
            seed=0, perturb_one_at_a_time=True,
            feasibility_mode="arithmetic",
        )
        flipped = cf.flipped.cpu()
        n_arith = int(flipped.sum())

        # Post-hoc combinatorial certification per (b, t) winner.
        certified = torch.zeros_like(flipped)
        for t in range(flipped.shape[1]):
            if not flipped[:, t].any():
                continue
            td_pert = td.clone()
            for k, dz in cf.delta.items():
                if k in td_pert:
                    td_pert[k] = (
                        td_pert[k].cpu() + dz[t].cpu()
                    ).to(td_pert[k].dtype).to(td_pert[k].device)
            cp_ok = is_feasible(td_pert, problem, mode="cp_sat", time_limit_s=2.0)
            certified[:, t] = cp_ok.cpu() & flipped[:, t]
        n_cert = int(certified.sum())

        # CF top family + per-mode agreement on certified cells only.
        cf_top = _compute_cf_top(cf.delta, flipped, names, fam_to_keys)
        rows: dict[str, list[int]] = {"proxy": [], "lp": []}
        for b in range(certified.shape[0]):
            for t in range(certified.shape[1]):
                if not bool(certified[b, t]):
                    continue
                for m in ("proxy", "lp"):
                    rows[m].append(int(int(lambda_top[m][b, t]) == int(cf_top[b, t])))
                    pooled[m].append(rows[m][-1])
        proxy_acc = sum(rows["proxy"]) / max(1, len(rows["proxy"]))
        lp_acc = sum(rows["lp"]) / max(1, len(rows["lp"]))
        print(f"[{problem} seed {s}] arith={n_arith} cert={n_cert}  "
              f"proxy={proxy_acc:.3f}  lp={lp_acc:.3f}")
        results.append(Result(problem, s, n_arith, n_cert, proxy_acc, lp_acc))

    if pooled["lp"]:
        pt, lo, hi = bootstrap_ci_diff(
            pooled["lp"], pooled["proxy"], n_resamples=10_000, seed=0
        )
        b01, b10, pv = mcnemar_pvalue(pooled["lp"], pooled["proxy"])
        n = len(pooled["lp"])
        print(f"\n[{problem} pooled] n_cert={n}  "
              f"proxy={sum(pooled['proxy'])/n:.3f}  lp={sum(pooled['lp'])/n:.3f}")
        print(f"    diff={pt:+.3f}  CI95=[{lo:+.3f}, {hi:+.3f}]  "
              f"McNemar p={pv:.2e}  b01={b01}  b10={b10}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            [r.__dict__ for r in results], indent=2,
        ))
        print(f"[runner] wrote {out}")
    return results


def cli(problem: str) -> int:
    """Argparse front-end used by the per-problem stubs."""
    parser = argparse.ArgumentParser(prog=f"adjudicate-{problem}")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint path template with {seed} placeholder")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--cf-shots", type=int, default=128)
    parser.add_argument("--num-loc", type=int, default=50 if problem != "fjsp" else 0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    adjudicate(
        problem=problem,
        seeds=args.seeds,
        num_loc=args.num_loc,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        cf_shots=args.cf_shots,
        ckpt_template=args.ckpt,
        out=args.out,
    )
    return 0
