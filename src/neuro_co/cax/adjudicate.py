"""Adjudicate Lambda-attribution backends via CP-counterfactual ground truth.

For each trained run dir, this module:

  1. Runs Lambda-attribution in all configured modes
     (`'proxy'`, `'lp'`, `'subgrad'`).
  2. Runs `cp_counterfactual` with one-feature-at-a-time perturbations
     to get a per-step *ground truth* of which constraint family
     a flipping perturbation lives in.
  3. Per (run_dir, step), records whether each Lambda-mode's
     top-ranked constraint family matches the CF's argmax-flipping
     constraint family.

The aggregated table feeds the paper-cax §4.3 headline: across
all (run_dir, step) pairs, what fraction of the time does each
Lambda backend pick the same top family as the CP-counterfactual?
High agreement = faithful; low = misled. Paired Wilcoxon over the
(seed, step) win/loss vector tests significance.

JSON layout per run_dir
-----------------------

  <run_dir>/adjudication.json::

    {
      "problem": "vrptw",
      "seed": 0,
      "constraint_names": ["capacity", "time_window", "spatial"],
      "modes": ["proxy", "lp", "subgrad"],
      "lambda_top_family_per_step": {
        "proxy": [[k, k, ...]] # [B, T]
        "lp":    [[...]],
        ...
      },
      "cf_flipped_per_step":           [[1, 0, ...]],   # [B, T]
      "cf_top_family_per_step":        [[k, ...]],       # [B, T] (-1 if no flip)
      "agreement_per_mode_per_step":   { "proxy": [[...]], "lp": [[...]] },
      "mean_agreement_per_mode":       { "proxy": 0.41, "lp": 0.83, ... }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from neuro_co.cax.benchmark import (
    _infer_problem,
    _instantiate,
    _load_ckpt,
    _load_hydra_cfg,
    _pick_device,
    _resolve_ckpt,
)
from neuro_co.cax.constraint_map import PROBLEM_CONSTRAINTS, get_constraints
from neuro_co.cax.cp_counterfactual import cp_counterfactual
from neuro_co.cax.lambda_attribution import lambda_attribution


_GENERIC_METHOD_KEYS = {
    "gradient", "ig", "integrated_gradients",
    "deeplift", "dl",
    "contrastive", "ctr",
}


def _top_family_per_step_for_method(
    method: str,
    model: Any,
    env: Any,
    td: Any,
    *,
    problem: str,
    families: list[tuple[str, tuple[str, ...]]],
    max_steps: int,
) -> torch.Tensor:
    """Return a ``[B, T]`` long tensor of the top-1 constraint family
    per decoding cell, for any of the supported attribution methods.

    The CAX backends (``proxy``, ``lp``, ``subgrad``) go through
    ``lambda_attribution``. Generic methods (``gradient``,
    ``ig``/``integrated_gradients``, ``deeplift``/``dl``,
    ``contrastive``/``ctr``) are run once per constraint family
    with that family's feature keys; the per-family scalar is the
    sum of ``node_scores`` over decoding cells, and we argmax to
    obtain the top family. K families means K backward-pass runs
    per cell.
    """
    method = method.lower()
    if method in {"proxy", "lp", "subgrad"}:
        mult_arg: str | None = None if method == "proxy" else method
        attr = lambda_attribution(
            model, env, td, problem=problem, max_steps=max_steps,
            multipliers=mult_arg,
        )
        return attr.top_family_per_step()

    if method not in _GENERIC_METHOD_KEYS:
        raise ValueError(
            f"Unknown attribution method {method!r}; expected one of "
            f"{{proxy, lp, subgrad, gradient, ig, deeplift, contrastive}}."
        )

    # Lazy imports so the public mirror does not require neuro-co-attr
    # for the CAX-only adjudication path.
    from neuro_co.xai.attribution.gradient import gradient_attribution
    from neuro_co.xai.attribution.ig import integrated_gradients
    from neuro_co.xai.attribution.deeplift import deeplift_attribution
    from neuro_co.xai.attribution.contrastive import contrastive_attribution

    policy = model.policy if hasattr(model, "policy") else model
    available_keys = set(td.keys())

    def run_attr(feature_keys: tuple[str, ...]) -> torch.Tensor:
        # Keep all Tensor-valued keys; the gradient attributor casts
        # int tensors to float32 internally (rl4co's CVRPTW stores
        # time_windows as int but it still contributes to gradient
        # mass once cast). We drop only bool tensors, which crash
        # the downstream env step on float arithmetic.
        present = tuple(
            k for k in feature_keys
            if k in available_keys
            and hasattr(td[k], "dtype")
            and td[k].dtype != torch.bool
        )
        B = int(td.batch_size[0])
        if not present:
            return torch.zeros(B, max_steps)
        try:
            if method == "gradient":
                trace = gradient_attribution(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps,
                )
            elif method in {"ig", "integrated_gradients"}:
                trace = integrated_gradients(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps, ig_steps=8, problem=problem,
                )
            elif method in {"deeplift", "dl"}:
                trace = deeplift_attribution(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps, problem=problem,
                )
            else:  # contrastive / ctr
                trace = contrastive_attribution(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps,
                )
        except RuntimeError as exc:
            # E.g. DeepLIFT can hit a shape mismatch deep in the
            # rescale hook on certain backbones, and the per-family
            # rollout may legitimately produce zero decoding steps
            # when the policy terminates immediately on a single
            # eligible action. Return an all-zero mass tensor so
            # the family-argmax falls back to the first family
            # rather than crashing the entire adjudication.
            print(
                f"[adjudicate] method={method!r} attribution failed "
                f"on feature_keys={present}: {type(exc).__name__}: {exc}; "
                f"falling back to zero scores for this family."
            )
            return torch.zeros(B, max_steps)
        # node_scores: [B, T, N] of per-node mass for this family.
        # Aggregate by summing over the node dimension.
        return trace.node_scores.sum(dim=-1).cpu()  # [B, T]

    per_family_total: list[torch.Tensor] = []
    for _name, keys in families:
        per_family_total.append(run_attr(keys))
    # Pad/truncate to the common T (the shortest run wins).
    T_min = min(t.shape[1] for t in per_family_total)
    per_family_total = [t[:, :T_min] for t in per_family_total]
    stacked = torch.stack(per_family_total, dim=-1)  # [B, T_min, K]
    return stacked.argmax(dim=-1)  # [B, T_min]

# Default per-feature epsilon for the three paper benchmarks.
# Override via `eps_per_problem[problem][feature_key]`.
DEFAULT_EPS_PER_PROBLEM: dict[str, dict[str, float]] = {
    "vrptw": {"locs": 30.0, "demand": 0.02, "time_windows": 30.0, "durations": 0.0},
    # FJSP: multiple eligible machines per op. proc_times shape
    # [B, M, N_ops]; values span 0-99. `num_eligible` controls
    # action flexibility; perturbing it can flip which machine
    # the policy picks, the natural CF signal for FJSP.
    "fjsp": {"proc_times": 20.0, "num_eligible": 1.0},
    # OP: probe showed locs eps=0.3 -> 31/32 flips, prize eps=0.5 -> 32/32,
    # max_length eps=1.0 -> 0 (one-sided shrink; ignored here).
    "op": {"locs": 0.3, "prize": 0.5, "max_length": 0.0},
}


@dataclass
class AdjudicationRow:
    """One row of the aggregate adjudication table."""

    problem: str
    seed: int
    run_dir: str
    mode: str
    mean_agreement: float
    n_flipped_steps: int
    n_total_steps: int


@dataclass
class AdjudicationReport:
    """Per-(run_dir) adjudication breakdown."""

    problem: str
    seed: int
    constraint_names: list[str]
    modes: list[str]
    lambda_top_per_step: dict[str, torch.Tensor]   # mode -> [B, T] long
    cf_flipped_per_step: torch.Tensor               # [B, T] bool
    cf_top_family_per_step: torch.Tensor            # [B, T] long, -1 if no flip
    agreement_per_mode_per_step: dict[str, torch.Tensor]
    mean_agreement_per_mode: dict[str, float] = field(default_factory=dict)


def adjudicate_run(
    run_dir: Path,
    *,
    modes: tuple[str, ...] = ("proxy", "lp", "subgrad"),
    num_instances: int = 4,
    max_steps: int = 4,
    cf_shots: int = 32,
    eps_per_problem: dict[str, dict[str, float]] | None = None,
    seed: int = 0,
    problem: str | None = None,
    feasibility_mode: str = "arithmetic",
    cp_sat_time_limit_s: float = 2.0,
) -> tuple[AdjudicationReport, list[AdjudicationRow]]:
    """Run all Lambda modes + CP-CF on one trained run dir; return report + rows.

    Side-effect: writes `<run_dir>/adjudication.json`.
    """
    cfg = _load_hydra_cfg(run_dir)
    problem = problem or _infer_problem(cfg)
    seed_meta = int(cfg.get("seed", 0))
    eps_table = eps_per_problem or DEFAULT_EPS_PER_PROBLEM
    eps_per_key = eps_table.get(problem.lower(), {})
    families = get_constraints(problem)
    constraint_names = [name for name, _ in families]

    import time as _time
    t0 = _time.time()
    print(f"[adjudicate] {run_dir.name} problem={problem} "
          f"modes={list(modes)} B={num_instances} T={max_steps} M={cf_shots} "
          f"feasibility={feasibility_mode}", flush=True)

    env, model = _instantiate(cfg)
    _load_ckpt(model, _resolve_ckpt(run_dir))
    device = _pick_device()
    model = model.to(device).eval()
    # Seed the instance generator so reruns of adjudicate produce
    # identical instances -- otherwise rl4co's `env.reset` draws a
    # fresh batch and the (b, t) cells aren't comparable across runs.
    torch.manual_seed(int(seed))
    td = env.reset(batch_size=num_instances).to(device)
    print(f"[adjudicate]   model loaded ({_time.time()-t0:.1f}s)", flush=True)

    # ---- Per-method top-family-per-step tensors ----
    lambda_top: dict[str, torch.Tensor] = {}
    for mode in modes:
        t_m = _time.time()
        print(f"[adjudicate]   running attribution method={mode!r} ...", flush=True)
        lambda_top[mode] = _top_family_per_step_for_method(
            mode, model, env, td,
            problem=problem,
            families=families,
            max_steps=max_steps,
        )
        print(f"[adjudicate]   {mode!r} done in {_time.time()-t_m:.1f}s", flush=True)

    # ---- CP-counterfactual ground truth ----
    feature_keys = tuple(eps_per_key.keys()) if eps_per_key else ()
    if not feature_keys:
        raise ValueError(
            f"No eps_per_problem entry for problem={problem!r}; pass "
            f"`eps_per_problem={{problem: {{key: eps, ...}}}}` explicitly."
        )
    t_cf = _time.time()
    print(f"[adjudicate]   running counterfactual search (M={cf_shots} shots, "
          f"feasibility={feasibility_mode}) ...", flush=True)
    cf = cp_counterfactual(
        model, env, td,
        problem=problem,
        feature_keys=feature_keys,
        epsilon=eps_per_key,
        sigma={k: v / 3 for k, v in eps_per_key.items()},
        max_shots=cf_shots,
        max_steps=max_steps,
        seed=seed,
        perturb_one_at_a_time=True,
        feasibility_mode=feasibility_mode,
        time_limit_s=cp_sat_time_limit_s,
    )
    n_flipped = int(cf.flipped.sum().item())
    print(f"[adjudicate]   counterfactual done in {_time.time()-t_cf:.1f}s "
          f"({n_flipped} flipped cells)", flush=True)

    # For each (b, t) where CF flipped, identify the constraint
    # family carrying the largest delta-L1 mass.
    cf_flipped = cf.flipped.cpu()
    cf_top = torch.full(cf_flipped.shape, fill_value=-1, dtype=torch.long)
    fam_to_keys = {name: tuple(keys) for name, keys in families}
    for b in range(cf_flipped.shape[0]):
        for t in range(cf_flipped.shape[1]):
            if not bool(cf_flipped[b, t]):
                continue
            best_fam_idx, best_mass = -1, -1.0
            for k_idx, fam_name in enumerate(constraint_names):
                mass = 0.0
                for fk in fam_to_keys.get(fam_name, ()):
                    if fk in cf.delta:
                        mass += float(cf.delta[fk][t, b].abs().sum().item())
                if mass > best_mass:
                    best_mass, best_fam_idx = mass, k_idx
            cf_top[b, t] = best_fam_idx

    # ---- Agreement per mode per step ----
    agree_per_mode: dict[str, torch.Tensor] = {}
    mean_agree: dict[str, float] = {}
    for mode in modes:
        top = lambda_top[mode].cpu()
        # Trim trace_T (Lambda) vs cf_T to the shared step count.
        T = min(top.shape[1], cf_top.shape[1])
        match = (top[:, :T] == cf_top[:, :T]) & cf_flipped[:, :T]
        agree_per_mode[mode] = match
        n_flips = int(cf_flipped[:, :T].sum().item())
        n_match = int(match.sum().item())
        mean_agree[mode] = float(n_match / n_flips) if n_flips > 0 else 0.0

    report = AdjudicationReport(
        problem=problem,
        seed=seed_meta,
        constraint_names=constraint_names,
        modes=list(modes),
        lambda_top_per_step=lambda_top,
        cf_flipped_per_step=cf_flipped,
        cf_top_family_per_step=cf_top,
        agreement_per_mode_per_step=agree_per_mode,
        mean_agreement_per_mode=mean_agree,
    )

    # Persist.
    out = run_dir / "adjudication.json"
    out.write_text(
        json.dumps(
            {
                "problem": problem,
                "seed": seed_meta,
                "constraint_names": constraint_names,
                "modes": list(modes),
                "lambda_top_family_per_step": {
                    m: lambda_top[m].cpu().tolist() for m in modes
                },
                "cf_flipped_per_step": cf_flipped.int().tolist(),
                "cf_top_family_per_step": cf_top.tolist(),
                "agreement_per_mode_per_step": {
                    m: agree_per_mode[m].int().tolist() for m in modes
                },
                "mean_agreement_per_mode": mean_agree,
                "n_flipped_steps": int(cf_flipped.sum().item()),
                "n_total_steps": int(cf_flipped.numel()),
            },
            indent=2,
        )
    )

    rows = [
        AdjudicationRow(
            problem=problem,
            seed=seed_meta,
            run_dir=str(run_dir),
            mode=m,
            mean_agreement=mean_agree[m],
            n_flipped_steps=int(cf_flipped.sum().item()),
            n_total_steps=int(cf_flipped.numel()),
        )
        for m in modes
    ]
    return report, rows


def adjudicate_runs(
    run_dirs: list[Path],
    *,
    modes: tuple[str, ...] = ("proxy", "lp", "subgrad"),
    num_instances: int = 4,
    max_steps: int = 4,
    cf_shots: int = 32,
    eps_per_problem: dict[str, dict[str, float]] | None = None,
    seed: int = 0,
    out_parquet: Path | None = None,
    feasibility_mode: str = "arithmetic",
    cp_sat_time_limit_s: float = 2.0,
) -> Any:
    """Loop over run dirs, run adjudication, optionally write aggregate parquet."""
    all_rows: list[AdjudicationRow] = []
    failures: list[tuple[Path, str]] = []
    for rd in run_dirs:
        try:
            _, rows = adjudicate_run(
                rd,
                modes=modes,
                num_instances=num_instances,
                max_steps=max_steps,
                cf_shots=cf_shots,
                eps_per_problem=eps_per_problem,
                seed=seed,
                feasibility_mode=feasibility_mode,
                cp_sat_time_limit_s=cp_sat_time_limit_s,
            )
            all_rows.extend(rows)
        except Exception as exc:  # surface, keep going
            failures.append((rd, f"{type(exc).__name__}: {exc}"))
    for rd, msg in failures:
        print(f"[cax-adjudicate] FAIL {rd}: {msg}")

    if out_parquet is not None and all_rows:
        try:
            import pandas as pd

            df = pd.DataFrame([row.__dict__ for row in all_rows])
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_parquet, index=False)
            print(f"[cax-adjudicate] wrote {out_parquet} ({len(df)} rows)")
            return df
        except ImportError:
            print("[cax-adjudicate] pandas not available, skipping parquet")
    return all_rows


def load_pairwise_matches(
    run_dirs: list[Path],
    *,
    modes: tuple[str, ...] = ("proxy", "lp", "subgrad"),
) -> dict[str, list[int]]:
    """Read per-run `adjudication.json` and stack per-mode match vectors.

    Returns `{mode: [match0, match1, ...]}` over every CF-flipped
    (run, b, t). Only CF-flipped cells contribute (where there's a
    ground truth). Used by `bootstrap_ci_diff` and McNemar's test.
    """
    out: dict[str, list[int]] = {m: [] for m in modes}
    out["_problem"] = []  # parallel labels for grouping by problem
    for rd in run_dirs:
        d = json.loads((Path(rd) / "adjudication.json").read_text())
        cf = d["cf_flipped_per_step"]
        for m in modes:
            agree = d["agreement_per_mode_per_step"][m]
            for b in range(len(cf)):
                for t in range(len(cf[b])):
                    if cf[b][t]:
                        out[m].append(int(agree[b][t]))
                        if m == modes[0]:
                            out["_problem"].append(d["problem"])
    return out


def bootstrap_ci_diff(
    a: list[int],
    b: list[int],
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on `mean(a) - mean(b)` for paired binaries.

    Returns `(point_estimate, lo, hi)`. Length-of-a must equal
    length-of-b (paired across the same flipped (run, b, t) cells).
    """
    import random as _random

    if len(a) != len(b):
        raise ValueError(f"paired vectors must match length; got {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = sum(a) / n - sum(b) / n
    rng = _random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        ai = [a[i] for i in idx]
        bi = [b[i] for i in idx]
        diffs.append(sum(ai) / n - sum(bi) / n)
    diffs.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = diffs[int(alpha * n_resamples)]
    hi = diffs[int((1.0 - alpha) * n_resamples)]
    return point, lo, hi


def mcnemar_pvalue(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Exact two-sided binomial test on McNemar discordant cells.

    Returns `(b01, b10, p_value)` where `b01` = "a wrong, b right",
    `b10` = "a right, b wrong". P-value under H0: discordant pairs
    split 50/50. Exact (no chi2 approx) so n can be small.
    """
    from math import comb

    if len(a) != len(b):
        raise ValueError(f"paired vectors must match length; got {len(a)} vs {len(b)}")
    b01 = sum(1 for ai, bi in zip(a, b, strict=False) if ai == 0 and bi == 1)
    b10 = sum(1 for ai, bi in zip(a, b, strict=False) if ai == 1 and bi == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    # Two-sided exact binomial: 2 * P(X <= k)
    p_one = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return b01, b10, float(min(1.0, 2.0 * p_one))


def main(argv: list[str] | None = None) -> int:
    """`neuroco-cax-adjudicate` CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco-cax-adjudicate")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--modes", nargs="+", default=["proxy", "lp", "subgrad"])
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--cf-shots", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    adjudicate_runs(
        run_dirs=args.run_dirs,
        modes=tuple(args.modes),
        num_instances=args.num_instances,
        max_steps=args.max_steps,
        cf_shots=args.cf_shots,
        out_parquet=args.out,
    )
    return 0


_ = PROBLEM_CONSTRAINTS  # imported for the docstring's family-name reference.
