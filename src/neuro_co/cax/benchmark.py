"""Benchmark Lambda-attribution across one or more run directories.

Each run dir is one `(problem, seed)` triple produced by
`neuroco-train`. We load its Hydra snapshot, re-instantiate
env + model, restore the checkpoint, sample a small batch of
fresh instances, run `lambda_attribution`, and dump:

    <run_dir>/lambda_attribution.json
    <run_dir>/lambda_attribution.parquet   (optional, --parquet)

JSON layout::

    {
        "problem": "vrptw",
        "num_instances": 8,
        "max_steps": 16,
        "constraint_names": ["capacity", "time_window", "spatial"],
        "feature_keys_per_family": [["demand"], [...], [...]],
        "mean_scores": [...],   # [K] mean over (B, T)
        "per_step_scores": [...],  # [T, K] mean over B
        "top_family_per_step": [[k, k, k, ...]] # [B, T]
    }

The aggregate parquet (across all run dirs) is produced by the
companion `neuroco-cax-benchmark` CLI when given multiple run
dirs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuro_co.cax.lambda_attribution import LambdaAttribution, lambda_attribution


@dataclass
class BenchmarkRow:
    """One row of the aggregate benchmark table."""

    problem: str
    seed: int
    run_dir: str
    constraint: str
    mean_score: float
    num_instances: int
    num_steps: int


def benchmark_run(
    run_dir: Path,
    *,
    num_instances: int = 8,
    max_steps: int | None = 16,
    problem: str | None = None,
    multipliers: str | None = None,
) -> tuple[LambdaAttribution, list[BenchmarkRow]]:
    """Run Lambda-attribution on a single run dir; return (trace, rows).

    Side-effect: writes `lambda_attribution.json` under `run_dir`.

    `multipliers` is forwarded to `lambda_attribution` (None=proxy,
    `'lp'`, `'subgrad'`, or a caller-supplied dict).
    """
    cfg = _load_hydra_cfg(run_dir)
    problem = problem or _infer_problem(cfg)
    seed = int(cfg.get("seed", 0))

    env, model = _instantiate(cfg)
    ckpt = _resolve_ckpt(run_dir)
    _load_ckpt(model, ckpt)

    device = _pick_device()
    model = model.to(device).eval()
    td = env.reset(batch_size=num_instances).to(device)

    trace = lambda_attribution(
        model,
        env,
        td,
        problem=problem,
        max_steps=max_steps,
        multipliers=multipliers,
    )

    rows = [
        BenchmarkRow(
            problem=problem,
            seed=seed,
            run_dir=str(run_dir),
            constraint=name,
            mean_score=float(trace.scores[..., k_idx].mean().item()),
            num_instances=trace.batch_size,
            num_steps=trace.num_steps,
        )
        for k_idx, name in enumerate(trace.constraint_names)
    ]

    suffix = f"_{multipliers}" if multipliers else ""
    out = run_dir / f"lambda_attribution{suffix}.json"
    out.write_text(
        json.dumps(
            {
                "problem": problem,
                "seed": seed,
                "multipliers_mode": multipliers or "proxy",
                "multipliers_values": (
                    trace.multipliers.tolist() if trace.multipliers is not None else None
                ),
                "num_instances": trace.batch_size,
                "max_steps": trace.num_steps,
                "constraint_names": trace.constraint_names,
                "feature_keys_per_family": [
                    list(keys) for keys in trace.feature_keys_per_family
                ],
                "mean_scores": [
                    float(trace.scores[..., k_idx].mean().item())
                    for k_idx in range(trace.num_families)
                ],
                "per_step_scores": trace.scores.mean(dim=0).tolist(),
                "top_family_per_step": trace.top_family_per_step().tolist(),
            },
            indent=2,
        )
    )
    return trace, rows


def benchmark_runs(
    run_dirs: list[Path],
    *,
    num_instances: int = 8,
    max_steps: int | None = 16,
    out_parquet: Path | None = None,
    multipliers: str | None = None,
) -> Any:
    """Loop over run dirs, run benchmark, optionally write aggregate parquet."""
    all_rows: list[BenchmarkRow] = []
    failures: list[tuple[Path, str]] = []
    for rd in run_dirs:
        try:
            _trace, rows = benchmark_run(
                rd,
                num_instances=num_instances,
                max_steps=max_steps,
                multipliers=multipliers,
            )
            all_rows.extend(rows)
        except Exception as exc:  # surface per-run, keep going
            failures.append((rd, f"{type(exc).__name__}: {exc}"))

    if failures:
        for rd, msg in failures:
            print(f"[cax-benchmark] FAIL {rd}: {msg}")

    if out_parquet is not None and all_rows:
        try:
            import pandas as pd

            df = pd.DataFrame([row.__dict__ for row in all_rows])
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_parquet, index=False)
            print(f"[cax-benchmark] wrote {out_parquet} ({len(df)} rows)")
            return df
        except ImportError:
            print("[cax-benchmark] pandas not available, skipping parquet")

    return all_rows


# ---------------------------------------------------------------------------
# Hydra loading helpers (mirror probes_cli pattern).
# ---------------------------------------------------------------------------


def _load_hydra_cfg(run_dir: Path) -> Any:
    from omegaconf import OmegaConf

    p = Path(run_dir) / ".hydra" / "config.yaml"
    if not p.is_file():
        raise FileNotFoundError(
            f"No .hydra/config.yaml under {run_dir}; pass a directory "
            "produced by a previous neuroco-train run."
        )
    return OmegaConf.load(str(p))


def _instantiate(cfg: Any) -> tuple[Any, Any]:
    from hydra.utils import instantiate

    env = instantiate(cfg.env)
    model = instantiate(cfg.model, env=env)
    return env, model


def _resolve_ckpt(run_dir: Path, override: str | None = None) -> str:
    if override:
        return str(override)
    cands = sorted(Path(run_dir).glob("checkpoints/*.ckpt"), key=lambda x: x.stat().st_mtime)
    if not cands:
        cands = sorted(
            Path(run_dir).rglob("checkpoints/last.ckpt"),
            key=lambda x: x.stat().st_mtime,
        )
    if not cands:
        raise FileNotFoundError(f"No checkpoint under {run_dir}/checkpoints/")
    return str(cands[-1])


def _load_ckpt(model: Any, ckpt_path: str) -> None:
    from neuro_co.xai.attribution import drop_baseline_keys

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = drop_baseline_keys(state)
    model.load_state_dict(state, strict=False)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_problem(cfg: Any) -> str:
    p = cfg.get("problem")
    if p:
        return str(p).lower()
    target = str(cfg.env.get("_target_", ""))
    # e.g. "rl4co.envs.routing.cvrptw.env.CVRPTWEnv" -> "vrptw"
    for k in ("vrptw", "cvrptw", "fjsp", "op"):
        if k in target.lower():
            return "vrptw" if k == "cvrptw" else k
    raise ValueError(f"Cannot infer problem name from cfg.env._target_={target!r}")
