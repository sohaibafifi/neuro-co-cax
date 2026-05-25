"""Per-feature baseline builders for gradient-axiomatic attribution.

Integrated Gradients and DeepLIFT both require a reference baseline
$x'$ against which to integrate / propagate. A zero-tensor baseline
is the textbook default but is *not* semantically valid for many
neural-CO inputs: zero coordinates land on the unit-square origin
(a corner, not a "neutral" location); zero `time_windows` invert
the [open, close] interval ordering; zero `proc_times` on an
ineligible (machine, op) cell is the rl4co encoding convention for
"this op cannot run here", so a zero-baseline on already-zero cells
contributes nothing and on eligible cells re-encodes the op as
non-eligible.

This module ships six baseline modes that we found semantically
valid in practice (ported from `ai4co-gnn/xai/engine/ig_attribution.py`
and extended with two FJSP / VRPTW-specific variants):

    zero-all                          x' = 0 everywhere
    mean-fill                         x' = mean(x along leading axis)
    zero-with-customers-at-depot      locs only: all customers -> depot coord
    zero-with-current-locs            locs only: identity (no perturbation)
    mean-fill-on-eligible             proc_times only: mean over eligibility-mask
    wide-window                       time_windows only: (0, planning_horizon)

`build_feature_baseline(td, feature_name, mode, problem)` is the
public entry; per-problem-per-feature defaults are locked in
`neuro_co.xai.defaults.DEFAULT_IG_BASELINE_PER_PROBLEM` (populated
by the sweep driver in `neuro_co.xai.baseline_sweep`).
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor


IG_BASELINE_MODES: tuple[str, ...] = (
    "zero-all",
    "mean-fill",
    "zero-with-customers-at-depot",
    "zero-with-current-locs",
    "mean-fill-on-eligible",
    "wide-window",
)


def _clone(value: Tensor) -> Tensor:
    return value.detach().clone()


def build_feature_baseline(
    td: TensorDict | Any,
    feature_name: str,
    mode: str,
    *,
    problem: str | None = None,
    planning_horizon: float = 1.0,
) -> Tensor:
    """Construct a baseline tensor for `feature_name` from `td`.

    Parameters
    ----------
    td : TensorDict
        Current decoding state. Must contain `feature_name`.
    feature_name : str
        Key to build the baseline for (e.g. `"locs"`, `"demand"`,
        `"proc_times"`, `"time_windows"`).
    mode : str
        One of `IG_BASELINE_MODES`.
    problem : str, optional
        Problem name; used by modes that need problem-specific
        structure (currently `mean-fill-on-eligible` for FJSP).
    planning_horizon : float
        Used by `wide-window` to set the closing time of each TW.

    Raises
    ------
    KeyError if `feature_name` is missing.
    ValueError if `mode` is unknown.
    ValueError if `mode` is incompatible with `feature_name`
        (e.g. `zero-with-customers-at-depot` on non-`locs`).
    """
    value = td.get(feature_name) if hasattr(td, "get") else td[feature_name]
    if not isinstance(value, Tensor):
        raise KeyError(f"Feature {feature_name!r} not a Tensor in td")

    mode = str(mode).strip()
    if mode not in IG_BASELINE_MODES:
        raise ValueError(
            f"Unknown baseline {mode!r}. Expected one of: {', '.join(IG_BASELINE_MODES)}"
        )

    if mode == "zero-all":
        return torch.zeros_like(value)

    if mode == "mean-fill":
        # Mean over the leading axis (batch); broadcast to original shape.
        # Cast through float so integer-dtyped features (e.g. rl4co's
        # time_windows on some envs) survive the reduction.
        v = value.detach().float()
        mean_val = v.mean(dim=0, keepdim=True)
        return mean_val.expand_as(v).clone().to(value.dtype)

    if mode == "zero-with-customers-at-depot":
        if feature_name != "locs":
            raise ValueError(
                "zero-with-customers-at-depot is locs-only; "
                f"got feature_name={feature_name!r}"
            )
        # value: [B, N, 2]; depot is node 0.
        depot = value[:, :1, :].detach()
        return depot.expand_as(value).clone()

    if mode == "zero-with-current-locs":
        if feature_name != "locs":
            raise ValueError(
                "zero-with-current-locs is locs-only; "
                f"got feature_name={feature_name!r}"
            )
        return _clone(value)

    if mode == "mean-fill-on-eligible":
        # For FJSP `proc_times` shape [B, M, N_ops]: take the mean
        # across eligible (machine, op) cells (where `ops_ma_adj==1`)
        # and broadcast to the full tensor on eligible cells only,
        # leaving ineligible cells at zero (rl4co encoding).
        if feature_name != "proc_times":
            raise ValueError(
                "mean-fill-on-eligible is proc_times-only; "
                f"got feature_name={feature_name!r}"
            )
        adj_key = "ops_ma_adj"
        adj = td.get(adj_key) if hasattr(td, "get") else td[adj_key]
        if not isinstance(adj, Tensor) or adj.shape != value.shape:
            raise ValueError(
                f"mean-fill-on-eligible requires {adj_key!r} with shape "
                f"{tuple(value.shape)}; got shape {tuple(adj.shape) if isinstance(adj, Tensor) else None}"
            )
        mask = adj.detach() > 0.5
        # Per-batch mean of value over its eligible cells.
        v = value.detach()
        # Avoid div by zero for batches with no eligible cell (degenerate).
        mask_f = mask.float()
        denom = mask_f.flatten(1).sum(dim=-1).clamp(min=1.0)  # [B]
        num = (v * mask_f).flatten(1).sum(dim=-1)             # [B]
        per_batch_mean = (num / denom)[:, None, None]        # [B, 1, 1]
        base = torch.zeros_like(v)
        base = torch.where(mask, per_batch_mean.expand_as(v), base)
        return base

    if mode == "wide-window":
        if feature_name != "time_windows":
            raise ValueError(
                "wide-window is time_windows-only; "
                f"got feature_name={feature_name!r}"
            )
        # value: [B, N, 2] with last dim = (open, close).
        base = torch.zeros_like(value)
        base[..., 1] = float(planning_horizon)
        return base

    raise AssertionError(f"unreachable; mode={mode!r}")


def baseline_dict_for_problem(
    td: TensorDict | Any,
    feature_keys: tuple[str, ...],
    *,
    problem: str,
    overrides: dict[str, str] | None = None,
    planning_horizon: float = 1.0,
) -> dict[str, Tensor]:
    """Build per-feature baselines for one decoding state.

    Reads `neuro_co.xai.defaults.DEFAULT_IG_BASELINE_PER_PROBLEM`
    for default per-feature picks; the optional `overrides` dict
    replaces individual entries (used by the sweep driver).

    Skips feature keys not listed in defaults (e.g. integer scalars
    like `max_length`, `to_choose`, `num_eligible`) silently; those
    keys are excluded from IG attribution entirely.
    """
    from neuro_co.xai.defaults import DEFAULT_IG_BASELINE_PER_PROBLEM

    per_problem = DEFAULT_IG_BASELINE_PER_PROBLEM.get(problem.lower(), {})
    if overrides:
        per_problem = {**per_problem, **overrides}

    out: dict[str, Tensor] = {}
    for key in feature_keys:
        if key not in per_problem:
            continue  # not in default map -> exclude from IG path
        mode = per_problem[key]
        out[key] = build_feature_baseline(
            td, key, mode, problem=problem, planning_horizon=planning_horizon
        )
    return out
