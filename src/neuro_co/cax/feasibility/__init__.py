"""Per-problem CO instance-feasibility checks for cp_counterfactual.

Each problem registers `is_feasible(td_batch) -> Tensor[bool]`
(shape `[B]`) that says, per batch element, whether the *instance*
satisfies the problem's structural constraints (non-negative
demand, non-empty TWs, positive capacity, non-negative processing
times, ...).

v0.3 (M3) backends are arithmetic — they verify field-value
sanity directly. v0.4 (M3.5) will swap in CP-SAT decision queries
("does there exist a feasible routing for this instance?") via
the existing `BASELINE_SOLVERS` registry. Arithmetic checks are
necessary but not sufficient -- they catch the trivial
infeasibilities a Gaussian perturbation would otherwise produce
(negative demand, inverted TWs, etc.) which is the common case
during counterfactual search.
"""

from __future__ import annotations

from typing import Any

import torch


def is_feasible(
    td: Any,
    problem: str,
    *,
    mode: str = "arithmetic",
    time_limit_s: float = 1.0,
) -> torch.Tensor:
    """Per-batch feasibility check. Returns `[B]` bool tensor.

    Parameters
    ----------
    mode
        `"arithmetic"` (M3, default) -- cheap field-value checks
        (demand >= 0, TW ordered, capacity > 0, ...). Fast
        (~0.1 ms / batch).
        `"cp_sat"` (M3.5) -- routes the candidate through the
        classical solver in `neuro_co.problems.BASELINE_SOLVERS`
        and accepts only instances for which a feasible routing
        / schedule exists. Slower (~0.5--2 s per batch element
        for VRPTW); use the arithmetic mode as a pre-filter.
    time_limit_s
        Per-instance solver budget when `mode="cp_sat"`. Ignored
        when `mode="arithmetic"`.
    """
    key = problem.lower()
    if mode not in ("arithmetic", "cp_sat"):
        raise ValueError(f"mode must be 'arithmetic' or 'cp_sat'; got {mode!r}")
    if key == "vrptw":
        if mode == "arithmetic":
            from neuro_co.cax.feasibility.vrptw import vrptw_is_feasible

            return vrptw_is_feasible(td)
        from neuro_co.cax.feasibility.vrptw_cp import vrptw_cp_is_feasible

        return vrptw_cp_is_feasible(td, time_limit_s=time_limit_s)
    if key == "op":
        if mode == "arithmetic":
            from neuro_co.cax.feasibility.op import op_is_feasible

            return op_is_feasible(td)
        from neuro_co.cax.feasibility.op_cp import op_cp_is_feasible

        return op_cp_is_feasible(td, time_limit_s=time_limit_s)
    if key == "fjsp":
        if mode == "arithmetic":
            from neuro_co.cax.feasibility.fjsp import fjsp_is_feasible

            return fjsp_is_feasible(td)
        from neuro_co.cax.feasibility.fjsp_cp import fjsp_cp_is_feasible

        return fjsp_cp_is_feasible(td, time_limit_s=time_limit_s)
    # Unknown problem -> conservatively assume feasible. Reported as
    # a known limitation in cp_counterfactual's report.method field.
    B = int(td.batch_size[0]) if hasattr(td, "batch_size") else 1
    return torch.ones(B, dtype=torch.bool)


__all__ = ["is_feasible"]
