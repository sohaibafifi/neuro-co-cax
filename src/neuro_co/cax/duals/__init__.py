"""Lagrangian-multiplier estimators for `lambda_attribution`.

Two backends, both producing the same dict shape
`{constraint_family: lambda_c}`:

  - `lp`, solve the problem's LP relaxation via OR-Tools GLOP,
    extract `constraint.dual_value()` per constraint, aggregate
    by family. Tight when the LP relaxation has a small integrality
    gap; **the v0.2 reference** for paper-cax §3.1.

  - `subgrad`, Beasley-style Lagrangian subgradient ascent on a
    relaxed subproblem. Doesn't need an LP solver; works with any
    combinatorial relaxation. Useful where the LP encoding is
    expensive (set-partitioning needs column-generation) or
    unavailable.

`get_multipliers(problem, instance, method=...)` dispatches; the
caller (`lambda_attribution(multipliers="lp" | "subgrad")`) picks
the backend. Paper-cax §4.X compares the two heads-up to test
whether the simpler subgradient is enough for the attribution
ranking to stabilise.

v0.2 ships VRPTW for both backends; JSSP/FJSP/OP/PDP are stubs.
"""

from __future__ import annotations

from typing import Any


def get_multipliers(
    problem: str,
    instance: dict[str, Any],
    *,
    method: str = "lp",
    **kwargs: Any,
) -> dict[str, float]:
    """Dispatch to the per-problem dual extractor.

    Parameters
    ----------
    problem
        Problem name (`'vrptw'`, `'fjsp'`, `'op'`).
    instance
        Flat dict of the instance fields the backend needs
        (`'locs'`, `'demand'`, `'time_windows'`, ...). Usually
        built from `td_to_instance(td, problem)` upstream.
    method
        `'lp'` (GLOP backend, default) or `'subgrad'` (Beasley).
    **kwargs
        Forwarded to the backend (`time_limit_s`, `max_iters`,
        `step_size`, ...).

    Returns
    -------
    dict[str, float]
        Constraint-family name -> Lagrangian multiplier (absolute
        value). Order matches `PROBLEM_CONSTRAINTS[problem]`.

    Raises
    ------
    NotImplementedError
        If the (problem, method) pair isn't wired yet.
    """
    key = (problem.lower(), method.lower())
    if key == ("vrptw", "lp"):
        from neuro_co.cax.duals.vrptw_lp import vrptw_lp_duals

        return vrptw_lp_duals(instance, **kwargs)
    if key == ("vrptw", "subgrad"):
        from neuro_co.cax.duals.vrptw_subgrad import vrptw_subgrad_duals

        return vrptw_subgrad_duals(instance, **kwargs)
    if key == ("op", "lp"):
        from neuro_co.cax.duals.op_lp import op_lp_duals

        return op_lp_duals(instance, **kwargs)
    if key == ("op", "subgrad"):
        from neuro_co.cax.duals.op_subgrad import op_subgrad_duals

        return op_subgrad_duals(instance, **kwargs)
    if key == ("fjsp", "lp"):
        from neuro_co.cax.duals.fjsp_lp import fjsp_lp_duals

        return fjsp_lp_duals(instance, **kwargs)
    if key == ("fjsp", "subgrad"):
        from neuro_co.cax.duals.fjsp_subgrad import fjsp_subgrad_duals

        return fjsp_subgrad_duals(instance, **kwargs)
    raise NotImplementedError(
        f"No multiplier estimator for problem={problem!r}, method={method!r}. "
        f"Paper-cax v1 covers VRPTW (lp, subgrad), OP (lp, subgrad), FJSP (lp, subgrad)."
    )


def td_to_instance(td: Any, problem: str, batch_idx: int = 0) -> dict[str, Any]:
    """Pull a single instance out of a batched TensorDict.

    The LP / subgrad encoders are per-instance; batch-wise multiplier
    extraction is a v0.3 thing. v0.2 takes the `batch_idx`-th
    instance and assumes its multipliers transfer to siblings
    (reasonable when instances are i.i.d. samples from the same
    generator).
    """
    keys_per_problem = {
        "vrptw": ("locs", "demand", "time_windows", "durations", "vehicle_capacity"),
        "fjsp": (
            "proc_times",
            "num_eligible",
            "ops_ma_adj",
            "ops_job_map",
            "ops_sequence_order",
            "pad_mask",
        ),
        "op": ("locs", "prize", "max_length"),
    }
    keys = keys_per_problem.get(problem.lower(), ())
    out: dict[str, Any] = {}
    for k in keys:
        if k not in td:
            continue
        v = td[k]
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        # Slice into batch_idx (single instance).
        if hasattr(v, "ndim") and v.ndim > 0 and v.shape[0] > batch_idx:
            out[k] = v[batch_idx]
        else:
            out[k] = v
    return out


__all__ = ["get_multipliers", "td_to_instance"]
