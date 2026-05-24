"""CVRPTW CSP feasibility-decision oracle (M3.5).

Stronger than the arithmetic check in `vrptw.py`. Answers the
question: "Does *any* feasible routing exist for instance
$x + \\zeta$?" by issuing a CSP feasibility-decision query
(no objective, stop at first solution) through
`neuro_co.problems.vrptw.cpsat.solve_cvrptw_cpsat`.

Acceptance is a constructive certificate: a feasibility-respecting
route demonstrably exists, and the perturbed instance is
policy-realisable. Failure within `time_limit_s` (timeout, no
solution) is treated as rejection rather than a certificate of
infeasibility, so the certified subset is conservative.

A CSP feasibility solve is far cheaper than a full CSP/COP
optimisation pass: the engine can stop at the first feasible
assignment instead of proving optimality. The check fires once
per surviving counterfactual candidate, so bulk searches should
pre-filter with the arithmetic mode and invoke this oracle only
on the per-cell winner.
"""

from __future__ import annotations

from typing import Any

import torch


def vrptw_cp_is_feasible(td: Any, *, time_limit_s: float = 1.0) -> torch.Tensor:
    """Per-batch CSP feasibility-decision check. Returns `[B]` bool tensor.

    Falls back to the arithmetic check
    (`neuro_co.cax.feasibility.vrptw.vrptw_is_feasible`) when the
    `ortools` extra is not installed; the pipeline still runs, just
    without the constructive feasibility certificate.
    """
    from neuro_co.cax.feasibility.vrptw import vrptw_is_feasible

    # Step 1: cheap arithmetic pre-filter. Any element failing here
    # is infeasible at the structural level; no need to invoke CSP.
    arithmetic_ok = vrptw_is_feasible(td)
    if not arithmetic_ok.any():
        return arithmetic_ok

    # Step 2: per-batch element CSP feasibility-only decision. The
    # solver runs in feasibility mode (no objective, halt at the
    # first feasible assignment); this is orders of magnitude cheaper
    # than the optimisation pass that a full CSP/COP solve would do,
    # and is the right semantics for "does a feasible routing exist
    # for x+zeta?". One call per surviving counterfactual candidate.
    try:
        from neuro_co.problems.vrptw.cpsat import solve_cvrptw_cpsat
    except ImportError:  # pragma: no cover - ortools is an optional extra
        return arithmetic_ok

    B = int(td.batch_size[0])
    cp_ok = arithmetic_ok.clone()
    for b in range(B):
        if not bool(arithmetic_ok[b]):
            continue
        try:
            single = _slice_batch(td, b)
            routes, _cost = solve_cvrptw_cpsat(
                single,
                max_runtime=float(time_limit_s),
                feasibility_only=True,
            )
            cp_ok[b] = bool(routes)   # non-empty route list = feasible
        except Exception:
            cp_ok[b] = False
    return cp_ok


def _slice_batch(td: Any, idx: int) -> Any:
    """Build a 1-element TensorDict view at batch index `idx`."""
    sub = td.clone(recurse=False)
    for k in list(sub.keys()):
        v = sub[k]
        if hasattr(v, "ndim") and v.ndim > 0 and v.shape[0] > idx:
            sub[k] = v[idx : idx + 1]
    sub.batch_size = (1,)
    return sub
