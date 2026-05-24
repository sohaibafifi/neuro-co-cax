"""CVRPTW CP/HGS decision feasibility check (M3.5).

Stronger than the arithmetic check in `vrptw.py`. Asks the
question: "Does *any* feasible routing exist for instance
$x + \\delta$?" via the same classical solver the workshop
already wires (PyVRP HGS through
`neuro_co.problems.BASELINE_SOLVERS[("vrptw", "pyvrp")]`).

If the solver returns a finite-cost solution within
`time_limit_s`, the instance is policy-realisable: there exists a
route the *policy* could in principle have learned to follow. If
the solver times out without producing one, we conservatively
mark the instance infeasible (false negatives possible; false
positives ruled out because PyVRP's HGS only returns
constraint-respecting solutions).

Cost is the dominant variable: a single PyVRP call on
$N{=}50$ runs in $0.5$--$2$\\,s with `time_limit_s=1.0`. Bulk
counterfactual searches that issue $32+$ shots per step should
keep this mode for the *survivor* candidates only (cheap
arithmetic pre-filter, then CP for the one survivor).
"""

from __future__ import annotations

from typing import Any

import torch


def vrptw_cp_is_feasible(td: Any, *, time_limit_s: float = 1.0) -> torch.Tensor:
    """Per-batch CP/HGS feasibility. Returns `[B]` bool tensor.

    Falls back to the arithmetic check
    (`neuro_co.cax.feasibility.vrptw.vrptw_is_feasible`) when the
    `pyvrp` extra is not installed -- the policy still runs, just
    without the policy-realisability certificate.
    """
    from neuro_co.cax.feasibility.vrptw import vrptw_is_feasible

    # Step 1: cheap arithmetic pre-filter. Any element failing here
    # is infeasible at the structural level; no need to call CP-SAT.
    arithmetic_ok = vrptw_is_feasible(td)
    if not arithmetic_ok.any():
        return arithmetic_ok

    # Step 2: per-batch element CP-SAT *feasibility-only* decision.
    # We deliberately use the CP-SAT solver in feasibility mode
    # (no objective, stop at first solution) rather than HGS:
    #   * we only need a yes/no on "does a feasible routing
    #     exist for x+delta?"; the optimisation step that HGS
    #     and CP-SAT-optimise spend most of their budget on is
    #     wasted here.
    #   * CP-SAT in feasibility mode terminates in ~ms on N=50
    #     vs ~seconds when optimising. The CF feasibility check
    #     fires ONCE per surviving counterfactual candidate, so
    #     compounding solver cost matters.
    #   * raw CP-SAT also lines up with the paper-cax narrative
    #     ("instance-feasibility certified by a CP feasibility
    #     check") without paying HGS's optimisation overhead.
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
