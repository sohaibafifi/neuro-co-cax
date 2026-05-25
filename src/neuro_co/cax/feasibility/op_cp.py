"""OP CP/SAT decision feasibility check (M3.5).

Upgrades the arithmetic check in `op.py` from "structural sanity"
to "does there exist a feasible OP tour for this instance?" via
the OR-Tools CP-SAT model registered as
`BASELINE_SOLVERS[("op", "ortools")]` in
`neuro_co.problems.op.ortools`.

The CP-SAT model returns an optimal prize-maximising tour if any
budget-respecting tour exists. We treat its returned status as
the feasibility certificate: `cost` is non-NaN  =>  a feasible
tour exists.

This is slow (1-3 s / instance on N=20-50). The CP-SAT mode is
only invoked for instances that already pass the arithmetic
check, as a stage-2 filter in `cp_counterfactual`.
"""

from __future__ import annotations

from math import isnan
from typing import Any

import torch


def op_cp_is_feasible(td: Any, *, time_limit_s: float = 2.0) -> torch.Tensor:
    """Per-batch CP-SAT feasibility for OP. Returns `[B]` bool.

    The underlying solver is called with `feasibility_only=True`: the
    prize-maximising objective is dropped and CP-SAT halts at the
    first feasible budget-respecting tour. This converts a slow COP
    pass into a CSP feasibility-decision and matches the VRPTW path.

    Stage-1 arithmetic check still pre-filters infeasible structural
    cases (negative budget, malformed prize tensor) before the CSP
    call fires.
    """
    from neuro_co.problems import BASELINE_SOLVERS, load_plugins

    load_plugins()  # ensure problem plug-ins register their solvers
    key = ("op", "ortools")
    from neuro_co.cax.feasibility.op import op_is_feasible

    # Stage 1: cheap arithmetic pre-filter. Any element failing here
    # is infeasible at the structural level; no need to invoke CSP.
    arithmetic_ok = op_is_feasible(td)
    if key not in BASELINE_SOLVERS or not arithmetic_ok.any():
        return arithmetic_ok

    solver = BASELINE_SOLVERS[key]
    B = int(td.batch_size[0])
    ok = arithmetic_ok.clone()
    for b in range(B):
        if not bool(arithmetic_ok[b]):
            continue
        instance = {
            "locs": td["locs"][b],
            "prize": td["prize"][b],
            "max_length": td["max_length"][b]
            if "max_length" in td
            else torch.tensor(2.0),
        }
        try:
            route, cost = solver(
                instance,
                max_runtime=time_limit_s,
                feasibility_only=True,
            )
            # `feasibility_only` returns a non-empty route + cost=0.0
            # on feasibility, empty + nan on infeasibility / timeout.
            ok[b] = bool(route) and not (
                cost is None or (isinstance(cost, float) and isnan(cost))
            )
        except Exception:
            ok[b] = False
    return ok
