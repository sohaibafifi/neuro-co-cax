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
    """Per-batch CP-SAT feasibility for OP. Returns `[B]` bool."""
    from neuro_co.problems import BASELINE_SOLVERS, load_plugins

    load_plugins()  # ensure problem plug-ins register their solvers
    key = ("op", "ortools")
    if key not in BASELINE_SOLVERS:
        # OR-Tools extra missing -- fall back to arithmetic.
        from neuro_co.cax.feasibility.op import op_is_feasible

        return op_is_feasible(td)

    solver = BASELINE_SOLVERS[key]
    B = int(td.batch_size[0])
    ok = torch.zeros(B, dtype=torch.bool)
    for b in range(B):
        instance = {
            "locs": td["locs"][b],
            "prize": td["prize"][b],
            "max_length": td["max_length"][b]
            if "max_length" in td
            else torch.tensor(2.0),
        }
        try:
            _route, cost = solver(instance, max_runtime=time_limit_s)
            ok[b] = not (cost is None or (isinstance(cost, float) and isnan(cost)))
        except Exception:
            ok[b] = False
    return ok
