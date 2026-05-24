"""FJSP CP/SAT decision feasibility check (M3.5).

Upgrades the arithmetic check in `fjsp.py` from "structural
sanity" to "does there exist a feasible FJSP schedule for this
instance?" via the OR-Tools CP-SAT model registered as
`BASELINE_SOLVERS[("fjsp", "ortools")]` in
`neuro_co.problems.fjsp.ortools`.

FJSP is always schedulable if every real op has at least one
eligible machine and processing times are non-negative -- the
arithmetic check already enforces this. The CP-SAT decision
upgrade adds a *bounded-makespan* certificate: it returns a
schedule whose makespan is finite, which rules out the pathological
case where the OR-Tools solver hits the time limit without finding
an incumbent (the arithmetic check would still pass).

Slow (~2-3 s / instance for N_jobs=10, M=5). Use the arithmetic
mode as a pre-filter; this routine is the stage-2 confirmation
when `feasibility_mode='cp_sat'` is requested.
"""

from __future__ import annotations

from math import isnan
from typing import Any

import torch


def fjsp_cp_is_feasible(td: Any, *, time_limit_s: float = 3.0) -> torch.Tensor:
    """Per-batch CP-SAT feasibility for FJSP. Returns `[B]` bool."""
    from neuro_co.problems import BASELINE_SOLVERS, load_plugins

    load_plugins()  # ensure problem plug-ins register their solvers
    key = ("fjsp", "ortools")
    if key not in BASELINE_SOLVERS:
        from neuro_co.cax.feasibility.fjsp import fjsp_is_feasible

        return fjsp_is_feasible(td)

    solver = BASELINE_SOLVERS[key]
    B = int(td.batch_size[0])
    ok = torch.zeros(B, dtype=torch.bool)
    for b in range(B):
        instance = {
            k: td[k][b]
            for k in (
                "proc_times",
                "ops_ma_adj",
                "ops_job_map",
                "ops_sequence_order",
                "num_eligible",
                "pad_mask",
            )
            if k in td
        }
        try:
            _schedule, cost = solver(instance, max_runtime=time_limit_s)
            ok[b] = not (cost is None or (isinstance(cost, float) and isnan(cost)))
        except Exception:
            ok[b] = False
    return ok
