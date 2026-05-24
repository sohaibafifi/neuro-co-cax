"""FJSP (Flexible Job-Shop Scheduling) LP-relaxation dual extractor.

Encoding: machine-flexible job-shop with a time-indexed continuous
relaxation. The classical disjunctive (machine non-overlap)
constraints are NP-hard inside an LP; we drop them, leaving the
*two* CAX-relevant constraint families intact:

  - `eligibility` : each op must run on exactly one of its eligible
    machines (sum_m x_{o,m} == 1 over the eligible set).
  - `precedence`  : intra-job ordering -- the completion time of op
    o+1 in the same job is at least the completion time of op o
    plus the (assignment-weighted) processing time of op o+1.

The LP relaxation is loose (no machine-resource pressure), so the
*absolute* C_max it returns is meaningless; the *relative* per-family
dual mass remains informative, which is exactly what
`lambda_attribution` consumes (same pattern as `vrptw_lp.py` /
`op_lp.py`).

Variables (LP-relaxed):
  x[o, m] in [0, 1]   forall (o, m)         (op assignment fraction)
  c[o]    in [0, T]   forall o              (completion time of op o)
  Cmax    in [0, T]                         (makespan upper bound)

Objective:
  minimise  Cmax

Constraints (by family, surfaced as duals):
  eligibility : sum_{m: eligible(o, m)} x[o, m] == 1   forall o
  precedence  : c[o_next] >= c[o] + sum_m p[o_next, m] * x[o_next, m]
                                       for consecutive (o, o_next) in same job
                c[o_first] >= sum_m p[o_first, m] * x[o_first, m]   (release)
                Cmax >= c[o_last]   for each last op of every job
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from neuro_co.cax.duals.lp import solve_lp_and_extract_duals


def fjsp_lp_duals(
    instance: dict[str, Any],
    *,
    time_limit_s: float = 10.0,
    aggregate: str = "mean",
) -> dict[str, float]:
    """Solve FJSP LP relaxation, return per-family mean |dual|."""
    # `proc_times` shape [M, N_ops]; `ops_ma_adj` shape [M, N_ops] (bool).
    proc = np.asarray(instance["proc_times"], dtype=float)
    if proc.ndim != 2:
        return {"eligibility": 0.0, "precedence": 0.0}
    M, N_ops = proc.shape
    if N_ops < 2 or M < 1:
        return {"eligibility": 0.0, "precedence": 0.0}

    # Eligibility mask: prefer `ops_ma_adj` if present, else infer from
    # nonzero `proc_times` (rl4co convention: ineligible -> p = 0).
    if "ops_ma_adj" in instance:
        elig = np.asarray(instance["ops_ma_adj"], dtype=float) > 0.5
    else:
        elig = proc > 0.0

    ops_job_map = np.asarray(
        instance.get("ops_job_map", np.zeros(N_ops, dtype=int)),
        dtype=int,
    )
    ops_sequence_order = np.asarray(
        instance.get("ops_sequence_order", np.arange(N_ops, dtype=int)),
        dtype=int,
    )

    # rl4co right-pads to N_ops_max with zero-eligibility slots; the
    # LP would be infeasible on those rows. Drop padded ops up front.
    if "pad_mask" in instance:
        pad = np.asarray(instance["pad_mask"], dtype=bool)
    else:
        pad = elig.sum(axis=0) == 0
    real_ops = [int(o) for o in range(N_ops) if not bool(pad[o])]
    if not real_ops:
        return {"eligibility": 0.0, "precedence": 0.0}

    # Loose makespan upper bound: every op runs sequentially at its
    # heaviest eligible machine.
    max_p = float(proc.max()) if proc.size else 1.0
    T_cap = max(1.0, max_p * float(N_ops))

    def _build(solver: Any) -> dict[str, list[Any]]:
        # ---- Variables ----
        x = {
            (o, m): solver.NumVar(0.0, 1.0, f"x_{o}_{m}")
            for o in real_ops
            for m in range(M)
        }
        c = {o: solver.NumVar(0.0, T_cap, f"c_{o}") for o in real_ops}
        Cmax = solver.NumVar(0.0, T_cap, "Cmax")

        # Pin ineligible (o, m) pairs to zero so they cannot pick up
        # mass during simplex (otherwise the LP would assign across
        # all machines uniformly and dilute the eligibility dual).
        for o in real_ops:
            for m in range(M):
                if not bool(elig[m, o]):
                    solver.Add(x[o, m] == 0.0, f"inelig_{o}_{m}")

        # ---- Objective ----
        solver.Minimize(Cmax)

        family_rows: dict[str, list[Any]] = {
            "eligibility": [],
            "precedence": [],
        }

        # ---- Eligibility: assignment-conservation, one row per op. ----
        for o in real_ops:
            family_rows["eligibility"].append(
                solver.Add(
                    solver.Sum(x[o, m] for m in range(M)) == 1.0,
                    f"assign_{o}",
                )
            )

        # ---- Precedence: intra-job order + release + makespan link. ----
        # Group real ops by job.
        jobs: dict[int, list[int]] = {}
        for o in real_ops:
            jobs.setdefault(int(ops_job_map[o]), []).append(o)
        for ops in jobs.values():
            ops.sort(key=lambda o: int(ops_sequence_order[o]))

        for ops in jobs.values():
            if not ops:
                continue
            o_first = ops[0]
            # Release: c[o_first] >= duration of o_first.
            family_rows["precedence"].append(
                solver.Add(
                    c[o_first]
                    >= solver.Sum(
                        float(proc[m, o_first]) * x[o_first, m] for m in range(M)
                    ),
                    f"release_{o_first}",
                )
            )
            # Sequential within-job constraints.
            for prev, nxt in pairwise(ops):
                family_rows["precedence"].append(
                    solver.Add(
                        c[nxt]
                        >= c[prev]
                        + solver.Sum(
                            float(proc[m, nxt]) * x[nxt, m] for m in range(M)
                        ),
                        f"prec_{prev}_{nxt}",
                    )
                )
            # Makespan link: Cmax >= c of last op of job.
            o_last = ops[-1]
            family_rows["precedence"].append(
                solver.Add(Cmax >= c[o_last], f"makespan_{o_last}")
            )

        return family_rows

    result = solve_lp_and_extract_duals(_build, time_limit_s=time_limit_s, aggregate=aggregate)
    return result.multipliers
