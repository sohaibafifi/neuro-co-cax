"""CVRPTW Lagrangian-subgradient dual estimator.

Relaxes the *capacity* and *time-window* constraint families into
the objective; the remaining subproblem is a soft-TSP-with-penalty
solved by **nearest-neighbour insertion** (a cheap, monotone
heuristic, Lagrangian duality does not require subproblem
optimality, only consistent improvement at the multiplier update).

Two dualised families surface multipliers:

  capacity     lambda_cap     penalises sum_v max(0, load_v - Q)
  time_window  lambda_tw      penalises sum_i max(0, a_i - l_i)

The `spatial` family stays in the subproblem as the routing cost
itself, so its multiplier is not estimated by this backend (set
to 0.0 in the report). The LP backend produces a meaningful
`spatial` dual (degree-constraint shadow prices), this is one
documented limitation of the subgrad backend; paper-cax §4.X uses
it as the head-to-head ablation.

The implementation is intentionally small (~150 LOC) so the
methodology is auditable. Production-quality VRPTW Lagrangian
relaxations live in Toth & Vigo 2014 ch.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuro_co.cax.duals.subgrad import subgradient_ascent
from neuro_co.cax.duals.vrptw_lp import _pad_depot, _pad_depot_tw, _to_scalar


def vrptw_subgrad_duals(
    instance: dict[str, Any],
    *,
    max_iters: int = 40,
    initial_step: float = 1.5,
    capacity_default: float = 1.0,
) -> dict[str, float]:
    """Return mean |lambda| per family from a small subgradient run."""
    locs = np.asarray(instance["locs"], dtype=float)
    N = locs.shape[0]
    if N < 3:
        return {"capacity": 0.0, "time_window": 0.0, "spatial": 0.0}

    demand = _pad_depot(np.asarray(instance.get("demand", np.zeros(N)), dtype=float), N)
    tw = _pad_depot_tw(
        np.asarray(
            instance.get("time_windows", np.tile([0.0, 1e6], (N, 1))),
            dtype=float,
        ),
        N,
    )
    dur = _pad_depot(np.asarray(instance.get("durations", np.zeros(N)), dtype=float), N)
    Q = _to_scalar(instance.get("vehicle_capacity", capacity_default))
    d = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)

    def subproblem(lam: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Greedy nearest-neighbour route construction with soft penalties.

        Returns (L_value, violations). `L_value` is the
        Lagrangian dual at `lam`: travel cost minus penalty-weighted
        slack. Violations are aggregate positive residuals.
        """
        unvisited = set(range(1, N))
        current = 0
        time = 0.0
        load = 0.0
        cost = 0.0
        cap_violation = 0.0
        tw_violation = 0.0

        while unvisited:
            # Soft-penalised cost of visiting each remaining node next.
            best_node = -1
            best_score = float("inf")
            for j in unvisited:
                arr = max(time + d[current, j], tw[j, 0])
                cap_excess = max(0.0, load + demand[j] - Q)
                tw_excess = max(0.0, arr - tw[j, 1])
                score = float(
                    d[current, j]
                    + lam.get("capacity", 0.0) * cap_excess
                    + lam.get("time_window", 0.0) * tw_excess
                )
                if score < best_score:
                    best_score = score
                    best_node = j
            assert best_node >= 0  # `unvisited` is non-empty here
            j = best_node
            arr = max(time + d[current, j], tw[j, 0])
            cap_violation += float(max(0.0, load + demand[j] - Q))
            tw_violation += float(max(0.0, arr - tw[j, 1]))
            cost += float(d[current, j])
            time = arr + float(dur[j])
            load += float(demand[j])
            if load > Q:
                cost += float(d[j, 0])
                current = 0
                time = 0.0
                load = 0.0
            else:
                current = j
            unvisited.discard(j)
        cost += float(d[current, 0])

        L_val = (
            cost
            - lam.get("capacity", 0.0) * cap_violation
            - lam.get("time_window", 0.0) * tw_violation
        )
        return float(L_val), {
            "capacity": float(cap_violation),
            "time_window": float(tw_violation),
        }

    report = subgradient_ascent(
        family_names=["capacity", "time_window"],
        subproblem=subproblem,
        max_iters=max_iters,
        initial_step=initial_step,
    )
    out = dict(report.multipliers)
    out.setdefault("capacity", 0.0)
    out.setdefault("time_window", 0.0)
    out["spatial"] = 0.0  # not dualised in this backend; see module docstring.
    return out
