"""OP (Orienteering Problem) Lagrangian-subgradient dual estimator.

Relaxes the **budget** constraint into the objective; the relaxed
subproblem becomes "maximise sum_i prize_i z_i with no travel
budget", which a greedy `prize / depot-distance` ordering solves
heuristically. The dual update is the budget violation
$\\sum_{i,j} d_{ij} x_{ij} - \\mathrm{max\\_length}$ at the current
multiplier.

Three CAX-relevant families surface multipliers:

  budget   : lambda_budget  -> mean over Beasley iterates of the
              positive part of the budget multiplier
  prize    : stays in the subproblem objective (every collected prize
              is rewarded directly); multiplier returned as 0.0
  spatial  : stays in the subproblem (greedy depot-anchored
              TSP order); multiplier returned as 0.0

Like the VRPTW / JSSP / FJSP subgrad backends, this is small (~120
LOC). On instances where the LP backend identifies budget as
binding, the subgradient often follows; on heuristically-feasible
instances, the multiplier stays at zero and the backend degenerates
to a proxy gradient, matching the negative-control framing in
paper-cax sec. 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuro_co.cax.duals.subgrad import subgradient_ascent


def op_subgrad_duals(
    instance: dict[str, Any],
    *,
    max_iters: int = 200,
    initial_step: float = 2.0,
) -> dict[str, float]:
    """Return mean |lambda| per family from a small subgradient run."""
    locs = np.asarray(instance["locs"], dtype=float)
    if locs.ndim != 2 or locs.shape[0] < 3:
        return {"prize": 0.0, "budget": 0.0, "spatial": 0.0}
    N = locs.shape[0]

    prize = np.asarray(instance.get("prize", np.ones(N)), dtype=float)
    ml = instance.get("max_length", np.array(2.0))
    ml_arr = np.asarray(ml, dtype=float)
    max_length = float(ml_arr.flatten()[0]) if ml_arr.size else 2.0

    # Pairwise distances. `d` is read in the subproblem and must
    # NOT be mutated (an earlier version aliased `d[0]` via a view
    # and silently poisoned the detour `d[cur, 0]` term).
    d = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)

    def subproblem(lam: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Lagrangian subproblem with the budget constraint dualised.

        Objective (the budget is in the Lagrangian, not in the
        subproblem's feasible set):
          max sum_i prize_i z_i
              - lambda_budget * (sum_{i,j} d_{ij} x_{ij} - max_length)

        Greedy: insert customers in decreasing
        `prize - lambda_budget * detour_dist` order, stopping when
        the marginal contribution drops to zero. The tour is allowed
        to exceed `max_length`; the resulting positive slack is the
        subgradient that drives `lambda_budget` upward at the next
        ascent step.
        """
        l_budget = lam.get("budget", 0.0)
        cur = 0
        total_dist = 0.0
        prize_collected = 0.0
        remaining = list(range(1, N))
        while remaining:
            # Pick the customer with highest marginal Lagrangian value
            # at the current state.
            best_idx, best_score, best_step = -1, 0.0, 0.0
            for i in remaining:
                step = float(d[cur, i] + d[i, 0] - d[cur, 0])
                score = float(prize[i]) - l_budget * step
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_step = step
            if best_idx == -1:
                break  # no positive-marginal customer left
            total_dist += best_step
            prize_collected += float(prize[best_idx])
            cur = best_idx
            remaining.remove(best_idx)
        # Close tour and compute slack -- positive when the unconstrained
        # subproblem chose a tour that violates the budget.
        total_dist += float(d[cur, 0])
        slack = total_dist - max_length
        # OP is a maximisation primal; convert to the equivalent
        # minimisation L_val = -prize + lambda * slack so the
        # subgradient ascent library (which maximises L_val for a
        # min primal) is consistent. The subgradient
        # dL/dlambda = slack is unchanged.
        L_val = -prize_collected + l_budget * slack
        return L_val, {"budget": float(slack)}

    # Feasible primal upper bound on the (negated) objective: the
    # empty tour collects zero prize, so L_val = -prize + lambda * slack
    # is upper-bounded by 0. Passing this UB enables Beasley's
    # `step = alpha * (UB - L) / ||g||^2` rule, which converges much
    # faster than the unconditional `alpha / (1 + iter)` fallback.
    report = subgradient_ascent(
        family_names=["budget"],
        subproblem=subproblem,
        max_iters=max_iters,
        initial_step=initial_step,
        upper_bound=0.0,
    )
    out = dict(report.multipliers)
    out.setdefault("budget", 0.0)
    out["prize"] = 0.0    # in subproblem objective
    out["spatial"] = 0.0  # enforced via greedy tour
    return out
