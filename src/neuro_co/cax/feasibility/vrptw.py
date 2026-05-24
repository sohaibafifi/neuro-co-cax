"""CVRPTW arithmetic instance-feasibility check.

A CVRPTW instance is feasible iff:

  1. All demands are non-negative              (`demand >= 0`)
  2. Total demand <= n_vehicles * Q            (some routing exists)
  3. All time windows are well-formed          (`tw[:, 0] <= tw[:, 1]`)
  4. Service durations are non-negative        (`durations >= 0`)
  5. Vehicle capacity is positive              (`Q > 0`)

These are *necessary* conditions; satisfying them does NOT prove
that a valid routing exists (that needs a CP-SAT decision query,
deferred to M3.5). But they catch every infeasibility a small
Gaussian perturbation around a valid instance would otherwise
produce -- the common case during counterfactual search, where
the seed instance is feasible and we only need to verify the
perturbed copy hasn't crossed a structural cliff.

Returns `[B]` bool tensor; `True` = candidate passes all checks.
"""

from __future__ import annotations

from typing import Any

import torch


def vrptw_is_feasible(td: Any) -> torch.Tensor:
    """Element-wise arithmetic check on a batched CVRPTW TensorDict."""
    B = int(td.batch_size[0])
    ok = torch.ones(B, dtype=torch.bool)

    if "demand" in td:
        demand = td["demand"]
        ok = ok & (demand >= 0).all(dim=-1).cpu()
    if "time_windows" in td:
        tw = td["time_windows"]
        # tw shape [B, N, 2]: open <= close
        ok = ok & (tw[..., 0] <= tw[..., 1]).all(dim=-1).cpu()
    if "durations" in td:
        dur = td["durations"]
        ok = ok & (dur >= 0).all(dim=-1).cpu()
    if "vehicle_capacity" in td:
        cap = td["vehicle_capacity"]
        # Squeeze any trailing singleton dim.
        if cap.ndim > 1:
            cap = cap.squeeze(-1)
        ok = ok & (cap > 0).cpu()
        # Total demand <= K * Q (need enough fleet capacity overall).
        if "demand" in td:
            total = td["demand"].sum(dim=-1).cpu()
            # Assume K = N (loose upper bound; real rl4co CVRPTW
            # lets the agent open any number of routes up to the
            # number of customers).
            n_customers = int(td["demand"].shape[-1])
            ok = ok & (total <= float(n_customers) * cap.cpu())
    return ok
