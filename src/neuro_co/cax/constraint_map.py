"""Per-problem constraint -> feature-key mapping.

`lambda_attribution` partitions the policy's gradient over feature
inputs by the *CO constraint family* each feature group participates
in. The mapping lives here so plug-in problems (vrptw, fjsp, op)
declare their own decomposition once, in one place.

The v0.1 implementation is heuristic: each constraint family is
linked to the feature tensors that *materially* parameterise it.
For VRPTW, `time_window` is parameterised by `time_windows` (the
[start, end] pairs) and the service `durations` that consume
budget. For JSSP, `precedence` is encoded by the row-structure of
`proc_times` (operations belonging to the same job).

A future v0.2 will compute true Lagrangian multipliers from the
LP relaxation of each problem's CP model (paper-cax milestone M2);
the current decomposition is a *proxy* that's already informative
and far simpler to ship.
"""

from __future__ import annotations

# (constraint_name, feature_keys). Order is significant only for
# reporting -- scores are returned in this exact order.
PROBLEM_CONSTRAINTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "vrptw": [
        ("capacity", ("demand", "demand_linehaul")),
        ("time_window", ("time_windows", "durations")),
        ("spatial", ("locs",)),
    ],
    "fjsp": [
        ("precedence", ("proc_times",)),
        ("eligibility", ("num_eligible",)),
    ],
    "op": [
        ("prize", ("prize",)),
        ("budget", ("max_length",)),
        ("spatial", ("locs",)),
    ],
}


def get_constraints(problem: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return the (constraint_name, feature_keys) list for `problem`.

    Falls back to a single `("all", ConceptBank.feature_keys)`
    entry when `problem` is not in `PROBLEM_CONSTRAINTS`, so the
    decomposition still runs (degenerate -- one bucket = the full
    gradient).
    """
    key = problem.lower()
    if key in PROBLEM_CONSTRAINTS:
        return PROBLEM_CONSTRAINTS[key]
    from neuro_co.xai import concept_registry

    bank = concept_registry.get(key)
    return [("all", tuple(bank.feature_keys))]
