"""Lambda-attribution: decompose a policy's per-step gradient by constraint family.

For a CO problem with constraint families `c_1, ..., c_K`
(capacity, time-window, precedence, ...), the v0.1
*Lambda-attribution* of decision `a_t` to constraint `c_k` is the
gradient-x-feature mass aggregated over the feature tensors that
parameterise `c_k`::

    Lambda_k(t) =  sum_{i in features(c_k), j in nodes}
                   | d log pi(a_t | s_t) / d x_{ij} * x_{ij} |

Where `features(c_k)` comes from
`neuro_co.cax.constraint_map.PROBLEM_CONSTRAINTS`. This is a
heuristic stand-in for the "true" Lagrangian-multiplier gradient
(see paper-cax milestone M2 for the formal version). It already
gives a CO-specific decomposition the four generic baselines
(gradient, IG, DeepLIFT, contrastive) cannot produce, because
they aggregate over *all* feature inputs and lose the
constraint-family attribution.

Cost: K calls to `gradient_attribution` per rollout (one per
constraint family). For a 4-step VRPTW rollout with 3 families,
this is roughly 3x a single gradient pass -- cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.cax.constraint_map import get_constraints


@dataclass
class LambdaAttribution:
    """Per-step attribution decomposed by constraint family.

    Attributes
    ----------
    constraint_names
        Ordered list of constraint family names.
    scores
        `[batch, T, K]` float tensor of `Lambda_k(t)`. Values are
        non-negative (we aggregate `|grad x feature|`).
    multipliers
        `[batch, K]` tensor of LP-relaxation duals, when available.
        v0.1 leaves this as `None`; milestone M2 populates it.
    per_family_node_scores
        `[K, B, T, N]` float tensor: per-family per-node attribution.
        Useful for downstream per-node, per-constraint heatmaps.
    feature_keys_per_family
        `[K]` list of tuples — the feature keys actually used per
        family (intersected with what's present in the TensorDict).
    """

    constraint_names: list[str]
    scores: torch.Tensor
    multipliers: torch.Tensor | None
    per_family_node_scores: torch.Tensor
    feature_keys_per_family: list[tuple[str, ...]]

    @property
    def num_families(self) -> int:
        return int(self.scores.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self.scores.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.scores.shape[1])

    def top_family_per_step(self) -> torch.Tensor:
        """`[B, T]` long tensor of the argmax constraint family per step."""
        return self.scores.argmax(dim=-1)


def lambda_attribution(
    policy: Any,
    env: Any,
    td: Any,
    *,
    problem: str,
    top_k: int = 5,
    max_steps: int | None = None,
    multipliers: str | dict[str, float] | None = None,
    multipliers_kwargs: dict[str, Any] | None = None,
) -> LambdaAttribution:
    """Compute Lambda-attribution per (step, constraint family).

    Runs `gradient_attribution` once per constraint family with
    `feature_keys` restricted to that family's parameterising
    tensors. Optionally weights each family's score by a
    Lagrangian multiplier `lambda_c*` from the LP relaxation
    (`multipliers='lp'`), subgradient ascent
    (`multipliers='subgrad'`), or a caller-supplied dict.

    Parameters
    ----------
    policy, env, td
        Standard rl4co triplet, same as `gradient_attribution`.
    problem
        Problem-bank key (`'vrptw'`, `'fjsp'`, `'op'`).
    top_k
        Top-k width forwarded to the underlying
        `gradient_attribution` call. Used only to populate
        `AttributionTrace.top_k_nodes`; the Lambda decomposition
        itself sums over all nodes.
    max_steps
        Optional cap on decoding-step count.
    multipliers
        - `None` (default) -- proxy mode: equal weight per family
          ("constraint-partitioned gradient", v0.1 behaviour).
        - `"lp"` -- LP-relaxation duals via OR-Tools GLOP
          (`neuro_co.cax.duals.lp` backend).
        - `"subgrad"` -- Beasley subgradient ascent
          (`neuro_co.cax.duals.subgrad` backend).
        - `dict[str, float]` -- caller-supplied multipliers
          keyed by constraint family name.
    multipliers_kwargs
        Forwarded to the backend
        (`time_limit_s`, `max_iters`, ...).

    Returns
    -------
    LambdaAttribution
    """
    # Local import to avoid a top-level neuro-co-xai dependency at
    # module-import time (lets `import neuro_co.cax` succeed even
    # if a downstream tool doesn't have the xai pkg fully wired).
    from neuro_co.xai.attribution import gradient_attribution

    families = get_constraints(problem)
    if not families:
        raise ValueError(
            f"No constraint families registered for problem={problem!r}. "
            "Add an entry to `neuro_co.cax.constraint_map.PROBLEM_CONSTRAINTS`."
        )

    # Filter family feature lists down to keys actually present in
    # the TensorDict — missing keys (depot-omitted variants, env
    # toggle flags) shouldn't crash the decomposition.
    available_keys = set(td.keys())
    filtered_families: list[tuple[str, tuple[str, ...]]] = []
    for name, keys in families:
        present = tuple(k for k in keys if k in available_keys)
        if present:
            filtered_families.append((name, present))
    if not filtered_families:
        raise ValueError(
            f"None of the feature keys declared for problem={problem!r} are "
            f"in the TensorDict (have: {sorted(available_keys)})."
        )

    per_family_traces = []
    for _, keys in filtered_families:
        trace = gradient_attribution(
            policy,
            env,
            td,
            feature_keys=keys,
            top_k=top_k,
            max_steps=max_steps,
        )
        per_family_traces.append(trace)

    # All traces share batch / steps / node count by construction.
    b = per_family_traces[0].batch_size
    t = min(tr.num_steps for tr in per_family_traces)
    n = per_family_traces[0].num_nodes
    k_families = len(filtered_families)

    family_names = [name for name, _ in filtered_families]
    mult_dict, mult_tensor = _resolve_multipliers(
        problem,
        td,
        family_names,
        multipliers,
        multipliers_kwargs or {},
    )

    scores = torch.zeros(b, t, k_families)
    per_family_node = torch.zeros(k_families, b, t, n)
    for k_idx, tr in enumerate(per_family_traces):
        # node_scores is already abs(grad x feat) per node per step.
        node = tr.node_scores[:, :t, :n]
        per_family_node[k_idx] = node
        # Scale per family by the (absolute) Lagrangian multiplier.
        weight = mult_dict.get(family_names[k_idx], 1.0) if mult_dict else 1.0
        scores[:, :, k_idx] = node.sum(dim=-1) * weight

    return LambdaAttribution(
        constraint_names=family_names,
        scores=scores,
        multipliers=mult_tensor,
        per_family_node_scores=per_family_node,
        feature_keys_per_family=[keys for _, keys in filtered_families],
    )


def _resolve_multipliers(
    problem: str,
    td: Any,
    family_names: list[str],
    spec: str | dict[str, float] | None,
    kwargs: dict[str, Any],
) -> tuple[dict[str, float] | None, torch.Tensor | None]:
    """Map the user's `multipliers=` spec onto a per-family weight dict.

    Returns (dict_for_weighting, tensor_for_reporting). The tensor
    is `[K]` (mean over batch) for downstream consumers; the dict
    drives the per-family score scaling.
    """
    if spec is None:
        return None, None
    if isinstance(spec, dict):
        d = {k: float(spec.get(k, 0.0)) for k in family_names}
        return d, torch.tensor([d[k] for k in family_names])
    method = str(spec).lower()
    if method not in {"lp", "subgrad"}:
        raise ValueError(
            f"multipliers must be 'lp', 'subgrad', a dict, or None; got {spec!r}"
        )
    from neuro_co.cax.duals import get_multipliers, td_to_instance

    instance = td_to_instance(td, problem, batch_idx=0)
    try:
        d_raw = get_multipliers(problem, instance, method=method, **kwargs)
    except NotImplementedError as exc:
        # Backend missing for this (problem, method) -> fall back to
        # equal weighting and warn so the caller notices.
        import warnings

        warnings.warn(
            f"[lambda_attribution] multipliers={spec!r} unsupported "
            f"for problem={problem!r}: {exc}. Falling back to proxy "
            f"(equal-weight) mode.",
            stacklevel=2,
        )
        return None, None
    d = {k: float(d_raw.get(k, 0.0)) for k in family_names}
    return d, torch.tensor([d[k] for k in family_names])
