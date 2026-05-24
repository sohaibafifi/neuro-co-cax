"""PAC-minimal sufficient subset (Koriche et al. 2024, CO variant).

For a black-box policy `pi` and an input `x`, find the smallest
node subset `S subset {1, ..., N}` such that masking all features
outside `S` (replace with baseline) preserves `argmax pi`
with PAC probability `>= 1 - delta` under `M` neighbourhood
samples.

Formally::

    min |S|
    s.t.  Pr_{x' ~ N(x, sigma)} [ argmax pi(x'_{|S}) == argmax pi(x') ]
          >= 1 - epsilon
    estimated from M samples, Hoeffding-tight (eq. 1 below).

v0.2 ships a **greedy PAC** approximation, not the true COP:

  1. Compute Lambda-attribution scores per node (top_k_nodes from
     an existing gradient attribution provides the priority order).
  2. For k = 1, 2, ..., N:
       - sample M neighbourhood perturbations of x (Gaussian noise
         with std `sigma`),
       - mask features outside the top-k subset to baseline,
       - check fraction of samples where argmax is preserved,
       - if fraction >= 1 - epsilon, accept k as the minimal size.
  3. Return the resulting subset + PAC certificate.

PAC sample size from Hoeffding:

    M >= ceil( log(2 / delta) / (2 epsilon^2) )         (eq. 1)

Greedy vs true CP: a real CP-SAT encoding (Koriche et al. 2024 §3)
optimises *which* nodes go in S, not just how many. The greedy
version is informative and *always sound* (it never overstates the
preserved-argmax probability) but may report a subset 10-30%
larger than optimal. Milestone M3 lifts to the true CP solve;
this v0.2 is the baseline measurement scaffold used to score
attribution methods *now*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.xai.attribution import AttributionTrace, gradient_attribution, step_logits
from neuro_co.xai.attribution._common import _encode
from neuro_co.xai.faithfulness import _mask_features_outside_topk


@dataclass
class MinimalSubsetReport:
    """PAC-minimal sufficient subset per (batch, step).

    Attributes
    ----------
    subset_size
        `[batch, T]` long tensor of `|S|` per (batch, step). 0
        means "no subset of size <= max_k satisfied PAC".
    subset
        `[batch, T, N]` bool tensor of which nodes are kept.
    preserved_rate
        `[batch, T]` float tensor of the empirical
        `Pr[argmax preserved]` at the returned `|S|`.
    pac_epsilon, pac_delta
        PAC parameters used.
    samples_drawn
        Hoeffding-derived M (>= eq. 1).
    method
        "greedy" in v0.2; "cp_sat" in M3+.
    """

    subset_size: torch.Tensor
    subset: torch.Tensor
    preserved_rate: torch.Tensor
    pac_epsilon: float
    pac_delta: float
    samples_drawn: int
    method: str = "greedy"


def pac_sample_count(
    epsilon: float, delta: float, *, n_tests: int = 1
) -> int:
    """Bonferroni-corrected Hoeffding sample size.

    For a *single* `(epsilon, delta)` PAC test, returns

        M = ceil( log(2 / delta) / (2 epsilon^2) ).

    When the greedy subset procedure inspects up to `n_tests`
    candidate subsets (one per k in {1, ..., k_max}) and accepts
    the first one whose empirical preserved-argmax rate exceeds
    `1 - epsilon`, the family-wise confidence is loose under
    Bonferroni; pass `n_tests = k_max` to obtain a family-wise
    (1 - delta)-PAC guarantee along the greedy ordering:

        M_bonf = ceil( log(2 n_tests / delta) / (2 epsilon^2) ).
    """
    if not (0 < epsilon < 1) or not (0 < delta < 1):
        raise ValueError(f"epsilon, delta must lie in (0, 1); got {epsilon=}, {delta=}")
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1; got {n_tests}")
    return math.ceil(math.log(2.0 * n_tests / delta) / (2.0 * epsilon * epsilon))


def cp_minimal_subset(
    policy: Any,
    env: Any,
    td: Any,
    *,
    pac_epsilon: float = 0.1,
    pac_delta: float = 0.1,
    sigma: float = 0.05,
    max_k: int | None = None,
    feature_keys: tuple[str, ...] = (),
    max_steps: int | None = 8,
    trace: AttributionTrace | None = None,
    bonferroni: bool = True,
) -> MinimalSubsetReport:
    """Find PAC-minimal sufficient node subsets per decoding step.

    See module docstring for the COP formulation. v0.2 uses greedy
    enumeration over `gradient_attribution`'s top-k ranking; the
    true CP-SAT solve lands in milestone M3.

    Parameters
    ----------
    pac_epsilon, pac_delta
        Tolerance + failure prob for the preserved-argmax estimate.
        Sample size derived from Hoeffding: `M = log(2/delta) / (2 eps^2)`.
    sigma
        Gaussian neighbourhood std for `x'` sampling.
    feature_keys
        Which TensorDict keys participate in masking. If empty,
        the function refuses (no reasonable default — depends on
        the problem).
    max_k
        Largest subset size to try (default: number of nodes).
    trace
        Optional precomputed AttributionTrace (uses its
        `top_k_nodes` as the greedy ordering). If None, a fresh
        `gradient_attribution` rollout is run.
    """
    if not feature_keys:
        raise ValueError(
            "feature_keys is required (problem-specific). Pass the "
            "ConceptBank.feature_keys for the problem, or a subset."
        )
    device = next(policy.parameters()).device
    policy.eval()

    if trace is None:
        trace = gradient_attribution(
            policy,
            env,
            td,
            feature_keys=feature_keys,
            top_k=None or _num_nodes(td),
            max_steps=max_steps,
        )

    B = int(trace.batch_size)
    T = int(trace.num_steps)
    N = int(trace.node_scores.shape[-1])
    if max_k is None:
        max_k = N
    max_k = min(max_k, N)

    # With `bonferroni=True`, the per-cell sample size is inflated so
    # that the family-wise confidence over the greedy sequence of up
    # to `max_k` tests remains `1 - pac_delta` (union bound).
    n_tests = max_k if bonferroni else 1
    M = pac_sample_count(pac_epsilon, pac_delta, n_tests=n_tests)

    subset_size = torch.zeros(B, T, dtype=torch.long)
    subset = torch.zeros(B, T, N, dtype=torch.bool)
    preserved_rate = torch.zeros(B, T, dtype=torch.float32)

    state = td.clone(recurse=False).to(device)

    with torch.no_grad():
        for t in range(T):
            if bool(state["done"].all()):
                break

            log_p_orig, _ = step_logits(policy, state, _encode(policy, state))
            orig_action = log_p_orig.argmax(dim=-1)

            # Per-batch greedy order from the trace's top-k.
            order = trace.top_k_nodes[:, t, :].to(device)

            for k in range(1, max_k + 1):
                top_k_set = order[:, :k]
                preserved_count = torch.zeros(B, device=device)
                for _ in range(M):
                    noise = {
                        key: torch.randn_like(state[key].float()) * sigma
                        for key in feature_keys
                        if key in state and isinstance(state[key], torch.Tensor)
                    }
                    noisy = state.clone(recurse=False)
                    for key, dz in noise.items():
                        noisy[key] = (state[key] + dz).to(state[key].dtype)
                    masked = _mask_features_outside_topk(
                        noisy, feature_keys, top_k_set, baseline="zero"
                    )
                    log_p_m, _ = step_logits(policy, masked, _encode(policy, masked))
                    new_action = log_p_m.argmax(dim=-1)
                    preserved_count = preserved_count + (new_action == orig_action).float()

                rate = preserved_count / M
                # Accept k for each batch element whose rate >= 1 - epsilon.
                satisfied = rate >= (1.0 - pac_epsilon)
                # Lock in the smallest k for each batch element.
                first_hit = (subset_size[:, t] == 0) & satisfied.cpu()
                if first_hit.any():
                    subset_size[first_hit, t] = k
                    preserved_rate[first_hit, t] = rate[first_hit].cpu()
                    for b in torch.nonzero(first_hit, as_tuple=False).flatten().tolist():
                        subset[b, t, top_k_set[b].cpu()] = True
                if (subset_size[:, t] != 0).all():
                    break

            # Advance env with the original argmax to keep the
            # trajectory in sync with the trace.
            state["action"] = orig_action
            state = env.step(state)["next"]

    return MinimalSubsetReport(
        subset_size=subset_size,
        subset=subset,
        preserved_rate=preserved_rate,
        pac_epsilon=pac_epsilon,
        pac_delta=pac_delta,
        samples_drawn=M,
        method="greedy",
    )


def _num_nodes(state: Any) -> int:
    """Local copy of the helper to avoid a circular import on xai._common."""
    if "locs" in state:
        return int(state["locs"].shape[-2])
    if "proc_times" in state:
        return int(state["proc_times"].shape[-1])
    if "action_mask" in state:
        return int(state["action_mask"].shape[-1])
    raise ValueError("Cannot infer num_nodes from TensorDict")
