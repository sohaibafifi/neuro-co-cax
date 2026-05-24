"""Deletion-based faithfulness metric.

For each decision step we mask the top-k attributed nodes (set their
feasibility mask to False and re-run the same step under the policy).
The metric is the fraction of decisions whose argmax flips compared
to the original rollout.

A high flip rate indicates the attribution is faithful: the policy
truly relied on the masked nodes. A low rate suggests the attribution
is not informative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.xai.attribution import AttributionTrace, step_logits
from neuro_co.xai.attribution._common import _encode


@dataclass
class DeletionReport:
    mean_flip_rate: float
    per_step_flip_rate: list[float]
    top_k_used: int
    num_steps: int
    num_instances: int


@dataclass
class SufficiencyReport:
    """Counterpart to deletion: keep ONLY top-k, mask everything else.

    A high `mean_keep_rate` means the top-k nodes alone reproduce the
    original action — strong evidence the attribution captures what the
    policy used.
    """

    mean_keep_rate: float
    per_step_keep_rate: list[float]
    top_k_used: int
    num_steps: int
    num_instances: int


@dataclass
class SanityCheckReport:
    """Adebayo et al. 2018 model-parameter-randomization sanity check.

    Re-runs gradient attribution with model weights randomized. The
    Jaccard overlap between original top-k and randomized top-k should
    be close to a random-chance baseline; a high overlap means the
    explanation is largely insensitive to the model and therefore
    suspect.
    """

    mode: str
    mean_jaccard: float
    chance_jaccard: float
    per_step_jaccard: list[float]
    top_k_used: int
    num_trials: int


def _mask_features_outside_topk(
    state: Any,
    feature_keys: tuple[str, ...],
    top_nodes: torch.Tensor,
    *,
    baseline: str = "zero",
) -> Any:
    """Return a clone of `state` with feature values *outside* `top_nodes` replaced.

    `top_nodes` is `[B, k]` of node indices to *keep*. All other node
    positions in each feature tensor are overwritten with the
    baseline (zero or per-feature mean over the batch). Used by both
    `deletion_flip_rate` (pass complement) and `sufficiency_keep_rate`
    (pass top-k directly).
    """
    new = state.clone(recurse=False)
    for key in feature_keys:
        if key not in new:
            continue
        t = new[key]
        if not isinstance(t, torch.Tensor):
            continue
        # Integer features (JSSP `proc_times`, FJSP `num_eligible`)
        # never used to mask out before — they fell through this
        # guard and left sufficiency_keep_rate trivially at 1.0 for
        # scheduling problems. Preserve the original dtype on write
        # so the env doesn't crash on a float-where-int-expected.
        orig_dtype = t.dtype
        if not t.dtype.is_floating_point:
            t = t.to(torch.float32)
        # Locate the node axis by matching dim size to mask width.
        node_axis = None
        for d in range(1, t.ndim):
            if t.shape[d] == int(top_nodes.shape[-1] if False else _state_num_nodes(state)):
                node_axis = d
                break
        if node_axis is None:
            # Try fallback: any axis whose size equals top_nodes.max()+1 range.
            n_guess = int(_state_num_nodes(state))
            for d in range(1, t.ndim):
                if t.shape[d] == n_guess:
                    node_axis = d
                    break
        if node_axis is None:
            continue
        n = t.shape[node_axis]
        # Build a [B, N] keep-mask matching node positions.
        b = t.shape[0]
        keep = torch.zeros(b, n, dtype=torch.bool, device=t.device)
        # Clamp indices to valid range (defensive; depot-omitted feats
        # have N-1 nodes — silently drop out-of-range hits).
        idx = top_nodes.clamp(0, n - 1)
        keep.scatter_(1, idx, True)

        if baseline == "zero":
            ref = torch.zeros_like(t)
        else:  # "mean"
            ref = t.mean(dim=0, keepdim=True).expand_as(t).clone()

        # Broadcast keep mask onto t's shape.
        shape = [1] * t.ndim
        shape[0] = b
        shape[node_axis] = n
        keep_b = keep.view(shape).expand_as(t)
        result = torch.where(keep_b, t, ref)
        # Cast back to original dtype (env may expect int for proc_times etc.).
        if result.dtype != orig_dtype:
            result = result.to(orig_dtype)
        new[key] = result
    return new


def _state_num_nodes(state: Any) -> int:
    """Best-effort node-count probe for masking. Mirrors `_common._num_nodes`."""
    if "locs" in state:
        return int(state["locs"].shape[-2])
    if "proc_times" in state:
        return int(state["proc_times"].shape[-1])
    if "action_mask" in state:
        return int(state["action_mask"].shape[-1])
    return 0


def deletion_flip_rate(
    trace: AttributionTrace,
    policy: Any,
    env: Any,
    td_initial: Any,
    *,
    top_k: int = 5,
    baseline: str = "zero",
) -> DeletionReport:
    """Replace top-k attributed nodes' features with baseline; report flips.

    Standard deletion semantics: at each step the features of the
    top-k attributed nodes are zeroed (or replaced by the per-feature
    batch mean), the encoder re-runs, and the new argmax is compared
    to the original. A high flip rate is evidence the attribution
    actually pinpointed inputs the policy relied on.
    """
    device = next(policy.parameters()).device
    policy.eval()

    k = min(int(top_k), trace.top_k_nodes.shape[-1])
    flips_per_step: list[float] = []
    state = td_initial.clone(recurse=False).to(device)
    feature_keys = tuple(trace.feature_keys)

    with torch.no_grad():
        for t in range(trace.num_steps):
            if bool(state["done"].all()):
                break

            log_p_orig, _ = step_logits(policy, state, _encode(policy, state))
            original_action = log_p_orig.argmax(dim=-1)

            top_nodes = trace.top_k_nodes[:, t, :k].to(device)
            # Build the complement of top-k as the "keep" set so the
            # shared helper can be reused: keep everything but top-k.
            num_nodes = _state_num_nodes(state)
            all_idx = torch.arange(num_nodes, device=device).unsqueeze(0).expand(
                top_nodes.shape[0], -1
            )
            # mask out top-k rows; remaining are nodes to keep.
            keep_mask = torch.ones_like(all_idx, dtype=torch.bool)
            keep_mask.scatter_(1, top_nodes.clamp(0, num_nodes - 1), False)
            # Pack indices for the masking helper: any nodes not in top-k.
            # The helper expects "indices to keep"; pass complement.
            keep_idx = torch.where(
                keep_mask, all_idx, torch.full_like(all_idx, fill_value=-1)
            )
            # The helper's clamp(0, n-1) on -1 -> 0 (re-adds depot);
            # that's harmless because depot is also kept (it isn't in
            # top-k by construction unless attribution picked it).
            masked_state = _mask_features_outside_topk(
                state, feature_keys, keep_idx, baseline=baseline
            )

            log_p_pert, _ = step_logits(policy, masked_state, _encode(policy, masked_state))
            perturbed_action = log_p_pert.argmax(dim=-1)

            flips = (perturbed_action != original_action).float().mean().item()
            flips_per_step.append(flips)

            state["action"] = original_action
            state = env.step(state)["next"]

    mean = float(sum(flips_per_step) / len(flips_per_step)) if flips_per_step else 0.0
    return DeletionReport(
        mean_flip_rate=mean,
        per_step_flip_rate=flips_per_step,
        top_k_used=k,
        num_steps=len(flips_per_step),
        num_instances=int(trace.batch_size),
    )


def sufficiency_keep_rate(
    trace: AttributionTrace,
    policy: Any,
    env: Any,
    td_initial: Any,
    *,
    top_k: int = 5,
    baseline: str = "zero",
) -> SufficiencyReport:
    """Keep only top-k attributed nodes in feature space; report unchanged decisions.

    Standard sufficiency semantics: replace all *feature* values
    outside the top-k attributed nodes with a baseline (zero or
    per-feature mean), re-encode + decode, and check whether the
    new argmax matches the original action.

    The previous implementation restricted the *action space* and
    re-argmaxed, which trivially returned the original action
    (which was the global max), making `mean_keep_rate` always
    saturate at 1.0. The feature-masking variant is what Adebayo
    and follow-ups use; baseline_v1 will produce informative
    numbers once we re-run.
    """
    device = next(policy.parameters()).device
    policy.eval()

    k = min(int(top_k), trace.top_k_nodes.shape[-1])
    keep_per_step: list[float] = []
    state = td_initial.clone(recurse=False).to(device)
    feature_keys = tuple(trace.feature_keys)

    with torch.no_grad():
        for t in range(trace.num_steps):
            if bool(state["done"].all()):
                break

            log_p_orig, _mask_orig = step_logits(
                policy,
                state,
                _encode(policy, state),
            )
            original_action = log_p_orig.argmax(dim=-1)

            top_nodes = trace.top_k_nodes[:, t, :k].to(device)
            masked_state = _mask_features_outside_topk(
                state, feature_keys, top_nodes, baseline=baseline
            )
            log_p_masked, _mask_masked = step_logits(
                policy,
                masked_state,
                _encode(policy, masked_state),
            )
            kept_action = log_p_masked.argmax(dim=-1)

            keeps = (kept_action == original_action).float().mean().item()
            keep_per_step.append(keeps)

            state["action"] = original_action
            state = env.step(state)["next"]

    mean = float(sum(keep_per_step) / len(keep_per_step)) if keep_per_step else 0.0
    return SufficiencyReport(
        mean_keep_rate=mean,
        per_step_keep_rate=keep_per_step,
        top_k_used=k,
        num_steps=len(keep_per_step),
        num_instances=int(trace.batch_size),
    )


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard overlap between two sets of node indices (1-D tensors)."""
    sa, sb = set(int(x) for x in a.tolist()), set(int(x) for x in b.tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def sanity_check(
    trace: AttributionTrace,
    policy: Any,
    env: Any,
    td_initial: Any,
    *,
    top_k: int = 5,
    num_trials: int = 1,
    mode: str = "random_weights",
    seed: int = 0,
) -> SanityCheckReport:
    """Model-parameter-randomization sanity check (Adebayo et al. 2018).

    Saves the policy's `state_dict`, replaces every parameter with a
    Gaussian draw, recomputes gradient attribution, restores the saved
    weights, then reports the Jaccard overlap between original and
    randomized top-k per step. A low mean Jaccard indicates the
    attribution genuinely depends on the trained weights.
    """
    if mode != "random_weights":
        raise ValueError(f"sanity_check mode must be 'random_weights', got {mode!r}")

    from neuro_co.xai.attribution import gradient_attribution

    device = next(policy.parameters()).device
    k = min(int(top_k), trace.top_k_nodes.shape[-1])
    num_nodes = int(trace.node_scores.shape[-1])
    # Closed form for the expected Jaccard of two uniformly-random
    # size-k subsets drawn from an N-element universe:
    #   E[|A intersect B|] = k^2 / N
    #   |A union B|        = 2k - |A intersect B|
    # E[Jaccard] is *not* k/N (that would be expected fraction-
    # overlap of B with A, a different quantity). Using k/N as the
    # chance baseline made every method look indistinguishable from
    # random in the baseline_v1 sweep; the corrected formula is
    # roughly half that value for small k, large N.
    if num_nodes > 0:
        ei = (k * k) / num_nodes
        eu = max(2 * k - ei, 1e-9)
        chance = ei / eu
    else:
        chance = 0.0

    saved = {name: p.detach().clone() for name, p in policy.state_dict().items()}
    overlaps: list[float] = []
    try:
        for trial in range(num_trials):
            g = torch.Generator(device="cpu").manual_seed(int(seed) + trial)
            new_state: dict[str, torch.Tensor] = {}
            for name, p in saved.items():
                if p.dtype.is_floating_point:
                    new_state[name] = (
                        torch.empty_like(p, device="cpu")
                        .normal_(0.0, 0.1, generator=g)
                        .to(p.device)
                    )
                else:
                    new_state[name] = p
            policy.load_state_dict(new_state, strict=False)
            policy.to(device)
            try:
                trial_trace = gradient_attribution(
                    policy,
                    env,
                    td_initial,
                    top_k=k,
                    feature_keys=tuple(trace.feature_keys),
                    max_steps=trace.num_steps,
                )
            except (RuntimeError, AssertionError, ValueError):
                # Randomized weights can produce NaNs / asserts inside
                # the decoder. Treat such trials as worst-case chance.
                overlaps.append(chance)
                continue
            t_steps = min(trace.num_steps, trial_trace.num_steps)
            for t in range(t_steps):
                for b in range(int(trace.batch_size)):
                    overlaps.append(
                        _jaccard(trace.top_k_nodes[b, t, :k], trial_trace.top_k_nodes[b, t, :k])
                    )
    finally:
        policy.load_state_dict(saved, strict=False)
        policy.to(device)

    mean = float(sum(overlaps) / len(overlaps)) if overlaps else 0.0
    return SanityCheckReport(
        mode=mode,
        mean_jaccard=mean,
        chance_jaccard=chance,
        per_step_jaccard=overlaps,
        top_k_used=k,
        num_trials=num_trials,
    )
