"""Shared primitives for attribution methods.

`AttributionTrace`, the feature-collection helpers, the rollout driver
used by single-pass gradient-based methods, and the trace-packing tail.
Method-specific files import from here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class AttributionTrace:
    """Per-step attribution for one rollout.

    Attributes
    ----------
    actions
        `[batch, T]` long tensor of taken action indices.
    log_probs
        `[batch, T]` float tensor of `log pi(a_t | s_t)` (or the
        contrast margin, depending on the method).
    node_scores
        `[batch, T, N]` float tensor, attribution per node per step.
    top_k_nodes
        `[batch, T, K]` long tensor of top-k node indices per step.
    top_k_scores
        `[batch, T, K]` float tensor of their attribution values.
    feature_keys
        Names of TensorDict fields used as feature inputs.
    """

    actions: Tensor
    log_probs: Tensor
    node_scores: Tensor
    top_k_nodes: Tensor
    top_k_scores: Tensor
    feature_keys: list[str]

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_nodes(self) -> int:
        return int(self.node_scores.shape[-1])


def drop_baseline_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Strip rl4co `baseline.*` critic-state keys from a checkpoint dict.

    rl4co's `WarmupBaseline` appends critic-net keys to the policy
    `state_dict()`; loading those into a fresh model with
    `strict=True` would fail. Use this helper before
    `model.load_state_dict(state, strict=False)` to keep the warning
    list short.
    """
    return {k: v for k, v in state.items() if not k.startswith("baseline.")}


def _setup(policy: Any, feature_keys: tuple[str, ...]) -> tuple[torch.device, list[str]]:
    """Resolve device, normalise feature keys, set eval mode."""
    if not feature_keys:
        raise ValueError(
            "feature_keys is required. Pass the problem-specific tuple "
            "(e.g. via `neuro_co.xai.concept_registry.get('vrptw').feature_keys`)."
        )
    keys = list(feature_keys)
    device = next(policy.parameters()).device
    policy.eval()
    return device, keys


def _collect_feature_tensors(td: Any, keys: list[str]) -> dict[str, Tensor]:
    """Pick out the requested feature fields and make them differentiable.

    Integer-valued features (e.g. CVRPTW's `time_windows`) are cast to
    `float32` so they accept `.requires_grad_()`. The cast tensor is
    used only for the gradient pass, the env's own state retains its
    native dtype.

    Boolean tensors (e.g.PDP's `to_deliver`, CVRP's `visited`) are
    *skipped*. Carrying them as float-32 grad-enabled tensors makes
    the downstream env step crash on `bitwise_and(float, bool)`, and
    a bool field has no meaningful gradient anyway -- masking and
    sufficiency / deletion at the bool level requires a different
    attribution primitive (e.g.saliency on bool-flip).
    """
    out: dict[str, Tensor] = {}
    for k in keys:
        if k not in td:
            continue
        t = td[k]
        if not isinstance(t, Tensor):
            continue
        if t.dtype == torch.bool:
            continue
        if not t.dtype.is_floating_point:
            t = t.to(torch.float32)
        out[k] = t.detach().clone().requires_grad_(True)
    return out


def _put_back(td: Any, feats: dict[str, Tensor]) -> Any:
    """Return a shallow copy of `td` with `feats` overlaid."""
    new = td.clone(recurse=False)
    for k, v in feats.items():
        new[k] = v
    return new


def _aggregate_node_scores(grad: Tensor, value: Tensor, num_nodes: int) -> Tensor:
    """Reduce `grad x value` to `[batch, num_nodes]`.

    The node axis can land anywhere except batch (axis 0): routing
    features are `[B, N, F]` or `[B, N]`, while JSSP features such as
    `proc_times` are `[B, M, N]`. We detect the node axis by matching
    `num_nodes` (or `num_nodes - 1` for VRPTW's depot-stripped fields)
    and sum every other non-batch axis.

    rl4co's CVRP family stores customer features as `[B, N-1]` (depot
    omitted); pad the result with a leading zero so the per-node sum
    aligns with the encoder output `[B, N, D]`. The depot's "demand"
    is 0 by definition, so this is semantically correct.
    """
    prod = grad * value
    if prod.ndim < 2:
        raise ValueError(f"unsupported feature rank: {grad.shape}")
    # Find the node axis (size matches num_nodes or num_nodes-1).
    node_dim = None
    for d in range(1, prod.ndim):
        if prod.shape[d] in (num_nodes, num_nodes - 1):
            node_dim = d
            break
    if node_dim is None:
        raise ValueError(
            f"cannot locate node axis in feature shape {tuple(prod.shape)} (num_nodes={num_nodes})"
        )
    reduce_dims = tuple(d for d in range(1, prod.ndim) if d != node_dim)
    per_node = prod.sum(dim=reduce_dims) if reduce_dims else prod
    # Move node axis to position 1 if it isn't already.
    if node_dim != 1:
        # Now shape is [B, N] after the sum; reorder if needed.
        per_node = per_node.transpose(1, node_dim) if per_node.ndim > 2 else per_node
    if per_node.shape[1] == num_nodes - 1:
        zeros = torch.zeros(per_node.shape[0], 1, dtype=per_node.dtype, device=per_node.device)
        per_node = torch.cat([zeros, per_node], dim=1)
    return per_node


def _num_nodes(state: Any) -> int:
    """Number of encoder nodes for the current env's TensorDict.

    Routing envs expose `locs` `[B, N, 2]` → N. Scheduling envs (JSSP)
    expose `proc_times` `[B, M, N]` → N (operation count). Falls back
    to `action_mask` width for envs without either, but note that the
    action mask covers actions not nodes (JSSP: num_jobs + 1), so the
    explicit branches above must come first.
    """
    if "locs" in state:
        return int(state["locs"].shape[-2])
    if "proc_times" in state:
        return int(state["proc_times"].shape[-1])
    if "action_mask" in state:
        return int(state["action_mask"].shape[-1])
    raise ValueError("Cannot infer num_nodes from TensorDict")


def _encode(policy: Any, td: Any) -> Any:
    """Encoder forward, returning whatever shape the policy emits.

    Some rl4co encoders return a `(node_emb, init_emb)` pair; others
    (e.g. L2D for JSSP) return a nested `((node_emb, ma_emb), None)`.
    Callers downstream (`_step_logits`, probes) unwrap as needed.
    """
    return policy.policy.encoder(td) if hasattr(policy, "policy") else policy.encoder(td)


def _all_terminal(mask: Tensor | None) -> bool:
    """True if every batch element's action mask is fully False.

    rl4co envs (notably JSSP) advertise terminal states by returning
    a row of all-False mask values. Continuing to call `env.step`
    with an argmax action from a fully-masked `log_p` (which is
    `-inf` everywhere, so argmax picks 0) triggers a CUDA
    index-out-of-bounds inside the env. The rollouts use this guard
    to break before that happens.
    """
    return mask is not None and not bool(mask.any())


def _safe_action(action: Tensor, mask: Tensor | None) -> Tensor:
    """Clamp action indices for batch elements whose mask row is empty.

    Returns a new tensor where any row with no valid action is
    replaced by `0` (a harmless index, the caller is expected to
    have already decided not to record / progress those envs). When
    `mask` is `None`, the input action is returned unchanged.
    """
    if mask is None:
        return action
    valid = mask.any(dim=-1)
    if bool(valid.all()):
        return action
    return torch.where(valid, action, torch.zeros_like(action))


def step_logits(policy: Any, td: Any, encoded: Any) -> tuple[Tensor, Tensor]:
    """Single decoding step, returning `(log_probs, action_mask)`.

    Compatible with rl4co's `AttentionModelPolicy` family. The encoder
    returns either a `Tensor` or a `(node_embeddings, init_embeddings)`
    pair; the decoder expects a `PrecomputedCache` built from the node
    embeddings.
    """
    decoder = policy.policy.decoder if hasattr(policy, "policy") else policy.decoder
    embeddings = encoded[0] if isinstance(encoded, tuple) else encoded
    if hasattr(decoder, "_precompute_cache"):
        cache = decoder._precompute_cache(embeddings, num_starts=0)
    else:
        cache = embeddings
    try:
        logits, mask = decoder(td, cache, num_starts=0)
    except TypeError:
        logits, mask = decoder(td, cache)
    masked = logits.masked_fill(~mask, float("-inf")) if mask is not None else logits
    log_p = torch.log_softmax(masked, dim=-1)
    return log_p, mask


def _pack_trace(
    actions: list[Tensor],
    logp: list[Tensor],
    scores: list[Tensor],
    top_k: int,
    feature_keys: list[str],
) -> AttributionTrace:
    """Stack per-step lists, compute top-k, return on CPU."""
    if not actions:
        raise RuntimeError("Policy terminated without taking any action")
    a = torch.stack(actions, dim=1)
    lp = torch.stack(logp, dim=1)
    sc = torch.stack(scores, dim=1)
    k = min(int(top_k), sc.shape[-1])
    top_scores, top_nodes = sc.topk(k=k, dim=-1)
    return AttributionTrace(
        actions=a.cpu(),
        log_probs=lp.cpu(),
        node_scores=sc.cpu(),
        top_k_nodes=top_nodes.cpu(),
        top_k_scores=top_scores.cpu(),
        feature_keys=list(feature_keys),
    )


# Driver-callback signature:
#   target_fn(log_p, step_idx) -> (action, scalar_to_backward, value_to_record)
TargetFn = Callable[[Tensor, int], tuple[Tensor, Tensor, Tensor]]


def _rollout_grad_x_feats(
    policy: Any,
    env: Any,
    td: Any,
    *,
    top_k: int,
    feature_keys: tuple[str, ...] | None,
    max_steps: int | None,
    target_fn: TargetFn,
) -> AttributionTrace:
    """Single-pass greedy rollout with per-step gradient-x-feature attribution.

    The method-specific `target_fn` decides:
      - which action drives the env step,
      - which scalar gets backpropagated to the input features,
      - which value is recorded in `log_probs` (e.g. `log pi(a_t)` for
        plain gradient, the contrast margin for contrastive).

    Used by `gradient_attribution` and `contrastive_attribution`. IG
    and DeepLIFT have their own rollouts (multi-pass / hook-based).
    """
    device, keys = _setup(policy, feature_keys)
    feats = _collect_feature_tensors(td.to(device), keys)
    if not feats:
        raise ValueError(
            f"None of {keys!r} are floating-point tensors in the TensorDict. "
            f"Available keys: {sorted(td.keys())}"
        )
    initial_keys = list(feats.keys())
    state = _put_back(td.to(device), feats)
    num_nodes = _num_nodes(state)
    batch = int(state.batch_size[0])

    actions_per_step: list[Tensor] = []
    logp_per_step: list[Tensor] = []
    scores_per_step: list[Tensor] = []

    step = 0
    while not bool(state["done"].all()):
        if max_steps is not None and step >= max_steps:
            break
        encoded = _encode(policy, state)
        log_p, mask = step_logits(policy, state, encoded)
        if _all_terminal(mask):
            break

        action, scalar, value = target_fn(log_p, step)
        action = _safe_action(action, mask)

        grads = torch.autograd.grad(
            outputs=scalar.sum(),
            inputs=list(feats.values()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        node_score = torch.zeros(batch, num_nodes, device=device)
        for g, v in zip(grads, feats.values(), strict=False):
            if g is None:
                continue
            node_score = (
                node_score + _aggregate_node_scores(g.detach(), v.detach(), num_nodes).abs()
            )

        actions_per_step.append(action.detach())
        logp_per_step.append(value.detach())
        scores_per_step.append(node_score)

        state = state.detach()
        state["action"] = action
        state = env.step(state)["next"]
        feats = _collect_feature_tensors(state, keys)
        state = _put_back(state, feats)
        step += 1

    return _pack_trace(actions_per_step, logp_per_step, scores_per_step, top_k, initial_keys)
