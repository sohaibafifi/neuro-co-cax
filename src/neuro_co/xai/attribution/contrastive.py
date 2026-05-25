"""Contrastive attribution: features that push toward `a` vs `b`.

For each decoding step we attribute the log-probability margin
`log pi(a_t) - log pi(b_t)` to the floating-point feature inputs.
Positive per-node mass means the feature pushes toward the chosen
`a_t`; negative pushes toward the alternative `b_t`. We aggregate
`|grad x feature|` (same shape as `gradient_attribution`) and store
the signed log-probability margin in `log_probs`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuro_co.xai.attribution._common import AttributionTrace, _rollout_grad_x_feats


def contrastive_attribution(
    policy: Any,
    env: Any,
    td: Any,
    *,
    feature_keys: tuple[str, ...],
    top_k: int = 5,
    max_steps: int | None = None,
    action_b: Tensor | None = None,
) -> AttributionTrace:
    """Attribute `log pi(a_t) - log pi(b_t)` per step.

    Parameters
    ----------
    action_b
        Optional `[batch, T]` long tensor of alternatives to contrast
        against. If `None`, the second-argmax over the action mask is
        used per step.
    """

    def target_fn(log_p: Tensor, step: int) -> tuple[Tensor, Tensor, Tensor]:
        action = log_p.detach().argmax(dim=-1)
        if action_b is not None and step < action_b.shape[1]:
            alt = action_b[:, step].to(log_p.device)
        else:
            masked = log_p.detach().clone()
            masked.scatter_(-1, action.unsqueeze(-1), float("-inf"))
            # If a batch element has only one valid action, every entry
            # of `masked` is -inf and `argmax` returns 0 — which may
            # itself be an *invalid* (already-masked) index. Detect via
            # the per-row max being non-finite and fall back to the
            # primary action (margin=0, contrast is a no-op for that
            # row instead of producing inf/NaN gradients).
            alt_max_vals, alt_argmax = masked.max(dim=-1)
            no_alt = ~torch.isfinite(alt_max_vals)
            alt = torch.where(no_alt, action, alt_argmax)
        a_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        b_logp = log_p.gather(-1, alt.unsqueeze(-1)).squeeze(-1)
        margin = a_logp - b_logp
        return action, margin, margin

    return _rollout_grad_x_feats(
        policy,
        env,
        td,
        top_k=top_k,
        feature_keys=feature_keys,
        max_steps=max_steps,
        target_fn=target_fn,
    )
