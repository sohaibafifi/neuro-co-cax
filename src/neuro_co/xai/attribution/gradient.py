"""Gradient x input attribution for autoregressive CO policies.

For each decoding step `t` we record:

- the action taken `a_t` (index of the chosen node);
- the log-probability `log pi(a_t | s_t)` under the policy;
- the gradient of that log-prob w.r.t. the node-feature tensor;
- the per-node attribution score = sum over feature dims of
  `(grad x feature)`;
- the top-k attributed node indices and their scores.

The implementation is policy-agnostic, anything implementing the
`__call__(td, ...) -> {"actions": ..., "log_likelihood": ...}` shape
that rl4co policies expose works. No custom encoder / decoder hooks
required.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from neuro_co.xai.attribution._common import AttributionTrace, _rollout_grad_x_feats


def gradient_attribution(
    policy: Any,
    env: Any,
    td: Any,
    *,
    feature_keys: tuple[str, ...],
    top_k: int = 5,
    max_steps: int | None = None,
) -> AttributionTrace:
    """Roll out `policy` on `env` and record gradient x input per step.

    The policy is rolled out greedily (argmax). At each decision step we
    backpropagate `log pi(a_t | s_t)` through the floating-point feature
    fields present in the initial state, then aggregate per-node scores
    by summing absolute gradient x feature over feature dims.
    """

    def target_fn(log_p: Tensor, _step: int) -> tuple[Tensor, Tensor, Tensor]:
        action = log_p.detach().argmax(dim=-1)
        chosen_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return action, chosen_logp, chosen_logp

    return _rollout_grad_x_feats(
        policy,
        env,
        td,
        top_k=top_k,
        feature_keys=feature_keys,
        max_steps=max_steps,
        target_fn=target_fn,
    )
