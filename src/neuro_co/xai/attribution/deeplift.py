"""DeepLIFT-Rescale attribution (Shrikumar et al. 2017).

For each decoding step `t`, the per-input contribution is::

    C_i = (x_i - x'_i) * m_i

with multipliers `m_i` produced by overriding every element-wise
nonlinearity's gradient with the Rescale rule
`(f(x) - f(x')) / (x - x')`. Linear / pointwise-equal regions fall
back to the standard gradient at `x`.

Two forward passes per step (reference, then input) plus one
backward pass. Lighter than IG (~2x vs `ig_steps`x) and uses a real
discrete reference instead of a path integral.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from neuro_co.xai.attribution._common import (
    AttributionTrace,
    _aggregate_node_scores,
    _all_terminal,
    _encode,
    _num_nodes,
    _pack_trace,
    _put_back,
    _safe_action,
    _setup,
    step_logits,
)

_RESCALE_NONLINEARITIES: tuple[type, ...] = (
    torch.nn.ReLU,
    torch.nn.GELU,
    torch.nn.SiLU,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
    torch.nn.LeakyReLU,
    torch.nn.ELU,
)


class _RescaleHook:
    """DeepLIFT-Rescale state holder for one element-wise nonlinearity.

    Records (input, output) at the reference and at the actual input,
    then rewrites the backward gradient with the Rescale multiplier
    `(f(x) - f(x_ref)) / (x - x_ref)` (Shrikumar et al. 2017).
    """

    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps
        self.mode: str = "off"  # "ref" | "x" | "off"
        self.in_ref: Tensor | None = None
        self.out_ref: Tensor | None = None
        self.in_x: Tensor | None = None
        self.out_x: Tensor | None = None

    def fwd(self, _mod: Any, inp: tuple[Tensor, ...], out: Tensor) -> None:
        x = inp[0]
        if self.mode == "ref":
            self.in_ref = x.detach()
            self.out_ref = out.detach()
        elif self.mode == "x":
            self.in_x = x.detach()
            self.out_x = out.detach()

    def bwd(
        self, _mod: Any, grad_in: tuple[Tensor | None, ...], grad_out: tuple[Tensor, ...]
    ) -> tuple[Tensor | None, ...]:
        if self.in_ref is None or self.in_x is None:
            return grad_in
        delta_in = self.in_x - self.in_ref
        delta_out = self.out_x - self.out_ref  # type: ignore[operator]
        g_out = grad_out[0]
        existing = grad_in[0]
        # FJSP / matrix-attention backbones can broadcast activations
        # across non-trivial axes (e.g. ops vs machines), producing
        # incompatible shapes between `delta_in`, `delta_out`, and
        # `g_out`. Skip the rescale rule on shape mismatch and let
        # autograd's regular gradient flow through.
        try:
            shapes_match = (
                delta_in.shape == delta_out.shape
                and g_out.shape == delta_out.shape
                and (existing is None or existing.shape == delta_in.shape)
            )
        except Exception:
            shapes_match = False
        if not shapes_match:
            return grad_in
        # Linear-rule fallback wherever |delta_in| is tiny: use the
        # pointwise derivative at x (which equals `grad_in` already).
        safe = delta_in.abs() > self.eps
        multiplier = torch.where(
            safe,
            delta_out / torch.where(safe, delta_in, torch.ones_like(delta_in)),
            torch.ones_like(delta_in),
        )
        product = g_out * multiplier
        if existing is None:
            new_grad = product
        else:
            new_grad = torch.where(safe, product, existing)
        return (new_grad, *tuple(grad_in[1:]))


def _register_rescale_hooks(
    policy: Any, modules: tuple[type, ...] = _RESCALE_NONLINEARITIES
) -> tuple[list[_RescaleHook], list[Any]]:
    """Attach forward + backward hooks on every element-wise nonlinearity."""
    hooks: list[_RescaleHook] = []
    handles: list[Any] = []
    for m in policy.modules():
        if isinstance(m, modules):
            h = _RescaleHook()
            handles.append(m.register_forward_hook(h.fwd))
            handles.append(m.register_full_backward_hook(h.bwd))
            hooks.append(h)
    return hooks, handles


def deeplift_attribution(
    policy: Any,
    env: Any,
    td: Any,
    *,
    feature_keys: tuple[str, ...],
    top_k: int = 5,
    max_steps: int | None = None,
    baseline: str | dict[str, str] | None = None,
    problem: str | None = None,
) -> AttributionTrace:
    """DeepLIFT-Rescale attribution with problem-aware baselines.

    Parameters
    ----------
    baseline
        - ``None``: use per-problem defaults from
          `neuro_co.xai.defaults.DEFAULT_IG_BASELINE_PER_PROBLEM`
          (requires `problem`).
        - ``str``: single mode applied to every feature (legacy
          `"zero"` / `"mean"` accepted as aliases for `zero-all` /
          `mean-fill`).
        - ``dict[str, str]``: per-feature override of the defaults.
    problem
        Problem name; required when ``baseline is None`` or a
        ``dict`` override.
    """
    from neuro_co.xai.baselines import build_feature_baseline, IG_BASELINE_MODES

    legacy_aliases = {"zero": "zero-all", "mean": "mean-fill"}
    if isinstance(baseline, str):
        baseline = legacy_aliases.get(baseline, baseline)
        if baseline not in IG_BASELINE_MODES:
            raise ValueError(
                f"baseline must be one of {IG_BASELINE_MODES}, got {baseline!r}"
            )
    if baseline is None and problem is None:
        raise ValueError(
            "Provide either an explicit `baseline=` or a `problem=` "
            "for the per-problem default to apply."
        )

    device, keys = _setup(policy, feature_keys)
    state = td.to(device)
    num_nodes = _num_nodes(state)
    batch = int(state.batch_size[0])

    hooks, handles = _register_rescale_hooks(policy)

    actions_per_step: list[Tensor] = []
    logp_per_step: list[Tensor] = []
    scores_per_step: list[Tensor] = []

    try:
        step = 0
        while not bool(state["done"].all()):
            if max_steps is not None and step >= max_steps:
                break

            # Probe action mask up front so we can exit before the
            # ref/x forward passes when every env is in a terminal
            # (all-False mask) state.
            with torch.no_grad():
                encoded_probe = _encode(policy, state)
                _, mask_probe = step_logits(policy, state, encoded_probe)
            if _all_terminal(mask_probe):
                break

            # ---- Build x and x_ref for this step's feature tensors.
            x_inputs: dict[str, Tensor] = {}
            for k in keys:
                if k not in state:
                    continue
                t = state[k]
                if not isinstance(t, Tensor):
                    continue
                if not t.dtype.is_floating_point:
                    t = t.to(torch.float32)
                x_inputs[k] = t.detach().clone()
            if not x_inputs:
                raise ValueError(f"None of {keys!r} are floating-point tensors in the TensorDict.")
            # Resolve per-feature baseline modes.
            if isinstance(baseline, str):
                mode_map = {k: baseline for k in x_inputs}
            else:
                from neuro_co.xai.defaults import DEFAULT_IG_BASELINE_PER_PROBLEM

                defaults = DEFAULT_IG_BASELINE_PER_PROBLEM.get(
                    (problem or "").lower(), {}
                )
                mode_map = {k: defaults[k] for k in x_inputs if k in defaults}
                if isinstance(baseline, dict):
                    mode_map.update({k: v for k, v in baseline.items() if k in x_inputs})
            # Drop features with no declared baseline.
            x_inputs = {k: v for k, v in x_inputs.items() if k in mode_map}
            if not x_inputs:
                # Skip step entirely; record an empty attribution.
                scores_per_step.append(torch.zeros(batch, num_nodes, device=device))
                step += 1
                continue
            refs = {
                k: build_feature_baseline(state, k, mode_map[k], problem=problem).to(device)
                for k in x_inputs
            }

            # ---- Reference forward (populate ref activations).
            for h in hooks:
                h.mode = "ref"
            with torch.no_grad():
                state_ref = _put_back(state, refs)
                encoded_ref = _encode(policy, state_ref)
                _ = step_logits(policy, state_ref, encoded_ref)

            # ---- Action selection from the real state, deterministically.
            for h in hooks:
                h.mode = "off"
            with torch.no_grad():
                encoded_det = _encode(policy, state)
                log_p_det, mask_det = step_logits(policy, state, encoded_det)
                action = _safe_action(log_p_det.argmax(dim=-1), mask_det)

            # ---- Input forward with grad; backward through Rescale multipliers.
            for h in hooks:
                h.mode = "x"
            x_diff: dict[str, Tensor] = {
                k: v.detach().clone().requires_grad_(True) for k, v in x_inputs.items()
            }
            state_x = _put_back(state, x_diff)
            encoded_x = _encode(policy, state_x)
            log_p, _mask = step_logits(policy, state_x, encoded_x)
            chosen_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            grads = torch.autograd.grad(
                outputs=chosen_logp.sum(),
                inputs=list(x_diff.values()),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for h in hooks:
                h.mode = "off"

            node_score = torch.zeros(batch, num_nodes, device=device)
            for k_name, g in zip(x_diff.keys(), grads, strict=False):
                if g is None:
                    continue
                delta = x_inputs[k_name] - refs[k_name]
                node_score = (
                    node_score + _aggregate_node_scores(g.detach(), delta.detach(), num_nodes).abs()
                )

            actions_per_step.append(action.detach())
            logp_per_step.append(chosen_logp.detach())
            scores_per_step.append(node_score)

            # Step env. Reset hook captures for the next step.
            for h in hooks:
                h.in_ref = h.out_ref = h.in_x = h.out_x = None
            state = state.detach()
            state["action"] = action
            state = env.step(state)["next"]
            step += 1
    finally:
        for handle in handles:
            handle.remove()

    return _pack_trace(actions_per_step, logp_per_step, scores_per_step, top_k, keys)
