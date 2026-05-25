"""Integrated Gradients (Sundararajan et al. 2017).

For each decoding step `t`:

    IG_i(x) = (x_i - x'_i) * integral_{alpha=0..1}
              d log pi(a_t | s_t(x' + alpha * (x - x')))/dx_i  dalpha

The baseline $x'$ is built per-feature by
`neuro_co.xai.baselines.build_feature_baseline`. Six modes are
available (`zero-all`, `mean-fill`, `zero-with-customers-at-depot`,
`zero-with-current-locs`, `mean-fill-on-eligible`, `wide-window`);
per-problem-per-feature defaults are locked in
`neuro_co.xai.defaults.DEFAULT_IG_BASELINE_PER_PROBLEM`.

Approximated as a midpoint Riemann sum with `ig_steps` samples per
decoding step. Cost: `ig_steps x num_steps` backward passes vs
`num_steps` for plain `gradient_attribution`.
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


def integrated_gradients(
    policy: Any,
    env: Any,
    td: Any,
    *,
    feature_keys: tuple[str, ...],
    top_k: int = 5,
    max_steps: int | None = None,
    ig_steps: int = 20,
    baseline: str | dict[str, str] | None = None,
    problem: str | None = None,
) -> AttributionTrace:
    """Integrated Gradients with problem-aware per-feature baselines.

    Parameters
    ----------
    baseline
        - ``None``: use the per-problem defaults from
          `neuro_co.xai.defaults.DEFAULT_IG_BASELINE_PER_PROBLEM`
          (requires `problem`).
        - ``str``: apply that single baseline mode to every
          ``feature_keys`` entry (legacy `"zero"` / `"mean"`
          accepted as aliases for `zero-all` / `mean-fill`).
        - ``dict[str, str]``: per-feature override of the defaults.
    problem
        Problem name; required when ``baseline is None`` or a
        ``dict`` (the dict acts as override on top of the default).
    """
    from neuro_co.xai.baselines import build_feature_baseline, IG_BASELINE_MODES

    legacy_aliases = {"zero": "zero-all", "mean": "mean-fill"}
    if isinstance(baseline, str):
        baseline = legacy_aliases.get(baseline, baseline)
        if baseline not in IG_BASELINE_MODES:
            raise ValueError(
                f"baseline must be one of {IG_BASELINE_MODES}, got {baseline!r}"
            )
    if ig_steps < 1:
        raise ValueError("ig_steps must be >= 1")
    if baseline is None and problem is None:
        raise ValueError(
            "Provide either an explicit `baseline=` or a `problem=` "
            "for the per-problem default to apply."
        )

    device, keys = _setup(policy, feature_keys)

    # --- Pass 1: deterministic rollout, no gradients, capture trajectory.
    state = td.to(device)
    num_nodes = _num_nodes(state)
    batch = int(state.batch_size[0])
    snapshots: list[Any] = []
    actions_per_step: list[Tensor] = []
    logp_per_step: list[Tensor] = []

    step = 0
    with torch.no_grad():
        while not bool(state["done"].all()):
            if max_steps is not None and step >= max_steps:
                break
            encoded = _encode(policy, state)
            log_p, mask = step_logits(policy, state, encoded)
            if _all_terminal(mask):
                break
            action = log_p.argmax(dim=-1)
            action = _safe_action(action, mask)
            chosen_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            snapshots.append(state.clone(recurse=False))
            actions_per_step.append(action.detach())
            logp_per_step.append(chosen_logp.detach())
            next_state = state.clone(recurse=False)
            next_state["action"] = action
            state = env.step(next_state)["next"]
            step += 1

    if not actions_per_step:
        raise RuntimeError("Policy terminated without taking any action")

    # --- Pass 2: per-step IG via midpoint Riemann sum.
    scores_per_step: list[Tensor] = []
    inputs_keys: list[str] = list(keys)
    for snap, action in zip(snapshots, actions_per_step, strict=False):
        inputs = {
            k: snap[k].detach().clone()
            for k in keys
            if k in snap and isinstance(snap[k], Tensor) and snap[k].dtype.is_floating_point
        }
        if not inputs:
            scores_per_step.append(torch.zeros(batch, num_nodes, device=device))
            continue
        inputs_keys = list(inputs.keys())
        # Build per-feature baseline using the requested mode(s).
        # Resolution order:
        #   1. legacy `baseline` as single mode string (applies to all)
        #   2. per-problem defaults from defaults.py
        #   3. dict override on top of defaults
        if isinstance(baseline, str):
            mode_map = {k: baseline for k in inputs}
        else:
            from neuro_co.xai.defaults import DEFAULT_IG_BASELINE_PER_PROBLEM

            defaults = DEFAULT_IG_BASELINE_PER_PROBLEM.get(
                (problem or "").lower(), {}
            )
            mode_map = {k: defaults[k] for k in inputs if k in defaults}
            if isinstance(baseline, dict):
                mode_map.update({k: v for k, v in baseline.items() if k in inputs})
        # Features with no baseline mode declared are dropped from IG.
        inputs = {k: v for k, v in inputs.items() if k in mode_map}
        if not inputs:
            scores_per_step.append(torch.zeros(batch, num_nodes, device=device))
            continue
        inputs_keys = list(inputs.keys())
        baselines = {
            k: build_feature_baseline(snap, k, mode_map[k], problem=problem).to(device)
            for k in inputs
        }
        accum = {k: torch.zeros_like(v) for k, v in inputs.items()}
        for i in range(ig_steps):
            alpha = (i + 0.5) / ig_steps  # midpoint rule
            interp = {
                k: (baselines[k] + alpha * (inputs[k] - baselines[k]))
                .detach()
                .clone()
                .requires_grad_(True)
                for k in inputs
            }
            state_interp = _put_back(snap, interp)
            encoded = _encode(policy, state_interp)
            log_p, _mask = step_logits(policy, state_interp, encoded)
            chosen_logp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            grads = torch.autograd.grad(
                outputs=chosen_logp.sum(),
                inputs=list(interp.values()),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for k_name, g in zip(interp.keys(), grads, strict=False):
                if g is not None:
                    accum[k_name] = accum[k_name] + g.detach() / ig_steps

        node_score = torch.zeros(batch, num_nodes, device=device)
        for k_name, ig in accum.items():
            attr = ig * (inputs[k_name] - baselines[k_name])
            node_score = (
                node_score + _aggregate_node_scores(attr, torch.ones_like(attr), num_nodes).abs()
            )
        scores_per_step.append(node_score)

    return _pack_trace(actions_per_step, logp_per_step, scores_per_step, top_k, inputs_keys)
