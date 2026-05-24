"""CP-certified counterfactuals for neural CO policies.

Standard Wachter (2017) counterfactuals minimise `||delta||` such
that `argmax pi(x + delta) != argmax pi(x)`. They make no
guarantee that `x + delta` is a *feasible* CO instance (capacity,
time-windows, precedence, ...). For a routing problem the
resulting perturbation might violate every constraint at once.

This module solves a Constraint Optimisation Problem instead::

    minimise   ||delta||_1
    subject to delta in [-epsilon, epsilon]^d
               instance(x + delta) is feasible       (CP / arithmetic)
               argmax pi(x + delta) != argmax pi(x)  (verified)

v0.3 ships **sample + verify** (this file): draw `max_shots`
Gaussian candidates per step, filter by per-problem feasibility
(`neuro_co.cax.feasibility.is_feasible`) AND argmax-flip
(verified by a forward pass), return the smallest-L1 survivor.
Honest framing: this is *CP-feasibility-verified* counterfactual,
not *CP-optimal*. A v0.4 milestone (M3.5) replaces the arithmetic
feasibility check with a true CP-SAT decision query ("does there
exist a feasible solution for `x + delta`?") using the
`BASELINE_SOLVERS` registry.

The v0.3 method is sufficient to ground the paper-cax §4.3
*feasibility rate* claim: 100% for `cp_counterfactual` (we only
return feasible candidates), ~0% for Wachter and RouteExplainer
(which never check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.cax.feasibility import is_feasible
from neuro_co.xai.attribution import step_logits
from neuro_co.xai.attribution._common import _encode


@dataclass
class CounterfactualReport:
    """Per-step counterfactual diagnostics.

    Attributes
    ----------
    delta
        `dict[feature_key, Tensor]` of perturbations. Shapes match
        the corresponding `td[key]`. Zero where no feasible flipping
        counterfactual was found within epsilon.
    flipped
        `[batch, T]` bool tensor -- True if a flip was found.
    new_action
        `[batch, T]` long tensor of the counterfactual argmax.
        Equals the original action where `flipped == False`.
    delta_l1
        `[batch, T]` float tensor of the L1 norm of the chosen delta.
    epsilon
        Per-feature L-infinity bound used during sampling.
    method
        `'sample_verify'` in v0.3; `'cp_sat_decision'` in M3.5.
    """

    delta: dict[str, torch.Tensor]
    flipped: torch.Tensor
    new_action: torch.Tensor
    delta_l1: torch.Tensor
    epsilon: float
    method: str = "sample_verify"


def cp_counterfactual(
    policy: Any,
    env: Any,
    td: Any,
    *,
    problem: str,
    epsilon: float | dict[str, float] = 0.1,
    max_shots: int = 32,
    sigma: float | dict[str, float] | None = None,
    feature_keys: tuple[str, ...] = (),
    max_steps: int | None = 8,
    time_limit_s: float = 1.0,
    seed: int = 0,
    perturb_one_at_a_time: bool = True,
    feasibility_mode: str = "arithmetic",
) -> CounterfactualReport:
    """Sample-and-verify CP-certified counterfactuals per decoding step.

    Parameters
    ----------
    policy, env, td
        Standard rl4co triplet.
    problem
        Selects the feasibility encoding from
        `neuro_co.cax.feasibility.<problem>`.
    epsilon
        L-infinity ball radius around each feature value.
    max_shots
        Number of candidate `delta` samples drawn per step.
    sigma
        Gaussian std for the candidate draw (default = epsilon / 3,
        so most samples land within the epsilon-ball).
    feature_keys
        Tensor keys to perturb (required -- depends on the problem).
    max_steps
        Cap on decoding-step count.
    time_limit_s
        Reserved for the M3.5 CP-SAT backend.
    seed
        RNG seed for reproducible sampling.
    """
    if not feature_keys:
        raise ValueError(
            "feature_keys is required. Pass the problem-specific tuple "
            "(e.g. via `neuro_co.xai.concept_registry.get('vrptw').feature_keys`)."
        )
    if feasibility_mode not in ("arithmetic", "cp_sat"):
        raise ValueError(
            f"feasibility_mode must be 'arithmetic' or 'cp_sat'; got "
            f"{feasibility_mode!r}"
        )
    device = next(policy.parameters()).device
    policy.eval()

    def _per_key(arg: float | dict[str, float] | None, default: float) -> dict[str, float]:
        if isinstance(arg, dict):
            return {k: float(arg.get(k, default)) for k in feature_keys}
        if arg is None:
            return {k: float(default) for k in feature_keys}
        return {k: float(arg) for k in feature_keys}

    eps_per_key = _per_key(epsilon, 0.1)
    sigma_per_key = _per_key(sigma, max(eps_per_key.values()) / 3.0)

    state = td.clone(recurse=False).to(device)
    B = int(state.batch_size[0])
    T_max = max_steps if max_steps is not None else 32

    delta_accum: dict[str, torch.Tensor] = {}
    # Pre-allocate per-key delta-storage shaped [T_max, *feature_shape].
    for key in feature_keys:
        if key not in state:
            continue
        t_ref = state[key]
        if not isinstance(t_ref, torch.Tensor):
            continue
        delta_accum[key] = torch.zeros(
            (T_max, *t_ref.shape), dtype=torch.float32, device=device
        )

    flipped = torch.zeros(B, T_max, dtype=torch.bool)
    new_action = torch.zeros(B, T_max, dtype=torch.long)
    delta_l1 = torch.zeros(B, T_max, dtype=torch.float32)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    actual_T = 0

    with torch.no_grad():
        for t in range(T_max):
            if bool(state["done"].all()):
                break
            actual_T = t + 1

            log_p_orig, _ = step_logits(policy, state, _encode(policy, state))
            orig_action = log_p_orig.argmax(dim=-1)
            new_action[:, t] = orig_action.cpu()

            # Track best (min-L1) feasible flipping candidate per batch element.
            best_l1 = torch.full((B,), float("inf"))
            best_delta: dict[str, torch.Tensor] = {
                k: torch.zeros_like(delta_accum[k][t]) for k in delta_accum
            }
            best_new_action = orig_action.clone()

            for shot in range(max_shots):
                # Pick which feature key(s) to perturb this shot.
                # `perturb_one_at_a_time=True` rotates through keys
                # round-robin -- prevents joint perturbations from
                # tripping feasibility on any one feature (the common
                # failure mode on real instances). With False, all
                # keys perturbed together (Wachter-style joint).
                if perturb_one_at_a_time and feature_keys:
                    active_keys = (feature_keys[shot % len(feature_keys)],)
                else:
                    active_keys = feature_keys
                shot_delta: dict[str, torch.Tensor] = {}
                shot_state = state.clone(recurse=False)
                for key in feature_keys:
                    if key not in state:
                        continue
                    t_ref = state[key].float()
                    if key in active_keys:
                        sig_k = sigma_per_key[key]
                        eps_k = eps_per_key[key]
                        noise = (
                            torch.empty_like(t_ref.cpu())
                            .normal_(0.0, sig_k, generator=gen)
                            .clamp_(-eps_k, eps_k)
                            .to(t_ref.device)
                        )
                    else:
                        noise = torch.zeros_like(t_ref)
                    shot_delta[key] = noise
                    shot_state[key] = (t_ref + noise).to(state[key].dtype)

                # Cheap arithmetic feasibility first (always).
                feas = is_feasible(shot_state, problem, mode="arithmetic")
                log_p_new, _ = step_logits(policy, shot_state, _encode(policy, shot_state))
                cand_action = log_p_new.argmax(dim=-1)
                cand_flipped = (cand_action != orig_action).cpu()

                # Per-element L1 of the candidate delta.
                cand_l1 = torch.zeros(B)
                for dz in shot_delta.values():
                    cand_l1 = cand_l1 + dz.detach().cpu().abs().flatten(1).sum(dim=-1)

                # Provisional accept: feasible + flipped + smaller L1.
                provisional = feas & cand_flipped & (cand_l1 < best_l1)

                # Stage 2 (M3.5): for cp_sat mode, run the expensive
                # CP-SAT decision only on provisional survivors. This
                # keeps the per-step cost bounded by O(1) solver call
                # rather than O(max_shots).
                if feasibility_mode == "cp_sat" and provisional.any():
                    cp_feas = is_feasible(
                        shot_state, problem, mode="cp_sat", time_limit_s=time_limit_s
                    )
                    provisional = provisional & cp_feas

                if provisional.any():
                    idx = provisional.nonzero(as_tuple=False).flatten().tolist()
                    for b in idx:
                        best_l1[b] = cand_l1[b]
                        best_new_action[b] = cand_action[b]
                        for key in best_delta:
                            best_delta[key][b] = shot_delta[key][b].detach()

            # Lock per-step results.
            for key in delta_accum:
                delta_accum[key][t] = best_delta[key]
            found = best_l1.isfinite()
            flipped[:, t] = found
            delta_l1[:, t] = torch.where(found, best_l1, torch.zeros_like(best_l1))
            new_action[:, t] = torch.where(found, best_new_action.cpu(), orig_action.cpu())

            # Advance env using original action so trajectory matches the trace.
            state["action"] = orig_action
            state = env.step(state)["next"]

    # Trim to actual T.
    delta_final: dict[str, torch.Tensor] = {
        k: v[:actual_T].cpu() for k, v in delta_accum.items()
    }
    eps_report = (
        float(epsilon) if isinstance(epsilon, (int, float)) else max(eps_per_key.values())
    )
    method_tag = (
        "sample_verify_cp_sat" if feasibility_mode == "cp_sat" else "sample_verify"
    )
    return CounterfactualReport(
        delta=delta_final,
        flipped=flipped[:, :actual_T],
        new_action=new_action[:, :actual_T],
        delta_l1=delta_l1[:, :actual_T],
        epsilon=eps_report,
        method=method_tag,
    )
