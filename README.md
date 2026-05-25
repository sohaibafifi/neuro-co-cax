# neuro-co-cax

**Constraint-Anchored Attribution (CAX)** is a constraint-indexed
post-hoc explanation protocol for neural combinatorial-optimisation
policies. It has three named components, anchored in the problem's
constraint model and its LP relaxation:

1. **CAX attribution score**: `Λ_k(s_t) = λ_k(x) · Σ_{n ∈ F(c_k)}
   |∇_{x_n} log π(a* | s_t) · x_n|`. The gradient term measures
   local policy sensitivity at the current decoding state; the
   `λ_k(x)` term reweights this sensitivity by LP-relaxation
   *dual surrogates* obtained from Beasley-style subgradient
   ascent on the per-problem constraint model.
2. **CAX counterfactual certificate**: minimum-L1 input
   perturbation that flips the policy's argmax *and* is accepted
   by a problem-specific CSP feasibility-decision oracle (solved
   by Google OR-Tools CP-SAT in feasibility-only mode).
3. **CAX sufficiency test**: a Bonferroni-corrected Hoeffding
   bound on how many top-ranked nodes must remain unmasked for the
   policy decision to be preserved with high probability under
   randomised masking of the rest.

The package also wraps four generic feature-attribution methods
(gradient × input, Integrated Gradients, DeepLIFT, contrastive
gradient) on a common `rl4co`-friendly interface, with per-feature
baselines calibrated on a sensitivity sweep, so the comparison
between feature-level and constraint-level explanations is fair.

## Install

```bash
pip install neuro-co-cax                  # core
pip install neuro-co-cax[ortools]         # CSP feasibility backend (CVRPTW, OP, FJSP)
pip install neuro-co-cax[ortools,plots]   # full (matplotlib + pandas)
```

## Quickstart

```python
import torch
from rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from neuro_co.cax.lambda_attribution import lambda_attribution
from neuro_co.cax.cp_counterfactual import cp_counterfactual

torch.manual_seed(0)
env = CVRPTWEnv(generator_params={"num_loc": 50})
policy = ...  # load your trained rl4co policy

td = env.reset(batch_size=4)

# 1) CAX attribution score with LP-dual-surrogate weighting.
attr = lambda_attribution(
    policy, env, td,
    problem="vrptw",
    max_steps=8,
    multipliers="lp",          # or "subgrad", or None (proxy gradient)
)
print(attr.constraint_names)   # ['capacity', 'time_window', 'spatial']
print(attr.scores.shape)       # [batch, T, K]

# 2) CAX counterfactual certificate.
cf = cp_counterfactual(
    policy, env, td,
    problem="vrptw",
    feature_keys=("locs", "demand", "time_windows"),
    epsilon={"locs": 30.0, "demand": 0.02, "time_windows": 30.0},
    max_shots=128,
    feasibility_mode="cp_sat",  # stage-2 CSP oracle on per-cell winner
)
print(cf.flipped.sum().item(), "feasibility-certified flips")
```

Generic feature attributors share the same call shape and feed the
same evaluation harness (deletion-curve faithfulness, PAC
sufficient subsets, top-1 family adjudication):

```python
from neuro_co.xai.attribution import (
    gradient_attribution,
    integrated_gradients,
    deeplift_attribution,
    contrastive_attribution,
)
from neuro_co.xai.faithfulness import deletion_flip_rate

trace = integrated_gradients(
    policy, env, td,
    feature_keys=("locs", "demand", "time_windows", "durations"),
    top_k=10, max_steps=8, ig_steps=8, problem="vrptw",
)
report = deletion_flip_rate(trace, policy, env, td, top_k=5, baseline="mean")
print(report.mean_flip_rate)
```

## Multi-method evaluation harness

Three CO problems (CVRPTW, Orienteering, Flexible Job-Shop
Scheduling), constructive CSP-certified counterfactuals, and four
evaluation lenses against the LP-anchored CAX variant. Runner
scripts under `scripts/`:

```bash
git clone -b dev https://github.com/sohaibafifi/neuro-co-cax
cd neuro-co-cax
pip install -e .[ortools,plots,dev]

# Pooled 5-method top-1 family adjudication, CSP-certified.
python -c "
from pathlib import Path
from neuro_co.cax.adjudicate import adjudicate_run
for p in ('vrptw', 'op', 'fjsp'):
    for s in (0, 1, 2):
        adjudicate_run(
            Path(f'outputs/{p}/train_seed{s}'),
            modes=('gradient', 'ig', 'deeplift', 'contrastive', 'lp'),
            num_instances=16, max_steps=8, cf_shots=128, seed=s,
            feasibility_mode='cp_sat',
        )
"

# Deletion-curve flip-rate AUC across seeds and problems.
for s in 0 1 2; do for p in vrptw op fjsp; do
  python scripts/run_faithfulness.py "$p" "$s"
done; done

# Bonferroni-PAC sufficient subset per method (CVRPTW).
for s in 0 1 2; do
  python scripts/run_pac_subset.py vrptw "$s"
done

# FJSP eligibility-mass discriminator (rank-aligned substrate
# where the top-1 family agreement saturates at 1.00 across all
# methods).
for s in 0 1 2; do
  python scripts/run_fjsp_discriminator.py "$s"
done

# Adjudication figure regen from the per-seed outputs.
python scripts/regen_adjudication_figure.py
```

## License

MIT. See [LICENSE](LICENSE).
