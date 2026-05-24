# neuro-co-cax

**Constraint-Anchored Attribution (CAX)**: a post-hoc explainer for neural
combinatorial-optimisation policies. Three primitives, one constraint-programming
relaxation:

1. **Λ-attribution**: gradient × feature decomposed by constraint family, scaled
   by the LP-relaxation Lagrangian multiplier `λ_k*` (mean / sum / max
   aggregation supported).
2. **Feasibility-certified counterfactual**: minimum-L1 perturbation that flips
   the policy's argmax **and** is accepted by a problem-specific CSP
   feasibility-decision oracle.
3. **Bonferroni-PAC sufficient subset**: family-wise (1-δ)-PAC Hoeffding test
   along the greedy attribution ordering.

Companion paper: *Constraint-Anchored Attribution: Feasibility-Certified
Counterfactuals and Bonferroni-PAC Sufficient Subsets for Neural CO Policies*
(arXiv: XXXX.XXXXX).

## Install

```bash
pip install neuro-co-cax                  # core
pip install neuro-co-cax[ortools]         # CSP feasibility backend (CVRPTW, OP, FJSP)
pip install neuro-co-cax[ortools,plots]   # full (with matplotlib + pandas)
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

# 1) Λ-attribution with LP-anchored multipliers.
attr = lambda_attribution(
    policy, env, td,
    problem="vrptw",
    max_steps=8,
    multipliers="lp",          # or "subgrad", or None (proxy gradient)
)
print(attr.constraint_names)   # ['capacity', 'time_window', 'spatial']
print(attr.scores.shape)       # [batch, T, K]

# 2) Feasibility-certified counterfactual.
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

## Reproducing paper numbers

```bash
git clone https://github.com/sohaibafifi/neuro-co-cax
cd neuro-co-cax
pip install -e .[ortools,plots,dev]
# 1) Train a policy (one per problem, three seeds each)
# 2) Adjudicate:
python examples/adjudicate_all.py --out results/
```

## Citation

```bibtex
@article{afifi2026cax,
  title   = {Constraint-Anchored Attribution: Feasibility-Certified
             Counterfactuals and Bonferroni-PAC Sufficient Subsets for
             Neural CO Policies},
  author  = {Afifi, Sohaib},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}
```

## License

MIT. See [LICENSE](LICENSE).
