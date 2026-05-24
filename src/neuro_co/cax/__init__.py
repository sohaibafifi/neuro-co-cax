"""Constraint-Anchored XAI (CAX).

CO-specific explanation methods that anchor attribution in the
problem's *constraints* rather than its features. v0.1 ships:

- `lambda_attribution(policy, env, td, *, problem)` -- gradient
  decomposition by constraint family (capacity, time_window,
  precedence, ...).
- `benchmark_runs([run_dir, ...])` -- batch driver: loops over
  trained-policy run dirs, runs Lambda-attribution, lands
  `lambda_attribution.json` per run plus an optional aggregate
  parquet.

Three primitives ship at paper-cax v1: `lambda_attribution`,
`cp_minimal_subset`, `cp_counterfactual`. CSP feasibility is
in `neuro_co.cax.feasibility`; LP / subgradient multiplier
backends in `neuro_co.cax.duals`.
"""

from __future__ import annotations

from neuro_co.cax.benchmark import BenchmarkRow, benchmark_run, benchmark_runs
from neuro_co.cax.constraint_map import PROBLEM_CONSTRAINTS, get_constraints
from neuro_co.cax.cp_counterfactual import CounterfactualReport, cp_counterfactual
from neuro_co.cax.cp_minimal_subset import (
    MinimalSubsetReport,
    cp_minimal_subset,
    pac_sample_count,
)
from neuro_co.cax.lambda_attribution import LambdaAttribution, lambda_attribution

__version__ = "0.1.0"

__all__ = [
    "PROBLEM_CONSTRAINTS",
    "BenchmarkRow",
    "CounterfactualReport",
    "LambdaAttribution",
    "MinimalSubsetReport",
    "__version__",
    "benchmark_run",
    "benchmark_runs",
    "cp_counterfactual",
    "cp_minimal_subset",
    "get_constraints",
    "lambda_attribution",
    "pac_sample_count",
]
