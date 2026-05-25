"""Per-problem default IG / DeepLIFT baselines.

These values are *seeded* with semantically defensible picks and
will be over-written by `neuro_co.xai.baseline_sweep.run_sweep`
once the sensitivity sweep has been executed.

Each problem maps feature_name -> baseline_mode from
`neuro_co.xai.baselines.IG_BASELINE_MODES`. Feature keys
intentionally absent from the dict are *excluded* from IG / DeepLIFT
attribution; they are typically integer scalars (`max_length`,
`to_choose`) or combinatorial masks (`num_eligible`, `ops_ma_adj`)
that have no meaningful continuous baseline.
"""

from __future__ import annotations


# Values locked by `neuro_co.xai.baseline_sweep.run_sweep` on
# trained seed-0 checkpoints (3 problems, batch 8, max_steps 4,
# ig_steps 8). Selection rule: lowest deletion-curve AUC among
# candidates that produce a 100%-valid baseline instance. Full
# sweep reports under `experiments/baseline_sweep/<problem>.json`.
DEFAULT_IG_BASELINE_PER_PROBLEM: dict[str, dict[str, str]] = {
    "vrptw": {
        "locs": "zero-with-customers-at-depot",  # sweep: AUC 0.289 vs 0.327 mean-fill
        "demand": "mean-fill",                   # sweep: AUC 0.289 vs 0.289 (tied)
        "time_windows": "mean-fill",             # only candidate that preserves (open<=close)
        "durations": "mean-fill",                # sweep: AUC 0.289 vs 0.289 (tied)
    },
    "op": {
        "locs": "zero-with-current-locs",        # sweep: AUC 0.290 vs 0.333 depot-collapse
        "prize": "zero-all",                     # sweep: AUC 0.333 vs 0.360 mean-fill
    },
    "fjsp": {
        "proc_times": "mean-fill-on-eligible",   # sweep: AUC 0.345 vs 0.358 zero-all
        # `num_eligible` is the float feature parameterising the
        # eligibility constraint family. Without it in this dict,
        # IG / DeepLIFT silently skip the eligibility family and
        # default to the first listed family in the ranking, which
        # corrupts the FJSP comparison; we therefore include it
        # with a zero baseline (no eligibility = "every op is
        # ineligible everywhere", the natural CF reference).
        "num_eligible": "zero-all",
    },
    # Below are seeded defaults for problems not in the sweep yet;
    # re-run `baseline_sweep.run_sweep` to lock.
    "jssp": {
        "proc_times": "mean-fill",
    },
    "flp": {
        "locs": "mean-fill",
        "distances": "mean-fill",
    },
    "pdp": {
        "locs": "zero-with-customers-at-depot",
    },
}


__all__ = ["DEFAULT_IG_BASELINE_PER_PROBLEM"]
