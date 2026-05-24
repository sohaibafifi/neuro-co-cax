"""Reproduce paper-cax §3 CVRPTW row.

Expected output on the released checkpoints (3 seeds, B=16, T=8, M=128):

    [vrptw pooled] n_cert=344  proxy=0.750  lp=0.965
        diff=+0.215  CI95=[+0.166, +0.265]  McNemar p=4.5e-17

Usage:

    # With paper checkpoints:
    python examples/adjudicate_vrptw.py \\
        --ckpt outputs/vrptw/train_seed{seed}/checkpoints/last.ckpt

    # Smoke run (untrained policy, noisy numbers):
    python examples/adjudicate_vrptw.py --seeds 0 --batch-size 4 --cf-shots 16

Combinatorial certification uses an OR-Tools CSP feasibility-decision
model for CVRPTW (install with `pip install neuro-co-cax[ortools]`);
the arithmetic check alone runs without that extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _runner import cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli("vrptw"))
