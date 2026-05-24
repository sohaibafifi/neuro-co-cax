"""Reproduce paper-cax §3 Orienteering-Problem row.

Expected output on the released checkpoints (3 seeds, B=16, T=8, M=128):

    [op pooled] n_cert=281  proxy=0.352  lp=0.772
        diff=+0.420  CI95=[+0.324, +0.512]  McNemar p=9.3e-15

OP combinatorial certification uses the OR-Tools CP-SAT
prize-collecting tour decision model (install with
`pip install neuro-co-cax[ortools]`).

Usage:

    python examples/adjudicate_op.py \\
        --ckpt outputs/op/train_seed{seed}/checkpoints/last.ckpt \\
        --num-loc 20

    # Smoke:
    python examples/adjudicate_op.py --seeds 0 --batch-size 4 --cf-shots 16
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _runner import cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli("op"))
