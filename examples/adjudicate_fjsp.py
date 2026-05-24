"""Reproduce paper-cax §3 FJSP rank-alignment row.

Expected output on the released checkpoints (3 seeds, B=16, T=8, M=128):

    [fjsp pooled] n_cert=59  proxy=1.000  lp=1.000
        diff=+0.000  CI95=[+0.000, +0.000]  McNemar p=1.00e+00

FJSP is the rank-aligned regime: the eligible-machine count tensor
dominates both the LP shadow price and the proxy gradient mass, so
the two backends agree on every CSP-certified flip. Combinatorial
certification uses an OR-Tools CP-SAT FJSP decision model (install
with `pip install neuro-co-cax[ortools]`).

Usage:

    python examples/adjudicate_fjsp.py \\
        --ckpt outputs/fjsp/train_seed{seed}/checkpoints/last.ckpt

    # Smoke:
    python examples/adjudicate_fjsp.py --seeds 0 --batch-size 4 --cf-shots 16
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _runner import cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli("fjsp"))
