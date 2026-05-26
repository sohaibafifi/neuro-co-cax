#!/usr/bin/env bash
# End-to-end reproduction pipeline.
#
# Trains three CO problems (CVRPTW, OP, FJSP) across three seeds,
# runs the full multi-method evaluation harness (top-1 family
# adjudication with CSP-certified counterfactuals, deletion-curve
# faithfulness, Bonferroni-PAC sufficient subset, FJSP eligibility
# discriminator), and regenerates the adjudication figure.
#
# Usage
# -----
#     bash scripts/reproduce_all.sh                # full run, 5 epochs/problem
#     EPOCHS=50 bash scripts/reproduce_all.sh      # production-quality training
#     PROBLEMS="vrptw op" bash scripts/reproduce_all.sh   # subset
#     SEEDS="0" bash scripts/reproduce_all.sh             # one seed only
#     SKIP_TRAIN=1 bash scripts/reproduce_all.sh   # reuse existing checkpoints
#
# Outputs
# -------
#     outputs/<problem>/train_seed<N>/checkpoints/last.ckpt   trained policies
#     outputs/<problem>/train_seed<N>/adjudication.json       B1 (per seed)
#     experiments/faithfulness/<problem>_seed<N>.json         B2 (per seed)
#     experiments/pac_subset/vrptw_seed<N>.json               B3 (per seed)
#     experiments/fjsp_discriminator/fjsp_seed<N>.json        B4 (per seed)
#     figs/adjudication.pdf                                   figure regen

set -euo pipefail

PROBLEMS="${PROBLEMS:-vrptw op fjsp}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-16}"
T_STEPS="${T_STEPS:-8}"
CF_SHOTS="${CF_SHOTS:-128}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

PY="${PYTHON:-python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=========================================================="
echo "Reproducing full pipeline"
echo "  problems: $PROBLEMS"
echo "  seeds:    $SEEDS"
echo "  epochs:   $EPOCHS    (set EPOCHS=... to override)"
echo "  batch:    $BATCH"
echo "  CF shots: $CF_SHOTS"
echo "  root:     $ROOT"
echo "=========================================================="

# ---------------------------------------------------------------
# 1) Training
# ---------------------------------------------------------------
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    for p in $PROBLEMS; do
        for s in $SEEDS; do
            ckpt="outputs/$p/train_seed$s/checkpoints/last.ckpt"
            if [[ -f "$ckpt" ]]; then
                echo "[train] $p seed=$s already exists at $ckpt, skipping"
                continue
            fi
            echo "[train] $p seed=$s ..."
            $PY scripts/train.py "$p" "$s" --epochs "$EPOCHS"
        done
    done
else
    echo "[train] skipped (SKIP_TRAIN=1)"
fi

# ---------------------------------------------------------------
# 2) Top-1 family adjudication with CSP-certified counterfactuals
# ---------------------------------------------------------------
echo "[adjudicate] running multi-method CSP-certified adjudication"
$PY - <<PYEOF
from pathlib import Path
from neuro_co.cax.adjudicate import adjudicate_run

problems = "$PROBLEMS".split()
seeds = [int(x) for x in "$SEEDS".split()]

for p in problems:
    for s in seeds:
        run_dir = Path(f"outputs/{p}/train_seed{s}")
        if not (run_dir / "checkpoints" / "last.ckpt").exists():
            print(f"[adjudicate] skip {p} seed={s}: no checkpoint")
            continue
        print(f"[adjudicate] {p} seed={s} ...")
        adjudicate_run(
            run_dir,
            modes=("gradient", "ig", "deeplift", "contrastive", "lp"),
            num_instances=$BATCH,
            max_steps=$T_STEPS,
            cf_shots=$CF_SHOTS,
            seed=s,
            feasibility_mode="cp_sat",
        )
PYEOF

# ---------------------------------------------------------------
# 3) Deletion-curve faithfulness
# ---------------------------------------------------------------
echo "[deletion] running deletion-curve flip-rate AUC"
for p in $PROBLEMS; do
    for s in $SEEDS; do
        ckpt="outputs/$p/train_seed$s/checkpoints/last.ckpt"
        if [[ ! -f "$ckpt" ]]; then continue; fi
        $PY scripts/run_faithfulness.py "$p" "$s"
    done
done

# ---------------------------------------------------------------
# 4) Bonferroni-PAC sufficient subset (CVRPTW only by default;
#    the experiment is single-problem in the paper)
# ---------------------------------------------------------------
if [[ " $PROBLEMS " == *" vrptw "* ]]; then
    echo "[pac] running Bonferroni-PAC sufficient subset on CVRPTW"
    for s in $SEEDS; do
        ckpt="outputs/vrptw/train_seed$s/checkpoints/last.ckpt"
        if [[ ! -f "$ckpt" ]]; then continue; fi
        $PY scripts/run_pac_subset.py vrptw "$s"
    done
fi

# ---------------------------------------------------------------
# 5) FJSP eligibility-mass discriminator
#    (rank-aligned substrate where top-1 family agreement saturates)
# ---------------------------------------------------------------
if [[ " $PROBLEMS " == *" fjsp "* ]]; then
    echo "[fjsp] running FJSP eligibility-mass discriminator"
    for s in $SEEDS; do
        ckpt="outputs/fjsp/train_seed$s/checkpoints/last.ckpt"
        if [[ ! -f "$ckpt" ]]; then continue; fi
        $PY scripts/run_fjsp_discriminator.py "$s"
    done
fi

# ---------------------------------------------------------------
# 6) Adjudication figure regen
# ---------------------------------------------------------------
echo "[figure] regenerating adjudication figure"
$PY scripts/regen_adjudication_figure.py

echo "=========================================================="
echo "Done. Inspect:"
echo "  experiments/faithfulness/*.json"
echo "  experiments/pac_subset/*.json"
echo "  experiments/fjsp_discriminator/*.json"
echo "  outputs/*/train_seed*/adjudication.json"
echo "  figs/adjudication.pdf"
echo "=========================================================="
