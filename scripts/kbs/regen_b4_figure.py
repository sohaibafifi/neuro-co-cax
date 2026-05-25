"""Regenerate Figure 1 (adjudication, 5 methods x 3 problems).

Reads CSP-certified adjudication.json across seeds 0/1/2 for each
of CVRPTW, OP, FJSP and renders a grouped bar chart of top-$1$
family agreement with the counterfactual-derived signal, mean +/-
std error across seeds. Overwrites
``papers/paper-cax/kbs/figs/cax_w1_adjudication.pdf``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROBLEMS = ("vrptw", "op", "fjsp")
PROBLEM_LABELS = {"vrptw": "CVRPTW", "op": "OP", "fjsp": "FJSP"}
METHODS = ("gradient", "ig", "deeplift", "contrastive", "lp")
METHOD_LABELS = {
    "gradient":    r"gradient $\times$ input",
    "ig":          "Integrated Gradients",
    "deeplift":    "DeepLIFT",
    "contrastive": "contrastive gradient",
    "lp":          r"CAX (LP-anchored $\Lambda$)",
}
COLORS = ("#888888", "#1f77b4", "#2ca02c", "#d62728", "#9467bd")


def collect():
    """Return (means, stds) each shape [n_problems, n_methods]."""
    means = np.zeros((len(PROBLEMS), len(METHODS)))
    stds = np.zeros((len(PROBLEMS), len(METHODS)))
    for pi, prob in enumerate(PROBLEMS):
        for mi, meth in enumerate(METHODS):
            per_seed = []
            for seed in (0, 1, 2):
                path = Path(f"outputs/{prob}/train_seed{seed}/adjudication.json")
                d = json.loads(path.read_text())
                per_seed.append(d["mean_agreement_per_mode"][meth])
            arr = np.array(per_seed)
            means[pi, mi] = arr.mean()
            stds[pi, mi] = arr.std(ddof=0)
    return means, stds


def main(out_path: Path):
    means, stds = collect()

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    n_problems = len(PROBLEMS)
    n_methods = len(METHODS)
    bar_w = 0.155
    x = np.arange(n_problems)

    for mi, meth in enumerate(METHODS):
        offset = (mi - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            means[:, mi],
            width=bar_w,
            yerr=stds[:, mi],
            color=COLORS[mi],
            label=METHOD_LABELS[meth],
            capsize=2.5,
            edgecolor="black",
            linewidth=0.4,
        )
        # Annotate top of bar.
        for pi, b in enumerate(bars):
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.012,
                f"{h:.2f}",
                ha="center", va="bottom",
                fontsize=6.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PROBLEM_LABELS[p] for p in PROBLEMS])
    ax.set_ylabel("top-$1$ family agreement")
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(1.0, color="k", linewidth=0.4, linestyle=":", alpha=0.4)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(Path("papers/paper-cax/kbs/figs/cax_w1_adjudication.pdf"))
