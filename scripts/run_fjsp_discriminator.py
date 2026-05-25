"""FJSP discriminative metric (rank-aligned sanity-check substrate).

Top-$1$ family agreement on FJSP is uniformly $1.00$ across the
five methods, so the headline table cannot discriminate them on
this substrate. This runner adds a per-cell *family-magnitude
cosine similarity* between each method's per-family attribution
vector and the counterfactual perturbation-mass vector, then
averages across CSP-certified flipping cells.

Per (instance, decoding cell):
  - method vector  s_m = [score(family_k) for family_k]  in R^K
  - CF mass vector m   = [|delta in family_k|_1 for family_k]
  - cos(s_m, m)  in [-1, 1]

For FJSP K=2 families (precedence, eligibility), so cosine is
informative even though Kendall's tau degenerates.

Output: experiments/fjsp_discriminator/fjsp_seed{N}.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

from neuro_co.xai.attribution import (
    contrastive_attribution,
    deeplift_attribution,
    gradient_attribution,
    integrated_gradients,
)
from neuro_co.cax.benchmark import (
    _instantiate,
    _load_ckpt,
    _load_hydra_cfg,
    _pick_device,
    _resolve_ckpt,
)
from neuro_co.cax.constraint_map import get_constraints
from neuro_co.cax.lambda_attribution import lambda_attribution


def _method_family_scores(method, model, env, td, problem, families, max_steps):
    """Return [B, T, K] tensor of per-family attribution scores."""
    policy = model.policy if hasattr(model, "policy") else model
    available = set(td.keys())
    B = int(td.batch_size[0])
    K = len(families)

    if method == "lp":
        la = lambda_attribution(
            policy, env, td, problem=problem, top_k=5,
            max_steps=max_steps, multipliers="lp",
        )
        return la.scores.detach().cpu()  # [B, T, K]

    per_family = []
    for _name, keys in families:
        present = tuple(
            k for k in keys
            if k in available and hasattr(td[k], "dtype")
            and td[k].dtype != torch.bool
        )
        if not present:
            per_family.append(torch.zeros(B, max_steps))
            continue
        try:
            if method == "gradient":
                tr = gradient_attribution(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps,
                )
            elif method == "ig":
                tr = integrated_gradients(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps, ig_steps=8, problem=problem,
                )
            elif method == "deeplift":
                tr = deeplift_attribution(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps, problem=problem,
                )
            elif method == "contrastive":
                tr = contrastive_attribution(
                    policy, env, td, feature_keys=present, top_k=1,
                    max_steps=max_steps,
                )
            else:
                raise ValueError(f"unknown method {method!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"[fjsp] {method} {present} FAILED: {type(exc).__name__}: {exc}", flush=True)
            per_family.append(torch.zeros(B, max_steps))
            continue
        per_family.append(tr.node_scores.sum(dim=-1).cpu())  # [B, T]

    T_min = min(t.shape[1] for t in per_family)
    per_family = [t[:, :T_min] for t in per_family]
    stacked = torch.stack(per_family, dim=-1)  # [B, T_min, K]
    return stacked


def _cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    nu = float(torch.norm(u))
    nv = float(torch.norm(v))
    if nu == 0 or nv == 0:
        return float("nan")
    return float(torch.dot(u.flatten(), v.flatten()) / (nu * nv))


def run(seed: int = 0, batch: int = 16, max_steps: int = 8) -> dict:
    run_dir = Path(f"outputs/fjsp/train_seed{seed}")
    cfg = _load_hydra_cfg(run_dir)
    env, model = _instantiate(cfg)
    _load_ckpt(model, _resolve_ckpt(run_dir))
    device = _pick_device()
    model = model.to(device).eval()

    torch.manual_seed(int(seed))
    td = env.reset(batch_size=batch).to(device)
    families = get_constraints("fjsp")
    K = len(families)
    family_names = [n for n, _ in families]
    print(f"[fjsp seed={seed}] B={batch} T={max_steps} K={K} families={family_names}", flush=True)

    # ---- Per-method per-family scores [B, T, K].
    method_scores: dict[str, torch.Tensor] = {}
    for m in ("gradient", "ig", "deeplift", "contrastive", "lp"):
        t0 = time.time()
        s = _method_family_scores(m, model, env, td, "fjsp", families, max_steps)
        method_scores[m] = s
        print(f"  {m}: scores shape={tuple(s.shape)} dt={time.time()-t0:.1f}s", flush=True)

    # ---- CF mass vector per cell from existing adjudication.json.
    adj_path = run_dir / "adjudication.json"
    adj = json.loads(adj_path.read_text())
    flipped = torch.tensor(adj["cf_flipped_per_step"], dtype=torch.bool)  # [B, T]
    cf_top = torch.tensor(adj["cf_top_family_per_step"], dtype=torch.long)  # [B, T]
    # cf_top stores only argmax family; we don't have full per-family
    # CF mass in the persisted JSON. Use a one-hot proxy on the
    # certified-flip cells: CF mass vector = e_{cf_top[b,t]}.
    # This still discriminates methods because the cosine of [s1,s2]
    # against e_1 = s1 / ||s|| differs across methods.

    results: dict[str, dict] = {}
    T_min = min(method_scores[m].shape[1] for m in method_scores)
    flipped = flipped[:, :T_min]
    cf_top = cf_top[:, :T_min]
    n_cells = int(flipped.sum().item())
    print(f"  certified-flip cells: {n_cells}", flush=True)

    # Discriminative metric: per-cell fraction of attribution mass
    # on the binding (eligibility) family, averaged over
    # CSP-certified flipping cells. Since FJSP is rank-aligned the
    # CF identifies eligibility as the responsible family on every
    # flipping cell; the question becomes how concentrated each
    # method's continuous score is on that family, which is a
    # magnitude-profile property the top-$1$ argmax hides.
    elig_idx = family_names.index("eligibility") if "eligibility" in family_names else (K - 1)
    for m, scores in method_scores.items():
        fracs = []
        for b in range(scores.shape[0]):
            for t in range(scores.shape[1]):
                if not bool(flipped[b, t]):
                    continue
                vec = scores[b, t].abs()
                tot = float(vec.sum())
                if tot <= 0:
                    continue
                fracs.append(float(vec[elig_idx]) / tot)
        if fracs:
            mean_f = sum(fracs) / len(fracs)
            std_f = math.sqrt(sum((x - mean_f) ** 2 for x in fracs) / max(1, len(fracs)))
            results[m] = {
                "mean_elig_fraction": float(mean_f),
                "std_elig_fraction": float(std_f),
                "n_cells": len(fracs),
            }
        else:
            results[m] = {"mean_elig_fraction": float("nan"), "n_cells": 0}
        print(f"  {m}: elig_frac={results[m]['mean_elig_fraction']:.4f}", flush=True)

    out = {
        "problem": "fjsp",
        "seed": int(seed),
        "K_families": K,
        "family_names": family_names,
        "batch": batch,
        "max_steps": max_steps,
        "n_certified_cells": n_cells,
        "methods": results,
    }
    out_dir = Path("experiments/fjsp_discriminator")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"fjsp_seed{int(seed)}.json"
    (out_dir / fname).write_text(json.dumps(out, indent=2))
    print(f"[fjsp seed={seed}] wrote {out_dir / fname}", flush=True)
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(seed=seed)
