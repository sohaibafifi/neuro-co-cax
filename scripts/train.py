"""Train a single (problem, seed) checkpoint with rl4co.

Writes ``outputs/{problem}/train_seed{seed}/checkpoints/last.ckpt``
which is the layout consumed by the adjudication, deletion-curve,
PAC, and FJSP-discriminator runners.

Usage
-----
::

    python scripts/train.py vrptw 0
    python scripts/train.py op    1 --epochs 20
    python scripts/train.py fjsp  2 --epochs 30 --num-jobs 10 --num-machines 5

Notes
-----
- CVRPTW and OP use ``AttentionModel`` from rl4co with
  REINFORCE-with-rollout-baseline.
- FJSP uses ``L2DModel`` (matrix-attention learning-to-dispatch).
- For a quick reproducibility smoke test, the defaults are
  ``--epochs 5`` and ``--num-instances-per-epoch 1280``.
  Production-quality checkpoints typically need 50-100 epochs.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch


def _build_env(problem: str, num_loc: int, num_jobs: int, num_machines: int):
    if problem in ("vrptw", "cvrptw"):
        m = importlib.import_module("rl4co.envs.routing.cvrptw.env")
        return m.CVRPTWEnv(generator_params={"num_loc": num_loc})
    if problem == "op":
        m = importlib.import_module("rl4co.envs.routing.op.env")
        return m.OPEnv(generator_params={"num_loc": num_loc})
    if problem == "fjsp":
        m = importlib.import_module("rl4co.envs.scheduling.fjsp.env")
        return m.FJSPEnv(
            generator_params={"num_jobs": num_jobs, "num_machines": num_machines}
        )
    raise ValueError(f"unknown problem {problem!r}")


def _build_model(problem: str, env):
    if problem in ("vrptw", "cvrptw", "op"):
        from rl4co.models import AttentionModel
        from rl4co.models.zoo.am.policy import AttentionModelPolicy

        policy = AttentionModelPolicy(
            env_name="cvrptw" if problem == "vrptw" else problem,
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
        )
        return AttentionModel(env=env, policy=policy)
    if problem == "fjsp":
        # rl4co's L2D model for FJSP (matrix-attention dispatcher).
        from rl4co.models import L2DModel

        return L2DModel(env=env)
    raise ValueError(f"unknown problem {problem!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("problem", choices=("vrptw", "cvrptw", "op", "fjsp"))
    p.add_argument("seed", type=int)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument(
        "--num-instances-per-epoch", type=int, default=1280,
        help="training instances drawn per epoch",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-loc", type=int, default=50)
    p.add_argument("--num-jobs", type=int, default=10)
    p.add_argument("--num-machines", type=int, default=5)
    p.add_argument(
        "--out-root", default="outputs",
        help="output root; checkpoint goes to {out}/{problem}/train_seed{seed}/checkpoints/",
    )
    p.add_argument(
        "--accelerator", default="auto",
        help="lightning accelerator (auto, cpu, gpu, mps)",
    )
    args = p.parse_args()

    problem = "vrptw" if args.problem == "cvrptw" else args.problem

    torch.manual_seed(args.seed)

    env = _build_env(problem, args.num_loc, args.num_jobs, args.num_machines)
    model = _build_model(problem, env)

    out_dir = Path(args.out_root) / problem / f"train_seed{args.seed}"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Hydra-style snapshot expected by `neuro_co.cax.benchmark._load_hydra_cfg`.
    hydra_dir = out_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = (
        f"problem: {problem}\n"
        f"seed: {args.seed}\n"
        f"env:\n"
        f"  _target_: {type(env).__module__}.{type(env).__name__}\n"
        f"  generator_params: {{}}\n"
        f"model:\n"
        f"  _target_: {type(model).__module__}.{type(model).__name__}\n"
    )
    (hydra_dir / "config.yaml").write_text(cfg_text)

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="last",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=1,
        enable_checkpointing=True,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
        default_root_dir=str(out_dir),
    )

    # rl4co models override `train_dataloader` so the trainer can fit
    # without passing an explicit dataset; the batch size is taken from
    # the model's hyperparameters when defined.
    if hasattr(model, "hparams") and hasattr(model.hparams, "batch_size"):
        model.hparams.batch_size = args.batch_size
    if hasattr(model, "hparams") and hasattr(model.hparams, "train_data_size"):
        model.hparams.train_data_size = args.num_instances_per_epoch

    trainer.fit(model)

    # rl4co's ModelCheckpoint saves to "<dirpath>/last.ckpt"; copy it
    # to "<dirpath>/last.ckpt" explicitly in case the callback wrote
    # a versioned file.
    candidate = ckpt_dir / "last.ckpt"
    if not candidate.exists():
        cands = sorted(ckpt_dir.glob("*.ckpt"))
        if cands:
            cands[-1].rename(candidate)
    print(f"wrote checkpoint: {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
