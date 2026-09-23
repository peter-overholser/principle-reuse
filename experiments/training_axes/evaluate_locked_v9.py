"""Evaluate one V9 checkpoint tree on locked roles only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.panel import model as model_library
from experiments.training_axes.design_v9 import V9Config
from experiments.training_axes.metrics_v9 import evaluate_checkpoint
from experiments.training_axes.task_v9 import TASKS


LOCKED_PREFIXES = (
    "lock_", "probe_lock_", "paired_lock_", "causal_lock_",
    "organization_lock_",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--structural-pairs", type=int, default=256)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"locked V9 output exists: {args.out}")
    paths = sorted(args.checkpoint_dir.glob("step-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no V9 checkpoints under {args.checkpoint_dir}")
    device = model_library.get_device(args.device)
    first = torch.load(paths[0], map_location=device, weights_only=True)
    config = V9Config(**first["config"])
    task = TASKS[config.scaffold]
    datasets = task.evaluation_sets(
        seed=0, n=args.eval_size, anchor_count=config.anchor_count,
        include_locked=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for path in paths:
            saved = torch.load(path, map_location=device, weights_only=True)
            if saved["config"] != first["config"]:
                raise ValueError(f"mixed V9 configs in {args.checkpoint_dir}")
            model = model_library.build(
                "transformer", task.VOCAB, task.SEQ_LEN, task.PAD,
                d=config.width, layers=config.layers, bottleneck_dim=0,
            ).to(device)
            model.load_state_dict(saved["model"])
            measured = evaluate_checkpoint(
                model, task, datasets, device, config.seed + saved["step"],
                config.anchor_count, include_locked=True,
                structural_pairs=args.structural_pairs,
            )
            locked = {
                key: value for key, value in measured.items()
                if key.startswith(LOCKED_PREFIXES)
            }
            row = {
                "config_id": config.config_id, "step": int(saved["step"]),
                **locked,
            }
            handle.write(json.dumps(row, allow_nan=True) + "\n")
            print(
                f"{config.config_id} step={saved['step']:>6} "
                f"lock_x={locked['lock_x_nonanchor_accuracy']:.3f} "
                f"lock_s={locked['lock_s_nonanchor_accuracy']:.3f}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
