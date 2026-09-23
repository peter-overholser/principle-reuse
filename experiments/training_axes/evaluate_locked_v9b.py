"""Evaluate one frozen V9-A endpoint on the committed V9-B lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.panel import model as model_library
from experiments.training_axes.design_v9 import V9Config
from experiments.training_axes.evaluate_locked_v9 import LOCKED_PREFIXES
from experiments.training_axes.metrics_v9 import evaluate_checkpoint
from experiments.training_axes.task_v9 import TASKS
from experiments.training_axes.v9b_common import ENDPOINT, derive_seed, verify_reveal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--structural-pairs", type=int, default=256)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"locked V9-B output exists: {args.out}")
    reveal = verify_reveal(args.commitment, args.reveal)
    path = args.checkpoint_dir / f"step-{ENDPOINT:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing V9-B endpoint checkpoint: {path}")

    device = model_library.get_device(args.device)
    saved = torch.load(path, map_location=device, weights_only=True)
    if int(saved["step"]) != ENDPOINT:
        raise ValueError("V9-B only evaluates the update-20,000 endpoint")
    config = V9Config(**saved["config"])
    task = TASKS[config.scaffold]
    dataset_seed = derive_seed(reveal, "dataset", config.scaffold)
    assay_seed = derive_seed(
        reveal, "assay", config.scaffold, config.seed, ENDPOINT
    )
    datasets = task.evaluation_sets(
        seed=dataset_seed, n=args.eval_size,
        anchor_count=config.anchor_count, include_locked=True,
    )
    model = model_library.build(
        "transformer", task.VOCAB, task.SEQ_LEN, task.PAD,
        d=config.width, layers=config.layers, bottleneck_dim=0,
    ).to(device)
    model.load_state_dict(saved["model"])
    measured = evaluate_checkpoint(
        model, task, datasets, device, assay_seed,
        config.anchor_count, include_locked=True,
        structural_pairs=args.structural_pairs,
    )
    locked = {
        key: value for key, value in measured.items()
        if key.startswith(LOCKED_PREFIXES)
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config_id": config.config_id, "step": ENDPOINT, **locked,
    }, allow_nan=True) + "\n")
    print(
        f"{config.config_id} step={ENDPOINT:>6} "
        f"lock_x={locked['lock_x_nonanchor_accuracy']:.3f} "
        f"lock_s={locked['lock_s_nonanchor_accuracy']:.3f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
