"""Repeat the development-only V9 causal assay on one frozen R2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.panel import model as model_library
from experiments.training_axes import metrics
from experiments.training_axes.design_v9 import V9Config
from experiments.training_axes.design_v9_r2 import R2_ARMS, R2_SEEDS
from experiments.training_axes.metrics_v9 import as_tuple
from experiments.training_axes.task_v9 import TASKS


EVAL_SEED_BASE = 9_920_000
CAUSAL_SEED_BASE = 9_930_000


def evaluate(
    checkpoint, out, device, *, replicates=4, eval_size=4096,
    n_pairs=1024, n_random=16, expected_step=20_000,
):
    checkpoint = Path(checkpoint)
    out = Path(out)
    if out.exists():
        rows = [line for line in out.read_text().splitlines() if line]
        if len(rows) == replicates:
            print(f"complete; skipping {out.stem}")
            return
        raise RuntimeError(f"partial V9-R2 reliability output: {out}")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = V9Config(**saved["config"])
    if (
        config.scaffold != "circle"
        or config.arm not in R2_ARMS
        or config.seed not in R2_SEEDS
        or int(saved["step"]) != expected_step
    ):
        raise ValueError(f"unexpected V9-R2 endpoint checkpoint: {checkpoint}")
    task = TASKS[config.scaffold]
    model = model_library.build(
        "transformer", task.VOCAB, task.SEQ_LEN, task.PAD,
        d=config.width, layers=config.layers, bottleneck_dim=0,
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    rows = []
    for replicate in range(replicates):
        datasets = task.evaluation_sets(
            seed=EVAL_SEED_BASE + 10_000 * config.seed + replicate,
            n=eval_size, anchor_count=config.anchor_count,
            include_locked=False,
        )
        causal = metrics.causal_reuse(
            model, as_tuple(datasets["source"]), as_tuple(datasets["dev_s"]),
            device,
            seed=CAUSAL_SEED_BASE + 10_000 * config.seed + replicate,
            n_pairs=n_pairs, n_random=n_random,
        )
        row = {
            "config_id": config.config_id,
            "seed": config.seed,
            "arm": config.arm,
            "scaffold": config.scaffold,
            "checkpoint_step": int(saved["step"]),
            "replicate": replicate,
            "eval_seed": EVAL_SEED_BASE + 10_000 * config.seed + replicate,
            "causal_seed": CAUSAL_SEED_BASE + 10_000 * config.seed + replicate,
            "eval_size": eval_size,
            "n_pairs": n_pairs,
            "n_random": n_random,
            **causal,
        }
        rows.append(row)
        print(
            f"{config.config_id} replicate={replicate} "
            f"target={causal['causal_target_to_target']:.3f} "
            f"source={causal['causal_source_to_target']:.3f} "
            f"random={causal['causal_random']:.3f} "
            f"valid={int(causal['causal_valid'])}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=4096)
    parser.add_argument("--n-pairs", type=int, default=1024)
    parser.add_argument("--n-random", type=int, default=16)
    parser.add_argument("--expected-step", type=int, default=20_000)
    args = parser.parse_args()
    evaluate(
        args.checkpoint, args.out, model_library.get_device(args.device),
        replicates=args.replicates, eval_size=args.eval_size,
        n_pairs=args.n_pairs, n_random=args.n_random,
        expected_step=args.expected_step,
    )


if __name__ == "__main__":
    main()
