"""Generate the registered difference-comparison principle-reuse experiment."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from experiments.training_axes.design import (
    CAPACITY_LEVELS, COMPRESSION_LEVELS, ExperimentConfig, write_manifest,
)
from experiments.training_axes import task_differences


CAPACITIES = ("tight", "moderate", "wide")
ABSTRACTIONS = ("concrete", "invariant", "latent")


def make_configs(seeds=range(6), smoke=False):
    configs = []
    grid = itertools.product(
        task_differences.DIFFICULTY_LEVELS, ABSTRACTIONS, CAPACITIES, seeds
    )
    for difficulty, abstraction, capacity, seed in grid:
        configs.append(ExperimentConfig(
            config_id=(
                f"v3_t-differences_c-none_d-{difficulty}_a-{abstraction}_"
                f"p-{capacity}_s-{int(seed):02d}"
            ),
            seed=int(seed),
            compression="none",
            weight_decay=COMPRESSION_LEVELS["none"],
            difficulty=difficulty,
            train_dists=task_differences.DIFFICULTY_LEVELS[difficulty],
            abstraction=abstraction,
            task_name="differences",
            capacity=capacity,
            width=CAPACITY_LEVELS[capacity],
            batch_size=32 if smoke else 256,
            steps=30 if smoke else 10_000,
        ))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="experiments/training_axes/task2_manifest.csv"
    )
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configs = make_configs(range(args.seeds), smoke=args.smoke)
    write_manifest(Path(args.out), configs)
    print(f"wrote {len(configs)} task-2 configurations to {args.out}")


if __name__ == "__main__":
    main()
