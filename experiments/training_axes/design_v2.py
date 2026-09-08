"""Generate the overnight V2 capacity x decay x difficulty x abstraction grid."""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import asdict
from pathlib import Path

from experiments.training_axes.design import (
    CAPACITY_LEVELS,
    COMPRESSION_LEVELS,
    DIFFICULTY_LEVELS,
    ExperimentConfig,
)


V2_CAPACITIES = ("tight", "moderate", "wide")
V2_DECAY_LEVELS = ("none", "high")
V2_ABSTRACTIONS = ("concrete", "invariant", "latent")

# V2 predates the multi-task ``task_name`` field added for Task 2. Keep the
# historical schema here so regenerating the manifest is byte-for-byte exact.
V2_FIELDS = (
    "config_id", "seed", "compression", "weight_decay", "difficulty",
    "train_dists", "abstraction", "capacity", "family", "width", "layers",
    "batch_size", "steps", "learning_rate", "invariant_weight",
    "representation_weight", "latent_weight", "mapping_weight",
)


def make_v2_configs(seeds=range(6), smoke: bool = False):
    configs = []
    grid = itertools.product(
        V2_DECAY_LEVELS, DIFFICULTY_LEVELS, V2_ABSTRACTIONS,
        V2_CAPACITIES, seeds,
    )
    for decay, difficulty, abstraction, capacity, seed in grid:
        configs.append(ExperimentConfig(
            config_id=(
                f"v2_c-{decay}_d-{difficulty}_a-{abstraction}_"
                f"p-{capacity}_s-{int(seed):02d}"
            ),
            seed=int(seed),
            compression=decay,
            weight_decay=COMPRESSION_LEVELS[decay],
            difficulty=difficulty,
            train_dists=DIFFICULTY_LEVELS[difficulty],
            abstraction=abstraction,
            capacity=capacity,
            width=CAPACITY_LEVELS[capacity],
            batch_size=32 if smoke else 256,
            steps=30 if smoke else 10_000,
        ))
    return configs


def write_v2_manifest(path: Path, configs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in configs:
        row = asdict(config)
        row["train_dists"] = " ".join(map(str, config.train_dists))
        rows.append({field: row[field] for field in V2_FIELDS})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V2_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/training_axes/v2_manifest.csv")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configs = make_v2_configs(range(args.seeds), smoke=args.smoke)
    write_v2_manifest(Path(args.out), configs)
    print(f"wrote {len(configs)} V2 configurations to {args.out}")


if __name__ == "__main__":
    main()
