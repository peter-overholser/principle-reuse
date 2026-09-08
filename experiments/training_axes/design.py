"""Configuration and manifest generation for the training-axes experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path


COMPRESSION_LEVELS = {
    "none": 0.0,
    "low": 1e-4,
    "medium": 1e-3,
    "high": 1e-2,
}

DIFFICULTY_LEVELS = {
    # Rank margin is the controlled difficulty variable. Small margins require
    # a more precise learned order than large margins.
    "easy": (5, 6, 7, 8, 9),
    "mixed": (1, 2, 3, 4, 5, 6, 7, 8, 9),
    "hard": (1, 2, 3),
}

ABSTRACTION_LEVELS = ("concrete", "invariant", "latent")

CAPACITY_LEVELS = {
    "tight": 32,
    "moderate": 64,
    "standard": 96,
    "wide": 128,
}


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str
    seed: int
    compression: str
    weight_decay: float
    difficulty: str
    train_dists: tuple[int, ...]
    abstraction: str
    task_name: str = "order"
    capacity: str = "standard"
    family: str = "transformer"
    width: int = 96
    layers: int = 2
    batch_size: int = 256
    steps: int = 10_000
    learning_rate: float = 1e-3
    invariant_weight: float = 1.0
    representation_weight: float = 0.1
    latent_weight: float = 1.0
    mapping_weight: float = 1.0


def make_configs(seeds=range(6), smoke: bool = False) -> list[ExperimentConfig]:
    configs = []
    grid = itertools.product(
        COMPRESSION_LEVELS.items(), DIFFICULTY_LEVELS.items(),
        ABSTRACTION_LEVELS, seeds,
    )
    for i, ((c_name, wd), (d_name, dists), abstraction, seed) in enumerate(grid):
        config_id = f"c-{c_name}_d-{d_name}_a-{abstraction}_s-{seed:02d}"
        configs.append(ExperimentConfig(
            config_id=config_id,
            seed=int(seed),
            compression=c_name,
            weight_decay=wd,
            difficulty=d_name,
            train_dists=tuple(dists),
            abstraction=abstraction,
            width=48 if smoke else 96,
            batch_size=32 if smoke else 256,
            steps=30 if smoke else 10_000,
        ))
    return configs


def write_manifest(path: str | Path, configs: list[ExperimentConfig]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in configs:
        row = asdict(config)
        row["train_dists"] = " ".join(map(str, config.train_dists))
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/training_axes/manifest.csv")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configs = make_configs(range(args.seeds), smoke=args.smoke)
    write_manifest(args.out, configs)
    print(f"wrote {len(configs)} configurations to {args.out}")


if __name__ == "__main__":
    main()
