"""Registered manifests for the V9-A Selection/Emergence study."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
from pathlib import Path


PILOT_SEEDS = tuple(range(500, 504))
MAIN_SEEDS = tuple(range(512, 524))
PILOT_STEPS = 10_000
REPAIR_STEPS = 20_000
MAIN_STEPS = 20_000
SCAFFOLDS = ("line", "circle")
EMERGENT_ARMS = (
    "a00_confounded", "a25_confounded", "a50_confounded",
    "a00_diverse", "a25_diverse", "a50_diverse",
)
CONTROL_ARMS = ("imposed", "eperm")
ARMS = EMERGENT_ARMS + CONTROL_ARMS


@dataclass(frozen=True)
class V9Config:
    config_id: str
    seed: int
    scaffold: str
    arm: str
    anchor_count: int
    coverage: str
    imposed: int
    permuted: int
    width: int = 64
    layers: int = 2
    batch_size: int = 256
    steps: int = 10_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


def arm_fields(arm):
    if arm in EMERGENT_ARMS:
        anchor = {"a00": 0, "a25": 3, "a50": 6}[arm[:3]]
        coverage = "diverse" if arm.endswith("diverse") else "confounded"
        return anchor, coverage, 0, 0
    if arm == "imposed":
        return 0, "confounded", 1, 0
    if arm == "eperm":
        return 0, "confounded", 1, 1
    raise ValueError(f"unknown V9 arm: {arm}")


def make_configs(seeds=PILOT_SEEDS, smoke=False, steps=PILOT_STEPS):
    configs = []
    for scaffold, arm, seed in itertools.product(SCAFFOLDS, ARMS, seeds):
        anchor, coverage, imposed, permuted = arm_fields(arm)
        configs.append(V9Config(
            config_id=f"v9_sc-{scaffold}_{arm}_s-{int(seed):03d}",
            seed=int(seed), scaffold=scaffold, arm=arm,
            anchor_count=anchor, coverage=coverage,
            imposed=imposed, permuted=permuted,
            width=32 if smoke else 64,
            batch_size=32 if smoke else 256,
            steps=30 if smoke else int(steps),
        ))
    ids = [config.config_id for config in configs]
    expected = len(SCAFFOLDS) * len(ARMS) * len(tuple(seeds))
    if len(configs) != expected or len(ids) != len(set(ids)):
        raise AssertionError("unexpected or duplicate V9 configurations")
    return configs


def write_manifest(path, configs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(config) for config in configs]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    fields = V9Config.__dataclass_fields__
    integer = {
        "seed", "anchor_count", "imposed", "permuted", "width", "layers",
        "batch_size", "steps",
    }
    floating = {"learning_rate", "weight_decay"}
    rows = []
    with Path(path).open() as handle:
        for raw in csv.DictReader(handle):
            values = {}
            for key in fields:
                value = raw[key]
                values[key] = int(value) if key in integer else (
                    float(value) if key in floating else value
                )
            rows.append(V9Config(**values))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("pilot", "repair", "main"), default="pilot"
    )
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    base = PILOT_SEEDS if args.phase in {"pilot", "repair"} else MAIN_SEEDS
    steps = {
        "pilot": PILOT_STEPS,
        "repair": REPAIR_STEPS,
        "main": MAIN_STEPS,
    }[args.phase]
    seeds = base if args.seeds is None else tuple(range(base[0], base[0] + args.seeds))
    default_names = {
        "pilot": "v9_pilot_manifest.csv",
        "repair": "v9_repair_manifest.csv",
        "main": "v9_manifest.csv",
    }
    output = args.out or Path("experiments/training_axes") / default_names[args.phase]
    configs = make_configs(seeds=seeds, smoke=args.smoke, steps=steps)
    write_manifest(output, configs)
    print(f"wrote {len(configs)} V9 {args.phase} configurations to {output}")


if __name__ == "__main__":
    main()
