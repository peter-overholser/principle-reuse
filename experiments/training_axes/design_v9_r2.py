"""Fresh-seed V9-R2 circle reliability-screen manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.training_axes.design_v9 import (
    MAIN_STEPS, V9Config, arm_fields, write_manifest,
)


R2_SEEDS = tuple(range(504, 512))
R2_ARMS = ("a00_confounded", "imposed", "a50_diverse")


def make_configs(seeds=R2_SEEDS, smoke=False):
    configs = []
    for arm in R2_ARMS:
        anchor, coverage, imposed, permuted = arm_fields(arm)
        for seed in seeds:
            configs.append(V9Config(
                config_id=f"v9r2_sc-circle_{arm}_s-{int(seed):03d}",
                seed=int(seed), scaffold="circle", arm=arm,
                anchor_count=anchor, coverage=coverage,
                imposed=imposed, permuted=permuted,
                width=32 if smoke else 64,
                batch_size=32 if smoke else 256,
                steps=30 if smoke else MAIN_STEPS,
            ))
    expected = len(R2_ARMS) * len(tuple(seeds))
    if len(configs) != expected or len({c.config_id for c in configs}) != expected:
        raise AssertionError("unexpected or duplicate V9-R2 configurations")
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument(
        "--out", type=Path,
        default=Path("experiments/training_axes/v9_r2_manifest.csv"),
    )
    args = parser.parse_args()
    seeds = R2_SEEDS if args.seeds is None else R2_SEEDS[:args.seeds]
    configs = make_configs(seeds=seeds, smoke=args.smoke)
    write_manifest(args.out, configs)
    print(f"wrote {len(configs)} V9-R2 configurations to {args.out}")


if __name__ == "__main__":
    main()
