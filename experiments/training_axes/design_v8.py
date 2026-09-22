"""Generate V8: fresh-lock decomposition of the structural objective."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from experiments.training_axes import task_v8
from experiments.training_axes.design import ExperimentConfig, write_manifest


SEEDS = tuple(range(400, 412))
TASKS = {
    "order": ("order_v8", task_v8.ORDER.DIFFICULTY_LEVELS["mixed"]),
    "differences": (
        "differences_v8", task_v8.DIFFERENCES.DIFFICULTY_LEVELS["mixed"]
    ),
}
FACTOR_ARMS = tuple(
    f"e{embedding}h{hidden}g{gap}"
    for embedding, hidden, gap in itertools.product((0, 1), repeat=3)
)
CONTROL_ARMS = ("concrete", "eperm", "gsep")
ARMS = CONTROL_ARMS + FACTOR_ARMS


def _config(task, task_name, dists, seed, arm, smoke=False):
    concrete = arm == "concrete"
    if arm in FACTOR_ARMS:
        embedding, hidden, gap = int(arm[1]), int(arm[3]), int(arm[5])
        mapping_control, gap_head = "valid", "shared"
    elif arm == "eperm":
        embedding, hidden, gap = 1, 0, 0
        mapping_control, gap_head = "permuted", "shared"
    elif arm == "gsep":
        embedding, hidden, gap = 0, 0, 1
        mapping_control, gap_head = "valid", "separate"
    else:
        embedding = hidden = gap = 0
        mapping_control, gap_head = "valid", "shared"
    return ExperimentConfig(
        config_id=f"v8_t-{task}_{arm}_s-{int(seed):03d}",
        seed=int(seed), compression="none", weight_decay=0.0,
        difficulty="mixed", train_dists=tuple(dists),
        abstraction="concrete", task_name=task_name,
        capacity="moderate", family="transformer", width=64, layers=2,
        batch_size=32 if smoke else 256, steps=30 if smoke else 10_000,
        learning_rate=0.001, bottleneck_dim=0,
        aux_schedule="none" if concrete else "throughout",
        aux_fraction=1.0, rule_signal=gap, bridge_signal=0,
        representation_signal=hidden, mapping_signal=embedding,
        output_invariant_signal=0 if concrete else 1,
        mapping_control=mapping_control, gap_head_mode=gap_head,
        source_only_eval=0, aux_mode="online_global",
        v8_split=task, v8_arm=arm,
    )


def make_configs(seeds=SEEDS, smoke=False):
    seeds = tuple(seeds)
    configs = []
    for (task, (task_name, dists)), seed, arm in itertools.product(
        TASKS.items(), seeds, ARMS
    ):
        configs.append(_config(task, task_name, dists, seed, arm, smoke))
    ids = [config.config_id for config in configs]
    expected = len(TASKS) * len(ARMS) * len(seeds)
    if len(configs) != expected or len(ids) != len(set(ids)):
        raise AssertionError("unexpected or duplicate V8 configurations")
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("experiments/training_axes/v8_manifest.csv"),
    )
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configs = make_configs(range(400, 400 + args.seeds), smoke=args.smoke)
    write_manifest(args.out, configs)
    print(f"wrote {len(configs)} V8 configurations to {args.out}")


if __name__ == "__main__":
    main()
