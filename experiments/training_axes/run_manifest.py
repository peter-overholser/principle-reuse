"""Print or execute commands for every row in a generated manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import shlex
import subprocess
import sys
from pathlib import Path


def command(row: dict, args) -> list[str]:
    value = [
        sys.executable, "-m", "experiments.training_axes.run",
        "--compression", row["compression"],
        "--difficulty", row["difficulty"],
        "--abstraction", row["abstraction"],
        "--seed", row["seed"],
    ]
    optional = {
        "task_name": "--task",
        "capacity": "--capacity", "family": "--family", "width": "--width",
        "layers": "--layers", "batch_size": "--batch-size", "steps": "--steps",
        "learning_rate": "--learning-rate", "invariant_weight": "--invariant-weight",
        "representation_weight": "--representation-weight",
        "latent_weight": "--latent-weight", "mapping_weight": "--mapping-weight",
    }
    for field, flag in optional.items():
        if row.get(field, ""):
            value += [flag, row[field]]
    if row.get("config_id", "").startswith("v2_"):
        expected = (
            f"v2_c-{row['compression']}_d-{row['difficulty']}_"
            f"a-{row['abstraction']}_p-{row['capacity']}_s-{int(row['seed']):02d}"
        )
        if row["config_id"] != expected:
            raise ValueError(f"unexpected V2 config_id: {row['config_id']}")
        value += ["--run-prefix", "v2"]
    if row.get("config_id", "").startswith("v3_t-differences_"):
        expected = (
            f"v3_t-differences_c-{row['compression']}_d-{row['difficulty']}_"
            f"a-{row['abstraction']}_p-{row['capacity']}_s-{int(row['seed']):02d}"
        )
        if row["config_id"] != expected:
            raise ValueError(f"unexpected task-2 config_id: {row['config_id']}")
        value += ["--run-prefix", "v3"]
    if args.device:
        value += ["--device", args.device]
    if args.results_dir:
        value += ["--results-dir", str(args.results_dir)]
    if args.checkpoints_dir:
        value += ["--checkpoints-dir", str(args.checkpoints_dir)]
    if args.no_save_checkpoints:
        value.append("--no-save-checkpoints")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="number of independent training processes to run concurrently",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--no-save-checkpoints", action="store_true")
    args = parser.parse_args()
    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle))[args.start:args.stop]
    indexed = list(enumerate(rows, start=args.start))
    commands = []
    for index, row in indexed:
        value = command(row, args)
        commands.append((index, value))
        print(f"[{index}] {shlex.join(value)}", flush=True)
    if not args.execute:
        return

    def execute(item):
        index, value = item
        completed = subprocess.run(value)
        return index, completed.returncode

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for index, returncode in pool.map(execute, commands):
            print(f"[{index}] finished with status {returncode}", flush=True)
            if returncode:
                failures.append((index, returncode))
    if failures:
        raise SystemExit(f"failed manifest rows: {failures}")


if __name__ == "__main__":
    main()
