"""Evaluate every registered V9 main trunk on locked roles."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import sys

from experiments.training_axes.design_v9 import read_manifest
from experiments.training_axes.analyze_preunlock_v9 import CHECKPOINTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    configs = read_manifest(args.manifest)
    commands = []
    for config in configs:
        output = args.out_dir / f"{config.config_id}.jsonl"
        if output.exists():
            rows = [line for line in output.read_text().splitlines() if line]
            if len(rows) == len(CHECKPOINTS):
                print(f"complete; skipping {config.config_id}")
                continue
            raise RuntimeError(f"partial locked V9 output: {output}")
        command = [
            sys.executable, "-m", "experiments.training_axes.evaluate_locked_v9",
            str(args.checkpoints / config.config_id), "--out", str(output),
        ]
        if args.device:
            command += ["--device", args.device]
        commands.append((config.config_id, command))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def execute(item):
        config_id, command = item
        return config_id, subprocess.run(command).returncode

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for config_id, status in pool.map(execute, commands):
            print(f"{config_id} finished with status {status}", flush=True)
            if status:
                failures.append((config_id, status))
    if failures:
        raise SystemExit(f"failed locked V9 jobs: {failures}")


if __name__ == "__main__":
    main()
