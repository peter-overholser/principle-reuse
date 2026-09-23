"""Run the V9-R2 endpoint reliability assay with bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import sys

from experiments.training_axes.design_v9 import read_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_r2_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    configs = read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for config in configs:
        checkpoint = (
            args.checkpoints / config.config_id / "step-020000.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing V9-R2 endpoint: {checkpoint}")
        output = args.out_dir / f"{config.config_id}.jsonl"
        command = [
            sys.executable, "-m",
            "experiments.training_axes.evaluate_v9_r2_reliability",
            str(checkpoint), "--out", str(output),
        ]
        if args.device:
            command += ["--device", args.device]
        commands.append((config.config_id, command))

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
        raise SystemExit(f"failed V9-R2 reliability jobs: {failures}")


if __name__ == "__main__":
    main()
