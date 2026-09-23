"""Evaluate all frozen V9-A endpoints on the committed V9-B lock."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

from experiments.training_axes.design_v9 import read_manifest
from experiments.training_axes.v9b_common import (
    ENDPOINT, verify_reveal, verify_v9b_freeze,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_manifest.csv"))
    parser.add_argument("--freeze", type=Path, default=Path("experiments/training_axes/v9b_preunlock_freeze.json"))
    parser.add_argument("--v9a-freeze", type=Path, default=Path("experiments/training_axes/v9_preunlock_freeze.json"))
    parser.add_argument("--commitment", type=Path, default=Path("experiments/training_axes/v9b_lock_commitment.json"))
    parser.add_argument("--reveal", type=Path, default=Path("experiments/training_axes/v9b_lock_reveal.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/training_axes/locked_results_v9b"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    verify_v9b_freeze(
        args.freeze, args.manifest, args.checkpoints, args.v9a_freeze,
        args.commitment,
    )
    verify_reveal(args.commitment, args.reveal)

    configs = read_manifest(args.manifest)
    commands = []
    for config in configs:
        output = args.out_dir / f"{config.config_id}.jsonl"
        if output.exists():
            rows = [json.loads(line) for line in output.read_text().splitlines() if line]
            if len(rows) == 1 and int(rows[0].get("step", -1)) == ENDPOINT:
                print(f"complete; skipping {config.config_id}")
                continue
            raise RuntimeError(f"partial locked V9-B output: {output}")
        command = [
            sys.executable, "-m", "experiments.training_axes.evaluate_locked_v9b",
            str(args.checkpoints / config.config_id),
            "--out", str(output),
            "--commitment", str(args.commitment),
            "--reveal", str(args.reveal),
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
        raise SystemExit(f"failed locked V9-B jobs: {failures}")


if __name__ == "__main__":
    main()
