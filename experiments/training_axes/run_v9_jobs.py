"""Run a V9 manifest with bounded process-level concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from experiments.training_axes.design_v9 import read_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-step-extension", action="store_true")
    args = parser.parse_args()
    configs = read_manifest(args.manifest)
    commands = []
    for index, config in enumerate(configs):
        command = [
            sys.executable, "-m", "experiments.training_axes.run_v9",
            "--config-json", json.dumps(asdict(config), separators=(",", ":")),
            "--results-dir", str(args.results_dir),
            "--checkpoints-dir", str(args.checkpoints_dir),
        ]
        if args.device:
            command += ["--device", args.device]
        if args.no_save_checkpoints:
            command.append("--no-save-checkpoints")
        if args.smoke:
            command.append("--smoke")
        if args.allow_step_extension:
            command.append("--allow-step-extension")
        commands.append((index, config.config_id, command))
        print(f"[{index}] {config.config_id}", flush=True)
    if not args.execute:
        return

    def execute(item):
        index, config_id, command = item
        completed = subprocess.run(command)
        return index, config_id, completed.returncode

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for index, config_id, status in pool.map(execute, commands):
            print(f"[{index}] {config_id} finished with status {status}", flush=True)
            if status:
                failures.append((index, config_id, status))
    if failures:
        raise SystemExit(f"failed V9 jobs: {failures}")


if __name__ == "__main__":
    main()
