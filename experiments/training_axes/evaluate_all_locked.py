"""Evaluate every run directory under a checkpoint root on locked splits."""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=2000)
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.checkpoint_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"no run directories under {args.checkpoint_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for run_dir in run_dirs:
        checkpoints = sorted(run_dir.glob("step-*.pt"))
        if not checkpoints:
            raise SystemExit(f"no checkpoints under {run_dir}")
        output = args.out_dir / f"{run_dir.name}.jsonl"
        if output.exists():
            rows = sum(1 for line in output.open() if line.strip())
            if rows != len(checkpoints):
                raise SystemExit(
                    f"partial locked output {output}: {rows}/{len(checkpoints)} rows; "
                    "move it aside before resuming"
                )
            print(f"complete; skipping {run_dir.name}", flush=True)
            continue
        command = [
            sys.executable, "-m", "experiments.training_axes.evaluate_locked",
            str(run_dir), "--out", str(output),
            "--eval-size", str(args.eval_size),
        ]
        if args.device:
            command += ["--device", args.device]
        pending.append((run_dir.name, command))

    print(
        f"checkpoint runs={len(run_dirs)} complete={len(run_dirs)-len(pending)} "
        f"pending={len(pending)} jobs={max(1, args.jobs)}",
        flush=True,
    )

    def execute(item):
        name, command = item
        completed = subprocess.run(command)
        return name, completed.returncode

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name, returncode in pool.map(execute, pending):
            print(f"{name} finished with status {returncode}", flush=True)
            if returncode:
                failures.append((name, returncode))
    if failures:
        raise SystemExit(f"locked evaluation failures: {failures}")


if __name__ == "__main__":
    main()
