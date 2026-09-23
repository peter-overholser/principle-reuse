"""Registered loading and paired summaries for V9."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from statistics import mean, stdev

from experiments.training_axes.design_v9 import read_manifest


T_CRIT = {4: 3.182446, 8: 2.364624, 12: 2.200985}


def load_results(results_dir, manifest_path, expected_steps):
    configs = read_manifest(manifest_path)
    expected = {config.config_id for config in configs}
    paths = sorted(Path(results_dir).glob("*.jsonl"))
    observed = {path.stem for path in paths}
    if expected != observed:
        raise ValueError(
            f"V9 result mismatch: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    by_key = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if tuple(int(row["step"]) for row in rows) != tuple(expected_steps):
            raise ValueError(f"wrong V9 checkpoints in {path}")
        for row in rows:
            leaked = [key for key in row if key.startswith("lock_")]
            if leaked:
                raise ValueError(f"locked V9 field leaked into {path}: {leaked[:3]}")
            key = row["config_id"], int(row["step"])
            if key in by_key:
                raise ValueError(f"duplicate V9 row: {key}")
            by_key[key] = row
    return configs, by_key, paths


def config_id(scaffold, arm, seed):
    return f"v9_sc-{scaffold}_{arm}_s-{int(seed):03d}"


def summarize(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        return {
            "mean": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "positive_seeds": 0,
            "seed_values": values,
        }
    center = mean(values)
    critical = T_CRIT[len(values)]
    half = critical * stdev(values) / math.sqrt(len(values))
    return {
        "mean": center, "ci_low": center - half, "ci_high": center + half,
        "positive_seeds": sum(value > 0 for value in values),
        "seed_values": values,
    }


def paired(by_key, scaffold, left, right, seeds, metric, step=10_000):
    return summarize([
        by_key[(config_id(scaffold, left, seed), step)][metric]
        - by_key[(config_id(scaffold, right, seed), step)][metric]
        for seed in seeds
    ])


def arm_mean(by_key, scaffold, arm, seeds, metric, step=10_000):
    return mean(
        float(by_key[(config_id(scaffold, arm, seed), step)][metric])
        for seed in seeds
    )


def fmt(value):
    return (
        f"{value['mean']:+.3f} "
        f"[{value['ci_low']:+.3f}, {value['ci_high']:+.3f}]"
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha256(paths, root=None):
    digest = hashlib.sha256()
    root = Path(root) if root is not None else None
    for path in sorted(Path(value) for value in paths):
        name = str(path.relative_to(root)) if root is not None else path.name
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def checkpoint_tree(checkpoint_root, config_ids, checkpoints):
    checkpoint_root = Path(checkpoint_root)
    directories = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    observed, expected = {path.name for path in directories}, set(config_ids)
    if observed != expected:
        raise ValueError(
            f"V9 checkpoint mismatch: missing={sorted(expected-observed)[:3]}, "
            f"extra={sorted(observed-expected)[:3]}"
        )
    expected_names = {f"step-{step:06d}.pt" for step in checkpoints}
    paths = []
    for directory in directories:
        run_paths = sorted(directory.glob("step-*.pt"))
        names = {path.name for path in run_paths}
        if names != expected_names:
            raise ValueError(f"wrong V9 checkpoint tree in {directory}")
        paths.extend(run_paths)
    return paths


def load_locked(results_dir, manifest_path, expected_steps):
    configs = read_manifest(manifest_path)
    expected = {config.config_id for config in configs}
    paths = sorted(Path(results_dir).glob("*.jsonl"))
    observed = {path.stem for path in paths}
    if observed != expected:
        raise ValueError(
            f"V9 locked mismatch: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    allowed_prefixes = (
        "lock_", "probe_lock_", "paired_lock_", "causal_lock_",
        "organization_lock_",
    )
    by_key = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if tuple(int(row["step"]) for row in rows) != tuple(expected_steps):
            raise ValueError(f"wrong locked V9 checkpoints in {path}")
        for row in rows:
            unexpected = [
                key for key in row
                if key not in {"config_id", "step"}
                and not key.startswith(allowed_prefixes)
            ]
            if unexpected:
                raise ValueError(f"non-locked V9 field in {path}: {unexpected[:3]}")
            key = row["config_id"], int(row["step"])
            if key in by_key:
                raise ValueError(f"duplicate locked V9 row: {key}")
            by_key[key] = row
    return configs, by_key, paths
