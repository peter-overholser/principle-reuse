"""Shared registered utilities for V8 development and locked analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from experiments.training_axes.design_v8 import ARMS, FACTOR_ARMS, SEEDS, TASKS


CHECKPOINTS = (0, 50, 100, 200, 400, 700, 1_000, 2_000, 4_000, 7_000, 10_000)
DIAGNOSTIC_STEPS = tuple(range(50, 2_001, 50)) + (4_000, 7_000, 10_000)
T_CRIT_95_DF11 = 2.200985


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def checkpoint_tree(checkpoint_root: Path):
    """Validate and return the complete registered V8 checkpoint tree."""
    directories = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    observed, expected = {path.name for path in directories}, expected_ids()
    if observed != expected:
        raise ValueError(
            f"V8 checkpoint directories mismatch: "
            f"missing={sorted(expected-observed)[:3]}, "
            f"extra={sorted(observed-expected)[:3]}"
        )
    paths = []
    expected_names = {f"step-{step:06d}.pt" for step in CHECKPOINTS}
    for directory in directories:
        run_paths = sorted(directory.glob("step-*.pt"))
        names = {path.name for path in run_paths}
        if names != expected_names:
            raise ValueError(
                f"wrong V8 checkpoint tree in {directory}: "
                f"missing={sorted(expected_names-names)[:3]}, "
                f"extra={sorted(names-expected_names)[:3]}"
            )
        paths.extend(run_paths)
    return paths


def checkpoint_tree_sha256(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative_name = f"{path.parent.name}/{path.name}"
        digest.update(relative_name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def config_id(task: str, arm: str, seed: int) -> str:
    return f"v8_t-{task}_{arm}_s-{seed:03d}"


def expected_ids():
    return {
        config_id(task, arm, seed)
        for task in TASKS for arm in ARMS for seed in SEEDS
    }


def validate_manifest(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed, expected = {row["config_id"] for row in rows}, expected_ids()
    if observed != expected or len(rows) != len(expected):
        raise ValueError(
            f"V8 manifest mismatch: missing={sorted(expected-observed)[:3]}, "
            f"extra={sorted(observed-expected)[:3]}"
        )
    return rows


def _exact_paths(results_dir: Path):
    paths = sorted(results_dir.glob("v8_*.jsonl"))
    observed, expected = {path.stem for path in paths}, expected_ids()
    if observed != expected:
        raise ValueError(
            f"V8 results mismatch: missing={sorted(expected-observed)[:3]}, "
            f"extra={sorted(observed-expected)[:3]}"
        )
    return paths


def load_development(results_dir: Path):
    paths, by_key = _exact_paths(results_dir), {}
    forbidden = ("test_combo_", "held_pair_", "test_both_")
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if tuple(int(row["step"]) for row in rows) != CHECKPOINTS:
            raise ValueError(f"wrong V8 checkpoints in {path}")
        for row in rows:
            leaked = [field for field in row if field.startswith(forbidden)]
            if leaked:
                raise ValueError(f"locked metric leaked into {path}: {leaked[:3]}")
            key = row["config_id"], int(row["step"])
            if key in by_key:
                raise ValueError(f"duplicate V8 row: {key}")
            by_key[key] = row
    return by_key, paths


def load_diagnostics(results_dir: Path):
    diagnostics_dir = results_dir / "_diagnostics"
    paths = sorted(diagnostics_dir.glob("v8_*.jsonl"))
    observed, expected = {path.stem for path in paths}, expected_ids()
    if observed != expected:
        raise ValueError(
            f"V8 diagnostics mismatch: missing={sorted(expected-observed)[:3]}, "
            f"extra={sorted(observed-expected)[:3]}"
        )
    by_key = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if tuple(int(row["step"]) for row in rows) != DIAGNOSTIC_STEPS:
            raise ValueError(f"wrong V8 diagnostic steps in {path}")
        for row in rows:
            key = row["config_id"], int(row["step"])
            if key in by_key:
                raise ValueError(f"duplicate V8 diagnostic row: {key}")
            by_key[key] = row
    return by_key, paths


def load_locked(results_dir: Path):
    paths, by_key = _exact_paths(results_dir), {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if tuple(int(row["step"]) for row in rows) != CHECKPOINTS:
            raise ValueError(f"wrong locked checkpoints in {path}")
        for row in rows:
            allowed = {"config_id", "step"}
            unexpected = [
                field for field in row
                if field not in allowed
                and not field.startswith(("test_combo_", "held_pair_", "test_both_"))
            ]
            if unexpected:
                raise ValueError(f"non-locked field in {path}: {unexpected[:3]}")
            key = row["config_id"], int(row["step"])
            if key in by_key:
                raise ValueError(f"duplicate locked V8 row: {key}")
            by_key[key] = row
    return by_key, paths


def summarize(values):
    values = [float(value) for value in values]
    if len(values) != len(SEEDS) or not all(math.isfinite(value) for value in values):
        raise ValueError("V8 registered contrasts require 12 finite paired values")
    center = mean(values)
    half = T_CRIT_95_DF11 * stdev(values) / math.sqrt(len(values))
    return {
        "mean": center, "ci_low": center - half, "ci_high": center + half,
        "positive_seeds": sum(value > 0 for value in values),
        "seed_values": values,
    }


def arm_mean(by_key, task, arm, metric, step=10_000):
    return mean(
        float(by_key[(config_id(task, arm, seed), step)][metric])
        for seed in SEEDS
    )


def paired_effect(by_key, task, left, right, metric, step=10_000):
    return summarize(
        float(by_key[(config_id(task, left, seed), step)][metric])
        - float(by_key[(config_id(task, right, seed), step)][metric])
        for seed in SEEDS
    )


def paired_effect_at_steps(by_key, task, left, right, metric, selected_steps):
    return summarize(
        float(by_key[(config_id(task, left, seed), selected_steps[config_id(task, left, seed)])][metric])
        - float(by_key[(config_id(task, right, seed), selected_steps[config_id(task, right, seed)])][metric])
        for seed in SEEDS
    )


def factorial_effect(by_key, task, factor, metric, step=10_000):
    index = {"embedding": 1, "hidden": 3, "gap": 5}[factor]
    values = []
    for seed in SEEDS:
        on, off = [], []
        for arm in FACTOR_ARMS:
            value = float(by_key[(config_id(task, arm, seed), step)][metric])
            (on if arm[index] == "1" else off).append(value)
        values.append(mean(on) - mean(off))
    return summarize(values)


def fmt(value):
    return f"{value['mean']:+.3f} [{value['ci_low']:+.3f}, {value['ci_high']:+.3f}]"
