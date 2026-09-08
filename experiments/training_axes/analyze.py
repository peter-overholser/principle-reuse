"""Run-level summaries and emergence times for the training-axes experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_ENDPOINTS = (
    "source_accuracy",
    "calib_combo_accuracy",
    "probe_target_r2",
    "embedding_rank_axis_cosine",
    "embedding_participation_rank",
    "causal_source_to_target",
    "parameter_l2",
)


def load_jsonl(paths) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def final_rows(rows: list[dict]) -> list[dict]:
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config_id"]].append(row)
    return [max(group, key=lambda row: row["step"]) for group in by_config.values()]


def first_persistent(group: list[dict], key: str, threshold: float):
    ordered = sorted(group, key=lambda row: row["step"])
    for index in range(len(ordered) - 1):
        values = (ordered[index].get(key), ordered[index + 1].get(key))
        if all(value is not None and np.isfinite(value) and value >= threshold for value in values):
            return ordered[index]["step"]
    return np.nan


def emergence_rows(rows: list[dict]) -> list[dict]:
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config_id"]].append(row)
    output = []
    for config_id, group in by_config.items():
        first = group[0]
        source_step = first_persistent(group, "source_accuracy", 0.95)
        reuse_step = first_persistent(group, "calib_combo_accuracy", 0.80)
        structure_step = first_persistent(group, "probe_target_r2", 0.50)
        output.append({
            "config_id": config_id,
            "compression": first["compression"],
            "difficulty": first["difficulty"],
            "abstraction": first["abstraction"],
            "seed": first["seed"],
            "source_emergence_step": source_step,
            "structure_emergence_step": structure_step,
            "reuse_emergence_step": reuse_step,
            "mastery_reuse_lag": reuse_step - source_step,
            "structure_reuse_lag": reuse_step - structure_step,
        })
    return output


def summarize_factor(rows: list[dict], factor: str) -> None:
    levels = defaultdict(list)
    for row in rows:
        levels[str(row[factor])].append(row)
    print(f"\nBy {factor}")
    header = f"{'level':<12}{'n':>5}" + "".join(f"{key[:14]:>16}" for key in DEFAULT_ENDPOINTS)
    print(header)
    for level, group in sorted(levels.items()):
        values = []
        for key in DEFAULT_ENDPOINTS:
            x = np.asarray([
                np.nan if (
                    key.startswith("causal_")
                    and not bool(float(row.get("causal_valid", 0)))
                ) else row.get(key, np.nan)
                for row in group
            ], dtype=float)
            values.append(float(np.nanmean(x)))
        print(f"{level:<12}{len(group):>5}" + "".join(f"{value:>16.3f}" for value in values))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", nargs="?", type=Path,
        default=Path("experiments/training_axes/results"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    paths = sorted(args.results.glob("*.jsonl")) if args.results.is_dir() else [args.results]
    rows = load_jsonl(paths)
    if not rows:
        raise SystemExit(f"no result rows found under {args.results}")
    endpoints = final_rows(rows)
    print(f"loaded {len(rows)} checkpoint rows from {len(endpoints)} training runs")
    factors = ["abstraction", "compression", "difficulty"]
    if any("capacity" in row for row in endpoints):
        factors.append("capacity")
    for factor in factors:
        summarize_factor(endpoints, factor)
    emergence = emergence_rows(rows)
    finite_lags = np.asarray([row["mastery_reuse_lag"] for row in emergence], dtype=float)
    observed = finite_lags[np.isfinite(finite_lags)]
    median = float(np.median(observed)) if len(observed) else np.nan
    print(
        "\nmastery-to-reuse lag: "
        f"median={median:.0f} steps; "
        f"observed in {len(observed)}/{len(finite_lags)} runs"
    )
    if args.out:
        write_csv(args.out, endpoints)
        write_csv(args.out.with_name(args.out.stem + "_emergence.csv"), emergence)
        print(f"wrote {args.out} and emergence companion")


if __name__ == "__main__":
    main()
