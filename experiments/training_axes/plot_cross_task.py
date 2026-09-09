"""Plot validation mastery and locked reuse for the two completed tasks."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OBJECTIVES = ("concrete", "invariant", "latent")
LABELS = {
    "concrete": "Concrete answers",
    "invariant": "Output invariance",
    "latent": "Structural auxiliary",
}
COLORS = {
    "concrete": "#4C78A8",
    "invariant": "#F58518",
    "latent": "#54A24B",
}
T_CRIT_95_DF5 = 2.570581835636305


def read_directory(path: Path) -> list[dict]:
    rows = []
    for result_path in sorted(path.glob("*.jsonl")):
        with result_path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def attach_metadata(development: list[dict], locked: list[dict]) -> None:
    metadata = {
        (row["config_id"], int(row["step"])): {
            "abstraction": row["abstraction"], "seed": int(row["seed"]),
        }
        for row in development
    }
    for row in locked:
        key = (row["config_id"], int(row["step"]))
        if key not in metadata:
            raise ValueError(f"locked row has no development match: {key}")
        row.update(metadata[key])


def seed_trajectory(rows: list[dict], metric: str) -> dict[str, dict[int, tuple]]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["abstraction"], int(row["seed"]), int(row["step"]))].append(
            float(row[metric])
        )
    output: dict[str, dict[int, tuple]] = {name: {} for name in OBJECTIVES}
    for objective in OBJECTIVES:
        steps = sorted({key[2] for key in grouped if key[0] == objective})
        for step in steps:
            seed_means = np.asarray([
                np.mean(grouped[(objective, seed, step)]) for seed in range(6)
            ])
            mean = float(np.mean(seed_means))
            half = T_CRIT_95_DF5 * float(np.std(seed_means, ddof=1)) / math.sqrt(6)
            output[objective][step] = (mean, mean - half, mean + half)
    return output


def draw_panel(ax, trajectories, title, show_ylabel=False, show_mastery=False):
    for objective in OBJECTIVES:
        points = trajectories[objective]
        steps = np.asarray(sorted(points))
        values = np.asarray([points[step] for step in steps])
        ax.plot(
            steps, values[:, 0], color=COLORS[objective], linewidth=2.1,
            label=LABELS[objective],
        )
        ax.fill_between(
            steps, values[:, 1], values[:, 2], color=COLORS[objective],
            alpha=0.13, linewidth=0,
        )
    if show_mastery:
        ax.axvline(400, color="#777777", linestyle="--", linewidth=1)
        ax.text(
            430, 0.455, "median persistent mastery",
            color="#666666", fontsize=8, rotation=90, va="bottom",
        )
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlim(0, 10_000)
    ax.set_ylim(0.40, 1.025)
    ax.set_xlabel("Training step")
    if show_ylabel:
        ax.set_ylabel("Accuracy")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-development", type=Path, default=ROOT / "results_v2")
    parser.add_argument("--v2-locked", type=Path, default=ROOT / "locked_results_v2")
    parser.add_argument("--task2-development", type=Path, default=ROOT / "results_task2")
    parser.add_argument("--task2-locked", type=Path, default=ROOT / "locked_results_task2")
    parser.add_argument(
        "--png", type=Path, default=ROOT.parents[1] / "assets/cross_task_overview.png"
    )
    parser.add_argument(
        "--pdf", type=Path, default=ROOT.parents[1] / "assets/cross_task_overview.pdf"
    )
    args = parser.parse_args()

    v2_dev = read_directory(args.v2_development)
    v2_lock = read_directory(args.v2_locked)
    task2_dev = read_directory(args.task2_development)
    task2_lock = read_directory(args.task2_locked)
    expected = (("V2 development", v2_dev, 324 * 11),
                ("V2 locked", v2_lock, 324 * 11),
                ("Task 2 development", task2_dev, 162 * 11),
                ("Task 2 locked", task2_lock, 162 * 11))
    for label, rows, count in expected:
        if len(rows) != count:
            raise ValueError(f"expected {count} rows for {label}, got {len(rows)}")
    attach_metadata(v2_dev, v2_lock)
    attach_metadata(task2_dev, task2_lock)

    panels = (
        seed_trajectory(v2_dev, "source_accuracy"),
        seed_trajectory(v2_lock, "test_combo_accuracy"),
        seed_trajectory(task2_dev, "source_accuracy"),
        seed_trajectory(task2_lock, "test_combo_accuracy"),
    )
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 160,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    draw_panel(axes[0, 0], panels[0], "A  Ordered relation: validation", True, True)
    draw_panel(axes[0, 1], panels[1], "B  Ordered relation: locked combination")
    draw_panel(axes[1, 0], panels[2], "C  Difference comparison: validation", True, True)
    draw_panel(axes[1, 1], panels[3], "D  Difference comparison: locked combination")
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Equal validation mastery, different development of reusable structure",
        fontsize=13, fontweight="bold",
    )
    for path in (args.png, args.pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, bbox_inches="tight")
    fig.savefig(args.pdf, bbox_inches="tight")
    print(f"wrote {args.png}")
    print(f"wrote {args.pdf}")


if __name__ == "__main__":
    main()
