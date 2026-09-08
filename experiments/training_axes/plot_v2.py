"""Plot validation mastery and locked deployment generalization for V2."""

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
REPO_ROOT = ROOT.parents[1]
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


def seed_trajectory(rows: list[dict], metric: str) -> dict[str, dict[int, tuple]]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["abstraction"], int(row["seed"]), int(row["step"]))].append(
            float(row[metric])
        )

    by_objective: dict[str, dict[int, tuple]] = {name: {} for name in OBJECTIVES}
    for objective in OBJECTIVES:
        steps = sorted({key[2] for key in grouped if key[0] == objective})
        for step in steps:
            seed_means = np.asarray([
                np.mean(grouped[(objective, seed, step)]) for seed in range(6)
            ])
            mean = float(np.mean(seed_means))
            half = T_CRIT_95_DF5 * float(np.std(seed_means, ddof=1)) / math.sqrt(6)
            by_objective[objective][step] = (mean, mean - half, mean + half)
    return by_objective


def draw_panel(ax, trajectories, ylabel, title, ylim):
    for objective in OBJECTIVES:
        points = trajectories[objective]
        steps = np.asarray(sorted(points))
        values = np.asarray([points[step] for step in steps])
        ax.plot(
            steps, values[:, 0], color=COLORS[objective], linewidth=2.2,
            label=LABELS[objective],
        )
        ax.fill_between(
            steps, values[:, 1], values[:, 2], color=COLORS[objective], alpha=0.14,
            linewidth=0,
        )
    ax.axvline(400, color="#777777", linestyle="--", linewidth=1)
    ax.text(
        430, ylim[0] + 0.025 * (ylim[1] - ylim[0]), "median mastery by step 400",
        color="#666666", fontsize=8, rotation=90, va="bottom",
    )
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlim(0, 10_000)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=ROOT / "results_v2")
    parser.add_argument("--locked", type=Path, default=ROOT / "locked_results_v2")
    parser.add_argument("--png", type=Path, default=REPO_ROOT / "assets/v2_overview.png")
    parser.add_argument("--pdf", type=Path, default=REPO_ROOT / "assets/v2_overview.pdf")
    args = parser.parse_args()

    development = read_directory(args.development)
    locked = read_directory(args.locked)
    expected = 324 * 11
    if len(development) != expected or len(locked) != expected:
        raise ValueError(
            f"expected {expected} rows in each directory; got "
            f"development={len(development)}, locked={len(locked)}"
        )

    metadata = {
        (row["config_id"], int(row["step"])): {
            "abstraction": row["abstraction"], "seed": int(row["seed"]),
        }
        for row in development
    }
    for row in locked:
        key = (row["config_id"], int(row["step"]))
        if key not in metadata:
            raise ValueError(f"locked row has no matching development row: {key}")
        row.update(metadata[key])

    source = seed_trajectory(development, "source_accuracy")
    deployment = seed_trajectory(locked, "test_combo_accuracy")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    draw_panel(
        axes[0], source, "Accuracy", "A  In-distribution validation", (0.42, 1.025),
    )
    draw_panel(
        axes[1], deployment, "Accuracy", "B  Locked new-domain combinations", (0.35, 1.025),
    )
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Validation mastery does not identify deployment generalization",
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
