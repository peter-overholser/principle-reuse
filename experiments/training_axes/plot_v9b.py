"""Plot the released V9-B anchor and coverage intervention."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path("experiments/training_axes/locked_results_v9b")
SEEDS = range(512, 524)
T_CRIT_11 = 2.200985


def config_id(scaffold, arm, seed):
    return f"v9_sc-{scaffold}_{arm}_s-{seed:03d}"


def load_rows():
    rows = {}
    for path in ROOT.glob("*.jsonl"):
        row = json.loads(path.read_text())
        rows[row["config_id"]] = row
    if len(rows) != 192:
        raise ValueError("expected 192 released V9-B endpoint rows")
    return rows


def summary(rows, scaffold, arm, metric):
    values = [rows[config_id(scaffold, arm, seed)][metric] for seed in SEEDS]
    center = mean(values)
    half = T_CRIT_11 * stdev(values) / math.sqrt(len(values))
    return values, center, half


def panel(ax, rows, scaffold, metric, ylabel):
    anchors = (0.0, 0.25, 0.5)
    conditions = (
        ("confounded", "#6b7280", "High confounding"),
        ("diverse", "#147d64", "Diverse coverage"),
    )
    for suffix, color, label in conditions:
        centers, halves = [], []
        for prefix in ("a00", "a25", "a50"):
            _, center, half = summary(
                rows, scaffold, f"{prefix}_{suffix}", metric
            )
            centers.append(center); halves.append(half)
        ax.errorbar(
            anchors, centers, yerr=halves, color=color, marker="o",
            linewidth=2.2, markersize=6, capsize=3, label=label,
        )
    for arm, color, linestyle, label in (
        ("imposed", "#c46a19", "--", "Imposed correspondence"),
        ("eperm", "#a23b3b", ":", "Permuted correspondence"),
    ):
        _, center, _ = summary(rows, scaffold, arm, metric)
        ax.axhline(center, color=color, linestyle=linestyle, linewidth=1.6, label=label)
    ax.set_xlim(-0.03, 0.53)
    ax.set_xticks(anchors, ("0", "0.25", "0.50"))
    ax.set_xlabel("Shared-anchor fraction")
    ax.set_ylabel(ylabel)
    ax.set_ylim((0.45, 1.02) if "accuracy" in metric else (0.0, 1.02))
    if "accuracy" in metric:
        ax.axhline(0.5, color="#b9b9b9", linewidth=1, zorder=0)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    rows = load_rows()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex="col")
    for row, scaffold in enumerate(("line", "circle")):
        panel(
            axes[row, 0], rows, scaffold, "lock_s_nonanchor_accuracy",
            "Locked non-anchor accuracy",
        )
        panel(
            axes[row, 1], rows, scaffold, "organization_lock_s",
            "Organization score",
        )
        axes[row, 0].text(
            -0.18, 1.04, scaffold.capitalize(), transform=axes[row, 0].transAxes,
            fontsize=12, fontweight="bold", va="top",
        )
    axes[0, 0].set_title("Structural-role generalization")
    axes[0, 1].set_title("Transferred internal organization")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Shared anchors and coverage diversity induce reusable structure",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0.03, 0.08, 1, 0.96))
    output = Path("assets")
    output.mkdir(exist_ok=True)
    fig.savefig(output / "v9b_emergence.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "v9b_emergence.pdf", bbox_inches="tight")
    print("wrote assets/v9b_emergence.png and assets/v9b_emergence.pdf")


if __name__ == "__main__":
    main()
