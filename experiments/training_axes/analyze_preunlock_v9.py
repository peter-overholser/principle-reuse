"""Registered V9 main source gate and development freeze."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.training_axes.analyze_v9_common import (
    arm_mean, checkpoint_tree, config_id, fmt, load_results, paired, sha256,
    tree_sha256,
)
from experiments.training_axes.analyze_v9_r2 import verify_gate
from experiments.training_axes.design_v9 import ARMS, MAIN_SEEDS, SCAFFOLDS


CHECKPOINTS = (
    0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 7000,
    10000, 12000, 15000, 20000,
)
BASELINE = "a00_confounded"
PRIMARY = "a50_diverse"
SOURCE_MARGINS = {
    "source_accuracy": 0.02,
    "source_brier_skill": 0.03,
    "source_log_loss": 0.02,
    "source_log_loss_near": 0.05,
}


def select_source_steps(by_key):
    selected = {}
    for scaffold in SCAFFOLDS:
        for arm in ARMS:
            for seed in MAIN_SEEDS:
                cid = config_id(scaffold, arm, seed)
                eligible = []
                for step in CHECKPOINTS:
                    row = by_key[(cid, step)]
                    if row["source_accuracy"] >= 0.95:
                        eligible.append((abs(row["source_log_loss"] - 0.01), step))
                if not eligible:
                    raise ValueError(f"no source-mastered V9 checkpoint: {cid}")
                selected[cid] = min(eligible)[1]
    return selected


def analyze(
    results, checkpoints, manifest, protocol, repair_protocol, r2_protocol,
    repair_gate, r2_gate, out_freeze,
):
    verify_gate(r2_gate)
    configs, by_key, result_paths = load_results(results, manifest, CHECKPOINTS)
    expected_count = len(SCAFFOLDS) * len(ARMS) * len(MAIN_SEEDS)
    if len(configs) != expected_count:
        raise ValueError("V9 main manifest is not the registered 192-run design")
    failures = []
    for config in configs:
        for step in (15_000, 20_000):
            row = by_key[(config.config_id, step)]
            if row["source_accuracy"] < 0.95:
                failures.append((config.config_id, step, "accuracy"))
        if by_key[(config.config_id, 20_000)]["source_brier_skill"] < 0.80:
            failures.append((config.config_id, 20_000, "brier"))

    equivalence = {}
    for scaffold in SCAFFOLDS:
        equivalence[scaffold] = {}
        for arm in ARMS:
            if arm == BASELINE:
                continue
            equivalence[scaffold][arm] = {
                metric: paired(
                    by_key, scaffold, arm, BASELINE, MAIN_SEEDS, metric,
                    step=20_000,
                )
                for metric in SOURCE_MARGINS
            }
    equivalence_pass = all(
        value["ci_low"] >= -SOURCE_MARGINS[metric]
        and value["ci_high"] <= SOURCE_MARGINS[metric]
        for scaffold in SCAFFOLDS
        for arm in equivalence[scaffold]
        for metric, value in equivalence[scaffold][arm].items()
    )
    source_pass = not failures and equivalence_pass
    selected_steps = select_source_steps(by_key) if not failures else {}

    development = {}
    for scaffold in SCAFFOLDS:
        development[scaffold] = {
            "imposed_minus_baseline_accuracy": paired(
                by_key, scaffold, "imposed", BASELINE, MAIN_SEEDS,
                "dev_s_nonanchor_accuracy", step=20_000,
            ),
            "primary_minus_baseline_accuracy": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "dev_s_nonanchor_accuracy", step=20_000,
            ),
            "primary_minus_baseline_O": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "organization_dev_s", step=20_000,
            ),
            "anchor_trend_diverse": paired(
                by_key, scaffold, "a50_diverse", "a00_diverse", MAIN_SEEDS,
                "dev_s_nonanchor_accuracy", step=20_000,
            ),
        }

    root = Path(__file__).resolve().parent
    checkpoint_paths = checkpoint_tree(
        checkpoints, [config.config_id for config in configs], CHECKPOINTS
    )
    integrity_files = {
        "protocol_sha256": protocol,
        "manifest_sha256": manifest,
        "repair_protocol_sha256": repair_protocol,
        "r2_protocol_sha256": r2_protocol,
        "repair_gate_sha256": repair_gate,
        "r2_gate_sha256": r2_gate,
        "repair_analysis_sha256": root / "analyze_v9_repair.py",
        "r2_analysis_sha256": root / "analyze_v9_r2.py",
        "r2_design_sha256": root / "design_v9_r2.py",
        "r2_reliability_evaluator_sha256": root / "evaluate_v9_r2_reliability.py",
        "r2_reliability_all_evaluator_sha256": root / "evaluate_all_v9_r2_reliability.py",
        "task_sha256": root / "task_v9.py",
        "design_sha256": root / "design_v9.py",
        "runner_sha256": root / "run_v9.py",
        "metrics_sha256": root / "metrics_v9.py",
        "common_analysis_sha256": root / "analyze_v9_common.py",
        "preunlock_analysis_sha256": Path(__file__),
        "locked_evaluator_sha256": root / "evaluate_locked_v9.py",
        "locked_all_evaluator_sha256": root / "evaluate_all_locked_v9.py",
        "locked_analysis_sha256": root / "analyze_locked_v9.py",
        "model_sha256": root.parent / "panel" / "model.py",
    }
    result = {
        "decision": {
            "unlock_allowed": source_pass,
            "source_mastery": not failures,
            "source_equivalence": equivalence_pass,
        },
        "source_failures": failures,
        "source_equivalence": equivalence,
        "source_matched_steps": selected_steps,
        "development_contrasts": development,
        "integrity": {
            **{key: sha256(path) for key, path in integrity_files.items()},
            "development_tree_sha256": tree_sha256(result_paths),
            "checkpoint_tree_sha256": tree_sha256(
                checkpoint_paths, root=Path(checkpoints)
            ),
            "development_files": len(result_paths),
            "checkpoint_files": len(checkpoint_paths),
        },
    }
    Path(out_freeze).write_text(json.dumps(result, indent=2) + "\n")
    return result


def report(result):
    decision = "unlock" if result["decision"]["unlock_allowed"] else "stop"
    lines = [
        "# V9-A main development freeze", "",
        "No locked role was constructed or evaluated.", "",
        f"Decision: **{decision}**.", "",
        "## Development contrasts", "",
    ]
    for scaffold in SCAFFOLDS:
        lines += [f"### {scaffold}", ""]
        for name, value in result["development_contrasts"][scaffold].items():
            lines.append(
                f"- `{name}`: {fmt(value)}; positive seeds "
                f"{value['positive_seeds']}/12."
            )
        lines.append("")
    lines += [
        "## Source gate", "",
        f"- Mastery: {'pass' if result['decision']['source_mastery'] else 'fail'}",
        f"- Equivalence: {'pass' if result['decision']['source_equivalence'] else 'fail'}",
        f"- Failures: {len(result['source_failures'])}",
        "", "## Integrity", "",
    ]
    for key, value in result["integrity"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/results_v9"))
    parser.add_argument("--checkpoints", type=Path, default=Path("experiments/training_axes/checkpoints_v9"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_manifest.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/training_axes/V9_PROTOCOL.md"))
    parser.add_argument("--repair-protocol", type=Path, default=Path("experiments/training_axes/V9_REPAIR_PROTOCOL.md"))
    parser.add_argument("--r2-protocol", type=Path, default=Path("experiments/training_axes/V9_R2_PROTOCOL.md"))
    parser.add_argument("--repair-gate", type=Path, default=Path("experiments/training_axes/v9_repair_gate.json"))
    parser.add_argument("--r2-gate", type=Path, default=Path("experiments/training_axes/v9_r2_gate.json"))
    parser.add_argument("--out-freeze", type=Path, default=Path("experiments/training_axes/v9_preunlock_freeze.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V9_PREUNLOCK_REPORT.md"))
    args = parser.parse_args()
    result = analyze(
        args.results, args.checkpoints, args.manifest, args.protocol,
        args.repair_protocol, args.r2_protocol, args.repair_gate, args.r2_gate,
        args.out_freeze,
    )
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
