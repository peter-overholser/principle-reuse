"""Registered development-only V8 analysis and locked-evaluation freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.training_axes.analyze_v8_common import (
    ARMS, CHECKPOINTS, SEEDS, TASKS, arm_mean, checkpoint_tree,
    checkpoint_tree_sha256, config_id, factorial_effect, fmt, load_development,
    load_diagnostics, paired_effect, sha256, tree_sha256, validate_manifest,
)


SOURCE_LOSS_TARGET = 0.01
SOURCE_EQUIVALENCE_MARGINS = {
    "source_accuracy": 0.02,
    "source_brier_skill": 0.03,
    "source_log_loss": 0.02,
    "source_log_loss_near": 0.05,
}


def source_matched_steps(by_key):
    selected = {}
    candidates = tuple(step for step in CHECKPOINTS if step >= 400)
    for task in TASKS:
        for arm in ARMS:
            for seed in SEEDS:
                run_id = config_id(task, arm, seed)
                eligible = [
                    step for step in candidates
                    if float(by_key[(run_id, step)]["source_accuracy"]) >= 0.95
                ]
                if not eligible:
                    raise ValueError(f"no mastered source checkpoint for {run_id}")
                selected[run_id] = min(
                    eligible,
                    key=lambda step: (
                        abs(float(by_key[(run_id, step)]["source_log_loss"])
                            - SOURCE_LOSS_TARGET),
                        step,
                    ),
                )
    return selected


def analyze(results_dir, checkpoints_dir, manifest, protocol):
    validate_manifest(manifest)
    by_key, paths = load_development(results_dir)
    diagnostics, diagnostic_paths = load_diagnostics(results_dir)
    checkpoint_paths = checkpoint_tree(checkpoints_dir)
    source_failures = []
    for task in TASKS:
        for arm in ARMS:
            for seed in SEEDS:
                run_id = config_id(task, arm, seed)
                for step in (7_000, 10_000):
                    if float(by_key[(run_id, step)]["source_accuracy"]) < 0.95:
                        source_failures.append((task, arm, seed, step, "accuracy"))
                if float(by_key[(run_id, 10_000)]["source_brier_skill"]) < 0.80:
                    source_failures.append((task, arm, seed, 10_000, "brier"))

    matched_steps = source_matched_steps(by_key)
    arm_means = {
        task: {
            arm: {
                metric: arm_mean(by_key, task, arm, metric)
                for metric in (
                    "source_accuracy", "source_log_loss", "source_log_loss_near",
                    "calib_combo_accuracy", "calib_combo_brier_skill",
                    "calib_combo_shortcut_agreement", "probe_target_r2",
                    "embedding_rank_axis_cosine",
                )
            }
            for arm in ARMS
        }
        for task in TASKS
    }
    contrasts = {}
    factorial = {}
    source_equivalence = {}
    for task in TASKS:
        contrasts[task] = {
            "full_minus_invariant": paired_effect(
                by_key, task, "e1h1g1", "e0h0g0", "calib_combo_accuracy"
            ),
            "eonly_minus_full": paired_effect(
                by_key, task, "e1h0g0", "e1h1g1", "calib_combo_accuracy"
            ),
            "nonembedding_minus_invariant": paired_effect(
                by_key, task, "e0h1g1", "e0h0g0", "calib_combo_accuracy"
            ),
            "eperm_minus_eonly": paired_effect(
                by_key, task, "eperm", "e1h0g0", "calib_combo_accuracy"
            ),
            "gshared_minus_gseparate": paired_effect(
                by_key, task, "e0h0g1", "gsep", "calib_combo_accuracy"
            ),
        }
        factorial[task] = {
            factor: factorial_effect(by_key, task, factor, "calib_combo_accuracy")
            for factor in ("embedding", "hidden", "gap")
        }
        source_equivalence[task] = {}
        for arm in ("e1h1g1", "e1h0g0", "e0h1g1"):
            source_equivalence[task][arm] = {
                metric: paired_effect(by_key, task, arm, "e0h0g0", metric)
                for metric in (
                    "source_accuracy", "source_brier_skill", "source_log_loss",
                    "source_log_loss_near",
                )
            }

    gradient_summary = {}
    for task in TASKS:
        gradient_summary[task] = {}
        for arm in ARMS:
            gradient_summary[task][arm] = {}
            for step in (400, 1_000, 2_000, 10_000):
                rows = [
                    diagnostics[(config_id(task, arm, seed), step)] for seed in SEEDS
                ]
                gradient_summary[task][arm][str(step)] = {
                    name: sum(float(row[name]) for row in rows) / len(rows)
                    for name in (
                        "loss_binary", "loss_invariant", "loss_representation",
                        "loss_latent", "loss_mapping", "gradient_share_binary",
                        "gradient_share_invariant", "gradient_share_representation",
                        "gradient_share_latent", "gradient_share_mapping",
                    )
                }

    equivalence_failures = []
    for task, arms in source_equivalence.items():
        for arm, values in arms.items():
            for metric, value in values.items():
                margin = SOURCE_EQUIVALENCE_MARGINS[metric]
                if value["ci_low"] < -margin or value["ci_high"] > margin:
                    equivalence_failures.append({
                        "task": task, "arm": arm, "metric": metric,
                        "margin": margin, "effect": value,
                    })

    unlock_allowed = not source_failures and not equivalence_failures

    module_root = Path(__file__).resolve().parent
    return {
        "decision": {
            "unlock_allowed": unlock_allowed,
            "label": (
                "unlock" if unlock_allowed else
                "source-mastery-failed" if source_failures else
                "source-fiber-equivalence-failed"
            ),
            "source_failures": source_failures,
            "source_equivalence_failures": equivalence_failures,
        },
        "source_loss_target": SOURCE_LOSS_TARGET,
        "source_equivalence_margins": SOURCE_EQUIVALENCE_MARGINS,
        "source_matched_steps": matched_steps,
        "arm_means": arm_means,
        "development_contrasts": contrasts,
        "development_factorial_effects": factorial,
        "source_equivalence": source_equivalence,
        "gradient_summary": gradient_summary,
        "integrity": {
            "protocol_sha256": sha256(protocol),
            "manifest_sha256": sha256(manifest),
            "results_tree_sha256": tree_sha256(paths),
            "diagnostics_tree_sha256": tree_sha256(diagnostic_paths),
            "analysis_sha256": sha256(Path(__file__)),
            "common_analysis_sha256": sha256(module_root / "analyze_v8_common.py"),
            "task_sha256": sha256(module_root / "task_v8.py"),
            "runner_sha256": sha256(module_root / "run.py"),
            "metrics_sha256": sha256(module_root / "metrics.py"),
            "model_sha256": sha256(module_root.parent / "panel" / "model.py"),
            "locked_evaluator_sha256": sha256(module_root / "evaluate_locked.py"),
            "locked_analysis_sha256": sha256(module_root / "analyze_locked_v8.py"),
            "result_files": len(paths), "diagnostic_files": len(diagnostic_paths),
            "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint_paths),
            "checkpoint_files": len(checkpoint_paths),
        },
    }


def report(result):
    lines = [
        "# V8 component-ablation development freeze", "",
        "No locked combination, held-query, or joint-shift metric was present.", "",
        f"Decision: **{result['decision']['label']}**.", "",
        "## Development-combination endpoint means", "",
        "| Task | Arm | Accuracy | Brier | Shortcut | Probe R2 | Embedding cosine | Source log loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for arm in ARMS:
            item = result["arm_means"][task][arm]
            lines.append(
                f"| `{task}` | `{arm}` | {item['calib_combo_accuracy']:.3f} | "
                f"{item['calib_combo_brier_skill']:.3f} | "
                f"{item['calib_combo_shortcut_agreement']:.3f} | "
                f"{item['probe_target_r2']:.3f} | "
                f"{item['embedding_rank_axis_cosine']:.3f} | "
                f"{item['source_log_loss']:.4f} |"
            )
    lines += ["", "## Prespecified development contrasts", ""]
    for task in TASKS:
        for name, value in result["development_contrasts"][task].items():
            lines.append(
                f"- `{task}/{name}`: {fmt(value)}; positive seeds "
                f"{value['positive_seeds']}/12."
            )
    lines += ["", "## Marginal factorial effects on accuracy", ""]
    for task in TASKS:
        for factor, value in result["development_factorial_effects"][task].items():
            lines.append(f"- `{task}/{factor}`: {fmt(value)}.")
    lines += [
        "", "## Source-fiber equivalence", "",
        "Every interval below must lie inside its registered margin before "
        "locked evaluation is permitted.", "",
        "| Task | Arm vs invariant | Metric | Difference [95% interval] | Margin | Pass |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for task in TASKS:
        for arm, values in result["source_equivalence"][task].items():
            for metric, value in values.items():
                margin = result["source_equivalence_margins"][metric]
                passed = value["ci_low"] >= -margin and value["ci_high"] <= margin
                lines.append(
                    f"| `{task}` | `{arm}` | `{metric}` | {fmt(value)} | "
                    f"+/-{margin:.2f} | {'yes' if passed else 'no'} |"
                )

    lines += ["", "## Objective and gradient diagnostics", ""]
    for task in TASKS:
        lines += [
            f"### {task}", "",
            "| Arm | Step | Binary share | Invariance | Hidden | Gap | Embedding |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            for step in (400, 1_000, 2_000, 10_000):
                value = result["gradient_summary"][task][arm][str(step)]
                lines.append(
                    f"| `{arm}` | {step} | {value['gradient_share_binary']:.3f} | "
                    f"{value['gradient_share_invariant']:.3f} | "
                    f"{value['gradient_share_representation']:.3f} | "
                    f"{value['gradient_share_latent']:.3f} | "
                    f"{value['gradient_share_mapping']:.3f} |"
                )
        lines.append("")
    integrity = result["integrity"]
    lines += [
        "", "## Integrity", "",
        f"- Protocol SHA-256: `{integrity['protocol_sha256']}`",
        f"- Manifest SHA-256: `{integrity['manifest_sha256']}`",
        f"- Development tree SHA-256: `{integrity['results_tree_sha256']}`",
        f"- Diagnostics tree SHA-256: `{integrity['diagnostics_tree_sha256']}`",
        f"- Analysis SHA-256: `{integrity['analysis_sha256']}`",
        f"- Common analysis SHA-256: `{integrity['common_analysis_sha256']}`",
        f"- Task SHA-256: `{integrity['task_sha256']}`",
        f"- Runner SHA-256: `{integrity['runner_sha256']}`",
        f"- Metrics SHA-256: `{integrity['metrics_sha256']}`",
        f"- Model SHA-256: `{integrity['model_sha256']}`",
        f"- Locked evaluator SHA-256: `{integrity['locked_evaluator_sha256']}`",
        f"- Locked analysis SHA-256: `{integrity['locked_analysis_sha256']}`",
        f"- Checkpoint tree SHA-256: `{integrity['checkpoint_tree_sha256']}`",
        f"- Complete result/diagnostic files: `{integrity['result_files']}` / "
        f"`{integrity['diagnostic_files']}`",
        f"- Complete checkpoint files: `{integrity['checkpoint_files']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/results_v8"))
    parser.add_argument("--checkpoints", type=Path, default=Path("experiments/training_axes/checkpoints_v8"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v8_manifest.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/training_axes/V8_PROTOCOL.md"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/training_axes/v8_preunlock_freeze.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V8_PREUNLOCK_REPORT.md"))
    args = parser.parse_args()
    result = analyze(args.results, args.checkpoints, args.manifest, args.protocol)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
