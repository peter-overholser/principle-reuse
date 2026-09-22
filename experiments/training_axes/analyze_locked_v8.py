"""Registered locked analysis for V8 structural-component ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.training_axes.analyze_v8_common import (
    ARMS, TASKS, arm_mean, checkpoint_tree, checkpoint_tree_sha256,
    factorial_effect, fmt, load_locked, paired_effect, paired_effect_at_steps,
    sha256, tree_sha256, validate_manifest,
)


def verify_freeze(manifest, protocol, freeze_path, checkpoints_dir):
    validate_manifest(manifest)
    freeze = json.loads(freeze_path.read_text())
    if not freeze["decision"]["unlock_allowed"]:
        raise ValueError("V8 source-validity gate did not permit unlock")
    root = Path(__file__).resolve().parent
    expected_hashes = {
        "protocol_sha256": sha256(protocol),
        "manifest_sha256": sha256(manifest),
        "analysis_sha256": sha256(root / "analyze_preunlock_v8.py"),
        "common_analysis_sha256": sha256(root / "analyze_v8_common.py"),
        "task_sha256": sha256(root / "task_v8.py"),
        "runner_sha256": sha256(root / "run.py"),
        "metrics_sha256": sha256(root / "metrics.py"),
        "model_sha256": sha256(root.parent / "panel" / "model.py"),
        "locked_evaluator_sha256": sha256(root / "evaluate_locked.py"),
        "locked_analysis_sha256": sha256(Path(__file__)),
        "checkpoint_tree_sha256": checkpoint_tree_sha256(
            checkpoint_tree(checkpoints_dir)
        ),
    }
    for key, value in expected_hashes.items():
        if freeze["integrity"].get(key) != value:
            raise ValueError(f"V8 changed after development freeze: {key}")
    return freeze


def analyze(results_dir, checkpoints_dir, manifest, protocol, freeze_path):
    freeze = verify_freeze(manifest, protocol, freeze_path, checkpoints_dir)

    by_key, paths = load_locked(results_dir)
    matched_steps = {
        key: int(value) for key, value in freeze["source_matched_steps"].items()
    }
    arm_means = {
        task: {
            arm: {
                metric: arm_mean(by_key, task, arm, metric)
                for metric in (
                    "test_combo_accuracy", "test_combo_brier_skill",
                    "test_combo_log_loss", "test_combo_shortcut_agreement",
                    "test_combo_shortcut_probability", "test_both_accuracy",
                    "held_pair_accuracy",
                )
            }
            for arm in ARMS
        }
        for task in TASKS
    }
    contrasts, factorial, matched = {}, {}, {}
    for task in TASKS:
        contrasts[task] = {
            "full_minus_invariant_accuracy": paired_effect(
                by_key, task, "e1h1g1", "e0h0g0", "test_combo_accuracy"
            ),
            "full_minus_invariant_brier": paired_effect(
                by_key, task, "e1h1g1", "e0h0g0", "test_combo_brier_skill"
            ),
            "eonly_minus_full_accuracy": paired_effect(
                by_key, task, "e1h0g0", "e1h1g1", "test_combo_accuracy"
            ),
            "nonembedding_minus_invariant_accuracy": paired_effect(
                by_key, task, "e0h1g1", "e0h0g0", "test_combo_accuracy"
            ),
            "nonembedding_minus_invariant_brier": paired_effect(
                by_key, task, "e0h1g1", "e0h0g0", "test_combo_brier_skill"
            ),
            "eperm_minus_eonly_accuracy": paired_effect(
                by_key, task, "eperm", "e1h0g0", "test_combo_accuracy"
            ),
            "eperm_minus_concrete_accuracy": paired_effect(
                by_key, task, "eperm", "concrete", "test_combo_accuracy"
            ),
            "gshared_minus_gseparate_accuracy": paired_effect(
                by_key, task, "e0h0g1", "gsep", "test_combo_accuracy"
            ),
        }
        factorial[task] = {
            factor: factorial_effect(by_key, task, factor, "test_combo_accuracy")
            for factor in ("embedding", "hidden", "gap")
        }
        matched[task] = {
            "full_minus_invariant_accuracy": paired_effect_at_steps(
                by_key, task, "e1h1g1", "e0h0g0", "test_combo_accuracy",
                matched_steps,
            ),
            "nonembedding_minus_invariant_accuracy": paired_effect_at_steps(
                by_key, task, "e0h1g1", "e0h0g0", "test_combo_accuracy",
                matched_steps,
            ),
        }

    hierarchy = {}
    for task in TASKS:
        item = contrasts[task]
        hierarchy[task] = {
            "full_replication": item["full_minus_invariant_accuracy"]["ci_low"] > 0,
            "eonly_equivalent_full": (
                item["eonly_minus_full_accuracy"]["ci_low"] >= -0.10
                and item["eonly_minus_full_accuracy"]["ci_high"] <= 0.10
            ),
            "nonembedding_superior": (
                item["nonembedding_minus_invariant_accuracy"]["ci_low"] > 0
            ),
        }
    both_tasks = {
        key: all(hierarchy[task][key] for task in TASKS)
        for key in ("full_replication", "eonly_equivalent_full", "nonembedding_superior")
    }
    any_nonembedding = any(
        hierarchy[task]["nonembedding_superior"] for task in TASKS
    )
    if not both_tasks["full_replication"]:
        route = "replication-failed"
    elif both_tasks["nonembedding_superior"]:
        route = "non-tying-effect-survives"
    elif both_tasks["eonly_equivalent_full"] and not any_nonembedding:
        route = "embedding-tying-dominates"
    else:
        route = "mixed-component-result"
    return {
        "decision": {"route": route, "both_task_hierarchy": both_tasks},
        "task_hierarchy": hierarchy,
        "arm_means": arm_means,
        "locked_contrasts": contrasts,
        "matched_source_loss_contrasts": matched,
        "locked_factorial_effects": factorial,
        "integrity": {
            "freeze_sha256": sha256(freeze_path),
            "locked_tree_sha256": tree_sha256(paths),
            "analysis_sha256": sha256(Path(__file__)),
            "locked_files": len(paths),
        },
    }


def report(result):
    lines = [
        "# V8 fresh-lock component-ablation report", "",
        f"Decision route: **{result['decision']['route']}**.", "",
        "## Endpoint locked means", "",
        "| Task | Arm | Accuracy | Brier | Shortcut agreement | Shortcut probability | Joint shift |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for arm in ARMS:
            item = result["arm_means"][task][arm]
            lines.append(
                f"| `{task}` | `{arm}` | {item['test_combo_accuracy']:.3f} | "
                f"{item['test_combo_brier_skill']:.3f} | "
                f"{item['test_combo_shortcut_agreement']:.3f} | "
                f"{item['test_combo_shortcut_probability']:.3f} | "
                f"{item['test_both_accuracy']:.3f} |"
            )
    lines += ["", "## Sequential hierarchy", ""]
    for task in TASKS:
        lines.append(f"### {task}")
        lines.append("")
        for name, value in result["locked_contrasts"][task].items():
            lines.append(
                f"- `{name}`: {fmt(value)}; positive seeds "
                f"{value['positive_seeds']}/12."
            )
        lines.append("")
    lines += ["## Matched-source-loss checkpoint contrasts", ""]
    for task in TASKS:
        for name, value in result["matched_source_loss_contrasts"][task].items():
            lines.append(f"- `{task}/{name}`: {fmt(value)}.")
    integrity = result["integrity"]
    lines += [
        "", "## Integrity", "",
        f"- Development freeze SHA-256: `{integrity['freeze_sha256']}`",
        f"- Locked tree SHA-256: `{integrity['locked_tree_sha256']}`",
        f"- Analysis SHA-256: `{integrity['analysis_sha256']}`",
        f"- Complete locked files: `{integrity['locked_files']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/locked_results_v8"))
    parser.add_argument("--checkpoints", type=Path, default=Path("experiments/training_axes/checkpoints_v8"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v8_manifest.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/training_axes/V8_PROTOCOL.md"))
    parser.add_argument("--freeze", type=Path, default=Path("experiments/training_axes/v8_preunlock_freeze.json"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/training_axes/v8_locked_analysis.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V8_LOCKED_REPORT.md"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_freeze(
            args.manifest, args.protocol, args.freeze, args.checkpoints
        )
        print("V8 development freeze verified; locked evaluation is authorized")
        return
    result = analyze(
        args.results, args.checkpoints, args.manifest, args.protocol, args.freeze
    )
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
