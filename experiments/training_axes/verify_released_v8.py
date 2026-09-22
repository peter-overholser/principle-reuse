"""Recompute and verify the checkpoint-free public V8 release.

The registered analyzer also verifies the unreleased model checkpoints before
opening the lock.  This public verifier starts from the frozen development and
locked JSONL trajectories, checks their recorded hashes and schemas, and then
recomputes every field in the released locked analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.training_axes.analyze_locked_v8 import report
from experiments.training_axes.analyze_v8_common import (
    ARMS,
    TASKS,
    arm_mean,
    factorial_effect,
    load_development,
    load_diagnostics,
    load_locked,
    paired_effect,
    paired_effect_at_steps,
    sha256,
    tree_sha256,
    validate_manifest,
)


def recompute(locked_dir: Path, freeze: dict) -> tuple[dict, list[Path]]:
    by_key, paths = load_locked(locked_dir)
    selected_steps = {
        key: int(value) for key, value in freeze["source_matched_steps"].items()
    }
    arm_means = {
        task: {
            arm: {
                metric: arm_mean(by_key, task, arm, metric)
                for metric in (
                    "test_combo_accuracy",
                    "test_combo_brier_skill",
                    "test_combo_log_loss",
                    "test_combo_shortcut_agreement",
                    "test_combo_shortcut_probability",
                    "test_both_accuracy",
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
                by_key,
                task,
                "e1h1g1",
                "e0h0g0",
                "test_combo_accuracy",
                selected_steps,
            ),
            "nonembedding_minus_invariant_accuracy": paired_effect_at_steps(
                by_key,
                task,
                "e0h1g1",
                "e0h0g0",
                "test_combo_accuracy",
                selected_steps,
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
        for key in (
            "full_replication",
            "eonly_equivalent_full",
            "nonembedding_superior",
        )
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
    }, paths


def assert_equivalent(computed, released, path="analysis") -> None:
    """Compare nested analysis objects across Python statistics versions."""
    if isinstance(computed, dict) and isinstance(released, dict):
        if computed.keys() != released.keys():
            raise ValueError(f"released V8 key mismatch at {path}")
        for key in computed:
            assert_equivalent(computed[key], released[key], f"{path}.{key}")
        return
    if isinstance(computed, list) and isinstance(released, list):
        if len(computed) != len(released):
            raise ValueError(f"released V8 length mismatch at {path}")
        for index, (left, right) in enumerate(zip(computed, released)):
            assert_equivalent(left, right, f"{path}[{index}]")
        return
    if isinstance(computed, float) and isinstance(released, (float, int)):
        if not math.isclose(computed, float(released), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"released V8 numeric mismatch at {path}")
        return
    if computed != released:
        raise ValueError(f"released V8 value mismatch at {path}")


def verify(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parent
    validate_manifest(args.manifest)
    freeze = json.loads(args.freeze.read_text())
    if not freeze["decision"]["unlock_allowed"]:
        raise ValueError("the released development freeze did not authorize unlock")

    development, development_paths = load_development(args.development)
    diagnostics, diagnostic_paths = load_diagnostics(args.development)
    if len(development) != 264 * 11 or len(diagnostics) != 264 * 43:
        raise ValueError("unexpected V8 development or diagnostic row count")

    expected_frozen_hashes = {
        "protocol_sha256": sha256(args.protocol),
        "manifest_sha256": sha256(args.manifest),
        "results_tree_sha256": tree_sha256(development_paths),
        "diagnostics_tree_sha256": tree_sha256(diagnostic_paths),
        "analysis_sha256": sha256(root / "analyze_preunlock_v8.py"),
        "common_analysis_sha256": sha256(root / "analyze_v8_common.py"),
        "task_sha256": sha256(root / "task_v8.py"),
        "runner_sha256": sha256(root / "run.py"),
        "metrics_sha256": sha256(root / "metrics.py"),
        "model_sha256": sha256(root.parent / "panel" / "model.py"),
        "locked_evaluator_sha256": sha256(root / "evaluate_locked.py"),
        "locked_analysis_sha256": sha256(root / "analyze_locked_v8.py"),
    }
    for key, value in expected_frozen_hashes.items():
        if freeze["integrity"].get(key) != value:
            raise ValueError(f"released V8 freeze mismatch: {key}")

    computed, locked_paths = recompute(args.locked, freeze)
    released = json.loads(args.analysis.read_text())
    for key, value in computed.items():
        assert_equivalent(value, released.get(key), key)

    expected_locked_integrity = {
        "freeze_sha256": sha256(args.freeze),
        "locked_tree_sha256": tree_sha256(locked_paths),
        "analysis_sha256": sha256(root / "analyze_locked_v8.py"),
        "locked_files": len(locked_paths),
    }
    if released.get("integrity") != expected_locked_integrity:
        raise ValueError("released V8 locked integrity mismatch")
    if report(released) != args.report.read_text():
        raise ValueError("released V8 report does not match the analysis JSON")
    return released


def main() -> None:
    root = Path("experiments/training_axes")
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=root / "results_v8")
    parser.add_argument("--locked", type=Path, default=root / "locked_results_v8")
    parser.add_argument("--manifest", type=Path, default=root / "v8_manifest.csv")
    parser.add_argument("--protocol", type=Path, default=root / "V8_PROTOCOL.md")
    parser.add_argument(
        "--freeze", type=Path, default=root / "v8_preunlock_freeze.json"
    )
    parser.add_argument(
        "--analysis", type=Path, default=root / "v8_locked_analysis.json"
    )
    parser.add_argument(
        "--report", type=Path, default=root / "V8_LOCKED_REPORT.md"
    )
    args = parser.parse_args()
    result = verify(args)
    print(
        "V8 public release verified: "
        f"{result['integrity']['locked_files']} locked runs; "
        f"route={result['decision']['route']}"
    )


if __name__ == "__main__":
    main()
