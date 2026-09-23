"""Registered locked hierarchy for V9-A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.training_axes.analyze_v9_common import (
    arm_mean, checkpoint_tree, config_id, fmt, load_locked, paired, sha256,
    tree_sha256,
)
from experiments.training_axes.analyze_preunlock_v9 import CHECKPOINTS
from experiments.training_axes.design_v9 import ARMS, MAIN_SEEDS, SCAFFOLDS


BASELINE = "a00_confounded"
PRIMARY = "a50_diverse"
ENDPOINT = 20_000


def verify_freeze(manifest, protocol, freeze_path, checkpoints):
    freeze = json.loads(Path(freeze_path).read_text())
    if not freeze["decision"]["unlock_allowed"]:
        raise ValueError("V9 source gate did not permit unlock")
    root = Path(__file__).resolve().parent
    files = {
        "protocol_sha256": protocol,
        "repair_protocol_sha256": root / "V9_REPAIR_PROTOCOL.md",
        "r2_protocol_sha256": root / "V9_R2_PROTOCOL.md",
        "manifest_sha256": manifest,
        "repair_gate_sha256": root / "v9_repair_gate.json",
        "r2_gate_sha256": root / "v9_r2_gate.json",
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
        "preunlock_analysis_sha256": root / "analyze_preunlock_v9.py",
        "locked_evaluator_sha256": root / "evaluate_locked_v9.py",
        "locked_all_evaluator_sha256": root / "evaluate_all_locked_v9.py",
        "locked_analysis_sha256": Path(__file__),
        "model_sha256": root.parent / "panel" / "model.py",
    }
    for key, path in files.items():
        if freeze["integrity"].get(key) != sha256(path):
            raise ValueError(f"V9 changed after freeze: {key}")
    configs = __import__(
        "experiments.training_axes.design_v9", fromlist=["read_manifest"]
    ).read_manifest(manifest)
    checkpoint_paths = checkpoint_tree(
        checkpoints, [config.config_id for config in configs], CHECKPOINTS
    )
    observed = tree_sha256(checkpoint_paths, root=Path(checkpoints))
    if observed != freeze["integrity"]["checkpoint_tree_sha256"]:
        raise ValueError("V9 changed after freeze: checkpoint tree")
    return freeze


def paired_at_selected(by_key, scaffold, left, right, metric, selected):
    values = []
    for seed in MAIN_SEEDS:
        left_id, right_id = config_id(scaffold, left, seed), config_id(scaffold, right, seed)
        values.append(
            by_key[(left_id, int(selected[left_id]))][metric]
            - by_key[(right_id, int(selected[right_id]))][metric]
        )
    from experiments.training_axes.analyze_v9_common import summarize
    return summarize(values)


def analyze(results, checkpoints, manifest, protocol, freeze_path):
    freeze = verify_freeze(manifest, protocol, freeze_path, checkpoints)
    configs, by_key, paths = load_locked(results, manifest, CHECKPOINTS)
    means, contrasts, hierarchy = {}, {}, {}
    for scaffold in SCAFFOLDS:
        metrics = (
            "lock_x_nonanchor_accuracy", "lock_s_nonanchor_accuracy",
            "lock_s_nonanchor_brier_skill", "lock_s_nonanchor_log_loss",
            "lock_joint_nonanchor_accuracy", "lock_held_source_accuracy",
            "organization_lock_s", "probe_lock_s_r2",
            "paired_lock_s_hidden_cka", "causal_lock_s_normalized",
            "causal_lock_s_valid",
        )
        means[scaffold] = {
            arm: {
                metric: arm_mean(
                    by_key, scaffold, arm, MAIN_SEEDS, metric, step=ENDPOINT
                )
                for metric in metrics
            }
            for arm in ARMS
        }
        contrasts[scaffold] = {
            "imposed_minus_baseline_accuracy": paired(
                by_key, scaffold, "imposed", BASELINE, MAIN_SEEDS,
                "lock_s_nonanchor_accuracy", step=ENDPOINT,
            ),
            "primary_minus_baseline_accuracy": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "lock_s_nonanchor_accuracy", step=ENDPOINT,
            ),
            "primary_minus_baseline_O": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "organization_lock_s", step=ENDPOINT,
            ),
            "primary_minus_baseline_probe": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "probe_lock_s_r2", step=ENDPOINT,
            ),
            "primary_minus_baseline_cka": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "paired_lock_s_hidden_cka", step=ENDPOINT,
            ),
            "primary_minus_baseline_causal": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "causal_lock_s_source_to_target", step=ENDPOINT,
            ),
            "anchor_trend_diverse": paired(
                by_key, scaffold, "a50_diverse", "a00_diverse", MAIN_SEEDS,
                "lock_s_nonanchor_accuracy", step=ENDPOINT,
            ),
            "primary_minus_baseline_joint": paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "lock_joint_nonanchor_accuracy", step=ENDPOINT,
            ),
            "matched_imposed_minus_baseline": paired_at_selected(
                by_key, scaffold, "imposed", BASELINE,
                "lock_s_nonanchor_accuracy", freeze["source_matched_steps"],
            ),
            "matched_primary_minus_baseline": paired_at_selected(
                by_key, scaffold, PRIMARY, BASELINE,
                "lock_s_nonanchor_accuracy", freeze["source_matched_steps"],
            ),
        }
        component_positive = sum(
            contrasts[scaffold][name]["mean"] > 0
            for name in (
                "primary_minus_baseline_probe", "primary_minus_baseline_cka",
                "primary_minus_baseline_causal",
            )
        )
        assay = contrasts[scaffold]["imposed_minus_baseline_accuracy"]["ci_low"] > 0
        behavior = contrasts[scaffold]["primary_minus_baseline_accuracy"]["ci_low"] > 0
        organization = (
            contrasts[scaffold]["primary_minus_baseline_O"]["ci_low"] > 0
            and component_positive >= 2
            and means[scaffold][PRIMARY]["causal_lock_s_valid"] >= (10.0 / 12.0)
        )
        dose = contrasts[scaffold]["anchor_trend_diverse"]["ci_low"] > 0
        hierarchy[scaffold] = {
            "assay_replication": assay,
            "behavioral_emergence": behavior,
            "organizational_emergence": organization,
            "dose_response": dose,
            "supported": assay and behavior and organization and dose,
        }
    if not all(hierarchy[s]["assay_replication"] for s in SCAFFOLDS):
        route = "assay-failed"
    elif all(hierarchy[s]["supported"] for s in SCAFFOLDS):
        route = "emergence-supported"
    elif any(hierarchy[s]["supported"] for s in SCAFFOLDS):
        route = "scaffold-specific"
    else:
        route = "imposed-only"
    return {
        "decision": {"route": route}, "hierarchy": hierarchy,
        "means": means, "contrasts": contrasts,
        "integrity": {
            "freeze_sha256": sha256(freeze_path),
            "locked_tree_sha256": tree_sha256(paths),
            "analysis_sha256": sha256(Path(__file__)),
            "locked_files": len(paths),
        },
    }


def report(result):
    lines = [
        "# V9-A locked Selection/Emergence report", "",
        f"Decision route: **{result['decision']['route']}**.", "",
        "## Hierarchy", "",
        "| Scaffold | Assay | Behavior | Organization | Dose | Supported |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for scaffold in SCAFFOLDS:
        item = result["hierarchy"][scaffold]
        mark = lambda key: "yes" if item[key] else "no"
        lines.append(
            f"| `{scaffold}` | {mark('assay_replication')} | "
            f"{mark('behavioral_emergence')} | {mark('organizational_emergence')} | "
            f"{mark('dose_response')} | {mark('supported')} |"
        )
    lines += ["", "## Registered contrasts", ""]
    for scaffold in SCAFFOLDS:
        lines += [f"### {scaffold}", ""]
        for name, value in result["contrasts"][scaffold].items():
            lines.append(
                f"- `{name}`: {fmt(value)}; positive seeds "
                f"{value['positive_seeds']}/12."
            )
        lines.append("")
    lines += [
        "## Endpoint means", "",
        "| Scaffold | Arm | Lock-s nonanchor | Joint | O | Probe | CKA | Causal |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scaffold in SCAFFOLDS:
        for arm in ARMS:
            item = result["means"][scaffold][arm]
            lines.append(
                f"| `{scaffold}` | `{arm}` | {item['lock_s_nonanchor_accuracy']:.3f} | "
                f"{item['lock_joint_nonanchor_accuracy']:.3f} | "
                f"{item['organization_lock_s']:.3f} | {item['probe_lock_s_r2']:.3f} | "
                f"{item['paired_lock_s_hidden_cka']:.3f} | "
                f"{item['causal_lock_s_normalized']:.3f} |"
            )
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
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/locked_results_v9"))
    parser.add_argument("--checkpoints", type=Path, default=Path("experiments/training_axes/checkpoints_v9"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_manifest.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/training_axes/V9_PROTOCOL.md"))
    parser.add_argument("--freeze", type=Path, default=Path("experiments/training_axes/v9_preunlock_freeze.json"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/training_axes/v9_locked_analysis.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V9_LOCKED_REPORT.md"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_freeze(args.manifest, args.protocol, args.freeze, args.checkpoints)
        print("V9 development freeze verified; locked evaluation is authorized")
        return
    result = analyze(
        args.results, args.checkpoints, args.manifest, args.protocol, args.freeze
    )
    args.out_json.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
