"""Registered feasibility gate for the four-seed V9 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.training_axes.analyze_v9_common import (
    arm_mean, fmt, load_results, paired,
)
from experiments.training_axes.design_v9 import ARMS, PILOT_SEEDS, SCAFFOLDS


CHECKPOINTS = (0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 7000, 10000)
BASELINE = "a00_confounded"
PRIMARY = "a50_diverse"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(results, manifest, protocol):
    configs, by_key, paths = load_results(results, manifest, CHECKPOINTS)
    if len(configs) != len(SCAFFOLDS) * len(ARMS) * len(PILOT_SEEDS):
        raise ValueError("V9 pilot manifest is not the registered 64-run design")
    mastery_failures = []
    for config in configs:
        for step in (7000, 10000):
            row = by_key[(config.config_id, step)]
            if row["source_accuracy"] < 0.95:
                mastery_failures.append((config.config_id, step, "accuracy"))
        if by_key[(config.config_id, 10000)]["source_brier_skill"] < 0.80:
            mastery_failures.append((config.config_id, 10000, "brier"))

    contrasts, means = {}, {}
    for scaffold in SCAFFOLDS:
        contrasts[scaffold] = {
            "imposed_minus_baseline_accuracy": paired(
                by_key, scaffold, "imposed", BASELINE, PILOT_SEEDS,
                "dev_s_nonanchor_accuracy",
            ),
            "imposed_minus_baseline_O": paired(
                by_key, scaffold, "imposed", BASELINE, PILOT_SEEDS,
                "organization_dev_s",
            ),
            "eperm_minus_imposed_accuracy": paired(
                by_key, scaffold, "eperm", "imposed", PILOT_SEEDS,
                "dev_s_nonanchor_accuracy",
            ),
            "primary_minus_baseline_accuracy": paired(
                by_key, scaffold, PRIMARY, BASELINE, PILOT_SEEDS,
                "dev_s_nonanchor_accuracy",
            ),
            "primary_minus_baseline_O": paired(
                by_key, scaffold, PRIMARY, BASELINE, PILOT_SEEDS,
                "organization_dev_s",
            ),
        }
        means[scaffold] = {
            arm: {
                metric: arm_mean(by_key, scaffold, arm, PILOT_SEEDS, metric)
                for metric in (
                    "source_accuracy", "dev_x_nonanchor_accuracy",
                    "dev_s_nonanchor_accuracy", "organization_dev_s",
                    "probe_dev_s_r2", "paired_dev_s_hidden_cka",
                    "causal_dev_s_normalized", "causal_dev_s_valid",
                )
            }
            for arm in ARMS
        }

    source_pass = not mastery_failures
    imposed_pass = all(
        contrasts[scaffold]["imposed_minus_baseline_accuracy"]["mean"] >= 0.25
        and contrasts[scaffold]["imposed_minus_baseline_accuracy"]["positive_seeds"] == 4
        and contrasts[scaffold]["imposed_minus_baseline_O"]["mean"] > 0
        and means[scaffold]["imposed"]["causal_dev_s_valid"] == 1.0
        for scaffold in SCAFFOLDS
    )
    semantic_pass = all(
        contrasts[scaffold]["eperm_minus_imposed_accuracy"]["mean"] <= -0.15
        for scaffold in SCAFFOLDS
    )
    emergence_signal = all(
        contrasts[scaffold]["primary_minus_baseline_accuracy"]["mean"] >= 0.10
        and contrasts[scaffold]["primary_minus_baseline_accuracy"]["positive_seeds"] >= 3
        and contrasts[scaffold]["primary_minus_baseline_O"]["mean"] > 0
        for scaffold in SCAFFOLDS
    )
    if not source_pass:
        route = "repair-source-mastery"
    elif not imposed_pass:
        route = "stop-invalid-assay"
    elif not semantic_pass:
        route = "stop-semantic-control-failed"
    elif not emergence_signal:
        route = "test-context-channel"
    else:
        route = "run-main"
    root = Path(__file__).resolve().parent
    integrity_files = {
        "protocol_sha256": protocol,
        "manifest_sha256": manifest,
        "task_sha256": root / "task_v9.py",
        "design_sha256": root / "design_v9.py",
        "runner_sha256": root / "run_v9.py",
        "metrics_sha256": root / "metrics_v9.py",
        "analysis_sha256": Path(__file__),
        "model_sha256": root.parent / "panel" / "model.py",
    }
    return {
        "decision": {
            "route": route, "source_pass": source_pass,
            "imposed_pass": imposed_pass, "semantic_pass": semantic_pass,
            "emergence_signal": emergence_signal,
        },
        "mastery_failures": mastery_failures,
        "means": means, "contrasts": contrasts,
        "integrity": {
            **{key: sha256(path) for key, path in integrity_files.items()},
            "results_files": len(paths),
        },
    }


def verify_gate(path):
    value = json.loads(Path(path).read_text())
    if value["decision"]["route"] != "run-main":
        raise ValueError("V9 pilot did not authorize main")
    root = Path(__file__).resolve().parent
    files = {
        "protocol_sha256": root / "V9_PROTOCOL.md",
        "manifest_sha256": root / "v9_pilot_manifest.csv",
        "task_sha256": root / "task_v9.py",
        "design_sha256": root / "design_v9.py",
        "runner_sha256": root / "run_v9.py",
        "metrics_sha256": root / "metrics_v9.py",
        "analysis_sha256": Path(__file__),
        "model_sha256": root.parent / "panel" / "model.py",
    }
    for key, file_path in files.items():
        if value["integrity"].get(key) != sha256(file_path):
            raise ValueError(f"V9 pilot code changed after gate: {key}")
    return value


def report(result):
    lines = [
        "# V9-A feasibility pilot", "",
        "No locked role was constructed or evaluated.", "",
        f"Decision route: **{result['decision']['route']}**.", "",
        "## Gate", "",
        "| Gate | Pass |", "|---|:---:|",
    ]
    for key, label in (
        ("source_pass", "Source mastery"),
        ("imposed_pass", "Imposed positive control"),
        ("semantic_pass", "Permuted semantic control"),
        ("emergence_signal", "A50 + diverse feasibility signal"),
    ):
        lines.append(f"| {label} | {'yes' if result['decision'][key] else 'no'} |")
    lines += ["", "## Registered contrasts", ""]
    for scaffold in SCAFFOLDS:
        lines += [f"### {scaffold}", ""]
        for name, value in result["contrasts"][scaffold].items():
            lines.append(
                f"- `{name}`: {fmt(value)}; positive seeds "
                f"{value['positive_seeds']}/4."
            )
        lines.append("")
    lines += [
        "## Endpoint means", "",
        "| Scaffold | Arm | Source | Dev-x nonanchor | Dev-s nonanchor | O | Probe R2 | CKA | Causal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scaffold in SCAFFOLDS:
        for arm in ARMS:
            item = result["means"][scaffold][arm]
            lines.append(
                f"| `{scaffold}` | `{arm}` | {item['source_accuracy']:.3f} | "
                f"{item['dev_x_nonanchor_accuracy']:.3f} | "
                f"{item['dev_s_nonanchor_accuracy']:.3f} | "
                f"{item['organization_dev_s']:.3f} | "
                f"{item['probe_dev_s_r2']:.3f} | "
                f"{item['paired_dev_s_hidden_cka']:.3f} | "
                f"{item['causal_dev_s_normalized']:.3f} |"
            )
    integrity = result["integrity"]
    lines += [
        "", "## Integrity", "",
        f"- Protocol SHA-256: `{integrity['protocol_sha256']}`",
        f"- Manifest SHA-256: `{integrity['manifest_sha256']}`",
        f"- Complete result files: `{integrity['results_files']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/results_v9_pilot"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_pilot_manifest.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/training_axes/V9_PROTOCOL.md"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/training_axes/v9_pilot_gate.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V9_PILOT_GATE.md"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_gate(args.out_json)
        print("V9 pilot gate and code hashes verified; main is authorized")
        return
    result = analyze(args.results, args.manifest, args.protocol)
    args.out_json.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
