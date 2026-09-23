"""Outcome-free endpoint gate and integrity freeze for V9-B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.training_axes.analyze_v9_common import (
    fmt, load_results, paired, sha256, tree_sha256,
)
from experiments.training_axes.analyze_preunlock_v9 import CHECKPOINTS
from experiments.training_axes.design_v9 import ARMS, MAIN_SEEDS, SCAFFOLDS
from experiments.training_axes.v9b_common import (
    ENDPOINT, EXPECTED_V9A_FAILURE, SOURCE_MARGINS,
    endpoint_checkpoint_tree, load_commitment, v9b_integrity_files,
    verify_stopped_v9a,
)


BASELINE = "a00_confounded"


def analyze(results, checkpoints, manifest, v9a_freeze, commitment, out_freeze):
    verify_stopped_v9a(
        v9a_freeze, manifest, checkpoints, results=results
    )
    configs, by_key, result_paths = load_results(results, manifest, CHECKPOINTS)
    expected = len(SCAFFOLDS) * len(ARMS) * len(MAIN_SEEDS)
    if len(configs) != expected:
        raise ValueError("V9-B requires all 192 registered V9-A trunks")

    endpoint_failures = []
    for config in configs:
        row = by_key[(config.config_id, ENDPOINT)]
        if row["source_accuracy"] < 0.95:
            endpoint_failures.append([config.config_id, ENDPOINT, "accuracy"])
        if row["source_brier_skill"] < 0.80:
            endpoint_failures.append([config.config_id, ENDPOINT, "brier"])

    equivalence = {}
    for scaffold in SCAFFOLDS:
        equivalence[scaffold] = {}
        for arm in ARMS:
            if arm == BASELINE:
                continue
            equivalence[scaffold][arm] = {
                metric: paired(
                    by_key, scaffold, arm, BASELINE, MAIN_SEEDS, metric,
                    step=ENDPOINT,
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
    endpoint_mastery = not endpoint_failures
    lock_authorized = endpoint_mastery and equivalence_pass

    root = Path(__file__).resolve().parent
    endpoint_paths = endpoint_checkpoint_tree(checkpoints, manifest)
    files = v9b_integrity_files(
        root, manifest, v9a_freeze, commitment
    )
    result = {
        "decision": {
            "lock_authorized": lock_authorized,
            "endpoint_mastery": endpoint_mastery,
            "source_equivalence": equivalence_pass,
        },
        "v9a_source_failure": EXPECTED_V9A_FAILURE,
        "endpoint_failures": endpoint_failures,
        "source_equivalence": equivalence,
        "lock_commitment": load_commitment(commitment)["commitment"],
        "integrity": {
            **{key: sha256(path) for key, path in files.items()},
            "development_tree_sha256": tree_sha256(result_paths),
            "endpoint_checkpoint_tree_sha256": tree_sha256(
                endpoint_paths, root=Path(checkpoints)
            ),
            "development_files": len(result_paths),
            "endpoint_checkpoints": len(endpoint_paths),
        },
    }
    Path(out_freeze).write_text(json.dumps(result, indent=2) + "\n")
    return result


def report(result):
    decision = "unlock-fresh-lock" if result["decision"]["lock_authorized"] else "stop"
    lines = [
        "# V9-B prospective endpoint freeze", "",
        "V9-A remains stopped. No V9-B locked example was constructed or evaluated.", "",
        f"Decision: **{decision}**.", "", "## Eligibility", "",
        "| Gate | Pass |", "|---|:---:|",
        f"| All 192 endpoints mastered | {'yes' if result['decision']['endpoint_mastery'] else 'no'} |",
        f"| Registered source equivalence | {'yes' if result['decision']['source_equivalence'] else 'no'} |",
        f"| V9-A stopped provenance verified | yes |", "",
        f"Endpoint failures: **{len(result['endpoint_failures'])}**.", "",
        "## Source equivalence", "",
    ]
    for scaffold, arms in result["source_equivalence"].items():
        lines += [f"### {scaffold}", ""]
        for arm, metrics in arms.items():
            rendered = ", ".join(
                f"{metric} {fmt(value)}" for metric, value in metrics.items()
            )
            lines.append(f"- `{arm}`: {rendered}.")
        lines.append("")
    lines += ["## Integrity", ""]
    for key, value in result["integrity"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", f"- Lock commitment: `{result['lock_commitment']}`"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/results_v9"))
    parser.add_argument("--checkpoints", type=Path, default=Path("experiments/training_axes/checkpoints_v9"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_manifest.csv"))
    parser.add_argument("--v9a-freeze", type=Path, default=Path("experiments/training_axes/v9_preunlock_freeze.json"))
    parser.add_argument("--commitment", type=Path, default=Path("experiments/training_axes/v9b_lock_commitment.json"))
    parser.add_argument("--out-freeze", type=Path, default=Path("experiments/training_axes/v9b_preunlock_freeze.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V9B_PREUNLOCK_REPORT.md"))
    args = parser.parse_args()
    result = analyze(
        args.results, args.checkpoints, args.manifest, args.v9a_freeze,
        args.commitment, args.out_freeze,
    )
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
