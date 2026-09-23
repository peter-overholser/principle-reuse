"""Registered locked hierarchy for the prospective V9-B confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from experiments.training_axes.analyze_v9_common import (
    config_id, fmt, sha256, summarize, tree_sha256,
)
from experiments.training_axes.design_v9 import ARMS, MAIN_SEEDS, SCAFFOLDS, read_manifest
from experiments.training_axes.v9b_common import (
    ENDPOINT, load_commitment, verify_reveal, verify_v9b_freeze,
)


BASELINE = "a00_confounded"
PRIMARY = "a50_diverse"


def load_locked(results_dir, manifest_path):
    configs = read_manifest(manifest_path)
    expected = {config.config_id for config in configs}
    paths = sorted(Path(results_dir).glob("*.jsonl"))
    observed = {path.stem for path in paths}
    if expected != observed:
        raise ValueError(
            f"V9-B locked mismatch: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    prefixes = (
        "lock_", "probe_lock_", "paired_lock_", "causal_lock_",
        "organization_lock_",
    )
    by_key = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != 1 or int(rows[0].get("step", -1)) != ENDPOINT:
            raise ValueError(f"wrong V9-B endpoint rows in {path}")
        row = rows[0]
        unexpected = [
            key for key in row
            if key not in {"config_id", "step"} and not key.startswith(prefixes)
        ]
        if unexpected:
            raise ValueError(f"non-locked V9-B field in {path}: {unexpected[:3]}")
        if row["config_id"] != path.stem:
            raise ValueError(f"V9-B config mismatch in {path}")
        by_key[row["config_id"]] = row
    return configs, by_key, paths


def values(by_key, scaffold, arm, metric):
    return [
        float(by_key[config_id(scaffold, arm, seed)][metric])
        for seed in MAIN_SEEDS
    ]


def paired(by_key, scaffold, left, right, metric):
    return summarize([
        left_value - right_value
        for left_value, right_value in zip(
            values(by_key, scaffold, left, metric),
            values(by_key, scaffold, right, metric),
        )
    ])


def interaction(by_key, scaffold, metric):
    result = []
    for seed in MAIN_SEEDS:
        get = lambda arm: float(by_key[config_id(scaffold, arm, seed)][metric])
        result.append(
            (get("a50_diverse") - get("a00_diverse"))
            - (get("a50_confounded") - get("a00_confounded"))
        )
    return summarize(result)


def analyze(
    results, checkpoints, manifest, freeze, v9a_freeze, commitment, reveal,
):
    frozen = verify_v9b_freeze(
        freeze, manifest, checkpoints, v9a_freeze, commitment
    )
    verify_reveal(commitment, reveal)
    configs, by_key, paths = load_locked(results, manifest)
    if len(configs) != 192:
        raise ValueError("V9-B requires all 192 registered endpoints")

    metrics = (
        "lock_x_nonanchor_accuracy", "lock_s_nonanchor_accuracy",
        "lock_s_nonanchor_brier_skill", "lock_s_nonanchor_log_loss",
        "lock_joint_nonanchor_accuracy", "lock_held_source_accuracy",
        "organization_lock_s", "probe_lock_s_r2",
        "paired_lock_s_hidden_cka", "causal_lock_s_normalized",
        "causal_lock_s_valid",
    )
    means, contrasts, hierarchy = {}, {}, {}
    for scaffold in SCAFFOLDS:
        means[scaffold] = {
            arm: {
                metric: mean(values(by_key, scaffold, arm, metric))
                for metric in metrics
            }
            for arm in ARMS
        }
        contrasts[scaffold] = {
            "imposed_minus_baseline_accuracy": paired(
                by_key, scaffold, "imposed", BASELINE,
                "lock_s_nonanchor_accuracy",
            ),
            "primary_minus_baseline_accuracy": paired(
                by_key, scaffold, PRIMARY, BASELINE,
                "lock_s_nonanchor_accuracy",
            ),
            "primary_minus_baseline_O": paired(
                by_key, scaffold, PRIMARY, BASELINE,
                "organization_lock_s",
            ),
            "primary_minus_baseline_probe": paired(
                by_key, scaffold, PRIMARY, BASELINE, "probe_lock_s_r2",
            ),
            "primary_minus_baseline_cka": paired(
                by_key, scaffold, PRIMARY, BASELINE,
                "paired_lock_s_hidden_cka",
            ),
            "primary_minus_baseline_causal": paired(
                by_key, scaffold, PRIMARY, BASELINE,
                "causal_lock_s_normalized",
            ),
            "anchor_trend_diverse": paired(
                by_key, scaffold, "a50_diverse", "a00_diverse",
                "lock_s_nonanchor_accuracy",
            ),
            "diversity_at_a00": paired(
                by_key, scaffold, "a00_diverse", "a00_confounded",
                "lock_s_nonanchor_accuracy",
            ),
            "diversity_at_a50": paired(
                by_key, scaffold, "a50_diverse", "a50_confounded",
                "lock_s_nonanchor_accuracy",
            ),
            "anchor_by_diversity_interaction": interaction(
                by_key, scaffold, "lock_s_nonanchor_accuracy"
            ),
            "primary_minus_baseline_joint": paired(
                by_key, scaffold, PRIMARY, BASELINE,
                "lock_joint_nonanchor_accuracy",
            ),
        }
        components = sum(
            contrasts[scaffold][name]["mean"] > 0
            for name in (
                "primary_minus_baseline_probe",
                "primary_minus_baseline_cka",
                "primary_minus_baseline_causal",
            )
        )
        assay = contrasts[scaffold]["imposed_minus_baseline_accuracy"]["ci_low"] > 0
        behavior = contrasts[scaffold]["primary_minus_baseline_accuracy"]["ci_low"] > 0
        organization = (
            contrasts[scaffold]["primary_minus_baseline_O"]["ci_low"] > 0
            and components >= 2
            and means[scaffold][PRIMARY]["causal_lock_s_valid"] >= (10.0 / 12.0)
        )
        dose = contrasts[scaffold]["anchor_trend_diverse"]["ci_low"] > 0
        hierarchy[scaffold] = {
            "assay_replication": assay,
            "behavioral_emergence": behavior,
            "organizational_emergence": organization,
            "anchor_dose": dose,
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
        "decision": {"route": route},
        "v9a_status": "stopped-interim-source-mastery",
        "endpoint_eligibility": frozen["decision"],
        "hierarchy": hierarchy,
        "means": means,
        "contrasts": contrasts,
        "integrity": {
            "freeze_sha256": sha256(freeze),
            "lock_commitment": load_commitment(commitment)["commitment"],
            "locked_tree_sha256": tree_sha256(paths),
            "analysis_sha256": sha256(Path(__file__)),
            "locked_files": len(paths),
        },
    }


def report(result):
    lines = [
        "# V9-B prospective locked Selection/Emergence report", "",
        "V9-A remains stopped under its original interim-mastery gate.", "",
        f"Decision route: **{result['decision']['route']}**.", "",
        "## Hierarchy", "",
        "| Scaffold | Assay | Behavior | Organization | Anchor dose | Supported |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for scaffold in SCAFFOLDS:
        item = result["hierarchy"][scaffold]
        mark = lambda key: "yes" if item[key] else "no"
        lines.append(
            f"| `{scaffold}` | {mark('assay_replication')} | "
            f"{mark('behavioral_emergence')} | {mark('organizational_emergence')} | "
            f"{mark('anchor_dose')} | {mark('supported')} |"
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
        f"- V9-B freeze SHA-256: `{integrity['freeze_sha256']}`",
        f"- Lock commitment: `{integrity['lock_commitment']}`",
        f"- Locked tree SHA-256: `{integrity['locked_tree_sha256']}`",
        f"- Analysis SHA-256: `{integrity['analysis_sha256']}`",
        f"- Complete locked files: `{integrity['locked_files']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/locked_results_v9b"))
    parser.add_argument("--checkpoints", type=Path, default=Path("experiments/training_axes/checkpoints_v9"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_manifest.csv"))
    parser.add_argument("--freeze", type=Path, default=Path("experiments/training_axes/v9b_preunlock_freeze.json"))
    parser.add_argument("--v9a-freeze", type=Path, default=Path("experiments/training_axes/v9_preunlock_freeze.json"))
    parser.add_argument("--commitment", type=Path, default=Path("experiments/training_axes/v9b_lock_commitment.json"))
    parser.add_argument("--reveal", type=Path, default=Path("experiments/training_axes/v9b_lock_reveal.txt"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/training_axes/v9b_locked_analysis.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V9B_LOCKED_REPORT.md"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_v9b_freeze(
            args.freeze, args.manifest, args.checkpoints,
            args.v9a_freeze, args.commitment,
        )
        verify_reveal(args.commitment, args.reveal)
        print("V9-B freeze and reveal verified; prospective lock is authorized")
        return
    result = analyze(
        args.results, args.checkpoints, args.manifest, args.freeze,
        args.v9a_freeze, args.commitment, args.reveal,
    )
    args.out_json.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
