"""Recompute and verify the checkpoint-free public V9-B release.

The registered preunlock programs also hashed the unreleased model
checkpoints. This verifier starts from the released development trajectories,
the prospective V9-B freeze, the revealed lock salt, and the locked endpoint
rows. It reconstructs the source gates and every released locked statistic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from experiments.training_axes.analyze_locked_v9b import (
    BASELINE, PRIMARY, interaction, load_locked, paired, report, values,
)
from experiments.training_axes.analyze_v9_common import (
    config_id, load_results, paired as development_paired, sha256,
    tree_sha256,
)
from experiments.training_axes.analyze_preunlock_v9 import CHECKPOINTS
from experiments.training_axes.design_v9 import ARMS, MAIN_SEEDS, SCAFFOLDS
from experiments.training_axes.v9b_common import (
    ENDPOINT, EXPECTED_V9A_FAILURE, SOURCE_MARGINS, load_commitment,
    verify_reveal,
)


def assert_equivalent(computed, released, path="analysis"):
    """Compare nested objects while treating registered NaNs as equal."""
    if isinstance(computed, dict) and isinstance(released, dict):
        if computed.keys() != released.keys():
            raise ValueError(f"released V9-B key mismatch at {path}")
        for key in computed:
            assert_equivalent(computed[key], released[key], f"{path}.{key}")
        return
    if isinstance(computed, list) and isinstance(released, list):
        if len(computed) != len(released):
            raise ValueError(f"released V9-B length mismatch at {path}")
        for index, (left, right) in enumerate(zip(computed, released)):
            assert_equivalent(left, right, f"{path}[{index}]")
        return
    if isinstance(computed, float) and isinstance(released, (float, int)):
        right = float(released)
        if math.isnan(computed) and math.isnan(right):
            return
        if not math.isclose(computed, right, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"released V9-B numeric mismatch at {path}")
        return
    if computed != released:
        raise ValueError(f"released V9-B value mismatch at {path}")


def recompute_development(results, manifest):
    configs, by_key, paths = load_results(results, manifest, CHECKPOINTS)
    if len(configs) != 192:
        raise ValueError("the public V9 development tree must contain 192 runs")
    failures = []
    for config in configs:
        for step in (15_000, ENDPOINT):
            if by_key[(config.config_id, step)]["source_accuracy"] < 0.95:
                failures.append([config.config_id, step, "accuracy"])
        if by_key[(config.config_id, ENDPOINT)]["source_brier_skill"] < 0.80:
            failures.append([config.config_id, ENDPOINT, "brier"])

    equivalence = {}
    for scaffold in SCAFFOLDS:
        equivalence[scaffold] = {}
        for arm in ARMS:
            if arm == BASELINE:
                continue
            equivalence[scaffold][arm] = {
                metric: development_paired(
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
    development = {}
    for scaffold in SCAFFOLDS:
        development[scaffold] = {
            "imposed_minus_baseline_accuracy": development_paired(
                by_key, scaffold, "imposed", BASELINE, MAIN_SEEDS,
                "dev_s_nonanchor_accuracy", step=ENDPOINT,
            ),
            "primary_minus_baseline_accuracy": development_paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "dev_s_nonanchor_accuracy", step=ENDPOINT,
            ),
            "primary_minus_baseline_O": development_paired(
                by_key, scaffold, PRIMARY, BASELINE, MAIN_SEEDS,
                "organization_dev_s", step=ENDPOINT,
            ),
            "anchor_trend_diverse": development_paired(
                by_key, scaffold, "a50_diverse", "a00_diverse", MAIN_SEEDS,
                "dev_s_nonanchor_accuracy", step=ENDPOINT,
            ),
        }
    endpoint_failures = []
    for config in configs:
        row = by_key[(config.config_id, ENDPOINT)]
        if row["source_accuracy"] < 0.95:
            endpoint_failures.append([config.config_id, ENDPOINT, "accuracy"])
        if row["source_brier_skill"] < 0.80:
            endpoint_failures.append([config.config_id, ENDPOINT, "brier"])
    return {
        "failures": failures,
        "equivalence": equivalence,
        "equivalence_pass": equivalence_pass,
        "development": development,
        "endpoint_failures": endpoint_failures,
    }, paths


def recompute_locked(results, endpoint_eligibility):
    configs, by_key, paths = load_locked(results, Path(
        "experiments/training_axes/v9_manifest.csv"
    ))
    if len(configs) != 192:
        raise ValueError("the public V9-B lock must contain 192 endpoints")
    metric_names = (
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
                for metric in metric_names
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
                by_key, scaffold, PRIMARY, BASELINE, "organization_lock_s",
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
        component_count = sum(
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
            and component_count >= 2
            and means[scaffold][PRIMARY]["causal_lock_s_valid"] >= 10 / 12
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
        "endpoint_eligibility": endpoint_eligibility,
        "hierarchy": hierarchy,
        "means": means,
        "contrasts": contrasts,
    }, paths


def verify(args):
    root = Path(__file__).resolve().parent
    reveal = verify_reveal(args.commitment, args.reveal)
    del reveal
    v9a = json.loads(args.v9a_freeze.read_text())
    v9b = json.loads(args.v9b_freeze.read_text())
    released = json.loads(args.analysis.read_text())

    if sha256(args.v9a_freeze) != v9b["integrity"]["v9a_freeze_sha256"]:
        raise ValueError("released V9-A freeze does not match V9-B provenance")
    if v9a["source_failures"] != [EXPECTED_V9A_FAILURE]:
        raise ValueError("released V9-A freeze has the wrong stopped failure")
    if v9a["decision"] != {
        "unlock_allowed": False,
        "source_mastery": False,
        "source_equivalence": True,
    }:
        raise ValueError("released V9-A stopped decision changed")

    computed_dev, development_paths = recompute_development(
        args.development, args.manifest
    )
    assert_equivalent(computed_dev["failures"], v9a["source_failures"], "V9-A failures")
    assert_equivalent(computed_dev["equivalence"], v9a["source_equivalence"], "V9-A equivalence")
    assert_equivalent(computed_dev["development"], v9a["development_contrasts"], "V9-A development")
    if tree_sha256(development_paths) != v9a["integrity"]["development_tree_sha256"]:
        raise ValueError("released V9 development tree hash mismatch")
    if tree_sha256(development_paths) != v9b["integrity"]["development_tree_sha256"]:
        raise ValueError("V9-B development provenance mismatch")
    if computed_dev["endpoint_failures"] != v9b["endpoint_failures"]:
        raise ValueError("V9-B endpoint failure set changed")
    assert_equivalent(computed_dev["equivalence"], v9b["source_equivalence"], "V9-B equivalence")

    public_hashes = {
        "v9b_protocol_sha256": root / "V9B_PROTOCOL.md",
        "v9b_commitment_file_sha256": args.commitment,
        "v9b_common_sha256": root / "v9b_common.py",
        "v9b_preunlock_analysis_sha256": root / "prepare_v9b.py",
        "v9b_locked_evaluator_sha256": root / "evaluate_locked_v9b.py",
        "v9b_locked_all_evaluator_sha256": root / "evaluate_all_locked_v9b.py",
        "v9b_locked_analysis_sha256": root / "analyze_locked_v9b.py",
        "v9a_freeze_sha256": args.v9a_freeze,
        "manifest_sha256": args.manifest,
        "task_sha256": root / "task_v9.py",
        "design_sha256": root / "design_v9.py",
        "metrics_sha256": root / "metrics_v9.py",
        "model_sha256": root.parent / "panel" / "model.py",
    }
    for key, path in public_hashes.items():
        if v9b["integrity"].get(key) != sha256(path):
            raise ValueError(f"released V9-B frozen file mismatch: {key}")
    if load_commitment(args.commitment)["commitment"] != v9b["lock_commitment"]:
        raise ValueError("released V9-B commitment changed")

    computed, locked_paths = recompute_locked(
        args.locked, v9b["decision"]
    )
    for key, value in computed.items():
        assert_equivalent(value, released.get(key), key)
    expected_integrity = {
        "freeze_sha256": sha256(args.v9b_freeze),
        "lock_commitment": v9b["lock_commitment"],
        "locked_tree_sha256": tree_sha256(locked_paths),
        "analysis_sha256": sha256(root / "analyze_locked_v9b.py"),
        "locked_files": len(locked_paths),
    }
    if released.get("integrity") != expected_integrity:
        raise ValueError("released V9-B locked integrity mismatch")
    if report(released) != args.report.read_text():
        raise ValueError("released V9-B report does not match its analysis")
    return released


def main():
    root = Path("experiments/training_axes")
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=root / "results_v9")
    parser.add_argument("--locked", type=Path, default=root / "locked_results_v9b")
    parser.add_argument("--manifest", type=Path, default=root / "v9_manifest.csv")
    parser.add_argument("--v9a-freeze", type=Path, default=root / "v9_preunlock_freeze.json")
    parser.add_argument("--v9b-freeze", type=Path, default=root / "v9b_preunlock_freeze.json")
    parser.add_argument("--commitment", type=Path, default=root / "v9b_lock_commitment.json")
    parser.add_argument("--reveal", type=Path, default=root / "v9b_lock_reveal.txt")
    parser.add_argument("--analysis", type=Path, default=root / "v9b_locked_analysis.json")
    parser.add_argument("--report", type=Path, default=root / "V9B_LOCKED_REPORT.md")
    args = parser.parse_args()
    result = verify(args)
    print(
        "V9-B public release verified: "
        f"{result['integrity']['locked_files']} locked endpoints; "
        f"route={result['decision']['route']}"
    )


if __name__ == "__main__":
    main()
