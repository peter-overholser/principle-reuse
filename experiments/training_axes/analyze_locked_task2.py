"""Registered endpoint analysis for the Task 2 locked deployment evaluation.

The inferential unit is the paired random seed. Objective contrasts first
average the fully crossed difficulty and capacity cells within each seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.training_axes import analyze_locked_v2 as base
from experiments.training_axes.analyze_preunlock_task2 import tree_sha256


ROOT = Path(__file__).resolve().parent
OBJECTIVES = ("concrete", "invariant", "latent")
DEPLOYMENT_SPLITS = ("test_combo", "held_pair", "test_both")
PAIR_FACTORS = ("difficulty", "capacity")
MARGINS = ("near", "middle", "far")
EXPECTED_CHECKPOINTS = {0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 7000, 10_000}
FROZEN_HASHES = {
    "manifest": "8bed6bf9ebf1a5779d60c348f79b90997b450305cc36de2226b6009c043feaec",
    "protocol": "c20c9e671c453183a3a8da79b6970ff98507c55972759a4706a096871d444f4f",
    "development_tree": "504a8503f2ee5c8297474f1181dae848f61e0cd0a0bdb0ba3d897e1525f04e7a",
    "preunlock_analysis": "83f1e72574b7e03ada27c31d2d700075d7ec4af8aa9d2436b29b0847d6763208",
    "preunlock_contrasts": "59507134dfc74a8b98c0c4e21a65cd0cc412d790c11298c7cdfeb5c70afd7c84",
    "preunlock_report": "d78794da6d5f2c742b7d0ca926f3506334d7e25ed692a986a6eb74038615e310",
}


def fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def render_contrast(result: dict) -> str:
    return (
        f"{fmt(result['mean'], signed=True)} "
        f"[{fmt(result['t_low'], signed=True)}, {fmt(result['t_high'], signed=True)}]"
    )


def verify_freeze(paths: dict[str, Path]) -> dict[str, str]:
    observed = {
        "manifest": base.sha256(paths["manifest"]),
        "protocol": base.sha256(paths["protocol"]),
        "development_tree": tree_sha256(paths["development"]),
        "preunlock_analysis": base.sha256(paths["preunlock_analysis"]),
        "preunlock_contrasts": base.sha256(paths["preunlock_contrasts"]),
        "preunlock_report": base.sha256(paths["preunlock_report"]),
    }
    mismatches = {
        key: {"expected": FROZEN_HASHES[key], "observed": value}
        for key, value in observed.items()
        if value != FROZEN_HASHES[key]
    }
    if mismatches:
        raise ValueError(f"pre-unlock freeze mismatch: {mismatches}")
    return observed


def validate_trajectories(counts: dict[str, int], rows: list[dict], label: str) -> None:
    if len(counts) != 162:
        raise ValueError(f"expected 162 {label} files, got {len(counts)}")
    if set(counts.values()) != {11}:
        raise ValueError(f"{label} rows per run are not uniformly 11: {set(counts.values())}")
    steps_by_config: dict[str, set[int]] = {}
    for row in rows:
        steps_by_config.setdefault(row["config_id"], set()).add(int(row["step"]))
    bad = [config_id for config_id, steps in steps_by_config.items() if steps != EXPECTED_CHECKPOINTS]
    if bad:
        raise ValueError(f"unexpected checkpoint schedule in {label}: {bad[:3]}")


def objective_means(rows: list[dict], outcomes: tuple[str, ...]) -> dict:
    return {
        objective: {
            outcome: float(np.mean([
                float(row[outcome]) for row in rows if row["abstraction"] == objective
            ]))
            for outcome in outcomes
        }
        for objective in OBJECTIVES
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def negative_cells(rows: list[dict], split: str) -> list[dict]:
    keyed = {
        (
            int(row["seed"]), str(row["difficulty"]), str(row["capacity"]),
            str(row["abstraction"]),
        ): float(row[f"{split}_accuracy"])
        for row in rows
    }
    output = []
    factors = sorted({
        (int(row["seed"]), str(row["difficulty"]), str(row["capacity"]))
        for row in rows
    })
    for seed, difficulty, capacity in factors:
        concrete = keyed[(seed, difficulty, capacity, "concrete")]
        latent = keyed[(seed, difficulty, capacity, "latent")]
        if latent < concrete:
            output.append({
                "seed": seed, "difficulty": difficulty, "capacity": capacity,
                "concrete": concrete, "latent": latent,
                "difference": latent - concrete,
            })
    return output


def contrast(
    rows: list[dict], outcome: str, treatment: str = "latent",
    control: str = "concrete", subset: dict[str, str] | None = None,
) -> dict:
    return base.contrast_row(rows, outcome, treatment, control, subset)


def build_contrasts(rows: list[dict]) -> dict:
    outcomes = (
        "source_accuracy", "source_brier_skill", "source_all_accuracy",
        "calib_combo_accuracy", "calib_combo_brier_skill",
        "test_combo_accuracy", "test_combo_brier_skill",
        "held_pair_accuracy", "held_pair_brier_skill",
        "test_both_accuracy", "test_both_brier_skill",
    )
    result = {
        "objective_means": objective_means(rows, outcomes),
        "primary": contrast(rows, "test_combo_accuracy"),
        "endpoint": {},
        "mastery_matched": {},
        "factor_conditioned": {},
        "signed_margin": {},
        "stability": {},
    }
    for split in DEPLOYMENT_SPLITS:
        result["endpoint"][split] = {
            "latent_minus_concrete_accuracy": contrast(rows, f"{split}_accuracy"),
            "invariant_minus_concrete_accuracy": contrast(
                rows, f"{split}_accuracy", "invariant", "concrete"
            ),
            "latent_minus_concrete_brier_skill": contrast(
                rows, f"{split}_brier_skill"
            ),
        }
        result["mastery_matched"][split] = {
            "latent_minus_concrete_accuracy": contrast(
                rows, f"mastery_{split}_accuracy"
            ),
            "latent_minus_concrete_brier_skill": contrast(
                rows, f"mastery_{split}_brier_skill"
            ),
        }
        result["signed_margin"][split] = {
            margin: contrast(rows, f"{split}_accuracy_{margin}")
            for margin in MARGINS
        }
    result["validation_equivalence"] = {
        "source_accuracy": contrast(rows, "source_accuracy"),
        "source_brier_skill": contrast(rows, "source_brier_skill"),
    }
    result["locked_minus_development"] = contrast(
        rows, "test_combo_minus_calib_accuracy"
    )
    result["factor_conditioned"] = {
        factor: {
            level: contrast(
                rows, "test_combo_accuracy", subset={factor: level}
            )
            for level in sorted({str(row[factor]) for row in rows})
        }
        for factor in PAIR_FACTORS
    }
    for split in DEPLOYMENT_SPLITS:
        positive, zero, negative = base.matched_cell_signs(rows, f"{split}_accuracy")
        result["stability"][split] = {
            "positive": positive, "zero": zero, "negative": negative,
            "total": positive + zero + negative,
            "negative_cells": negative_cells(rows, split),
        }
    return result


def make_report(contrasts: dict, integrity: dict, report_path: Path) -> None:
    means = contrasts["objective_means"]
    validation = contrasts["validation_equivalence"]
    primary = contrasts["primary"]
    accuracy_equivalent = (
        validation["source_accuracy"]["t_low"] > -0.02
        and validation["source_accuracy"]["t_high"] < 0.02
    )
    brier_equivalent = (
        validation["source_brier_skill"]["t_low"] > -0.03
        and validation["source_brier_skill"]["t_high"] < 0.03
    )
    primary_positive = primary["t_low"] > 0
    attenuation = contrasts["locked_minus_development"]
    attenuation_detected = attenuation["t_high"] < 0 or attenuation["t_low"] > 0

    lines = [
        "# Task 2 locked deployment report",
        "",
        "Generated after the pre-unlock freeze and the one-time evaluation of the",
        "registered `test_combo`, `held_pair`, and `test_both` splits.",
        "",
        "## Integrity",
        "",
        f"- Registered runs: {integrity['manifest_runs']}.",
        f"- Development files/rows: {integrity['development_files']}/{integrity['development_rows']}.",
        f"- Locked files/rows: {integrity['locked_files']}/{integrity['locked_rows']}.",
        f"- Checkpoints per run: {integrity['rows_per_run']}; final step: {integrity['final_step']}.",
        f"- Development tree SHA-256 (frozen): `{integrity['freeze_hashes']['development_tree']}`.",
        f"- Locked tree SHA-256: `{integrity['locked_tree_sha256']}`.",
        "",
        "All files join one-to-one with the frozen manifest, every trajectory has the",
        "registered checkpoint schedule, and no run or checkpoint was selected using a",
        "locked outcome.",
        "",
        "## Validation equivalence",
        "",
        "| Endpoint source metric | Structural auxiliary − concrete | Equivalence margin |",
        "|---|---:|---:|",
        f"| Accuracy | {render_contrast(validation['source_accuracy'])} | ±0.02 |",
        f"| Brier skill | {render_contrast(validation['source_brier_skill'])} | ±0.03 |",
        "",
        (
            "Both intervals lie wholly within the inherited equivalence margins."
            if accuracy_equivalent and brier_equivalent
            else "At least one inherited equivalence criterion is not satisfied."
        ),
        "",
        "## Confirmatory result",
        "",
        "Seed-paired mean differences and 95% t intervals use six seeds; each seed",
        "contrast averages the nine crossed difficulty-by-capacity cells.",
        "",
        "| Locked outcome | Structural auxiliary − concrete | Output invariant − concrete |",
        "|---|---:|---:|",
    ]
    for split in DEPLOYMENT_SPLITS:
        item = contrasts["endpoint"][split]
        lines.append(
            f"| `{split}` accuracy | "
            f"{render_contrast(item['latent_minus_concrete_accuracy'])} | "
            f"{render_contrast(item['invariant_minus_concrete_accuracy'])} |"
        )
    lines += [
        "",
        "The registered primary contrast is structural auxiliary minus concrete on",
        f"`test_combo`: {render_contrast(primary)}. "
        + (
            "Its interval excludes zero in the predicted direction."
            if primary_positive
            else "Its interval does not exclude zero in the predicted direction."
        ),
        "",
        "Relative to the development `calib_combo` effect, the locked `test_combo`",
        f"effect changes by {render_contrast(attenuation)}. "
        + (
            "This difference is detectable at the unadjusted 95% level."
            if attenuation_detected
            else "There is no detectable attenuation or amplification at the 95% level."
        ),
        "",
        "### Objective means",
        "",
        "| Outcome | Concrete | Output invariant | Structural auxiliary |",
        "|---|---:|---:|---:|",
    ]
    for outcome in (
        "source_accuracy", "source_all_accuracy", "calib_combo_accuracy",
        "test_combo_accuracy", "held_pair_accuracy", "test_both_accuracy",
    ):
        lines.append(
            f"| `{outcome}` | {means['concrete'][outcome]:.4f} | "
            f"{means['invariant'][outcome]:.4f} | {means['latent'][outcome]:.4f} |"
        )

    lines += [
        "",
        "### Proper-score contrast",
        "",
        "| Locked outcome | Structural auxiliary − concrete Brier skill |",
        "|---|---:|",
    ]
    for split in DEPLOYMENT_SPLITS:
        lines.append(
            f"| `{split}` | "
            f"{render_contrast(contrasts['endpoint'][split]['latent_minus_concrete_brier_skill'])} |"
        )

    lines += [
        "",
        "### Performance at first persistent validation mastery",
        "",
        "| Locked outcome | Accuracy contrast | Brier-skill contrast |",
        "|---|---:|---:|",
    ]
    for split in DEPLOYMENT_SPLITS:
        item = contrasts["mastery_matched"][split]
        lines.append(
            f"| `{split}` | {render_contrast(item['latent_minus_concrete_accuracy'])} | "
            f"{render_contrast(item['latent_minus_concrete_brier_skill'])} |"
        )

    lines += [
        "",
        "### Signed-margin breakdown",
        "",
        "| Split | Near | Middle | Far |",
        "|---|---:|---:|---:|",
    ]
    for split in DEPLOYMENT_SPLITS:
        cells = [
            render_contrast(contrasts["signed_margin"][split][margin])
            for margin in MARGINS
        ]
        lines.append(f"| `{split}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Stability and scope",
        "",
    ]
    for split in DEPLOYMENT_SPLITS:
        signs = contrasts["stability"][split]
        lines.append(
            f"- `{split}`: positive in {signs['positive']}/{signs['total']} matched cells, "
            f"zero in {signs['zero']}, negative in {signs['negative']}."
        )
    lines += [
        "",
        "The negative primary cells are concentrated in particular",
        "seed-by-capacity realizations:",
        "",
        "| Seed | Difficulty | Capacity | Concrete | Structural | Difference |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in contrasts["stability"]["test_combo"]["negative_cells"]:
        lines.append(
            f"| {item['seed']} | {item['difficulty']} | {item['capacity']} | "
            f"{item['concrete']:.4f} | {item['latent']:.4f} | "
            f"{item['difference']:+.4f} |"
        )
    lines += [
        "",
        "All six seed-level primary contrasts are reported below.",
        "",
        "| Seed | Structural auxiliary − concrete `test_combo` accuracy |",
        "|---:|---:|",
    ]
    for seed, value in enumerate(primary["seed_values"]):
        lines.append(f"| {seed} | {fmt(value, signed=True)} |")
    lines += [
        "",
        f"Paired bootstrap interval: [{fmt(primary['bootstrap_low'], signed=True)}, "
        f"{fmt(primary['bootstrap_high'], signed=True)}].",
        "",
        "### Factor-conditioned primary contrasts",
        "",
        "These secondary intervals are not multiplicity-adjusted.",
    ]
    for factor in PAIR_FACTORS:
        lines += [
            "",
            f"#### {factor}",
            "",
            "| Level | Structural auxiliary − concrete `test_combo` accuracy |",
            "|---|---:|",
        ]
        for level, item in contrasts["factor_conditioned"][factor].items():
            lines.append(f"| `{level}` | {render_contrast(item)} |")

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "Task 2 confirms the registered behavioral prediction: under equivalent"
            " familiar-validation mastery, structural auxiliary supervision produces"
            " greater reuse on an independently locked rendering combination."
            if primary_positive and accuracy_equivalent and brier_equivalent
            else "Task 2 does not satisfy every condition of the registered confirmatory claim."
        ),
        "The held-quadruple and joint-shift outcomes delimit how far that result extends.",
        "The development analysis independently linked the behavioral difference to",
        "transferred probes, aligned rank geometry, and valid causal interventions.",
        "",
        "This is a qualitative replication on a more compositional but related synthetic",
        "task. Structural supervision supplies privileged information: the result shows",
        "that training objectives can select reusable internal organization after benchmark",
        "mastery, not that models spontaneously discover principles or that the effect",
        "already generalizes to natural-language reasoning or alignment.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked", type=Path, default=ROOT / "locked_results_task2")
    parser.add_argument("--development", type=Path, default=ROOT / "results_task2")
    parser.add_argument("--manifest", type=Path, default=ROOT / "task2_manifest.csv")
    parser.add_argument("--protocol", type=Path, default=ROOT / "TASK2_PROTOCOL.md")
    parser.add_argument("--cells", type=Path, default=ROOT / "task2_locked_cells.csv")
    parser.add_argument("--contrasts", type=Path, default=ROOT / "task2_locked_contrasts.json")
    parser.add_argument("--report", type=Path, default=ROOT / "TASK2_LOCKED_REPORT.md")
    args = parser.parse_args()

    paths = {
        "manifest": args.manifest,
        "protocol": args.protocol,
        "development": args.development,
        "preunlock_analysis": ROOT / "analyze_preunlock_task2.py",
        "preunlock_contrasts": ROOT / "task2_preunlock_contrasts.json",
        "preunlock_report": ROOT / "TASK2_PREUNLOCK_REPORT.md",
    }
    freeze_hashes = verify_freeze(paths)
    manifest = base.read_manifest(args.manifest)
    development_rows, development_counts = base.read_jsonl_directory(args.development)
    locked_rows, locked_counts = base.read_jsonl_directory(args.locked)
    validate_trajectories(development_counts, development_rows, "development")
    validate_trajectories(locked_counts, locked_rows, "locked")

    base.PAIR_FACTORS = PAIR_FACTORS
    development_endpoints = base.endpoints(development_rows)
    locked_endpoints = base.endpoints(locked_rows)
    rows = base.join_endpoints(manifest, development_endpoints, locked_endpoints)
    base.add_mastery_matched_metrics(rows, development_rows, locked_rows)
    if len(rows) != 162:
        raise ValueError(f"expected 162 joined endpoints, got {len(rows)}")

    integrity = {
        "manifest_runs": len(manifest),
        "development_files": len(development_counts),
        "development_rows": len(development_rows),
        "locked_files": len(locked_counts),
        "locked_rows": len(locked_rows),
        "rows_per_run": 11,
        "final_step": 10_000,
        "freeze_hashes": freeze_hashes,
        "locked_tree_sha256": tree_sha256(args.locked),
    }
    contrasts = build_contrasts(rows)
    write_csv(args.cells, rows)
    args.contrasts.write_text(json.dumps(
        {"integrity": integrity, **contrasts}, indent=2, sort_keys=True,
        allow_nan=False,
    ) + "\n")
    make_report(contrasts, integrity, args.report)
    print(json.dumps(integrity, indent=2))
    print(f"primary: {render_contrast(contrasts['primary'])}")
    print(f"wrote {args.cells}")
    print(f"wrote {args.contrasts}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
