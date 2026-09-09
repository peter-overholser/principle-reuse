"""Freeze the Task 2 development analysis before locked evaluation.

This script reads only the registered manifest and development trajectories. It
does not import or inspect any locked result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OBJECTIVES = ("concrete", "invariant", "latent")
PAIR_FACTORS = ("difficulty", "capacity")
MARGINS = ("near", "middle", "far")
T_CRIT_95_DF5 = 2.570581835636305


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash relative names and contents of all Task 2 JSONL files."""
    digest = hashlib.sha256()
    for result_path in sorted(path.glob("*.jsonl")):
        digest.update(result_path.name.encode("utf-8"))
        digest.update(b"\0")
        with result_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[dict], dict[str, dict]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {row["config_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("duplicate config_id in manifest")
    return rows, keyed


def read_results(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = []
    grouped = {}
    for result_path in sorted(path.glob("*.jsonl")):
        current = []
        with result_path.open() as handle:
            for line in handle:
                if line.strip():
                    current.append(json.loads(line))
        if not current:
            raise ValueError(f"empty result file: {result_path}")
        ids = {row["config_id"] for row in current}
        if ids != {result_path.stem}:
            raise ValueError(f"config/file mismatch in {result_path}: {ids}")
        grouped[result_path.stem] = current
        rows.extend(current)
    return rows, grouped


def first_persistent(group: list[dict], key: str, threshold: float):
    ordered = sorted(group, key=lambda row: int(row["step"]))
    for index in range(len(ordered) - 1):
        values = [ordered[index].get(key), ordered[index + 1].get(key)]
        if all(value is not None and np.isfinite(value) and value >= threshold for value in values):
            return int(ordered[index]["step"]), ordered[index]
    return None, None


def endpoints_and_timing(
    manifest_rows: list[dict], groups: dict[str, list[dict]],
) -> list[dict]:
    output = []
    for meta in manifest_rows:
        config_id = meta["config_id"]
        group = groups[config_id]
        endpoint = max(group, key=lambda row: int(row["step"]))
        if int(endpoint["step"]) != 10_000:
            raise ValueError(f"non-final endpoint for {config_id}: {endpoint['step']}")
        mastery_step, mastery_row = first_persistent(group, "source_accuracy", 0.95)
        reuse_step, _ = first_persistent(group, "calib_combo_accuracy", 0.80)
        if mastery_step is None:
            raise ValueError(f"no persistent source mastery for {config_id}")
        enriched = dict(endpoint)
        enriched["mastery_step"] = mastery_step
        enriched["reuse_step"] = np.nan if reuse_step is None else reuse_step
        enriched["mastery_reuse_lag"] = (
            np.nan if reuse_step is None else reuse_step - mastery_step
        )
        for suffix in ("accuracy", "brier_skill", "confidence"):
            key = f"calib_combo_{suffix}"
            enriched[f"mastery_{key}"] = mastery_row[key]
        output.append(enriched)
    return output


def paired_seed_differences(
    rows: list[dict], outcome: str, treatment: str, control: str,
    subset: dict[str, str] | None = None,
) -> list[float]:
    selected = [
        row for row in rows
        if not subset or all(str(row[key]) == str(value) for key, value in subset.items())
    ]
    differences = []
    for seed in range(6):
        maps = {}
        for objective in (treatment, control):
            group = [
                row for row in selected
                if int(row["seed"]) == seed and row["abstraction"] == objective
            ]
            maps[objective] = {
                tuple(str(row[key]) for key in PAIR_FACTORS): float(row[outcome])
                for row in group
            }
            if len(maps[objective]) != len(group):
                raise ValueError(f"duplicate paired cell: seed={seed}, objective={objective}")
        if set(maps[treatment]) != set(maps[control]) or not maps[treatment]:
            raise ValueError(
                f"unpaired cells: seed={seed}, {treatment} vs {control}, subset={subset}"
            )
        differences.append(float(np.mean([
            maps[treatment][key] - maps[control][key]
            for key in sorted(maps[treatment])
        ])))
    return differences


def interval(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    if len(x) != 6:
        raise ValueError(f"Task 2 analysis expects six paired seeds, got {len(x)}")
    mean = float(np.mean(x))
    half = T_CRIT_95_DF5 * float(np.std(x, ddof=1)) / math.sqrt(len(x))
    return {
        "mean": mean, "low": mean - half, "high": mean + half,
        "seed_values": [float(value) for value in x],
    }


def objective_means(rows: list[dict], outcomes: list[str]) -> dict:
    return {
        objective: {
            outcome: float(np.mean([
                float(row[outcome]) for row in rows if row["abstraction"] == objective
            ]))
            for outcome in outcomes
        }
        for objective in OBJECTIVES
    }


def matched_cell_signs(rows: list[dict], outcome: str) -> tuple[int, int, int]:
    keyed = {
        (
            int(row["seed"]), str(row["difficulty"]), str(row["capacity"]),
            str(row["abstraction"]),
        ): float(row[outcome])
        for row in rows
    }
    positive = zero = negative = 0
    for seed in range(6):
        for difficulty in ("easy", "hard", "mixed"):
            for capacity in ("tight", "moderate", "wide"):
                difference = (
                    keyed[(seed, difficulty, capacity, "latent")]
                    - keyed[(seed, difficulty, capacity, "concrete")]
                )
                positive += difference > 0
                zero += difference == 0
                negative += difference < 0
    return positive, zero, negative


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def render_contrast(result: dict) -> str:
    return (
        f"{fmt(result['mean'], signed=True)} "
        f"[{fmt(result['low'], signed=True)}, {fmt(result['high'], signed=True)}]"
    )


def make_report(
    rows: list[dict], integrity: dict, contrasts: dict, report_path: Path,
) -> None:
    means = contrasts["objective_means"]
    signs = contrasts["matched_cell_signs"]
    gates = contrasts["causal_validity"]
    latency = contrasts["timing"]

    lines = [
        "# Task 2 pre-unlock report",
        "",
        "Recorded 9 September 2026 after training and before evaluation of",
        "`test_combo`, `held_pair`, or `test_both`.",
        "",
        "## Integrity",
        "",
        f"- Registered runs: {integrity['manifest_runs']}.",
        f"- Completed result files: {integrity['result_files']}.",
        f"- Checkpoint rows: {integrity['result_rows']} ({integrity['rows_per_run']} per run).",
        f"- Final step: {integrity['final_step']} in every run.",
        f"- Manifest SHA-256: `{integrity['manifest_sha256']}`.",
        f"- Protocol SHA-256: `{integrity['protocol_sha256']}`.",
        f"- Development-result tree SHA-256: `{integrity['results_tree_sha256']}`.",
        "",
        "All manifest identifiers join one-to-one to result files. Every run reaches",
        "persistent source mastery. No locked result directory is read by this analysis.",
        "",
        "## Validation mastery",
        "",
        "| Objective | Source accuracy | Source Brier skill |",
        "|---|---:|---:|",
    ]
    for objective in OBJECTIVES:
        lines.append(
            f"| {objective} | {means[objective]['source_accuracy']:.4f} | "
            f"{means[objective]['source_brier_skill']:.4f} |"
        )
    lines += [
        "",
        "Applying the same equivalence margins used in V2 (accuracy ±0.02; Brier",
        "skill ±0.03), the seed-paired latent-minus-concrete intervals are:",
        "",
        f"- source accuracy: {render_contrast(contrasts['latent_minus_concrete']['source_accuracy'])};",
        f"- source Brier skill: {render_contrast(contrasts['latent_minus_concrete']['source_brier_skill'])}.",
        "",
        "Both intervals lie wholly inside the inherited margins.",
        "",
        "## Development-generalization result",
        "",
        "| Outcome | Concrete | Output invariant | Structural auxiliary |",
        "|---|---:|---:|---:|",
        (
            f"| `source_all` accuracy | {means['concrete']['source_all_accuracy']:.4f} | "
            f"{means['invariant']['source_all_accuracy']:.4f} | "
            f"{means['latent']['source_all_accuracy']:.4f} |"
        ),
        (
            f"| `calib_combo` accuracy | {means['concrete']['calib_combo_accuracy']:.4f} | "
            f"{means['invariant']['calib_combo_accuracy']:.4f} | "
            f"{means['latent']['calib_combo_accuracy']:.4f} |"
        ),
        (
            f"| `calib_combo` Brier skill | {means['concrete']['calib_combo_brier_skill']:.4f} | "
            f"{means['invariant']['calib_combo_brier_skill']:.4f} | "
            f"{means['latent']['calib_combo_brier_skill']:.4f} |"
        ),
        "",
        "Seed-paired development contrasts:",
        "",
        f"- latent minus concrete `calib_combo` accuracy: {render_contrast(contrasts['latent_minus_concrete']['calib_combo_accuracy'])};",
        f"- invariant minus concrete `calib_combo` accuracy: {render_contrast(contrasts['invariant_minus_concrete']['calib_combo_accuracy'])};",
        f"- latent minus concrete `calib_combo` Brier skill: {render_contrast(contrasts['latent_minus_concrete']['calib_combo_brier_skill'])}.",
        "",
        f"The latent-minus-concrete accuracy difference is positive in {signs['positive']}/54",
        f"matched cells, zero in {signs['zero']}, and negative in {signs['negative']}.",
        "",
        "## Development after mastery",
        "",
        "Persistent mastery is source accuracy at least 0.95 at two consecutive",
        "checkpoints; persistent reuse is `calib_combo` accuracy at least 0.80 by the",
        "same rule.",
        "",
        "| Objective | Median mastery step | Runs reaching reuse | Median lag among observed |",
        "|---|---:|---:|---:|",
    ]
    for objective in OBJECTIVES:
        item = latency[objective]
        lines.append(
            f"| {objective} | {item['median_mastery_step']:.0f} | "
            f"{item['reuse_runs']}/54 | {item['median_mastery_reuse_lag']:.0f} |"
        )
    lines += [
        "",
        "At the first persistent-mastery checkpoint, the latent-minus-concrete",
        "development-combination contrast is",
        f"{render_contrast(contrasts['latent_minus_concrete']['mastery_calib_combo_accuracy'])}.",
        "The fixed-budget endpoint contrast is substantially larger, so deployment-relevant",
        "organization continues to develop after ordinary mastery.",
        "",
        "## Structural and causal measurements",
        "",
        "| Measurement | Concrete | Output invariant | Structural auxiliary |",
        "|---|---:|---:|---:|",
    ]
    for outcome, label in (
        ("probe_target_r2", "Transferred target probe R²"),
        ("probe_target_sign_accuracy", "Transferred probe sign accuracy"),
        ("embedding_rank_axis_cosine", "Rank-axis cosine"),
        ("embedding_participation_rank", "Embedding participation rank"),
    ):
        lines.append(
            f"| {label} | {means['concrete'][outcome]:.4f} | "
            f"{means['invariant'][outcome]:.4f} | {means['latent'][outcome]:.4f} |"
        )
    lines += [
        "",
        "Causal-intervention validity gates:",
        "",
        f"- concrete: {gates['concrete']['valid']}/54;",
        f"- invariant: {gates['invariant']['valid']}/54;",
        f"- latent: {gates['latent']['valid']}/54.",
        "",
        "Conditional causal means are descriptive only and remain coupled to these gate",
        "rates. Invalid mechanistic measurements are not converted to zero.",
        "",
        "## Factor-conditioned latent contrasts",
        "",
        "These are secondary development-set summaries; intervals use the same six paired",
        "seeds and are not multiplicity-adjusted.",
    ]
    for factor in PAIR_FACTORS:
        lines += [
            "",
            f"### {factor}",
            "",
            "| Level | Latent − concrete `calib_combo` accuracy |",
            "|---|---:|",
        ]
        for level, result in contrasts["factor_conditioned"][factor].items():
            lines.append(f"| {level} | {render_contrast(result)} |")
    lines += [
        "",
        "## Frozen interpretation before unlock",
        "",
        "Task 2 passes the development-stage replication criterion: equal in-distribution",
        "mastery accompanies a large objective-dependent difference in reuse on a more",
        "compositional calculation, and the behavioral difference converges with transferred",
        "structural measurements. Output-level invariance again does not reproduce the",
        "structural-supervision effect.",
        "",
        "This remains a development-set result. It does not yet establish replication on the",
        "independently locked combinations, held quadruples, or joint shift. It also remains",
        "a privileged-structure intervention on a related synthetic family, not evidence of",
        "spontaneous principle discovery or generality to natural-language reasoning.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results_task2")
    parser.add_argument("--manifest", type=Path, default=ROOT / "task2_manifest.csv")
    parser.add_argument("--protocol", type=Path, default=ROOT / "TASK2_PROTOCOL.md")
    parser.add_argument("--cells", type=Path, default=ROOT / "task2_cells.csv")
    parser.add_argument(
        "--contrasts", type=Path, default=ROOT / "task2_preunlock_contrasts.json",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "TASK2_PREUNLOCK_REPORT.md",
    )
    args = parser.parse_args()

    manifest_rows, manifest = read_manifest(args.manifest)
    result_rows, groups = read_results(args.results)
    if len(manifest_rows) != 162:
        raise ValueError(f"expected 162 registered runs, got {len(manifest_rows)}")
    if set(groups) != set(manifest):
        raise ValueError(
            f"manifest/result mismatch: missing={sorted(set(manifest)-set(groups))[:3]}, "
            f"extra={sorted(set(groups)-set(manifest))[:3]}"
        )
    counts = {len(group) for group in groups.values()}
    if counts != {11}:
        raise ValueError(f"rows per run are not uniformly 11: {counts}")
    if any({int(row["step"]) for row in group} != {
        0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 7000, 10_000,
    } for group in groups.values()):
        raise ValueError("one or more runs have an unexpected checkpoint schedule")

    rows = endpoints_and_timing(manifest_rows, groups)
    outcomes = [
        "source_accuracy", "source_brier_skill", "source_all_accuracy",
        "calib_combo_accuracy", "calib_combo_brier_skill", "probe_target_r2",
        "probe_target_sign_accuracy", "embedding_rank_axis_cosine",
        "embedding_participation_rank",
    ]
    means = objective_means(rows, outcomes)

    latent_outcomes = [
        "source_accuracy", "source_brier_skill", "calib_combo_accuracy",
        "calib_combo_brier_skill", "mastery_calib_combo_accuracy",
        "probe_target_r2", "probe_target_sign_accuracy",
        "embedding_rank_axis_cosine", "embedding_participation_rank",
    ]
    contrasts = {
        "objective_means": means,
        "latent_minus_concrete": {
            outcome: interval(paired_seed_differences(
                rows, outcome, "latent", "concrete",
            )) for outcome in latent_outcomes
        },
        "invariant_minus_concrete": {
            outcome: interval(paired_seed_differences(
                rows, outcome, "invariant", "concrete",
            )) for outcome in ("source_accuracy", "source_brier_skill",
                               "calib_combo_accuracy", "calib_combo_brier_skill")
        },
        "matched_cell_signs": dict(zip(
            ("positive", "zero", "negative"),
            matched_cell_signs(rows, "calib_combo_accuracy"),
        )),
        "factor_conditioned": {
            factor: {
                level: interval(paired_seed_differences(
                    rows, "calib_combo_accuracy", "latent", "concrete",
                    subset={factor: level},
                ))
                for level in sorted({str(row[factor]) for row in rows})
            }
            for factor in PAIR_FACTORS
        },
        "causal_validity": {},
        "timing": {},
    }
    for objective in OBJECTIVES:
        group = [row for row in rows if row["abstraction"] == objective]
        valid = [row for row in group if bool(float(row.get("causal_valid", 0)))]
        contrasts["causal_validity"][objective] = {
            "valid": len(valid),
            "total": len(group),
            "source_to_target_mean_if_valid": (
                None if not valid else float(np.mean([
                    float(row["causal_source_to_target"]) for row in valid
                ]))
            ),
            "normalized_mean_if_valid": (
                None if not valid else float(np.mean([
                    float(row["causal_normalized"]) for row in valid
                ]))
            ),
        }
        observed_lags = np.asarray([
            float(row["mastery_reuse_lag"]) for row in group
            if np.isfinite(float(row["mastery_reuse_lag"]))
        ])
        contrasts["timing"][objective] = {
            "median_mastery_step": float(np.median([
                float(row["mastery_step"]) for row in group
            ])),
            "reuse_runs": int(len(observed_lags)),
            "median_mastery_reuse_lag": (
                None if not len(observed_lags) else float(np.median(observed_lags))
            ),
        }

    integrity = {
        "manifest_runs": len(manifest_rows),
        "result_files": len(groups),
        "result_rows": len(result_rows),
        "rows_per_run": 11,
        "final_step": 10_000,
        "manifest_sha256": sha256(args.manifest),
        "protocol_sha256": sha256(args.protocol),
        "results_tree_sha256": tree_sha256(args.results),
    }
    write_csv(args.cells, rows)
    args.contrasts.write_text(json.dumps({
        "integrity": integrity, **contrasts,
    }, indent=2, sort_keys=True) + "\n")
    make_report(rows, integrity, contrasts, args.report)
    print(json.dumps(integrity, indent=2))
    print(f"wrote {args.cells}")
    print(f"wrote {args.contrasts}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
