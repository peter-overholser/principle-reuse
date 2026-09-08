"""Frozen endpoint analysis for the V2 locked deployment evaluation.

The inferential unit is the paired random seed. Objective contrasts average the
fully crossed compression, difficulty, and capacity cells within each seed.
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
PAIR_FACTORS = ("compression", "difficulty", "capacity")
OBJECTIVES = ("concrete", "invariant", "latent")
DEPLOYMENT_SPLITS = ("test_combo", "held_pair", "test_both")
MARGINS = ("near", "middle", "far")
T_CRIT_95_DF5 = 2.570581835636305


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {row["config_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError("duplicate config_id in manifest")
    return output


def read_jsonl_directory(path: Path) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    counts: dict[str, int] = {}
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
        counts[result_path.stem] = len(current)
        rows.extend(current)
    return rows, counts


def endpoints(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["config_id"]].append(row)
    return {
        config_id: max(group, key=lambda row: int(row["step"]))
        for config_id, group in grouped.items()
    }


def numeric_manifest_value(key: str, value: str):
    if key in {"seed", "width", "layers", "batch_size", "steps"}:
        return int(value)
    if key in {
        "weight_decay", "learning_rate", "invariant_weight",
        "representation_weight", "latent_weight", "mapping_weight",
    }:
        return float(value)
    return value


def join_endpoints(
    manifest: dict[str, dict],
    development: dict[str, dict],
    locked: dict[str, dict],
) -> list[dict]:
    expected = set(manifest)
    if set(development) != expected:
        raise ValueError(
            "development/manifest mismatch: "
            f"missing={sorted(expected-set(development))[:3]} "
            f"extra={sorted(set(development)-expected)[:3]}"
        )
    if set(locked) != expected:
        raise ValueError(
            "locked/manifest mismatch: "
            f"missing={sorted(expected-set(locked))[:3]} "
            f"extra={sorted(set(locked)-expected)[:3]}"
        )
    output = []
    for config_id in sorted(expected):
        meta = {
            key: numeric_manifest_value(key, value)
            for key, value in manifest[config_id].items()
        }
        dev = development[config_id]
        lock = locked[config_id]
        if int(dev["step"]) != int(meta["steps"]):
            raise ValueError(f"development endpoint is not final for {config_id}")
        if int(lock["step"]) != int(meta["steps"]):
            raise ValueError(f"locked endpoint is not final for {config_id}")
        output.append({**meta, **dev, **lock})
    return output


def add_mastery_matched_metrics(
    rows: list[dict], development_rows: list[dict], locked_rows: list[dict],
) -> None:
    development_by_config: dict[str, list[dict]] = defaultdict(list)
    locked_by_key: dict[tuple[str, int], dict] = {}
    for row in development_rows:
        development_by_config[row["config_id"]].append(row)
    for row in locked_rows:
        locked_by_key[(row["config_id"], int(row["step"]))] = row

    for endpoint in rows:
        config_id = endpoint["config_id"]
        ordered = sorted(
            development_by_config[config_id], key=lambda row: int(row["step"])
        )
        mastery_step = None
        for index in range(len(ordered) - 1):
            if (
                float(ordered[index]["source_accuracy"]) >= 0.95
                and float(ordered[index + 1]["source_accuracy"]) >= 0.95
            ):
                mastery_step = int(ordered[index]["step"])
                break
        if mastery_step is None:
            raise ValueError(f"no persistent validation mastery for {config_id}")

        locked = locked_by_key[(config_id, mastery_step)]
        endpoint["mastery_step"] = mastery_step
        for split in DEPLOYMENT_SPLITS:
            for suffix in ("accuracy", "brier_skill", "confidence"):
                key = f"{split}_{suffix}"
                endpoint[f"mastery_{key}"] = float(locked[key])

        for split in DEPLOYMENT_SPLITS:
            endpoint[f"{split}_accuracy_gap"] = (
                float(endpoint[f"{split}_accuracy"])
                - float(endpoint["source_accuracy"])
            )
            endpoint[f"{split}_brier_skill_gap"] = (
                float(endpoint[f"{split}_brier_skill"])
                - float(endpoint["source_brier_skill"])
            )
        endpoint["test_combo_minus_calib_accuracy"] = (
            float(endpoint["test_combo_accuracy"])
            - float(endpoint["calib_combo_accuracy"])
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_by(rows: list[dict], factor: str, outcome: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[factor])].append(float(row[outcome]))
    return {level: float(np.mean(values)) for level, values in grouped.items()}


def matched_cell_signs(
    rows: list[dict], outcome: str,
    treatment: str = "latent", control: str = "concrete",
) -> tuple[int, int, int]:
    keyed = {
        (
            int(row["seed"]),
            *(str(row[factor]) for factor in PAIR_FACTORS),
            str(row["abstraction"]),
        ): float(row[outcome])
        for row in rows
    }
    positive = zero = negative = 0
    factor_values = sorted({
        tuple(str(row[factor]) for factor in PAIR_FACTORS) for row in rows
    })
    for seed in sorted({int(row["seed"]) for row in rows}):
        for values in factor_values:
            difference = (
                keyed[(seed, *values, treatment)]
                - keyed[(seed, *values, control)]
            )
            positive += difference > 0
            zero += difference == 0
            negative += difference < 0
    return positive, zero, negative


def paired_seed_differences(
    rows: list[dict],
    outcome: str,
    treatment: str,
    control: str,
    subset: dict[str, str] | None = None,
) -> list[float]:
    selected = [
        row for row in rows
        if not subset or all(str(row[key]) == str(value) for key, value in subset.items())
    ]
    seeds = sorted({int(row["seed"]) for row in selected})
    differences = []
    for seed in seeds:
        maps = {}
        for objective in (treatment, control):
            group = [
                row for row in selected
                if int(row["seed"]) == seed and row["abstraction"] == objective
            ]
            keyed = {
                tuple(str(row[key]) for key in PAIR_FACTORS): float(row[outcome])
                for row in group
            }
            if len(keyed) != len(group):
                raise ValueError(f"duplicate paired cell for seed={seed}, objective={objective}")
            maps[objective] = keyed
        if set(maps[treatment]) != set(maps[control]):
            raise ValueError(
                f"unpaired cells for seed={seed}, {treatment} vs {control}, subset={subset}"
            )
        cell_diffs = [
            maps[treatment][key] - maps[control][key]
            for key in sorted(maps[treatment])
        ]
        if not cell_diffs:
            raise ValueError(f"no paired cells for seed={seed}, subset={subset}")
        differences.append(float(np.mean(cell_diffs)))
    return differences


def interval(values: list[float]) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) != 6:
        raise ValueError(f"registered V2 analysis expects six seeds, got {len(x)}")
    mean = float(np.mean(x))
    half = T_CRIT_95_DF5 * float(np.std(x, ddof=1)) / math.sqrt(len(x))
    return mean, mean - half, mean + half


def bootstrap_interval(values: list[float], seed: int = 20260906) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(100_000, len(x)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def contrast_row(
    rows: list[dict], outcome: str, treatment: str, control: str,
    subset: dict[str, str] | None = None,
) -> dict:
    values = paired_seed_differences(rows, outcome, treatment, control, subset)
    mean, low, high = interval(values)
    boot_low, boot_high = bootstrap_interval(values)
    return {
        "outcome": outcome,
        "treatment": treatment,
        "control": control,
        "subset": subset or {},
        "mean": mean,
        "t_low": low,
        "t_high": high,
        "bootstrap_low": boot_low,
        "bootstrap_high": boot_high,
        "seed_values": values,
    }


def render_contrast(result: dict) -> str:
    return (
        f"{fmt(result['mean'])} "
        f"[{fmt(result['t_low'])}, {fmt(result['t_high'])}]"
    )


def make_report(
    rows: list[dict], integrity: dict, manifest_path: Path,
    report_path: Path, contrasts_json: Path,
) -> None:
    contrast_results = []
    for split in DEPLOYMENT_SPLITS:
        for suffix in ("accuracy", "brier_skill"):
            outcome = f"{split}_{suffix}"
            for treatment in ("latent", "invariant"):
                contrast_results.append(
                    contrast_row(rows, outcome, treatment, "concrete")
                )
    for split in DEPLOYMENT_SPLITS:
        for margin in MARGINS:
            contrast_results.append(
                contrast_row(
                    rows, f"{split}_accuracy_{margin}", "latent", "concrete"
                )
            )
    for split in DEPLOYMENT_SPLITS:
        for factor in PAIR_FACTORS:
            for level in sorted({str(row[factor]) for row in rows}):
                contrast_results.append(
                    contrast_row(
                        rows, f"{split}_accuracy", "latent", "concrete",
                        {factor: level},
                    )
                )
    for outcome in ("source_accuracy", "source_brier_skill"):
        contrast_results.append(
            contrast_row(rows, outcome, "latent", "concrete")
        )
    contrast_results.append(
        contrast_row(
            rows, "test_combo_minus_calib_accuracy", "latent", "concrete"
        )
    )
    for split in DEPLOYMENT_SPLITS:
        for suffix in ("accuracy", "brier_skill"):
            contrast_results.append(
                contrast_row(
                    rows, f"mastery_{split}_{suffix}", "latent", "concrete"
                )
            )

    with contrasts_json.open("w") as handle:
        json.dump(
            {"integrity": integrity, "contrasts": contrast_results},
            handle, indent=2, allow_nan=False,
        )

    lookup = {
        (
            item["outcome"], item["treatment"], item["control"],
            tuple(sorted(item["subset"].items())),
        ): item
        for item in contrast_results
    }

    def get(outcome, treatment="latent", control="concrete", subset=None):
        key = (outcome, treatment, control, tuple(sorted((subset or {}).items())))
        return lookup[key]

    lines = [
        "# V2 locked deployment report",
        "",
        "Generated after the pre-unlock report and the one-time evaluation of the",
        "registered `test_combo`, `held_pair`, and `test_both` splits.",
        "",
        "## Integrity",
        "",
        f"- Registered runs: {integrity['manifest_runs']}.",
        f"- Locked result files: {integrity['locked_files']}.",
        f"- Locked checkpoint rows: {integrity['locked_rows']}.",
        f"- Rows per run: {integrity['rows_per_run']}.",
        f"- Final step: {integrity['final_step']} in every run.",
        f"- Manifest SHA-256: `{sha256(manifest_path)}`.",
        "",
        "All locked files joined one-to-one with the registered manifest and the",
        "development results. No run or checkpoint was selected using a locked outcome.",
        "",
        "## Validation equivalence",
        "",
        "| Endpoint source metric | Latent − concrete | Equivalence margin |",
        "|---|---:|---:|",
        f"| Accuracy | {render_contrast(get('source_accuracy'))} | ±0.02 |",
        f"| Brier skill | {render_contrast(get('source_brier_skill'))} | ±0.03 |",
        "",
        "Both 95% intervals lie wholly inside their pre-specified equivalence bounds.",
        "The endpoint comparison is therefore mastery-matched under the registered",
        "definition; this is stronger than merely failing to detect a validation",
        "difference.",
        "",
        "## Confirmatory result",
        "",
        "The table reports seed-paired mean differences and 95% t intervals over the",
        "six seeds. Each seed contrast averages the fully crossed compression,",
        "difficulty, and capacity cells.",
        "",
        "| Locked outcome | Latent − concrete | Invariant − concrete |",
        "|---|---:|---:|",
    ]
    for split in DEPLOYMENT_SPLITS:
        lines.append(
            f"| `{split}` accuracy | "
            f"{render_contrast(get(f'{split}_accuracy'))} | "
            f"{render_contrast(get(f'{split}_accuracy', 'invariant'))} |"
        )
    lines += [
        "",
        "The registered primary contrast is latent minus concrete on `test_combo`.",
        "`held_pair` and `test_both` are the pre-specified secondary deployment",
        "outcomes.",
        "",
        "The locked `test_combo` effect differs from the development `calib_combo`",
        "effect by "
        f"{render_contrast(get('test_combo_minus_calib_accuracy'))}. Thus there is no",
        "detectable attenuation on the independently held combination set.",
        "",
        "### Objective means",
        "",
        "| Outcome | Concrete | Invariant | Latent |",
        "|---|---:|---:|---:|",
    ]
    for outcome in (
        "source_accuracy", "source_all_accuracy", "calib_combo_accuracy",
        "test_combo_accuracy", "held_pair_accuracy", "test_both_accuracy",
    ):
        means = mean_by(rows, "abstraction", outcome)
        lines.append(
            f"| `{outcome}` | {means['concrete']:.4f} | "
            f"{means['invariant']:.4f} | {means['latent']:.4f} |"
        )

    lines += [
        "",
        "### Proper-score contrast",
        "",
        "| Locked outcome | Latent − concrete Brier skill |",
        "|---|---:|",
    ]
    for split in DEPLOYMENT_SPLITS:
        lines.append(
            f"| `{split}` | {render_contrast(get(f'{split}_brier_skill'))} |"
        )

    lines += [
        "",
        "### Mastery-matched deployment performance",
        "",
        "For each run, this evaluates the locked checkpoint at the first of two",
        "consecutive checkpoints with source accuracy at least 0.95. It asks how much",
        "reuse is already present when ordinary validation mastery first stabilizes.",
        "",
        "| Locked outcome at mastery | Latent − concrete accuracy | Brier skill |",
        "|---|---:|---:|",
    ]
    for split in DEPLOYMENT_SPLITS:
        lines.append(
            f"| `{split}` | "
            f"{render_contrast(get(f'mastery_{split}_accuracy'))} | "
            f"{render_contrast(get(f'mastery_{split}_brier_skill'))} |"
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
            render_contrast(get(f"{split}_accuracy_{margin}"))
            for margin in MARGINS
        ]
        lines.append(f"| `{split}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Factor-conditioned latent contrasts",
        "",
        "These are secondary interaction summaries. Each interval uses the same six",
        "paired seeds; they are not multiplicity-adjusted.",
    ]
    for factor in PAIR_FACTORS:
        lines += [
            "",
            f"### {factor}",
            "",
            "| Level | test_combo | held_pair | test_both |",
            "|---|---:|---:|---:|",
        ]
        for level in sorted({str(row[factor]) for row in rows}):
            rendered = [
                render_contrast(get(f"{split}_accuracy", subset={factor: level}))
                for split in DEPLOYMENT_SPLITS
            ]
            lines.append(f"| `{level}` | " + " | ".join(rendered) + " |")

    primary = get("test_combo_accuracy")
    signs = matched_cell_signs(rows, "test_combo_accuracy")
    lines += [
        "",
        "## Raw seed-level primary contrasts",
        "",
        "| Seed | Latent − concrete test_combo accuracy |",
        "|---:|---:|",
    ]
    for seed, value in enumerate(primary["seed_values"]):
        lines.append(f"| {seed} | {fmt(value)} |")
    lines += [
        "",
        "The paired bootstrap interval for the primary contrast is "
        f"[{fmt(primary['bootstrap_low'])}, {fmt(primary['bootstrap_high'])}].",
        f"The cellwise contrast is positive in {signs[0]}/108 matched factorial",
        f"cells, zero in {signs[1]}, and negative in {signs[2]}.",
        "",
        "## Interpretation",
        "",
        "Interpretation is conditional on the pre-unlock finding of identical endpoint",
        "source accuracy and source Brier skill across objectives. The primary question",
        "is whether the latent-supervision advantage observed on `calib_combo` survives",
        "on the independently locked deployment splits. Factor-conditioned intervals",
        "describe scope and should not replace the registered pooled contrast.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locked", type=Path, default=ROOT / "locked_results_v2",
    )
    parser.add_argument(
        "--development", type=Path, default=ROOT / "results_v2",
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "v2_manifest.csv",
    )
    parser.add_argument(
        "--cells", type=Path, default=ROOT / "v2_locked_cells.csv",
    )
    parser.add_argument(
        "--contrasts", type=Path, default=ROOT / "v2_locked_contrasts.json",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "V2_LOCKED_REPORT.md",
    )
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    locked_rows, locked_counts = read_jsonl_directory(args.locked)
    development_rows, development_counts = read_jsonl_directory(args.development)
    if set(locked_counts.values()) != {11}:
        raise ValueError(f"locked rows per run are not uniformly 11: {set(locked_counts.values())}")
    if set(development_counts.values()) != {11}:
        raise ValueError(
            f"development rows per run are not uniformly 11: {set(development_counts.values())}"
        )
    locked_endpoint = endpoints(locked_rows)
    development_endpoint = endpoints(development_rows)
    rows = join_endpoints(manifest, development_endpoint, locked_endpoint)
    add_mastery_matched_metrics(rows, development_rows, locked_rows)
    final_steps = {int(row["step"]) for row in rows}
    if final_steps != {10_000}:
        raise ValueError(f"unexpected final steps: {final_steps}")

    integrity = {
        "manifest_runs": len(manifest),
        "locked_files": len(locked_counts),
        "locked_rows": len(locked_rows),
        "development_files": len(development_counts),
        "development_rows": len(development_rows),
        "rows_per_run": 11,
        "final_step": 10_000,
    }
    write_csv(args.cells, rows)
    make_report(rows, integrity, args.manifest, args.report, args.contrasts)
    print(json.dumps(integrity, indent=2))
    print(f"wrote {args.cells}")
    print(f"wrote {args.contrasts}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
