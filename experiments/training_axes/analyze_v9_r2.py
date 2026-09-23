"""Registered gate for the fresh-seed V9-R2 causal reliability screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from experiments.training_axes.analyze_v9_common import (
    fmt, load_results, sha256, summarize,
)
from experiments.training_axes.design_v9_r2 import R2_ARMS, R2_SEEDS


CHECKPOINTS = (
    0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 7000,
    10000, 12000, 15000, 20000,
)
BASELINE = "a00_confounded"
IMPOSED = "imposed"
PRIMARY = "a50_diverse"


def verify_repair_stop(path, base_protocol, repair_protocol, repair_manifest):
    value = json.loads(Path(path).read_text())
    if value["decision"]["route"] != "stop-invalid-assay":
        raise ValueError("V9-R2 requires the registered invalid-assay stop")
    checks = {
        "protocol_sha256": base_protocol,
        "repair_protocol_sha256": repair_protocol,
        "manifest_sha256": repair_manifest,
    }
    for key, target in checks.items():
        if value["integrity"].get(key) != sha256(target):
            raise ValueError(f"V9 repair provenance mismatch: {key}")
    return value


def config_map(configs):
    result = {}
    for config in configs:
        key = config.arm, config.seed
        if key in result:
            raise ValueError(f"duplicate V9-R2 config: {key}")
        result[key] = config.config_id
    expected = {(arm, seed) for arm in R2_ARMS for seed in R2_SEEDS}
    if set(result) != expected:
        raise ValueError("V9-R2 manifest is not the registered 24-run design")
    return result


def paired_training(by_key, ids, left, right, metric):
    return summarize([
        float(by_key[(ids[(left, seed)], 20_000)][metric])
        - float(by_key[(ids[(right, seed)], 20_000)][metric])
        for seed in R2_SEEDS
    ])


def load_reliability(directory, ids):
    directory = Path(directory)
    expected_ids = set(ids.values())
    paths = sorted(directory.glob("*.jsonl"))
    if {path.stem for path in paths} != expected_ids:
        raise ValueError("V9-R2 reliability files do not match the manifest")
    by_model = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if [int(row["replicate"]) for row in rows] != [0, 1, 2, 3]:
            raise ValueError(f"wrong V9-R2 reliability replicates: {path}")
        if any(int(row["checkpoint_step"]) != 20_000 for row in rows):
            raise ValueError(f"wrong V9-R2 reliability checkpoint: {path}")
        if any(int(row["eval_size"]) != 4096 for row in rows):
            raise ValueError(f"wrong V9-R2 eval size: {path}")
        if any(int(row["n_pairs"]) != 1024 for row in rows):
            raise ValueError(f"wrong V9-R2 intervention count: {path}")
        if any(int(row["n_random"]) != 16 for row in rows):
            raise ValueError(f"wrong V9-R2 random-control count: {path}")
        by_model[path.stem] = rows
    return by_model, paths


def reliability_summary(ids, by_model):
    arms = {}
    seed_details = {}
    for arm in R2_ARMS:
        arm_rows = []
        seed_details[arm] = {}
        for seed in R2_SEEDS:
            rows = by_model[ids[(arm, seed)]]
            valid_replicates = sum(float(row["causal_valid"]) == 1.0 for row in rows)
            target = mean(float(row["causal_target_to_target"]) for row in rows)
            source = mean(float(row["causal_source_to_target"]) for row in rows)
            random = mean(float(row["causal_random"]) for row in rows)
            detail = {
                "valid_replicates": valid_replicates,
                "reliable": valid_replicates >= 3,
                "target": target,
                "source": source,
                "random": random,
                "target_minus_random": target - random,
                "source_minus_random": source - random,
            }
            seed_details[arm][str(seed)] = detail
            arm_rows.append(detail)
        arms[arm] = {
            "reliable_seeds": sum(row["reliable"] for row in arm_rows),
            "target": summarize([row["target"] for row in arm_rows]),
            "random": summarize([row["random"] for row in arm_rows]),
            "target_minus_random": summarize([
                row["target_minus_random"] for row in arm_rows
            ]),
            "source_minus_random": summarize([
                row["source_minus_random"] for row in arm_rows
            ]),
        }
    return arms, seed_details


def analyze(
    results, reliability_dir, manifest, base_protocol, repair_protocol,
    r2_protocol, repair_gate,
):
    verify_repair_stop(
        repair_gate, base_protocol, repair_protocol,
        Path("experiments/training_axes/v9_repair_manifest.csv"),
    )
    configs, by_key, result_paths = load_results(results, manifest, CHECKPOINTS)
    ids = config_map(configs)
    by_model, reliability_paths = load_reliability(reliability_dir, ids)

    source_failures = []
    for config in configs:
        for step in (15_000, 20_000):
            if by_key[(config.config_id, step)]["source_accuracy"] < 0.95:
                source_failures.append((config.config_id, step, "accuracy"))
        if by_key[(config.config_id, 20_000)]["source_brier_skill"] < 0.80:
            source_failures.append((config.config_id, 20_000, "brier"))

    behavior = {
        "imposed_minus_baseline_accuracy": paired_training(
            by_key, ids, IMPOSED, BASELINE, "dev_s_nonanchor_accuracy"
        ),
        "primary_minus_baseline_accuracy": paired_training(
            by_key, ids, PRIMARY, BASELINE, "dev_s_nonanchor_accuracy"
        ),
        "primary_minus_baseline_O": paired_training(
            by_key, ids, PRIMARY, BASELINE, "organization_dev_s"
        ),
    }
    reliability, seed_details = reliability_summary(ids, by_model)

    source_pass = not source_failures
    imposed_behavior = behavior["imposed_minus_baseline_accuracy"]
    primary_behavior = behavior["primary_minus_baseline_accuracy"]
    organization = behavior["primary_minus_baseline_O"]
    behavior_pass = (
        imposed_behavior["mean"] >= 0.25
        and imposed_behavior["ci_low"] > 0
        and imposed_behavior["positive_seeds"] == 8
        and primary_behavior["mean"] >= 0.10
        and primary_behavior["ci_low"] > 0
        and primary_behavior["positive_seeds"] >= 7
        and organization["ci_low"] > 0
    )
    counts_pass = (
        reliability[BASELINE]["reliable_seeds"] <= 2
        and reliability[IMPOSED]["reliable_seeds"] >= 6
        and reliability[PRIMARY]["reliable_seeds"] >= 7
    )
    effect_pass = True
    for arm in (IMPOSED, PRIMARY):
        item = reliability[arm]
        effect_pass = effect_pass and (
            item["target"]["mean"] >= 0.75
            and item["target_minus_random"]["mean"] >= 0.10
            and item["target_minus_random"]["ci_low"] > 0
            and item["source_minus_random"]["mean"] >= 0.10
            and item["source_minus_random"]["ci_low"] > 0
        )
    reliability_pass = counts_pass and effect_pass
    if not source_pass:
        route = "stop-source-mastery"
    elif not behavior_pass:
        route = "stop-behavioral-replication"
    elif not reliability_pass:
        route = "stop-causal-assay-unreliable"
    else:
        route = "run-main"

    root = Path(__file__).resolve().parent
    integrity_files = {
        "base_protocol_sha256": base_protocol,
        "repair_protocol_sha256": repair_protocol,
        "r2_protocol_sha256": r2_protocol,
        "repair_gate_sha256": repair_gate,
        "manifest_sha256": manifest,
        "task_sha256": root / "task_v9.py",
        "design_sha256": root / "design_v9.py",
        "r2_design_sha256": root / "design_v9_r2.py",
        "runner_sha256": root / "run_v9.py",
        "jobs_sha256": root / "run_v9_jobs.py",
        "metrics_sha256": root / "metrics_v9.py",
        "common_analysis_sha256": root / "analyze_v9_common.py",
        "reliability_evaluator_sha256": root / "evaluate_v9_r2_reliability.py",
        "reliability_all_evaluator_sha256": root / "evaluate_all_v9_r2_reliability.py",
        "r2_analysis_sha256": Path(__file__),
        "preunlock_analysis_sha256": root / "analyze_preunlock_v9.py",
        "locked_evaluator_sha256": root / "evaluate_locked_v9.py",
        "locked_all_evaluator_sha256": root / "evaluate_all_locked_v9.py",
        "locked_analysis_sha256": root / "analyze_locked_v9.py",
        "main_manifest_sha256": root / "v9_manifest.csv",
        "model_sha256": root.parent / "panel" / "model.py",
    }
    return {
        "decision": {
            "route": route,
            "source_pass": source_pass,
            "behavior_pass": behavior_pass,
            "reliability_pass": reliability_pass,
            "reliability_counts_pass": counts_pass,
            "reliability_effects_pass": effect_pass,
        },
        "source_failures": source_failures,
        "behavior": behavior,
        "reliability": reliability,
        "seed_details": seed_details,
        "integrity": {
            **{key: sha256(path) for key, path in integrity_files.items()},
            "training_files": len(result_paths),
            "reliability_files": len(reliability_paths),
        },
    }


def verify_gate(path):
    value = json.loads(Path(path).read_text())
    if value["decision"]["route"] != "run-main":
        raise ValueError("V9-R2 did not authorize the main study")
    root = Path(__file__).resolve().parent
    files = {
        "base_protocol_sha256": root / "V9_PROTOCOL.md",
        "repair_protocol_sha256": root / "V9_REPAIR_PROTOCOL.md",
        "r2_protocol_sha256": root / "V9_R2_PROTOCOL.md",
        "repair_gate_sha256": root / "v9_repair_gate.json",
        "manifest_sha256": root / "v9_r2_manifest.csv",
        "task_sha256": root / "task_v9.py",
        "design_sha256": root / "design_v9.py",
        "r2_design_sha256": root / "design_v9_r2.py",
        "runner_sha256": root / "run_v9.py",
        "jobs_sha256": root / "run_v9_jobs.py",
        "metrics_sha256": root / "metrics_v9.py",
        "common_analysis_sha256": root / "analyze_v9_common.py",
        "reliability_evaluator_sha256": root / "evaluate_v9_r2_reliability.py",
        "reliability_all_evaluator_sha256": root / "evaluate_all_v9_r2_reliability.py",
        "r2_analysis_sha256": Path(__file__),
        "preunlock_analysis_sha256": root / "analyze_preunlock_v9.py",
        "locked_evaluator_sha256": root / "evaluate_locked_v9.py",
        "locked_all_evaluator_sha256": root / "evaluate_all_locked_v9.py",
        "locked_analysis_sha256": root / "analyze_locked_v9.py",
        "main_manifest_sha256": root / "v9_manifest.csv",
        "model_sha256": root.parent / "panel" / "model.py",
    }
    for key, file_path in files.items():
        if value["integrity"].get(key) != sha256(file_path):
            raise ValueError(f"V9-R2 changed after gate: {key}")
    return value


def report(result):
    lines = [
        "# V9-R2 fresh-seed causal reliability gate", "",
        "No locked role was constructed or evaluated.", "",
        f"Decision route: **{result['decision']['route']}**.", "",
        "## Gate", "", "| Gate | Pass |", "|---|:---:|",
        f"| Source mastery | {'yes' if result['decision']['source_pass'] else 'no'} |",
        f"| Behavioral replication | {'yes' if result['decision']['behavior_pass'] else 'no'} |",
        f"| Causal reliability | {'yes' if result['decision']['reliability_pass'] else 'no'} |",
        "", "## Behavioral contrasts", "",
    ]
    for name, value in result["behavior"].items():
        lines.append(
            f"- `{name}`: {fmt(value)}; positive seeds "
            f"{value['positive_seeds']}/8."
        )
    lines += [
        "", "## Reliability", "",
        "| Arm | Reliable seeds | Target | Target-random | Source-random |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in R2_ARMS:
        item = result["reliability"][arm]
        lines.append(
            f"| `{arm}` | {item['reliable_seeds']}/8 | "
            f"{item['target']['mean']:.3f} | {fmt(item['target_minus_random'])} | "
            f"{fmt(item['source_minus_random'])} |"
        )
    lines += ["", "## Per-seed replicate validity", ""]
    for arm in R2_ARMS:
        values = ", ".join(
            f"{seed}:{result['seed_details'][arm][str(seed)]['valid_replicates']}/4"
            for seed in R2_SEEDS
        )
        lines.append(f"- `{arm}`: {values}")
    failures = result["source_failures"]
    lines += ["", "## Source failures", ""]
    lines.extend(
        [f"- `{cid}` at {step}: {metric}" for cid, step, metric in failures]
        if failures else ["None."]
    )
    lines += ["", "## Integrity", ""]
    for key, value in result["integrity"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/training_axes/results_v9_r2"))
    parser.add_argument("--reliability", type=Path, default=Path("experiments/training_axes/results_v9_r2_reliability"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/training_axes/v9_r2_manifest.csv"))
    parser.add_argument("--base-protocol", type=Path, default=Path("experiments/training_axes/V9_PROTOCOL.md"))
    parser.add_argument("--repair-protocol", type=Path, default=Path("experiments/training_axes/V9_REPAIR_PROTOCOL.md"))
    parser.add_argument("--r2-protocol", type=Path, default=Path("experiments/training_axes/V9_R2_PROTOCOL.md"))
    parser.add_argument("--repair-gate", type=Path, default=Path("experiments/training_axes/v9_repair_gate.json"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/training_axes/v9_r2_gate.json"))
    parser.add_argument("--out-report", type=Path, default=Path("experiments/training_axes/V9_R2_GATE.md"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_gate(args.out_json)
        print("V9-R2 gate and code hashes verified; main is authorized")
        return
    result = analyze(
        args.results, args.reliability, args.manifest, args.base_protocol,
        args.repair_protocol, args.r2_protocol, args.repair_gate,
    )
    args.out_json.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    rendered = report(result)
    args.out_report.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
