"""Integrity and commit--reveal utilities for V9-B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.training_axes.analyze_v9_common import (
    checkpoint_tree, sha256, tree_sha256,
)
from experiments.training_axes.analyze_preunlock_v9 import CHECKPOINTS
from experiments.training_axes.design_v9 import read_manifest


ENDPOINT = 20_000
EXPECTED_V9A_FAILURE = [
    "v9_sc-circle_a00_diverse_s-519", 15_000, "accuracy",
]
SOURCE_MARGINS = {
    "source_accuracy": 0.02,
    "source_brier_skill": 0.03,
    "source_log_loss": 0.02,
    "source_log_loss_near": 0.05,
}


def load_commitment(path):
    value = json.loads(Path(path).read_text())
    if value.get("scheme") != "sha256-strip-utf8-v1":
        raise ValueError("unsupported V9-B commitment scheme")
    commitment = value.get("commitment", "")
    if len(commitment) != 64 or any(c not in "0123456789abcdef" for c in commitment):
        raise ValueError("invalid V9-B SHA-256 commitment")
    return value


def verify_reveal(commitment_path, reveal_path):
    commitment = load_commitment(commitment_path)["commitment"]
    reveal = Path(reveal_path).read_text().strip()
    observed = hashlib.sha256(reveal.encode("utf-8")).hexdigest()
    if observed != commitment:
        raise ValueError("V9-B reveal does not match the registered commitment")
    return reveal


def derive_seed(reveal, *domain):
    payload = "v9b-lock-v1\0" + reveal + "\0" + "\0".join(map(str, domain))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def endpoint_checkpoint_tree(checkpoint_root, manifest_path):
    checkpoint_root = Path(checkpoint_root)
    configs = read_manifest(manifest_path)
    expected = {config.config_id for config in configs}
    directories = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    observed = {path.name for path in directories}
    if observed != expected:
        raise ValueError(
            f"V9-B checkpoint mismatch: missing={sorted(expected-observed)[:3]}, "
            f"extra={sorted(observed-expected)[:3]}"
        )
    paths = []
    for config in configs:
        path = checkpoint_root / config.config_id / f"step-{ENDPOINT:06d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing V9-B endpoint checkpoint: {path}")
        paths.append(path)
    return sorted(paths)


def v9a_integrity_files(root, manifest_path):
    root = Path(root)
    return {
        "protocol_sha256": root / "V9_PROTOCOL.md",
        "manifest_sha256": Path(manifest_path),
        "repair_protocol_sha256": root / "V9_REPAIR_PROTOCOL.md",
        "r2_protocol_sha256": root / "V9_R2_PROTOCOL.md",
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
        "locked_analysis_sha256": root / "analyze_locked_v9.py",
        "model_sha256": root.parent / "panel" / "model.py",
    }


def verify_stopped_v9a(freeze_path, manifest_path, checkpoints, results=None):
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    expected_decision = {
        "unlock_allowed": False,
        "source_mastery": False,
        "source_equivalence": True,
    }
    if freeze.get("decision") != expected_decision:
        raise ValueError("V9-B requires the registered stopped V9-A decision")
    if freeze.get("source_failures") != [EXPECTED_V9A_FAILURE]:
        raise ValueError("V9-A did not stop for the registered sole interim failure")
    root = Path(__file__).resolve().parent
    for key, path in v9a_integrity_files(root, manifest_path).items():
        if freeze["integrity"].get(key) != sha256(path):
            raise ValueError(f"V9-A provenance mismatch before V9-B: {key}")
    configs = read_manifest(manifest_path)
    checkpoint_paths = checkpoint_tree(
        checkpoints, [config.config_id for config in configs], CHECKPOINTS
    )
    observed = tree_sha256(checkpoint_paths, root=Path(checkpoints))
    if observed != freeze["integrity"]["checkpoint_tree_sha256"]:
        raise ValueError("V9-A checkpoint tree changed before V9-B")
    if results is not None:
        result_paths = sorted(Path(results).glob("*.jsonl"))
        if tree_sha256(result_paths) != freeze["integrity"]["development_tree_sha256"]:
            raise ValueError("V9-A development results changed before V9-B")
    return freeze


def v9b_integrity_files(root, manifest_path, v9a_freeze, commitment_path):
    root = Path(root)
    return {
        "v9b_protocol_sha256": root / "V9B_PROTOCOL.md",
        "v9b_commitment_file_sha256": Path(commitment_path),
        "v9b_common_sha256": root / "v9b_common.py",
        "v9b_preunlock_analysis_sha256": root / "prepare_v9b.py",
        "v9b_locked_evaluator_sha256": root / "evaluate_locked_v9b.py",
        "v9b_locked_all_evaluator_sha256": root / "evaluate_all_locked_v9b.py",
        "v9b_locked_analysis_sha256": root / "analyze_locked_v9b.py",
        "v9a_freeze_sha256": Path(v9a_freeze),
        "manifest_sha256": Path(manifest_path),
        "task_sha256": root / "task_v9.py",
        "design_sha256": root / "design_v9.py",
        "metrics_sha256": root / "metrics_v9.py",
        "model_sha256": root.parent / "panel" / "model.py",
    }


def verify_v9b_freeze(
    freeze_path, manifest_path, checkpoints, v9a_freeze, commitment_path,
):
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze.get("decision", {}).get("lock_authorized") is not True:
        raise ValueError("V9-B preunlock freeze did not authorize evaluation")
    if freeze.get("v9a_source_failure") != EXPECTED_V9A_FAILURE:
        raise ValueError("V9-B freeze has the wrong V9-A failure provenance")
    root = Path(__file__).resolve().parent
    files = v9b_integrity_files(
        root, manifest_path, v9a_freeze, commitment_path
    )
    for key, path in files.items():
        if freeze["integrity"].get(key) != sha256(path):
            raise ValueError(f"V9-B changed after preunlock freeze: {key}")
    paths = endpoint_checkpoint_tree(checkpoints, manifest_path)
    observed = tree_sha256(paths, root=Path(checkpoints))
    if observed != freeze["integrity"]["endpoint_checkpoint_tree_sha256"]:
        raise ValueError("V9-B endpoint checkpoint tree changed after freeze")
    commitment = load_commitment(commitment_path)["commitment"]
    if freeze.get("lock_commitment") != commitment:
        raise ValueError("V9-B lock commitment changed after freeze")
    return freeze
