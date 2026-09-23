#!/usr/bin/env bash
set -euo pipefail

# Native, no-Docker launcher for managed GB10 systems. Point TRAINING_PYTHON at
# any administrator-provided Python with NumPy and CUDA-enabled PyTorch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${TRAINING_PYTHON:-python3}"
JOBS="${TRAINING_JOBS:-1}"

cd "${PROJECT_ROOT}"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/principle_training_pycache"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/principle_training_cache"
export PANEL_SIMPLE_ATTN=1

run_python() {
  "${PYTHON_BIN}" "$@"
}

usage() {
  printf '%s\n' \
    "usage: $0 check|smoke|gate|pilot|analyze|v2-smoke|v2|analyze-v2|unlock-v2|task2-smoke|task2|analyze-task2|unlock-task2|v8-smoke|v8|analyze-v8|unlock-v8|analyze-locked-v8|verify-released-v8|v9-smoke|v9-pilot|analyze-v9-pilot|v9-repair|analyze-v9-repair|v9-r2-smoke|v9-r2|v9-r2-assay|analyze-v9-r2|v9|analyze-v9|unlock-v9|analyze-locked-v9|v9b-smoke|prepare-v9b|verify-v9b|unlock-v9b|analyze-locked-v9b|verify-released-v9b" \
    "" \
    "Environment overrides:" \
    "  TRAINING_PYTHON  Python executable with CUDA PyTorch" \
    "  TRAINING_JOBS    concurrent pilot processes (default: ${JOBS})"
}

case "${1:-}" in
  check)
    uname -m
    nvidia-smi || true
    run_python -c \
      'import platform, numpy, torch; print("python_arch", platform.machine()); print("numpy", numpy.__version__); print("torch", torch.__version__); print("cuda_runtime", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); print("capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); print("compiled_arches", torch.cuda.get_arch_list()); assert torch.cuda.is_available(); x=torch.randn(1024,1024,device="cuda",requires_grad=True); y=(x@x.T).square().mean(); y.backward(); torch.cuda.synchronize(); print("cuda_forward_backward", "PASS", float(y))'
    ;;
  smoke)
    run_python -m experiments.training_axes.task
    run_python -m experiments.training_axes.run \
      --smoke --device cuda --unlock-test --no-save-checkpoints \
      --results-dir /tmp/principle_training_smoke --overwrite
    ;;
  gate)
    for abstraction in concrete invariant latent; do
      run_python -m experiments.training_axes.run \
        --compression medium --difficulty mixed \
        --abstraction "${abstraction}" --seed 0 \
        --steps 4000 --batch-size 128 --eval-size 500 \
        --structural-pairs 256 --device cuda \
        --results-dir experiments/training_axes/gate_results \
        --checkpoints-dir experiments/training_axes/gate_checkpoints
    done
    ;;
  pilot)
    run_python -m experiments.training_axes.run_manifest \
      experiments/training_axes/pilot_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute
    ;;
  analyze)
    run_python -m experiments.training_axes.analyze \
      experiments/training_axes/results \
      --out experiments/training_axes/pilot_cells.csv
    ;;
  v2-smoke)
    run_python -m experiments.training_axes.run \
      --run-prefix v2 --compression none --difficulty mixed \
      --abstraction concrete --capacity wide --smoke --device cuda \
      --results-dir /tmp/principle_training_v2_smoke \
      --checkpoints-dir /tmp/principle_training_v2_smoke_checkpoints --overwrite
    run_python -m experiments.training_axes.run \
      --run-prefix v2 --compression high --difficulty hard \
      --abstraction latent --capacity tight --smoke --device cuda \
      --results-dir /tmp/principle_training_v2_smoke \
      --checkpoints-dir /tmp/principle_training_v2_smoke_checkpoints --overwrite
    ;;
  v2)
    run_python -m experiments.training_axes.run_manifest \
      experiments/training_axes/v2_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute \
      --results-dir experiments/training_axes/results_v2 \
      --checkpoints-dir experiments/training_axes/checkpoints_v2
    ;;
  analyze-v2)
    run_python -m experiments.training_axes.analyze \
      experiments/training_axes/results_v2 \
      --out experiments/training_axes/v2_cells.csv
    ;;
  unlock-v2)
    run_python -m experiments.training_axes.evaluate_all_locked \
      experiments/training_axes/checkpoints_v2 \
      --out-dir experiments/training_axes/locked_results_v2 \
      --device cuda --jobs "${JOBS}"
    ;;
  task2-smoke)
    run_python -m experiments.training_axes.run \
      --task differences --run-prefix v3 --compression none \
      --difficulty mixed --abstraction concrete --capacity wide \
      --smoke --device cuda --no-save-checkpoints \
      --results-dir /tmp/principle_task2_smoke --overwrite
    run_python -m experiments.training_axes.run \
      --task differences --run-prefix v3 --compression none \
      --difficulty hard --abstraction latent --capacity tight \
      --smoke --device cuda --no-save-checkpoints \
      --results-dir /tmp/principle_task2_smoke --overwrite
    ;;
  task2)
    run_python -m experiments.training_axes.run_manifest \
      experiments/training_axes/task2_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute \
      --results-dir experiments/training_axes/results_task2 \
      --checkpoints-dir experiments/training_axes/checkpoints_task2
    ;;
  analyze-task2)
    run_python -m experiments.training_axes.analyze \
      experiments/training_axes/results_task2 \
      --out experiments/training_axes/task2_cells.csv
    ;;
  unlock-task2)
    run_python -m experiments.training_axes.evaluate_all_locked \
      experiments/training_axes/checkpoints_task2 \
      --out-dir experiments/training_axes/locked_results_task2 \
      --device cuda --jobs "${JOBS}"
    ;;
  v8-smoke)
    smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/principle_v8_smoke.XXXXXX")"
    smoke_manifest="${smoke_root}/manifest.csv"
    run_python -m experiments.training_axes.design_v8 \
      --smoke --seeds 1 --out "${smoke_manifest}"
    run_python -m experiments.training_axes.run_manifest \
      "${smoke_manifest}" --device cuda --jobs 1 --execute \
      --no-save-checkpoints --results-dir "${smoke_root}/results"
    ;;
  v8)
    run_python -m experiments.training_axes.run_manifest \
      experiments/training_axes/v8_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute \
      --results-dir experiments/training_axes/results_v8 \
      --checkpoints-dir experiments/training_axes/checkpoints_v8
    ;;
  analyze-v8)
    run_python -m experiments.training_axes.analyze_preunlock_v8
    ;;
  unlock-v8)
    run_python -m experiments.training_axes.analyze_locked_v8 --verify-only
    run_python -m experiments.training_axes.evaluate_all_locked \
      experiments/training_axes/checkpoints_v8 \
      --out-dir experiments/training_axes/locked_results_v8 \
      --device cuda --jobs "${JOBS}"
    ;;
  analyze-locked-v8)
    run_python -m experiments.training_axes.analyze_locked_v8
    ;;
  verify-released-v8)
    run_python -m experiments.training_axes.verify_released_v8
    ;;
  v9-smoke)
    run_python -m unittest experiments.training_axes.test_v9 -v
    smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/principle_v9_smoke.XXXXXX")"
    smoke_manifest="${smoke_root}/manifest.csv"
    run_python -m experiments.training_axes.design_v9 \
      --phase pilot --smoke --seeds 1 --out "${smoke_manifest}"
    run_python -m experiments.training_axes.run_v9_jobs \
      "${smoke_manifest}" --device cuda --jobs 1 --execute --smoke \
      --no-save-checkpoints --results-dir "${smoke_root}/results" \
      --checkpoints-dir "${smoke_root}/checkpoints"
    ;;
  v9-pilot)
    run_python -m unittest experiments.training_axes.test_v9 -v
    run_python -m experiments.training_axes.run_v9_jobs \
      experiments/training_axes/v9_pilot_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute \
      --results-dir experiments/training_axes/results_v9_pilot \
      --checkpoints-dir experiments/training_axes/checkpoints_v9_pilot
    ;;
  analyze-v9-pilot)
    run_python -m experiments.training_axes.analyze_v9_pilot
    ;;
  v9-repair)
    if [[ ! -f experiments/training_axes/v9_pilot_gate.json ]]; then
      printf '%s\n' "missing v9_pilot_gate.json; run analyze-v9-pilot first" >&2
      exit 2
    fi
    run_python -c \
      'import json; from pathlib import Path; value=json.loads(Path("experiments/training_axes/v9_pilot_gate.json").read_text()); assert value["decision"]["route"] == "repair-source-mastery"'
    run_python -m experiments.training_axes.run_v9_jobs \
      experiments/training_axes/v9_repair_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute --allow-step-extension \
      --results-dir experiments/training_axes/results_v9_pilot \
      --checkpoints-dir experiments/training_axes/checkpoints_v9_pilot
    ;;
  analyze-v9-repair)
    run_python -m experiments.training_axes.analyze_v9_repair
    ;;
  v9-r2-smoke)
    run_python -m unittest \
      experiments.training_axes.test_v9 experiments.training_axes.test_v9_r2 -v
    ;;
  v9-r2)
    if [[ ! -f experiments/training_axes/v9_repair_gate.json ]]; then
      printf '%s\n' "missing v9_repair_gate.json; run analyze-v9-repair first" >&2
      exit 2
    fi
    run_python -m experiments.training_axes.run_v9_jobs \
      experiments/training_axes/v9_r2_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute \
      --results-dir experiments/training_axes/results_v9_r2 \
      --checkpoints-dir experiments/training_axes/checkpoints_v9_r2
    ;;
  v9-r2-assay)
    run_python -m experiments.training_axes.evaluate_all_v9_r2_reliability \
      experiments/training_axes/checkpoints_v9_r2 \
      --out-dir experiments/training_axes/results_v9_r2_reliability \
      --device cuda --jobs "${JOBS}"
    ;;
  analyze-v9-r2)
    run_python -m experiments.training_axes.analyze_v9_r2
    ;;
  v9)
    if [[ ! -f experiments/training_axes/v9_r2_gate.json ]]; then
      printf '%s\n' "missing v9_r2_gate.json; run analyze-v9-r2 first" >&2
      exit 2
    fi
    run_python -m experiments.training_axes.analyze_v9_r2 --verify-only
    run_python -m experiments.training_axes.run_v9_jobs \
      experiments/training_axes/v9_manifest.csv \
      --device cuda --jobs "${JOBS}" --execute \
      --results-dir experiments/training_axes/results_v9 \
      --checkpoints-dir experiments/training_axes/checkpoints_v9
    ;;
  analyze-v9)
    run_python -m experiments.training_axes.analyze_preunlock_v9
    ;;
  unlock-v9)
    run_python -m experiments.training_axes.analyze_locked_v9 --verify-only
    run_python -m experiments.training_axes.evaluate_all_locked_v9 \
      experiments/training_axes/checkpoints_v9 \
      --out-dir experiments/training_axes/locked_results_v9 \
      --device cuda --jobs "${JOBS}"
    ;;
  analyze-locked-v9)
    run_python -m experiments.training_axes.analyze_locked_v9
    ;;
  v9b-smoke)
    run_python -m unittest \
      experiments.training_axes.test_v9 experiments.training_axes.test_v9b
    ;;
  prepare-v9b)
    if [[ -e experiments/training_axes/v9b_lock_reveal.txt ]]; then
      printf '%s\n' "V9-B reveal is already present; use a new salt for replication" >&2
      exit 2
    fi
    run_python -m experiments.training_axes.prepare_v9b
    ;;
  verify-v9b)
    run_python -m experiments.training_axes.analyze_locked_v9b --verify-only
    ;;
  unlock-v9b)
    run_python -m experiments.training_axes.analyze_locked_v9b --verify-only
    run_python -m experiments.training_axes.evaluate_all_locked_v9b \
      experiments/training_axes/checkpoints_v9 \
      --out-dir experiments/training_axes/locked_results_v9b \
      --device cuda --jobs "${JOBS}"
    ;;
  analyze-locked-v9b)
    run_python -m experiments.training_axes.analyze_locked_v9b
    ;;
  verify-released-v9b)
    run_python -m experiments.training_axes.verify_released_v9b
    ;;
  *)
    usage
    exit 2
    ;;
esac
