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
    "usage: $0 check|smoke|gate|pilot|analyze|v2-smoke|v2|analyze-v2|unlock-v2|task2-smoke|task2|analyze-task2|unlock-task2" \
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
  *)
    usage
    exit 2
    ;;
esac
