# Reproducing the released experiments

## 1. Environment

Analysis requires Python 3.10 or newer, NumPy, and Matplotlib. Training also
requires PyTorch 2.4 or newer. CUDA is recommended for the full 324-run sweep,
but the task check and a smoke run work on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
```

All commands below run from the repository root.

## 2. Validate the task and software path

```bash
python -m experiments.training_axes.task

python -m experiments.training_axes.run \
  --smoke --device cpu --unlock-test --no-save-checkpoints \
  --results-dir /tmp/principle-reuse-smoke --overwrite
```

The smoke run deliberately touches the complete evaluation path but writes no
scientific result into the repository.

## 3. Regenerate the registered grid

```bash
python -m experiments.training_axes.design_v2 \
  --out /tmp/v2_manifest_regenerated.csv

cmp /tmp/v2_manifest_regenerated.csv \
  experiments/training_axes/v2_manifest.csv
```

The comparison should produce no output. The manifest contains 324 rows: two
weight-decay conditions by three difficulty conditions by three objectives by
three widths by six paired seeds.

## 4. Train

Normal training evaluates only the training, in-distribution validation, and
development-generalization splits. It saves checkpoints for a separate locked
evaluation.

```bash
python -m experiments.training_axes.run_manifest \
  experiments/training_axes/v2_manifest.csv \
  --device cuda --jobs 2 --execute \
  --results-dir experiments/training_axes/results_v2 \
  --checkpoints-dir experiments/training_axes/checkpoints_v2
```

Each configuration writes its own JSONL trajectory and checkpoint directory.
Existing result files are protected by default, so interrupted sweeps can be
resumed at the job level.

The native GB10 launcher wraps the same commands:

```bash
export TRAINING_PYTHON="$PWD/.venv/bin/python"
export TRAINING_JOBS=2
bash experiments/training_axes/remote/gb10_native.sh check
bash experiments/training_axes/remote/gb10_native.sh v2-smoke
bash experiments/training_axes/remote/gb10_native.sh v2
```

## 5. Freeze the pre-unlock analysis

The committed
[`V2_PREUNLOCK_REPORT.md`](experiments/training_axes/V2_PREUNLOCK_REPORT.md)
records the completed-run count, validation-equivalence result, development-set
contrast, interpretation, manifest hash, and protocol hash before any locked
outcome was inspected.

For a new replication, write and hash the equivalent report before continuing.
Do not use locked outcomes to choose runs, checkpoints, exclusions, or model
specifications.

## 6. Evaluate the locked splits once

```bash
python -m experiments.training_axes.evaluate_all_locked \
  experiments/training_axes/checkpoints_v2 \
  --out-dir experiments/training_axes/locked_results_v2 \
  --device cuda --jobs 2
```

## 7. Reproduce the confirmatory analysis

```bash
python -m experiments.training_axes.analyze_locked_v2
python -m experiments.training_axes.plot_v2
```

`analyze_locked_v2.py` enforces the registered 324-run join, eleven rows per
run, a common 10,000-step endpoint, and seed-paired inference. Invalid causal
measurements remain missing rather than being converted to zero.

## 8. Verify the released files

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

The checksum file covers all released manifests, protocols, freezes, locked
reports, summary outputs, analysis code, figures, and raw-result directories,
including V8 and V9-B.

## 9. Reproduce Task 2

Task 2 uses the composed rule `a - b > c - d`. Its registered grid contains
162 configurations: three objectives by three difficulty regimes by three
widths by six paired seeds, with weight decay fixed at zero.

Validate the generator and regenerate the manifest:

```bash
python -m experiments.training_axes.task_differences
python -m experiments.training_axes.design_task2 \
  --out /tmp/task2_manifest_regenerated.csv
cmp /tmp/task2_manifest_regenerated.csv \
  experiments/training_axes/task2_manifest.csv
```

Train without touching the locked splits:

```bash
python -m experiments.training_axes.run_manifest \
  experiments/training_axes/task2_manifest.csv \
  --device cuda --jobs 2 --execute \
  --results-dir experiments/training_axes/results_task2 \
  --checkpoints-dir experiments/training_axes/checkpoints_task2
```

Before evaluating locked data, reproduce and freeze the development analysis:

```bash
python -m experiments.training_axes.analyze_preunlock_task2
```

The released freeze is recorded in `TASK2_PREUNLOCK_FREEZE.md`. A new
replication should create its own freeze before proceeding. Then evaluate the
three locked splits once and reproduce the confirmatory analysis:

```bash
python -m experiments.training_axes.evaluate_all_locked \
  experiments/training_axes/checkpoints_task2 \
  --out-dir experiments/training_axes/locked_results_task2 \
  --device cuda --jobs 2

python -m experiments.training_axes.analyze_locked_task2
python -m experiments.training_axes.plot_cross_task
```

`analyze_locked_task2.py` refuses to run unless all 162 development and locked
files have eleven registered checkpoints, all identifiers join one-to-one, and
the manifest, protocol, development-result tree, pre-unlock analysis, frozen
contrasts, and pre-unlock report still match their recorded SHA-256 digests.

## 10. Verify the released V8 study

V8 is a 264-run fresh-lock decomposition of the structural objective. The
public release includes the exact frozen implementation, protocol, manifest,
development trajectories, gradient diagnostics, locked trajectories, and
registered reports. Model checkpoints are omitted because the complete tree is
substantially larger than the analysis artifact.

Recompute every released locked statistic and verify the frozen source trees,
code hashes, locked tree, analysis JSON, and report with:

```bash
python -m experiments.training_axes.verify_released_v8
```

Expected output:

```text
V8 public release verified: 264 locked runs; route=embedding-tying-dominates
```

The verifier intentionally does not replace the preregistered unlock path. The
original `analyze_locked_v8.py` additionally verifies all 2,904 frozen model
checkpoints before constructing or reading the locked evaluation. The public
verifier begins from the released JSONL trajectories and verifies them against
the hashes recorded during that unlock.

To train an independent replication, first validate the implementation and
regenerate the manifest:

```bash
python -m unittest experiments.training_axes.test_v8
python -m experiments.training_axes.design_v8 \
  --out /tmp/v8_manifest_regenerated.csv
cmp /tmp/v8_manifest_regenerated.csv \
  experiments/training_axes/v8_manifest.csv
```

Then train the complete manifest:

```bash
python -m experiments.training_axes.run_manifest \
  experiments/training_axes/v8_manifest.csv \
  --device cuda --jobs 2 --execute \
  --results-dir /tmp/principle-reuse-v8/results \
  --checkpoints-dir /tmp/principle-reuse-v8/checkpoints
```

An independent replication should create and preserve its own development
freeze before constructing locked examples. The registered sequence is:

```bash
python -m experiments.training_axes.analyze_preunlock_v8 \
  --results /tmp/principle-reuse-v8/results \
  --checkpoints /tmp/principle-reuse-v8/checkpoints \
  --out-json /tmp/principle-reuse-v8/preunlock.json \
  --out-report /tmp/principle-reuse-v8/PREUNLOCK.md

python -m experiments.training_axes.evaluate_all_locked \
  /tmp/principle-reuse-v8/checkpoints \
  --out-dir /tmp/principle-reuse-v8/locked \
  --device cuda --jobs 2

python -m experiments.training_axes.analyze_locked_v8 \
  --results /tmp/principle-reuse-v8/locked \
  --checkpoints /tmp/principle-reuse-v8/checkpoints \
  --freeze /tmp/principle-reuse-v8/preunlock.json \
  --out-json /tmp/principle-reuse-v8/locked_analysis.json \
  --out-report /tmp/principle-reuse-v8/LOCKED_REPORT.md
```

Do not run the locked evaluator unless the independent pre-unlock report says
`Decision: unlock`.

## 11. Verify the released V9-B study

V9-B is a prospective outcome-role confirmation over 192 frozen V9 endpoints.
The public artifact contains the complete fourteen-checkpoint development
tree, the stopped V9-A freeze, the subsequent endpoint-only V9-B freeze, the
commitment and post-evaluation reveal, and every locked endpoint row. Model
checkpoints are omitted.

Recompute both source gates, verify the commit--reveal pair and frozen code
hashes, reconstruct every locked statistic, and compare the registered report:

```bash
python -m experiments.training_axes.verify_released_v9b
```

Expected output:

```text
V9-B public release verified: 192 locked endpoints; route=emergence-supported
```

Regenerate the public V9-B figure with:

```bash
python -m experiments.training_axes.plot_v9b
```

The original sequence is recorded in
[`V9B_RUNBOOK.md`](experiments/training_axes/V9B_RUNBOOK.md). Its critical
ordering was:

1. preserve V9-A's registered stop;
2. hash all 192 update-20,000 checkpoints and verify endpoint mastery and
   source equivalence while the reveal was absent;
3. transfer the committed reveal only after the outcome-free freeze;
4. evaluate exactly one locked endpoint per model; and
5. run the registered hierarchy once.

For an independent replication, generate a new salt and commitment and create
a new untouched lock. The released reveal is public by design and must not be
reused as a hidden test.
