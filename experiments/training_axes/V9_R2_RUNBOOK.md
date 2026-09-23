# V9-R2 GB10 runbook

V9-R2 is a fresh-seed, development-only reliability screen. It trains 24
circle models and repeats the causal assay four times at the frozen 20k
endpoint. No locked role is constructed or evaluated. Do not launch V9 main
unless `analyze-v9-r2` returns `run-main`.

## 1. Sync code to the Spark

From the local repository root:

```bash
rsync -az --progress \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude 'experiments/training_axes/results*/' \
  --exclude 'experiments/training_axes/checkpoints*/' \
  --exclude 'experiments/training_axes/locked_results*/' \
  ./ dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/
```

The command does not delete or overwrite the existing V9 result directories.

## 2. Enter the environment

```bash
ssh dpo10@s35624.humboldt.edu
cd /home/dpo10/Desktop/work/principle-reuse
source .venv-gb10/bin/activate

export TRAINING_PYTHON="$PWD/.venv-gb10/bin/python3"
export TRAINING_JOBS=2
```

Verify that the preceding registered stop is present:

```bash
"$TRAINING_PYTHON" -c \
'import json; print(json.load(open("experiments/training_axes/v9_repair_gate.json"))["decision"]["route"])'
```

Expected output:

```text
stop-invalid-assay
```

## 3. Smoke test

```bash
bash experiments/training_axes/remote/gb10_native.sh v9-r2-smoke \
  2>&1 | tee experiments/training_axes/v9_r2_smoke.log
```

This runs the deterministic tests, three 30-update training jobs, and a small
end-to-end causal-assay evaluation. It must finish without an exception.

## 4. Train the 24 fresh-seed models

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh v9-r2 \
  > experiments/training_axes/v9_r2_run.log 2>&1 &
echo $!
```

Monitor:

```bash
tail -f experiments/training_axes/v9_r2_run.log
```

Training is complete when `jobs -l` is empty, all launcher statuses are zero,
and the following reports 24 endpoints at update 20,000:

```bash
find experiments/training_axes/results_v9_r2 \
  -maxdepth 1 -name '*.jsonl' -exec tail -n 1 {} \; |
"$TRAINING_PYTHON" -c \
'import sys,json; from collections import Counter; print(Counter(json.loads(x)["step"] for x in sys.stdin if x.strip()))'
```

Expected output:

```text
Counter({20000: 24})
```

Confirm that every trajectory has all 14 checkpoints:

```bash
"$TRAINING_PYTHON" - <<'PY'
from collections import Counter
from pathlib import Path
root = Path("experiments/training_axes/results_v9_r2")
print(Counter(sum(1 for line in p.open() if line.strip()) for p in root.glob("*.jsonl")))
PY
```

Expected output:

```text
Counter({14: 24})
```

If a trajectory is partial, rerun the same `v9-r2` command; it resumes exactly
from its latest saved checkpoint.

## 5. Run the frozen-model reliability assay

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh v9-r2-assay \
  > experiments/training_axes/v9_r2_assay.log 2>&1 &
echo $!
```

Monitor:

```bash
tail -f experiments/training_axes/v9_r2_assay.log
```

The assay is complete when there are 24 files and each contains four rows:

```bash
"$TRAINING_PYTHON" - <<'PY'
from collections import Counter
from pathlib import Path
root = Path("experiments/training_axes/results_v9_r2_reliability")
paths = list(root.glob("*.jsonl"))
print("files", len(paths))
print("rows", Counter(sum(1 for line in p.open() if line.strip()) for p in paths))
PY
```

Expected output:

```text
files 24
rows Counter({4: 24})
```

The reliability launcher is resumable: complete four-row files are skipped.

## 6. Run the registered gate

```bash
bash experiments/training_axes/remote/gb10_native.sh analyze-v9-r2 \
  2>&1 | tee experiments/training_axes/v9_r2_analysis.log
```

Only this output authorizes the main study:

```text
Decision route: **run-main**.
```

Any other route is a scientific stop. Do not run `v9`, `unlock-v9`, or any
locked evaluator.

## 7. Pull V9-R2 results locally

From the local repository root:

```bash
rsync -az --progress \
  dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/results_v9_r2/ \
  experiments/training_axes/results_v9_r2/

rsync -az --progress \
  dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/results_v9_r2_reliability/ \
  experiments/training_axes/results_v9_r2_reliability/

rsync -az --progress \
  'dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/{V9_R2_GATE.md,v9_r2_gate.json,v9_r2_analysis.log}' \
  experiments/training_axes/
```

Checkpoint transfer is unnecessary unless an audit or additional registered
diagnostic is needed.

## 8. Main study, only after a pass

If V9-R2 returns `run-main`:

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh v9 \
  > experiments/training_axes/v9_run.log 2>&1 &
echo $!
```

The existing V9 runbook then governs main completion, pre-unlock analysis, and
locked release. The main uses untouched seeds 512--523.
