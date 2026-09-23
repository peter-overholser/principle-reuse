# V9-B GB10 runbook

V9-A remains stopped. V9-B prospectively evaluates the frozen update-20,000
checkpoints on one freshly salted instance of the untouched locked roles. It
does not retrain or extend any model.

## 1. Sync the registered code without the reveal

From the local repository root:

```bash
rsync -az --progress \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '.v9b_lock_reveal' \
  --exclude 'experiments/training_axes/results*/' \
  --exclude 'experiments/training_axes/checkpoints*/' \
  --exclude 'experiments/training_axes/locked_results*/' \
  ./ dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/
```

Do not transfer `.v9b_lock_reveal` yet.

## 2. Enter the environment and test the implementation

```bash
ssh dpo10@s35624.humboldt.edu
cd /home/dpo10/Desktop/work/principle-reuse
source .venv-gb10/bin/activate

export TRAINING_PYTHON="$PWD/.venv-gb10/bin/python3"
export TRAINING_JOBS=2

bash experiments/training_axes/remote/gb10_native.sh v9b-smoke \
  2>&1 | tee experiments/training_axes/v9b_smoke.log
```

The smoke test must finish with `OK`.

## 3. Create the outcome-free V9-B freeze

The reveal must still be absent. Run:

```bash
test ! -e experiments/training_axes/v9b_lock_reveal.txt

bash experiments/training_axes/remote/gb10_native.sh prepare-v9b \
  2>&1 | tee experiments/training_axes/v9b_preunlock_analysis.log
```

This rehashes the complete V9-A checkpoint tree and can take several minutes.
Proceed only if the report says:

```text
Decision: unlock-fresh-lock.
```

It must also report 192 mastered endpoints, source equivalence passing, zero
endpoint failures, and 192 endpoint checkpoints.

Do **not** run `unlock-v9`; V9-A remains stopped.

## 4. Reveal only after the freeze

In a second local terminal, from the local repository root:

```bash
rsync -az --progress \
  .v9b_lock_reveal \
  dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/v9b_lock_reveal.txt
```

Back on the Spark, verify authorization without constructing outcomes:

```bash
bash experiments/training_axes/remote/gb10_native.sh verify-v9b
```

Expected final line:

```text
V9-B freeze and reveal verified; prospective lock is authorized
```

## 5. Evaluate the fresh lock

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh unlock-v9b \
  > experiments/training_axes/v9b_unlock.log 2>&1 &
echo $!
```

Monitor:

```bash
tail -f experiments/training_axes/v9b_unlock.log
```

The evaluation is complete when `jobs -l` is empty, the last launcher job has
status 0, and the following prints `192`:

```bash
find experiments/training_axes/locked_results_v9b \
  -maxdepth 1 -name '*.jsonl' | wc -l
```

Every file must contain exactly one update-20,000 row:

```bash
"$TRAINING_PYTHON" - <<'PY'
import json
from collections import Counter
from pathlib import Path

paths = sorted(Path("experiments/training_axes/locked_results_v9b").glob("*.jsonl"))
counts = Counter()
steps = Counter()
for path in paths:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    counts[len(rows)] += 1
    steps.update(row["step"] for row in rows)
print("row counts", counts)
print("steps", steps)
PY
```

Expected:

```text
row counts Counter({1: 192})
steps Counter({20000: 192})
```

## 6. Analyze exactly once

```bash
bash experiments/training_axes/remote/gb10_native.sh analyze-locked-v9b \
  2>&1 | tee experiments/training_axes/v9b_locked_analysis.log
```

## 7. Pull the registered artifacts locally

From the local repository root:

```bash
rsync -az --progress \
  dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/locked_results_v9b/ \
  experiments/training_axes/locked_results_v9b/

rsync -az --progress \
  'dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/{V9B_PREUNLOCK_REPORT.md,v9b_preunlock_freeze.json,v9b_preunlock_analysis.log,V9B_LOCKED_REPORT.md,v9b_locked_analysis.json,v9b_locked_analysis.log}' \
  experiments/training_axes/
```

After the locked evaluation is complete, publish the reveal alongside the
commitment so third parties can verify the commit--reveal sequence. The reveal
must never be transferred before the preunlock freeze, but it is no longer
secret after the one-time lock has been evaluated.
