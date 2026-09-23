# V9-A GB10 runbook

V9-A is gated. The original 10,000-update pilot returned
`repair-source-mastery`. Preserve that result, continue the same 64 trajectories
to 20,000 updates, and do not launch the 192-trunk main study unless
`analyze-v9-repair` returns `run-main`.

## Sync code to the Spark

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

## Environment

```bash
ssh dpo10@s35624.humboldt.edu
cd /home/dpo10/Desktop/work/principle-reuse
source .venv-gb10/bin/activate

export TRAINING_PYTHON="$PWD/.venv-gb10/bin/python3"
export TRAINING_JOBS=2
```

## Preserve the registered 10k pilot

Before continuing, archive the original development-only result and gate. This
does not copy the large checkpoint tree; the existing update-10,000 checkpoints
remain untouched and are the exact continuation boundary.

```bash
cd /home/dpo10/Desktop/work/principle-reuse

archive=experiments/training_axes/v9_pilot_10k_archive
if [ ! -e "$archive" ]; then
  mkdir -p "$archive"
  cp -a experiments/training_axes/results_v9_pilot "$archive"/
  cp -a experiments/training_axes/V9_PILOT_GATE.md "$archive"/
  cp -a experiments/training_axes/v9_pilot_gate.json "$archive"/
  cp -a experiments/training_axes/v9_pilot_analysis.log "$archive"/ 2>/dev/null || true
fi
```

Confirm that the continuation inputs exist:

```bash
find experiments/training_axes/results_v9_pilot \
  -maxdepth 1 -name '*.jsonl' | wc -l

find experiments/training_axes/checkpoints_v9_pilot \
  -name 'step-010000.pt' | wc -l
```

Both commands must print `64`.

## Smoke test

```bash
bash experiments/training_axes/remote/gb10_native.sh v9-smoke \
  2>&1 | tee experiments/training_axes/v9_smoke.log
```

The command must finish with `OK` from the deterministic tests and status
0 for all sixteen one-seed smoke configurations.

## Original feasibility pilot

The commands below document the original run. Do not rerun them now.

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh v9-pilot \
  > experiments/training_axes/v9_pilot_run.log 2>&1 &
echo $!
```

Monitor:

```bash
tail -f experiments/training_axes/v9_pilot_run.log
```

The launcher is done when `jobs -l` is empty and the final manifest job reports
status 0. Confirm 64 result files:

```bash
find experiments/training_axes/results_v9_pilot \
  -maxdepth 1 -name '*.jsonl' | wc -l
```

Analyze without constructing a locked role:

```bash
bash experiments/training_axes/remote/gb10_native.sh analyze-v9-pilot \
  2>&1 | tee experiments/training_axes/v9_pilot_analysis.log
```

This original analysis returned `repair-source-mastery`; the 20k continuation
below is the registered response.

## Continue the feasibility pilot to 20k

The continuation restores parameters, optimizer state, and all registered RNG
states. It appends updates 12k, 15k, and 20k to the existing result files.

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh v9-repair \
  > experiments/training_axes/v9_repair_run.log 2>&1 &
echo $!
```

Monitor:

```bash
tail -f experiments/training_axes/v9_repair_run.log
```

The continuation is finished when `jobs -l` is empty, the final launcher lines
report status 0, and this check reports 64 files ending at update 20,000:

```bash
find experiments/training_axes/results_v9_pilot \
  -maxdepth 1 -name '*.jsonl' -exec tail -n 1 {} \; |
"$TRAINING_PYTHON" -c \
'import sys,json; from collections import Counter; print(Counter(json.loads(x)["step"] for x in sys.stdin if x.strip()))'
```

Expected output:

```text
Counter({20000: 64})
```

Every file should now contain the 14 registered checkpoints:

```bash
"$TRAINING_PYTHON" - <<'PY'
from collections import Counter
from pathlib import Path
root = Path("experiments/training_axes/results_v9_pilot")
print(Counter(sum(1 for line in p.open() if line.strip()) for p in root.glob("*.jsonl")))
PY
```

Expected output:

```text
Counter({14: 64})
```

Run the repair analysis:

```bash
bash experiments/training_axes/remote/gb10_native.sh analyze-v9-repair \
  2>&1 | tee experiments/training_axes/v9_repair_analysis.log
```

The observed repair route was `stop-invalid-assay`. Continue with the fresh-seed
causal reliability screen in `V9_R2_RUNBOOK.md`; do not launch the main study
from the repair result.

## Main study

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh v9 \
  > experiments/training_axes/v9_run.log 2>&1 &
echo $!
```

The repaired main study trains fresh seeds for 20,000 updates. After all 192
result files complete, confirm that each ends at 20,000 and then run:

```bash
bash experiments/training_axes/remote/gb10_native.sh analyze-v9 \
  2>&1 | tee experiments/training_axes/v9_preunlock_analysis.log
```

Proceed only if the report says `Decision: unlock`.

## Locked release

```bash
nohup bash experiments/training_axes/remote/gb10_native.sh unlock-v9 \
  > experiments/training_axes/v9_unlock.log 2>&1 &
echo $!
```

Then:

```bash
bash experiments/training_axes/remote/gb10_native.sh analyze-locked-v9 \
  2>&1 | tee experiments/training_axes/v9_locked_analysis.log
```

## Pull results locally

From the local repository root:

```bash
rsync -az --progress \
  dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/results_v9_pilot/ \
  experiments/training_axes/results_v9_pilot/

rsync -az --progress \
  'dpo10@s35624.humboldt.edu:/home/dpo10/Desktop/work/principle-reuse/experiments/training_axes/{V9_PILOT_GATE.md,v9_pilot_gate.json,V9_REPAIR_GATE.md,v9_repair_gate.json,v9_repair_analysis.log}' \
  experiments/training_axes/
```

After a main or locked run, use the same pattern for `results_v9/`,
`locked_results_v9/`, `V9_PREUNLOCK_REPORT.md`, `v9_preunlock_freeze.json`,
`V9_LOCKED_REPORT.md`, and `v9_locked_analysis.json`.
