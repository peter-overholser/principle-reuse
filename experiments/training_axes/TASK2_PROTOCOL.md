# Task 2: cross-domain difference comparison

Recorded before running task 2.

## Scientific purpose

Task 2 tests whether the V1/V2 mastery--reuse dissociation extends beyond a
first-order total-order comparison. The latent rule is the composed comparison

```text
a - b > c - d
```

over integer values 0--9. The second grammar expresses the logically equivalent
rule `b - a < d - c`. Six disjoint token alphabets render the same values.

The task requires composing four values and two differences, then comparing the
results. The reusable principle is translation/cancellation structure rather
than direct pairwise order alone.

## Split discipline

- Training: eight alphabet--grammar combinations and non-held quadruples.
- ID validation (`source`): new samples with the same combinations, quadruple
  split, and margin support as training.
- Development generalization (`source_all`, `calib_combo`): expanded margin
  support or two reserved alphabet--grammar combinations.
- Locked deployment (`test_combo`, `held_pair`, `test_both`): two untouched
  combinations, held latent quadruples, or both shifts.

A fixed 25 percent of latent quadruples is held out within every signed-margin
stratum. Locked sets are not evaluated during the registered run.

## Factorial design

- Objective: concrete, output-invariant, latent-structure.
- Evidence regime: hard margins 1--3, easy margins 7--17, mixed margins 1--17.
- Capacity: transformer widths 32, 64, 128.
- Seeds: 0--5.
- Weight decay: fixed at zero following its replicated behavioral null in V2.
- Training: 10,000 steps, batch size 256, two layers.

Total: 3 x 3 x 3 x 6 = 162 runs.

## Primary analysis

The primary development contrast is latent versus concrete `calib_combo`
accuracy at step 10,000, conditional on persistent ID mastery. The confirmatory
outcome is the same contrast on locked `test_combo`. `held_pair` and `test_both`
are secondary deployment outcomes.

The primary cross-task question is qualitative replication: equal ID mastery
with materially greater latent-objective reuse in both the order task and the
difference-comparison task. Effect sizes need not be equal.

## Interpretation safeguards

- Task 2 is related to, but computationally more compositional than, Task 1. It
  is a scope extension rather than evidence across unrelated real-world tasks.
- Signed-margin regression and alphabet mapping remain explicit privileged
  structure; they test the effect of rewarding a reusable principle, not its
  spontaneous discovery.
- Capacity, objective, and evidence-regime interactions are secondary.
- Causal results are missing when the intervention-validity gate fails.
- No cell or checkpoint is selected using locked outcomes.

## Remote launch

```bash
cd /home/dpo10/Desktop/work/principle-reuse
source .venv-gb10/bin/activate
export TRAINING_PYTHON="$PWD/.venv-gb10/bin/python3"
export TRAINING_JOBS=2

bash experiments/training_axes/remote/gb10_native.sh task2-smoke
bash experiments/training_axes/remote/gb10_native.sh task2 \
  2>&1 | tee experiments/training_axes/task2_run.log
```

The registered run is resumable. After all 162 runs complete:

```bash
bash experiments/training_axes/remote/gb10_native.sh analyze-task2 \
  2>&1 | tee experiments/training_axes/task2_analysis.log
```
