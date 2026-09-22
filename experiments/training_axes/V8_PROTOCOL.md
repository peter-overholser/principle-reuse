# V8 protocol: which part of the structural objective matters?

Recorded after V7C was designed and before any V8 development or locked
outcome was evaluated. V8 is the smallest experiment that can decide whether
the paper's earlier effect is attributable mainly to explicit embedding
correspondence, to non-embedding structural supervision, or to their
combination. It is not a test of every part of the developmental thesis.

## E0: checkpoint reanalysis before new training

`analyze_e0.py` performs four read-only analyses on existing checkpoints:

1. On the original order and difference studies, it reports agreement and
   probability assigned to the confounded grammar shortcut, plus sensitivity
   to swapping the relation symbol while holding the item order fixed.
2. It decomposes the locked contrast into the structured model's gain above
   chance and the concrete model's loss below chance.
3. On V6 `frozen_adapter`, it tests the flip-wrapper account per example using
   agreement with the negated frozen-core decision and a regression of the
   complete logit on the core logit.
4. On V5 early-supervision runs, it measures parameter drift relative to the
   update-2,000 withdrawal boundary.

E0 is diagnostic. Its outputs cannot alter V8 arms, thresholds, hierarchy, or
fresh-lock generator.

## Tasks and fresh data roles

V8 repeats both synthetic tasks at width 64 with two transformer blocks and
mixed difficulty:

- `order_v8`: compare two ranks;
- `differences_v8`: compare two signed differences.

The V8 generator is a new task instance, not a new name for an old split. It
uses new item-token permutations, new latent holdout seeds, new evaluation
seeds, and rotated alphabet roles. The 12 alphabet--grammar cells have these
roles:

- eight source cells used for optimization and constrained validation;
- two development combination cells, alphabets 2 and 3 under grammar 1;
- two locked combination cells, alphabets 0 and 1 under grammar 1.

Within each task, the locked evaluation also contains held latent queries in
source cells and the joint combination-plus-latent shift. Training constructs
neither locked examples nor locked metrics. `evaluate_locked.py` constructs
them only after the development freeze authorizes release.

This rotation guards against reusing the old held alphabets. It does not remove
the two-grammar confound; the later Strata/V9 design is responsible for that
stronger test.

## Objective decomposition

Every non-concrete arm retains output invariance `I`. The three structural
components are crossed as a complete 2 x 2 x 2 factorial:

- `E`: align the centered item-embedding blocks across alphabets using the
  known rank correspondence;
- `H`: align the final hidden representation of paired renderings;
- `G`: regress the signed latent gap from the final hidden representation with
  one shared head.

The factorial arms are named `eEhHgG`, from `e0h0g0` through `e1h1g1`.
`e0h0g0` is therefore the invariant-only baseline, `e1h0g0` is E-only,
`e0h1g1` is HG without E, and `e1h1g1` is the full objective.

Three controls are added:

- `concrete`: binary task loss only, with I/E/H/G all absent;
- `eperm`: I plus E, but each alphabet is aligned under a different fixed
  random permutation of rank identities;
- `gsep`: I plus G, with a separate gap-regression head per alphabet rather
  than a shared head.

Thus the two registered mechanism comparisons are semantic embedding
correspondence (`E-only` versus `E-perm`) and shared latent readout
(`G-shared` versus `G-separate`).

All objective weights retain the historical values: binary and I weight 1,
H weight 0.1, and G and E weight 1. No bottleneck is used. Auxiliary objectives
are active throughout all 10,000 updates. Batch size is 256 and learning rate
is 0.001 with the existing 200-update warmup.

## Runs and statistical unit

Use fresh paired seeds 400--411:

```text
2 tasks x 11 arms x 12 seeds = 264 trajectories.
```

Seed is the statistical unit. All arm contrasts are paired within task and
seed. Registered intervals are two-sided 95% Student-t intervals with 11
degrees of freedom. No task's hierarchy gates the other task's analysis.

## Measurements

Behavior is measured at updates 0, 50, 100, 200, 400, 700, 1,000, 2,000,
4,000, 7,000, and 10,000. The development phase may access source and
development-combination metrics only. It reports accuracy, Brier skill,
log-loss, confidence, distance-stratified behavior, shortcut agreement,
transferred signed-gap probes, paired invariance, embedding structure, the
existing guarded interchange assay, and parameter structure.

At every 50 updates through 2,000 and again at 4,000, 7,000, and 10,000, the
runner logs the unweighted value of each loss component. It also logs the L2
norm of the parameter gradient from each *weighted* component separately.
The reported gradient share is that component's norm divided by the sum of the
five component norms. This is a diagnostic allocation, not an additive
decomposition of the joint gradient.

## Development validity and source fiber

Every arm and seed must have source accuracy at least 0.95 at updates 7,000
and 10,000, and source Brier skill at least 0.80 at update 10,000.

For each task, the full, E-only, and HG arms are separately compared with I on
source behavior. Every paired interval must fit inside these registered
equivalence margins:

| Metric | Margin |
|---|---:|
| Source accuracy | +/-0.02 |
| Source Brier skill | +/-0.03 |
| Source log-loss | +/-0.02 |
| Near-margin source log-loss | +/-0.05 |

Failure of either mastery or equivalence blocks locked release. This is the
validation-fiber discipline: an apparent deployment difference is not called
organization if the contributing models already differ materially on the
constrained source domain.

As a fixed robustness analysis, each run also selects one checkpoint among
updates 400--10,000 with source accuracy at least 0.95. Selection minimizes
absolute distance to source log-loss 0.01, breaking ties toward the earlier
checkpoint. This rule is applied before locked outcomes exist, stored in the
development freeze, and reused unchanged for locked comparisons.

## Locked hierarchy

The endpoint primary outcome is locked held-combination accuracy. Testing is
sequential within each task:

1. **Full-objective replication:** full minus I has a 95% interval wholly
   above zero.
2. **Embedding sufficiency:** E-only minus full is contained in the
   equivalence band [-0.10, +0.10].
3. **Non-embedding survival:** HG minus I has a 95% interval wholly above
   zero.

The cross-task route is:

- if test 1 fails on either task: `replication-failed`;
- if test 1 passes on both, test 2 passes on both, and test 3 fails on each
  task: `embedding-tying-dominates`;
- if tests 1 and 3 pass on both: `non-tying-effect-survives`;
- otherwise: `mixed-component-result`.

Locked Brier skill, shortcut diagnostics, held-latent and joint-shift results,
the source-log-loss-matched contrasts, all factorial main effects, E-perm, and
G-separate are fixed secondary analyses. They do not replace a failed primary
hierarchy.

## Integrity and interpretation limits

Before unlocking, the analyzer freezes SHA-256 hashes of the protocol,
manifest, task generator, runner, metrics, model constructor, locked evaluator,
both analysis programs, common analysis code, development result tree,
gradient-diagnostic tree, and every registered checkpoint. The locked analyzer
refuses any code, protocol, or checkpoint mismatch and refuses missing, extra,
duplicate, or non-locked rows.

V8 can localize the current effect to objective components. It cannot by
itself establish emergence from data, entrenchment, use-dependence,
compounding, or scaffold generality. Those are separate Strata/V9 claims and
require separate manipulations and independent organization assays.
