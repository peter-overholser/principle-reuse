# V2 validation and generalization protocol

Recorded before inspection of V2 outcomes.

## Inferential hierarchy

The experiment distinguishes four data roles. They must not be pooled or
renamed after results are known.

1. **Training distribution**: `TRAIN_COMBOS`, non-held rank pairs, and the rank
   distances assigned by the experimental difficulty condition.
2. **In-distribution validation**: `source`. These are fresh examples from the
   same combinations, non-held pairs, and distance support as training. This set
   measures task mastery and optimization quality, not principle reuse.
3. **Development generalization**: `source_all` and `calib_combo`.
   `source_all` expands the distance support while retaining training domain
   combinations. `calib_combo` introduces reserved alphabet--grammar
   combinations while retaining non-held latent pairs. These sets may diagnose
   and calibrate the experimental design but cannot serve as final test evidence.
4. **Locked deployment generalization**: `test_combo`, `held_pair`, and
   `test_both`. These respectively test unseen domain combinations, unseen
   latent relations, and their conjunction. They are evaluated only after the
   V2 analysis specification and exclusion rules are frozen.

## Required validation equivalence

The primary generalization comparison is interpretable only among runs that
master the in-distribution validation task. Mastery is defined prospectively as
`source_accuracy >= 0.95` at two consecutive measured checkpoints. Endpoint
comparisons additionally report source Brier skill and confidence to detect
differences hidden by saturated accuracy.

Validation equality is an equivalence claim, not a failure to reject a
difference. For the primary abstraction contrasts, report paired seed-level
differences and a 95% interval against these smallest effects of interest:

- accuracy: plus or minus 0.02;
- Brier skill: plus or minus 0.03.

If equivalence is not supported, report generalization conditional on validation
performance and do not describe the groups as mastery-matched.

## Primary estimands

1. **Fixed-budget deployment performance**: locked accuracy and Brier skill at
   step 10,000, conditional on validation mastery.
2. **Generalization gap**: locked deployment performance minus in-distribution
   validation performance on the same metric.
3. **Mastery-matched deployment performance**: locked performance at the first
   checkpoint satisfying persistent validation mastery. This separates reusable
   structure available at mastery from structure acquired through later
   training.
4. **Development lag**: steps from persistent validation mastery to persistent
   development-generalization success.

The fixed-budget estimand is primary. Mastery-matched and temporal estimands are
secondary but prespecified.

## Deployment tests

- `test_combo` is the primary deployment outcome: new domain compositions with
  familiar latent relations.
- `held_pair` is the secondary relational-generalization outcome.
- `test_both` is the stress test and most distant deployment condition.
- Accuracy and Brier skill are co-primary behavioral summaries; confidence is a
  calibration diagnostic.

Near, middle, and far distance strata are reported for every behavioral split.
No stratum replaces the aggregate primary outcome.

## Model selection and multiplicity

- No run, checkpoint, width, seed, or objective is selected using locked test
  performance.
- Failed runs are rerun with the identical registered configuration; they are
  not silently excluded.
- The primary contrast is latent versus concrete on `test_combo` at step 10,000.
- Invariant versus concrete, capacity effects, weight-decay effects, interactions,
  `held_pair`, and `test_both` are secondary.
- Report all registered cells and all six seeds. Use hierarchical/factorial
  estimates and uncertainty rather than choosing the best cell.

## Structural and causal evidence

Probe, alignment, effective-rank, and intervention measurements do not replace
behavioral deployment evaluation. They address mechanism after behavioral
generalization is established.

Intervention results use a two-part report:

1. the fraction of runs satisfying the prespecified intervention-validity gate;
2. normalized causal transfer among valid runs.

Invalid intervention values are missing mechanistic measurements, not zero
effects and not ordinary observations to average.

## Decision language

The strongest supported conclusion requires all three conditions:

1. validation equivalence within the prespecified bounds;
2. a replicated difference on locked deployment generalization;
3. convergent, explicitly qualified structural or causal evidence.

If only the second condition holds, report a generalization difference without
claiming equal mastery. If deployment outcomes do not diverge, development-set
differences are exploratory and cannot support the headline claim.
