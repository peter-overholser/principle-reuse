# V2 pre-unlock report

Recorded 2026-09-06 after training and before evaluation of `test_combo`,
`held_pair`, or `test_both`.

## Integrity

- Registered runs: 324.
- Completed result files: 324.
- Checkpoint rows: 3,564 (11 per run).
- Final step: 10,000 in every run.
- Logged failed manifest rows: none detected.
- Manifest SHA-256:
  `69651ec8f7129706b0dd4730921657a4ea028e9ad33e414f1b5c3c152d3497ed`.
- Validation protocol SHA-256:
  `bfbe9723cf3eb2528607c54a240700a9d7fbc85ef159c11bdf826e1794d1deed`.

## Validation result

All factorial groups have mean endpoint source accuracy 1.000. Seed-averaged
latent-minus-concrete differences at step 10,000 are numerically 0.000 for
source accuracy and source Brier skill. Source confidence differs only below
the displayed precision. The endpoint data therefore satisfy the registered
validation-equivalence margins, subject to the ceiling limitation of these
metrics.

## Primary development-set contrast

The registered primary objective contrast, evaluated here on `calib_combo`
rather than the still-locked deployment set, is:

- latent minus concrete accuracy: +0.4616;
- seed-level 95% t interval: [0.3910, 0.5322];
- six seed-averaged contrasts: +0.494, +0.490, +0.388, +0.376, +0.551, +0.469.

Latent-minus-concrete target-probe R-squared is +1.6797, with a seed-level 95%
interval of [1.2680, 2.0914].

Invariant-minus-concrete accuracy is -0.0186, with interval
[-0.0771, 0.0398]. Output-level invariance again shows no reliable benefit.

## Factor summaries

Mean `calib_combo` accuracy:

- abstraction: concrete 0.453, invariant 0.435, latent 0.915;
- weight decay: none 0.603, high 0.599;
- evidence structure: easy 0.561, mixed 0.608, hard 0.634;
- capacity: tight 0.543, moderate 0.587, wide 0.674.

The capacity main effect is driven primarily by non-latent objectives. Wide
minus tight accuracy is +0.191 for concrete, +0.170 for invariant, and +0.033
for latent. The corresponding six-seed intervals all include zero. Capacity
effects are therefore secondary and not yet conclusive.

High weight decay strongly reduces parameter norm but has effectively zero
behavioral effect within every abstraction condition. It should not be
described as improving generalization.

## Mechanistic gate

Causal-intervention validity rates are:

- concrete: 13/108;
- invariant: 12/108;
- latent: 101/108.

Invalid cells remain missing mechanistic measurements. Valid-cell causal means
must be reported together with these gate rates.

## Frozen interpretation before unlock

V2 replicates the V1 development-set dissociation with stronger replication:
equal endpoint in-distribution mastery accompanies a large abstraction-dependent
difference in reuse and transferred latent structure. It does not support a
weight-decay account. Architectural capacity may benefit non-latent reuse, but
the estimate is unstable across seeds and is much smaller than the abstraction
effect.

The decisive confirmatory question remains untouched: whether the preregistered
latent-versus-concrete difference replicates on locked `test_combo`, followed by
`held_pair` and `test_both`. No claim about locked deployment performance is
made in this report.
