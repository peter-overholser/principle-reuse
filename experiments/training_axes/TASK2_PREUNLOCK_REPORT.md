# Task 2 pre-unlock report

Recorded 9 September 2026 after training and before evaluation of
`test_combo`, `held_pair`, or `test_both`.

## Integrity

- Registered runs: 162.
- Completed result files: 162.
- Checkpoint rows: 1782 (11 per run).
- Final step: 10000 in every run.
- Manifest SHA-256: `8bed6bf9ebf1a5779d60c348f79b90997b450305cc36de2226b6009c043feaec`.
- Protocol SHA-256: `c20c9e671c453183a3a8da79b6970ff98507c55972759a4706a096871d444f4f`.
- Development-result tree SHA-256: `504a8503f2ee5c8297474f1181dae848f61e0cd0a0bdb0ba3d897e1525f04e7a`.

All manifest identifiers join one-to-one to result files. Every run reaches
persistent source mastery. No locked result directory is read by this analysis.

## Validation mastery

| Objective | Source accuracy | Source Brier skill |
|---|---:|---:|
| concrete | 0.9996 | 0.9988 |
| invariant | 0.9996 | 0.9988 |
| latent | 0.9999 | 0.9999 |

Applying the same equivalence margins used in V2 (accuracy ±0.02; Brier
skill ±0.03), the seed-paired latent-minus-concrete intervals are:

- source accuracy: +0.0004 [+0.0002, +0.0006];
- source Brier skill: +0.0011 [+0.0006, +0.0015].

Both intervals lie wholly inside the inherited margins.

## Development-generalization result

| Outcome | Concrete | Output invariant | Structural auxiliary |
|---|---:|---:|---:|
| `source_all` accuracy | 0.9721 | 0.9717 | 0.9932 |
| `calib_combo` accuracy | 0.4532 | 0.4417 | 0.9260 |
| `calib_combo` Brier skill | -1.1482 | -1.1909 | 0.7181 |

Seed-paired development contrasts:

- latent minus concrete `calib_combo` accuracy: +0.4728 [+0.3825, +0.5631];
- invariant minus concrete `calib_combo` accuracy: -0.0115 [-0.0986, +0.0755];
- latent minus concrete `calib_combo` Brier skill: +1.8662 [+1.5158, +2.2166].

The latent-minus-concrete accuracy difference is positive in 51/54
matched cells, zero in 0, and negative in 3.

## Development after mastery

Persistent mastery is source accuracy at least 0.95 at two consecutive
checkpoints; persistent reuse is `calib_combo` accuracy at least 0.80 by the
same rule.

| Objective | Median mastery step | Runs reaching reuse | Median lag among observed |
|---|---:|---:|---:|
| concrete | 400 | 11/54 | 1000 |
| invariant | 400 | 6/54 | 600 |
| latent | 400 | 46/54 | 1600 |

At the first persistent-mastery checkpoint, the latent-minus-concrete
development-combination contrast is
+0.1089 [+0.0517, +0.1661].
The fixed-budget endpoint contrast is substantially larger, so deployment-relevant
organization continues to develop after ordinary mastery.

## Structural and causal measurements

| Measurement | Concrete | Output invariant | Structural auxiliary |
|---|---:|---:|---:|
| Transferred target probe R² | -0.6260 | -0.5084 | 0.7441 |
| Transferred probe sign accuracy | 0.4646 | 0.4751 | 0.9320 |
| Rank-axis cosine | 0.1376 | 0.1531 | 0.9998 |
| Embedding participation rank | 28.8410 | 28.7899 | 7.7585 |

Causal-intervention validity gates:

- concrete: 11/54;
- invariant: 10/54;
- latent: 48/54.

Conditional causal means are descriptive only and remain coupled to these gate
rates. Invalid mechanistic measurements are not converted to zero.

## Factor-conditioned latent contrasts

These are secondary development-set summaries; intervals use the same six paired
seeds and are not multiplicity-adjusted.

### difficulty

| Level | Latent − concrete `calib_combo` accuracy |
|---|---:|
| easy | +0.5472 [+0.4307, +0.6637] |
| hard | +0.3779 [+0.1247, +0.6311] |
| mixed | +0.4933 [+0.2420, +0.7446] |

### capacity

| Level | Latent − concrete `calib_combo` accuracy |
|---|---:|
| moderate | +0.4301 [+0.3421, +0.5181] |
| tight | +0.6393 [+0.5205, +0.7580] |
| wide | +0.3490 [+0.0741, +0.6239] |

## Frozen interpretation before unlock

Task 2 passes the development-stage replication criterion: equal in-distribution
mastery accompanies a large objective-dependent difference in reuse on a more
compositional calculation, and the behavioral difference converges with transferred
structural measurements. Output-level invariance again does not reproduce the
structural-supervision effect.

This remains a development-set result. It does not yet establish replication on the
independently locked combinations, held quadruples, or joint shift. It also remains
a privileged-structure intervention on a related synthetic family, not evidence of
spontaneous principle discovery or generality to natural-language reasoning.
