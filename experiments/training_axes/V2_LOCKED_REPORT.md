# V2 locked deployment report

Generated after the pre-unlock report and the one-time evaluation of the
registered `test_combo`, `held_pair`, and `test_both` splits.

## Integrity

- Registered runs: 324.
- Locked result files: 324.
- Locked checkpoint rows: 3564.
- Rows per run: 11.
- Final step: 10000 in every run.
- Manifest SHA-256: `69651ec8f7129706b0dd4730921657a4ea028e9ad33e414f1b5c3c152d3497ed`.

All locked files joined one-to-one with the registered manifest and the
development results. No run or checkpoint was selected using a locked outcome.

## Validation equivalence

| Endpoint source metric | Latent − concrete | Equivalence margin |
|---|---:|---:|
| Accuracy | +0.0000 [+0.0000, +0.0000] | ±0.02 |
| Brier skill | +0.0000 [+0.0000, +0.0000] | ±0.03 |

Both 95% intervals lie wholly inside their pre-specified equivalence bounds.
The endpoint comparison is therefore mastery-matched under the registered
definition; this is stronger than merely failing to detect a validation
difference.

## Confirmatory result

The table reports seed-paired mean differences and 95% t intervals over the
six seeds. Each seed contrast averages the fully crossed compression,
difficulty, and capacity cells.

| Locked outcome | Latent − concrete | Invariant − concrete |
|---|---:|---:|
| `test_combo` accuracy | +0.4846 [+0.3867, +0.5825] | -0.0022 [-0.0553, +0.0510] |
| `held_pair` accuracy | +0.0791 [+0.0616, +0.0967] | +0.0046 [-0.0018, +0.0110] |
| `test_both` accuracy | +0.3997 [+0.2941, +0.5053] | -0.0105 [-0.0538, +0.0327] |

The registered primary contrast is latent minus concrete on `test_combo`.
`held_pair` and `test_both` are the pre-specified secondary deployment
outcomes.

The locked `test_combo` effect differs from the development `calib_combo`
effect by +0.0231 [-0.0091, +0.0552]. Thus there is no
detectable attenuation on the independently held combination set.

### Objective means

| Outcome | Concrete | Invariant | Latent |
|---|---:|---:|---:|
| `source_accuracy` | 1.0000 | 1.0000 | 1.0000 |
| `source_all_accuracy` | 0.9358 | 0.9427 | 0.9623 |
| `calib_combo_accuracy` | 0.4534 | 0.4347 | 0.9149 |
| `test_combo_accuracy` | 0.4342 | 0.4321 | 0.9189 |
| `held_pair_accuracy` | 0.8365 | 0.8411 | 0.9156 |
| `test_both_accuracy` | 0.4651 | 0.4546 | 0.8648 |

### Proper-score contrast

| Locked outcome | Latent − concrete Brier skill |
|---|---:|
| `test_combo` | +1.9380 [+1.5474, +2.3286] |
| `held_pair` | +0.3205 [+0.2515, +0.3894] |
| `test_both` | +1.5998 [+1.1777, +2.0220] |

### Mastery-matched deployment performance

For each run, this evaluates the locked checkpoint at the first of two
consecutive checkpoints with source accuracy at least 0.95. It asks how much
reuse is already present when ordinary validation mastery first stabilizes.

| Locked outcome at mastery | Latent − concrete accuracy | Brier skill |
|---|---:|---:|
| `test_combo` | +0.0712 [-0.0053, +0.1477] | +0.3240 [+0.0278, +0.6203] |
| `held_pair` | +0.0389 [+0.0188, +0.0590] | +0.1601 [+0.0923, +0.2278] |
| `test_both` | +0.0336 [-0.0340, +0.1013] | +0.1779 [-0.0751, +0.4310] |

### Signed-margin breakdown

| Split | Near | Middle | Far |
|---|---:|---:|---:|
| `test_combo` | +0.3650 [+0.2996, +0.4305] | +0.5026 [+0.4008, +0.6044] | +0.5886 [+0.4556, +0.7216] |
| `held_pair` | +0.1802 [+0.1223, +0.2381] | +0.0211 [+0.0033, +0.0389] | +0.0255 [+0.0119, +0.0391] |
| `test_both` | +0.2734 [+0.1727, +0.3741] | +0.4467 [+0.3407, +0.5527] | +0.4996 [+0.3706, +0.6287] |

## Factor-conditioned latent contrasts

These are secondary interaction summaries. Each interval uses the same six
paired seeds; they are not multiplicity-adjusted.

### compression

| Level | test_combo | held_pair | test_both |
|---|---:|---:|---:|
| `high` | +0.4887 [+0.3947, +0.5827] | +0.0776 [+0.0629, +0.0923] | +0.3983 [+0.2865, +0.5100] |
| `none` | +0.4806 [+0.3738, +0.5873] | +0.0806 [+0.0562, +0.1050] | +0.4011 [+0.2942, +0.5079] |

### difficulty

| Level | test_combo | held_pair | test_both |
|---|---:|---:|---:|
| `easy` | +0.4614 [+0.3706, +0.5522] | +0.0072 [-0.0061, +0.0204] | +0.3947 [+0.2899, +0.4995] |
| `hard` | +0.4345 [+0.3647, +0.5044] | +0.1235 [+0.0705, +0.1765] | +0.3314 [+0.2604, +0.4023] |
| `mixed` | +0.5579 [+0.3922, +0.7236] | +0.1067 [+0.0951, +0.1183] | +0.4730 [+0.3022, +0.6437] |

### capacity

| Level | test_combo | held_pair | test_both |
|---|---:|---:|---:|
| `moderate` | +0.4841 [+0.2912, +0.6769] | +0.0701 [+0.0452, +0.0949] | +0.3953 [+0.1942, +0.5963] |
| `tight` | +0.5581 [+0.3562, +0.7600] | +0.1049 [+0.0846, +0.1252] | +0.4678 [+0.2749, +0.6606] |
| `wide` | +0.4117 [+0.2166, +0.6068] | +0.0624 [+0.0353, +0.0894] | +0.3360 [+0.1849, +0.4870] |

## Raw seed-level primary contrasts

| Seed | Latent − concrete test_combo accuracy |
|---:|---:|
| 0 | +0.5110 |
| 1 | +0.5003 |
| 2 | +0.3755 |
| 3 | +0.3848 |
| 4 | +0.6263 |
| 5 | +0.5098 |

The paired bootstrap interval for the primary contrast is [+0.4205, +0.5477].
The cellwise contrast is positive in 107/108 matched factorial
cells, zero in 0, and negative in 1.

## Interpretation

Interpretation is conditional on the pre-unlock finding of identical endpoint
source accuracy and source Brier skill across objectives. The primary question
is whether the latent-supervision advantage observed on `calib_combo` survives
on the independently locked deployment splits. Factor-conditioned intervals
describe scope and should not replace the registered pooled contrast.
