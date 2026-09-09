# Task 2 locked deployment report

Generated after the pre-unlock freeze and the one-time evaluation of the
registered `test_combo`, `held_pair`, and `test_both` splits.

## Integrity

- Registered runs: 162.
- Development files/rows: 162/1782.
- Locked files/rows: 162/1782.
- Checkpoints per run: 11; final step: 10000.
- Development tree SHA-256 (frozen): `504a8503f2ee5c8297474f1181dae848f61e0cd0a0bdb0ba3d897e1525f04e7a`.
- Locked tree SHA-256: `5c91ccf8f76fc75510f2cb5b77fa52c77cc50f5501190d0a7326d3494bdf24ba`.

All files join one-to-one with the frozen manifest, every trajectory has the
registered checkpoint schedule, and no run or checkpoint was selected using a
locked outcome.

## Validation equivalence

| Endpoint source metric | Structural auxiliary − concrete | Equivalence margin |
|---|---:|---:|
| Accuracy | +0.0004 [+0.0002, +0.0006] | ±0.02 |
| Brier skill | +0.0011 [+0.0006, +0.0015] | ±0.03 |

Both intervals lie wholly within the inherited equivalence margins.

## Confirmatory result

Seed-paired mean differences and 95% t intervals use six seeds; each seed
contrast averages the nine crossed difficulty-by-capacity cells.

| Locked outcome | Structural auxiliary − concrete | Output invariant − concrete |
|---|---:|---:|
| `test_combo` accuracy | +0.4668 [+0.3698, +0.5639] | -0.0097 [-0.1262, +0.1069] |
| `held_pair` accuracy | +0.0221 [+0.0075, +0.0366] | -0.0002 [-0.0042, +0.0038] |
| `test_both` accuracy | +0.4681 [+0.3729, +0.5633] | -0.0094 [-0.1257, +0.1069] |

The registered primary contrast is structural auxiliary minus concrete on
`test_combo`: +0.4668 [+0.3698, +0.5639]. Its interval excludes zero in the predicted direction.

Relative to the development `calib_combo` effect, the locked `test_combo`
effect changes by -0.0060 [-0.0561, +0.0442]. There is no detectable attenuation or amplification at the 95% level.

### Objective means

| Outcome | Concrete | Output invariant | Structural auxiliary |
|---|---:|---:|---:|
| `source_accuracy` | 0.9996 | 0.9996 | 0.9999 |
| `source_all_accuracy` | 0.9721 | 0.9717 | 0.9932 |
| `calib_combo_accuracy` | 0.4532 | 0.4417 | 0.9260 |
| `test_combo_accuracy` | 0.4523 | 0.4426 | 0.9191 |
| `held_pair_accuracy` | 0.9700 | 0.9698 | 0.9921 |
| `test_both_accuracy` | 0.4492 | 0.4398 | 0.9173 |

### Proper-score contrast

| Locked outcome | Structural auxiliary − concrete Brier skill |
|---|---:|
| `test_combo` | +1.8434 [+1.4581, +2.2286] |
| `held_pair` | +0.0876 [+0.0314, +0.1439] |
| `test_both` | +1.8493 [+1.4710, +2.2275] |

### Performance at first persistent validation mastery

| Locked outcome | Accuracy contrast | Brier-skill contrast |
|---|---:|---:|
| `test_combo` | +0.1118 [+0.0482, +0.1755] | +0.4251 [+0.1742, +0.6760] |
| `held_pair` | +0.0038 [-0.0184, +0.0260] | +0.0135 [-0.0649, +0.0918] |
| `test_both` | +0.1080 [+0.0503, +0.1656] | +0.4219 [+0.1977, +0.6460] |

### Signed-margin breakdown

| Split | Near | Middle | Far |
|---|---:|---:|---:|
| `test_combo` | +0.3120 [+0.2458, +0.3782] | +0.4268 [+0.3127, +0.5410] | +0.5180 [+0.4127, +0.6233] |
| `held_pair` | +0.0576 [+0.0526, +0.0627] | +0.0069 [+0.0036, +0.0103] | +0.0159 [-0.0056, +0.0375] |
| `test_both` | +0.3144 [+0.2449, +0.3838] | +0.4308 [+0.3102, +0.5515] | +0.5161 [+0.4161, +0.6162] |

## Stability and scope

- `test_combo`: positive in 50/54 matched cells, zero in 0, negative in 4.
- `held_pair`: positive in 37/54 matched cells, zero in 13, negative in 4.
- `test_both`: positive in 50/54 matched cells, zero in 0, negative in 4.

The negative primary cells are concentrated in particular
seed-by-capacity realizations:

| Seed | Difficulty | Capacity | Concrete | Structural | Difference |
|---:|---|---|---:|---:|---:|
| 0 | hard | wide | 0.8845 | 0.3535 | -0.5310 |
| 3 | hard | wide | 0.8540 | 0.6440 | -0.2100 |
| 5 | easy | tight | 0.9270 | 0.8925 | -0.0345 |
| 5 | hard | moderate | 0.9520 | 0.7575 | -0.1945 |

All six seed-level primary contrasts are reported below.

| Seed | Structural auxiliary − concrete `test_combo` accuracy |
|---:|---:|
| 0 | +0.4077 |
| 1 | +0.4585 |
| 2 | +0.5925 |
| 3 | +0.3308 |
| 4 | +0.4749 |
| 5 | +0.5365 |

Paired bootstrap interval: [+0.3992, +0.5328].

### Factor-conditioned primary contrasts

These secondary intervals are not multiplicity-adjusted.

#### difficulty

| Level | Structural auxiliary − concrete `test_combo` accuracy |
|---|---:|
| `easy` | +0.5146 [+0.3914, +0.6378] |
| `hard` | +0.3654 [+0.1565, +0.5742] |
| `mixed` | +0.5205 [+0.2396, +0.8015] |

#### capacity

| Level | Structural auxiliary − concrete `test_combo` accuracy |
|---|---:|
| `moderate` | +0.4318 [+0.3258, +0.5378] |
| `tight` | +0.6467 [+0.5239, +0.7696] |
| `wide` | +0.3220 [-0.0006, +0.6446] |

## Interpretation

Task 2 confirms the registered behavioral prediction: under equivalent familiar-validation mastery, structural auxiliary supervision produces greater reuse on an independently locked rendering combination.
The held-quadruple and joint-shift outcomes delimit how far that result extends.
The development analysis independently linked the behavioral difference to
transferred probes, aligned rank geometry, and valid causal interventions.

This is a qualitative replication on a more compositional but related synthetic
task. Structural supervision supplies privileged information: the result shows
that training objectives can select reusable internal organization after benchmark
mastery, not that models spontaneously discover principles or that the effect
already generalizes to natural-language reasoning or alignment.
