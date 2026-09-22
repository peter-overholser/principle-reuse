# V8 fresh-lock component-ablation report

Decision route: **embedding-tying-dominates**.

## Endpoint locked means

| Task | Arm | Accuracy | Brier | Shortcut agreement | Shortcut probability | Joint shift |
|---|---|---:|---:|---:|---:|---:|
| `order` | `concrete` | 0.435 | -1.253 | 0.565 | 0.565 | 0.378 |
| `order` | `eperm` | 0.420 | -1.295 | 0.580 | 0.580 | 0.392 |
| `order` | `gsep` | 0.505 | -0.929 | 0.495 | 0.499 | 0.469 |
| `order` | `e0h0g0` | 0.426 | -1.284 | 0.574 | 0.574 | 0.412 |
| `order` | `e0h0g1` | 0.474 | -1.057 | 0.526 | 0.527 | 0.491 |
| `order` | `e0h1g0` | 0.372 | -1.512 | 0.628 | 0.628 | 0.359 |
| `order` | `e0h1g1` | 0.408 | -1.344 | 0.592 | 0.591 | 0.420 |
| `order` | `e1h0g0` | 0.934 | 0.741 | 0.066 | 0.066 | 0.849 |
| `order` | `e1h0g1` | 0.995 | 0.982 | 0.005 | 0.006 | 0.973 |
| `order` | `e1h1g0` | 0.915 | 0.659 | 0.085 | 0.085 | 0.854 |
| `order` | `e1h1g1` | 0.991 | 0.970 | 0.009 | 0.009 | 0.980 |
| `differences` | `concrete` | 0.506 | -0.929 | 0.494 | 0.495 | 0.510 |
| `differences` | `eperm` | 0.682 | -0.205 | 0.318 | 0.318 | 0.690 |
| `differences` | `gsep` | 0.440 | -1.204 | 0.560 | 0.561 | 0.451 |
| `differences` | `e0h0g0` | 0.491 | -0.985 | 0.509 | 0.509 | 0.493 |
| `differences` | `e0h0g1` | 0.625 | -0.473 | 0.375 | 0.377 | 0.617 |
| `differences` | `e0h1g0` | 0.400 | -1.356 | 0.600 | 0.599 | 0.408 |
| `differences` | `e0h1g1` | 0.583 | -0.636 | 0.417 | 0.418 | 0.578 |
| `differences` | `e1h0g0` | 0.996 | 0.985 | 0.004 | 0.005 | 0.994 |
| `differences` | `e1h0g1` | 0.997 | 0.991 | 0.003 | 0.003 | 0.997 |
| `differences` | `e1h1g0` | 0.992 | 0.967 | 0.008 | 0.008 | 0.990 |
| `differences` | `e1h1g1` | 0.988 | 0.956 | 0.012 | 0.012 | 0.985 |

## Sequential hierarchy

### order

- `full_minus_invariant_accuracy`: +0.565 [+0.389, +0.741]; positive seeds 12/12.
- `full_minus_invariant_brier`: +2.254 [+1.542, +2.966]; positive seeds 12/12.
- `eonly_minus_full_accuracy`: -0.057 [-0.087, -0.027]; positive seeds 1/12.
- `nonembedding_minus_invariant_accuracy`: -0.018 [-0.137, +0.101]; positive seeds 7/12.
- `nonembedding_minus_invariant_brier`: -0.060 [-0.534, +0.414]; positive seeds 7/12.
- `eperm_minus_eonly_accuracy`: -0.514 [-0.655, -0.373]; positive seeds 0/12.
- `eperm_minus_concrete_accuracy`: -0.015 [-0.180, +0.151]; positive seeds 6/12.
- `gshared_minus_gseparate_accuracy`: -0.031 [-0.257, +0.194]; positive seeds 7/12.

### differences

- `full_minus_invariant_accuracy`: +0.497 [+0.275, +0.718]; positive seeds 10/12.
- `full_minus_invariant_brier`: +1.942 [+1.059, +2.825]; positive seeds 10/12.
- `eonly_minus_full_accuracy`: +0.008 [+0.001, +0.015]; positive seeds 9/12.
- `nonembedding_minus_invariant_accuracy`: +0.091 [-0.218, +0.401]; positive seeds 6/12.
- `nonembedding_minus_invariant_brier`: +0.349 [-0.895, +1.593]; positive seeds 6/12.
- `eperm_minus_eonly_accuracy`: -0.314 [-0.479, -0.148]; positive seeds 0/12.
- `eperm_minus_concrete_accuracy`: +0.176 [+0.028, +0.325]; positive seeds 9/12.
- `gshared_minus_gseparate_accuracy`: +0.186 [-0.075, +0.446]; positive seeds 10/12.

## Matched-source-loss checkpoint contrasts

- `order/full_minus_invariant_accuracy`: +0.148 [-0.026, +0.321].
- `order/nonembedding_minus_invariant_accuracy`: -0.081 [-0.195, +0.034].
- `differences/full_minus_invariant_accuracy`: +0.262 [+0.039, +0.485].
- `differences/nonembedding_minus_invariant_accuracy`: +0.095 [-0.103, +0.294].

## Integrity

- Development freeze SHA-256: `e02a7a69cacbf6147c9d719a2d146b7b6e074047e598cbd0d4adb36978611478`
- Locked tree SHA-256: `7165bcbe1a830f3df5a35a678de117796ec371f40094975abd072dec0e209e70`
- Analysis SHA-256: `c960c06596e36585878e7f03962d2f9728d501b882790193159d736710cd251b`
- Complete locked files: `264`
