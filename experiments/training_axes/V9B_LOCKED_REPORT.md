# V9-B prospective locked Selection/Emergence report

V9-A remains stopped under its original interim-mastery gate.

Decision route: **emergence-supported**.

## Hierarchy

| Scaffold | Assay | Behavior | Organization | Anchor dose | Supported |
|---|:---:|:---:|:---:|:---:|:---:|
| `line` | yes | yes | yes | yes | yes |
| `circle` | yes | yes | yes | yes | yes |

## Registered contrasts

### line

- `imposed_minus_baseline_accuracy`: +0.265 [+0.155, +0.375]; positive seeds 11/12.
- `primary_minus_baseline_accuracy`: +0.334 [+0.264, +0.403]; positive seeds 12/12.
- `primary_minus_baseline_O`: +0.717 [+0.633, +0.800]; positive seeds 12/12.
- `primary_minus_baseline_probe`: +0.445 [+0.308, +0.582]; positive seeds 12/12.
- `primary_minus_baseline_cka`: +0.793 [+0.746, +0.839]; positive seeds 12/12.
- `primary_minus_baseline_causal`: +nan [+nan, +nan]; positive seeds 0/12.
- `anchor_trend_diverse`: +0.196 [+0.054, +0.339]; positive seeds 11/12.
- `diversity_at_a00`: +0.138 [-0.014, +0.289]; positive seeds 9/12.
- `diversity_at_a50`: +0.091 [+0.049, +0.134]; positive seeds 11/12.
- `anchor_by_diversity_interaction`: -0.046 [-0.202, +0.110]; positive seeds 4/12.
- `primary_minus_baseline_joint`: +0.317 [+0.174, +0.461]; positive seeds 12/12.

### circle

- `imposed_minus_baseline_accuracy`: +0.264 [+0.147, +0.382]; positive seeds 11/12.
- `primary_minus_baseline_accuracy`: +0.275 [+0.203, +0.347]; positive seeds 12/12.
- `primary_minus_baseline_O`: +0.688 [+0.611, +0.764]; positive seeds 12/12.
- `primary_minus_baseline_probe`: +0.471 [+0.383, +0.559]; positive seeds 12/12.
- `primary_minus_baseline_cka`: +0.680 [+0.592, +0.768]; positive seeds 12/12.
- `primary_minus_baseline_causal`: +nan [+nan, +nan]; positive seeds 0/12.
- `anchor_trend_diverse`: +0.246 [+0.146, +0.346]; positive seeds 12/12.
- `diversity_at_a00`: +0.029 [-0.085, +0.143]; positive seeds 7/12.
- `diversity_at_a50`: +0.189 [+0.114, +0.264]; positive seeds 12/12.
- `anchor_by_diversity_interaction`: +0.160 [+0.026, +0.294]; positive seeds 8/12.
- `primary_minus_baseline_joint`: +0.176 [+0.048, +0.304]; positive seeds 9/12.

## Endpoint means

| Scaffold | Arm | Lock-s nonanchor | Joint | O | Probe | CKA | Causal |
|---|---|---:|---:|---:|---:|---:|---:|
| `line` | `a00_confounded` | 0.630 | 0.566 | 0.156 | 0.266 | 0.113 | nan |
| `line` | `a25_confounded` | 0.711 | 0.722 | 0.494 | 0.394 | 0.339 | nan |
| `line` | `a50_confounded` | 0.872 | 0.856 | 0.810 | 0.661 | 0.770 | 0.999 |
| `line` | `a00_diverse` | 0.767 | 0.790 | 0.466 | 0.341 | 0.453 | nan |
| `line` | `a25_diverse` | 0.902 | 0.874 | 0.674 | 0.550 | 0.639 | nan |
| `line` | `a50_diverse` | 0.963 | 0.884 | 0.872 | 0.711 | 0.905 | 1.000 |
| `line` | `imposed` | 0.894 | 0.819 | 0.846 | 0.935 | 0.692 | nan |
| `line` | `eperm` | 0.644 | 0.507 | 0.197 | 0.399 | 0.097 | nan |
| `circle` | `a00_confounded` | 0.615 | 0.499 | 0.113 | 0.184 | 0.067 | nan |
| `circle` | `a25_confounded` | 0.740 | 0.412 | 0.488 | 0.360 | 0.354 | nan |
| `circle` | `a50_confounded` | 0.701 | 0.455 | 0.628 | 0.591 | 0.461 | nan |
| `circle` | `a00_diverse` | 0.644 | 0.426 | 0.184 | -0.037 | 0.190 | nan |
| `circle` | `a25_diverse` | 0.839 | 0.449 | 0.587 | 0.276 | 0.570 | nan |
| `circle` | `a50_diverse` | 0.890 | 0.675 | 0.801 | 0.655 | 0.747 | 1.000 |
| `circle` | `imposed` | 0.880 | 0.680 | 0.771 | 0.922 | 0.641 | nan |
| `circle` | `eperm` | 0.599 | 0.537 | 0.091 | 0.117 | 0.153 | nan |

## Integrity

- V9-B freeze SHA-256: `1c37741c289c7bfeae9204194a7023d7f1c9aca843a819c2b848ee9ec208123e`
- Lock commitment: `32981c290d654269b5b364c608493d9948538ad9a14252846e4dc8f06aaded93`
- Locked tree SHA-256: `d7d23f7d14405803e71842f6dacd54d0a232349e5e5cb254c3ac2f2201743450`
- Analysis SHA-256: `4d7ab0cfa038c58bf7261a6ac89931982b4149432b3666379927e13e05006714`
- Complete locked files: `192`
