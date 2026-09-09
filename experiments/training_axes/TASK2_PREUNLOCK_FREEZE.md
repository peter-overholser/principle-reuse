# Task 2 pre-unlock freeze

Recorded 9 September 2026 before any Task 2 locked deployment evaluation.

| Artifact | SHA-256 |
|---|---|
| `task2_manifest.csv` | `8bed6bf9ebf1a5779d60c348f79b90997b450305cc36de2226b6009c043feaec` |
| `TASK2_PROTOCOL.md` | `c20c9e671c453183a3a8da79b6970ff98507c55972759a4706a096871d444f4f` |
| Scientific protocol content through the `Remote launch` heading | `fb7fa05c25c2bdb932a6e07acd4ac142772aee175b427388efff6f41b6f5b1b6` |
| `results_task2/*.jsonl` tree | `504a8503f2ee5c8297474f1181dae848f61e0cd0a0bdb0ba3d897e1525f04e7a` |
| `analyze_preunlock_task2.py` | `83f1e72574b7e03ada27c31d2d700075d7ec4af8aa9d2436b29b0847d6763208` |
| `task2_preunlock_contrasts.json` | `59507134dfc74a8b98c0c4e21a65cd0cc412d790c11298c7cdfeb5c70afd7c84` |
| `TASK2_PREUNLOCK_REPORT.md` | `d78794da6d5f2c742b7d0ca926f3506334d7e25ed692a986a6eb74038615e310` |

The result-tree digest is computed by sorting JSONL files by basename and, for
each file, hashing its basename, a null byte, its exact contents, and a final
null byte. The implementation is `tree_sha256` in
`analyze_preunlock_task2.py`.

Frozen primary locked estimand: the seed-paired structural-auxiliary minus
concrete accuracy difference on `test_combo` at step 10,000, conditional on the
already-established source-mastery equivalence. `held_pair` and `test_both` are
secondary. No run, seed, capacity, difficulty level, checkpoint, or exclusion
will be selected using locked outcomes.
