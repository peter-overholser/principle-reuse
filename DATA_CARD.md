# Released data card

## Summary

This repository contains measurements from 942 small-transformer training runs
across four registered studies. The V2, Task 2, and V8 studies contribute 8,250
development rows and 8,250 separately evaluated locked rows at eleven
checkpoints per run. V9 contributes 2,688 development rows from 192 runs at
fourteen checkpoints and 192 prospectively locked endpoint rows. V8 also
contains 11,352 objective and gradient-diagnostic rows.

The data are generated from a controlled synthetic relational task. They
contain no human subjects, personal information, web-scraped material, or model
outputs derived from private corpora.

## Experimental factors

### V2: direct order comparison

| Factor | Levels |
|---|---|
| Training objective | concrete, output invariant, structural auxiliary |
| Weight decay | none, high |
| Evidence difficulty | easy, mixed, hard |
| Transformer capacity | tight, moderate, wide |
| Paired seed | 0–5 |

The Cartesian product contains 324 registered configurations.

### Task 2: comparison of differences

| Factor | Levels |
|---|---|
| Training objective | concrete, output invariant, structural auxiliary |
| Evidence difficulty | easy, mixed, hard |
| Transformer capacity | tight, moderate, wide |
| Paired seed | 0–5 |

Weight decay is fixed at zero following its V2 behavioral null. The Cartesian
product contains 162 registered configurations.

### V8: fresh-lock objective decomposition

| Factor | Levels |
|---|---|
| Relational computation | direct order, comparison of differences |
| Objective components | embedding correspondence E, hidden matching H, shared gap readout G |
| Factorial objective | complete 2 x 2 x 2 crossing of E, H, and G, always with output invariance |
| Controls | concrete, permuted E, alphabet-specific G |
| Transformer | width 64, two blocks |
| Paired seed | 400–411 |

The Cartesian product contains 264 registered configurations. V8 uses new item
token permutations, latent holdout seeds, evaluation seeds, and rotated
development and locked alphabet roles. Its development and lock therefore do
not reuse the earlier studies' held combinations.

### V9-B: emergent sharing from anchors and coverage

| Factor | Levels |
|---|---|
| Latent scaffold | line, circle |
| Shared-anchor fraction | 0, 0.25, 0.50 |
| Lexicon--grammar coverage | high confounding, diverse |
| Controls | imposed correct correspondence, imposed permuted correspondence |
| Transformer | width 64, two blocks |
| Paired seed | 512–523 |

The Cartesian product contains 192 configurations. The six anchor/coverage
arms receive ordinary task labels only. Primary outcomes exclude every entity
that can serve as a shared anchor. V9-A stopped under its original universal
interim-mastery gate because one run had not reached criterion at update
15,000. V9-B was registered subsequently and used all frozen update-20,000
models after every endpoint satisfied mastery and source equivalence. A
commit--reveal salt fixed the concrete locked sample before evaluation.

## Data roles

| Split | Role | Change relative to training |
|---|---|---|
| `source` | In-distribution validation | Fresh examples; same combinations, relation support, and difficulty support |
| `source_all` | Development generalization | Familiar domain combinations; expanded relational distances |
| `calib_combo` | Development generalization | Reserved alphabet–grammar combinations |
| `test_combo` | Primary locked deployment test | Independently reserved domain combinations |
| `held_pair` | Secondary locked test | Unseen latent pairs or quadruples in familiar combinations |
| `test_both` | Locked stress test | New domain combination and unseen latent case |
| `dev_s` / `lock_s` | V9 structural-role test | A lexicon seen with only one source grammar is evaluated under a nonlocal grammar |
| `dev_joint` / `lock_joint` | V9 joint stress test | Structural-role shift plus held latent pair |

## Files

- `v2_manifest.csv`: registered run configurations.
- `results_v2/*.jsonl`: development trajectory for each configuration.
- `locked_results_v2/*.jsonl`: locked trajectory for each configuration.
- `v2_cells.csv`: development endpoint summaries.
- `v2_locked_cells.csv`: joined endpoint and mastery-matched summaries.
- `v2_locked_contrasts.json`: registered contrasts and intervals.
- `V2_LOCKED_REPORT.md`: human-readable confirmatory report.
- `task2_manifest.csv`: registered difference-comparison configurations.
- `results_task2/*.jsonl`: Task 2 development trajectories.
- `locked_results_task2/*.jsonl`: Task 2 locked trajectories.
- `task2_locked_cells.csv`: joined Task 2 endpoint and mastery summaries.
- `task2_locked_contrasts.json`: registered Task 2 contrasts and intervals.
- `TASK2_LOCKED_REPORT.md`: human-readable Task 2 confirmatory report.
- `v8_manifest.csv`: registered V8 configurations.
- `results_v8/*.jsonl`: V8 development trajectories.
- `results_v8/_diagnostics/*.jsonl`: V8 loss and gradient diagnostics.
- `locked_results_v8/*.jsonl`: V8 locked trajectories.
- `v8_preunlock_freeze.json`: machine-readable V8 development freeze.
- `v8_locked_analysis.json`: machine-readable V8 confirmatory analysis.
- `V8_LOCKED_REPORT.md`: human-readable V8 confirmatory report.
- `v9_manifest.csv`: registered 192-run V9 configuration grid.
- `results_v9/*.jsonl`: fourteen-checkpoint V9 development trajectories.
- `v9_preunlock_freeze.json`: machine-readable stopped V9-A freeze.
- `v9b_preunlock_freeze.json`: prospective V9-B endpoint and integrity freeze.
- `v9b_lock_commitment.json` and `v9b_lock_reveal.txt`: the public
  commit--reveal pair for the concrete locked instance.
- `locked_results_v9b/*.jsonl`: one locked update-20,000 endpoint per run.
- `v9b_locked_analysis.json` and `V9B_LOCKED_REPORT.md`: machine-readable and
  human-readable V9-B confirmatory results.

JSONL files are keyed by `config_id` and `step`. Each locked file contains only
locked outcomes and joins to its development trajectory by those two fields.

## Statistical unit

The independently randomized training run is the observational unit. Paired
seed contrasts first average the crossed factorial cells within each seed:
weight decay, difficulty, and capacity for V2; difficulty and capacity for Task
2. V8 and V9 use paired seed-level arm contrasts directly. Checkpoints and
evaluation cases are not treated as independent replicates.

## Known limitations

- The V2 and Task 2 structural objectives receive privileged latent labels and
  correspondence information. V8 localizes most of their transferable effect
  to correct cross-alphabet item correspondence.
- Both tasks use known synthetic generators and closely related underlying
  order/difference relations.
- V9 adds a circular scaffold and ordinary-label emergence, but remains a
  small synthetic study. It does not establish the same process in pretrained
  language models or natural corpora.
- V9-B reuses development-selected training seeds with a new untouched outcome
  role; it is not an independent retraining replication.
- The V9 endpoint result does not yet distinguish durable solution selection
  from faster development followed by eventual catch-up.
- In-distribution validation is at ceiling at the endpoint.
- Weight decay changed parameter norms but did not act as an effective
  behavioral compression intervention.
- Mechanistic measurements exploit known synthetic ground truth and will not
  transfer automatically to natural-language settings.
- The release supports no claim about normative alignment.

## Recommended use

The data are suitable for reproducing the registered contrasts, checking
alternative hierarchical summaries, studying post-mastery trajectories,
analyzing the V8 objective factorial, examining anchor and coverage effects in
V9, and developing diagnostics for reusable internal structure. Any
exploratory reanalyses should be labeled as such and should preserve the
distinction among training, development, and locked splits.
