# Released data card

## Summary

This repository contains checkpoint-level measurements from 750 small
transformer training runs across three registered studies and two relational
computations. Every run has a separately evaluated locked deployment set.
There are eleven primary checkpoints per run, producing 8,250 development rows
and 8,250 locked rows. V8 additionally contains 11,352 objective and gradient
diagnostic rows.

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

## Data roles

| Split | Role | Change relative to training |
|---|---|---|
| `source` | In-distribution validation | Fresh examples; same combinations, relation support, and difficulty support |
| `source_all` | Development generalization | Familiar domain combinations; expanded relational distances |
| `calib_combo` | Development generalization | Reserved alphabet–grammar combinations |
| `test_combo` | Primary locked deployment test | Independently reserved domain combinations |
| `held_pair` | Secondary locked test | Unseen latent pairs or quadruples in familiar combinations |
| `test_both` | Locked stress test | New domain combination and unseen latent case |

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

JSONL files are keyed by `config_id` and `step`. Each locked file contains only
locked outcomes and joins to its development trajectory by those two fields.

## Statistical unit

The independently randomized training run is the observational unit. Paired
seed contrasts first average the crossed factorial cells within each seed:
weight decay, difficulty, and capacity for V2; difficulty and capacity for Task
2. Checkpoints and evaluation cases are not treated as independent replicates.

## Known limitations

- The V2 and Task 2 structural objectives receive privileged latent labels and
  correspondence information. V8 localizes most of their transferable effect
  to correct cross-alphabet item correspondence.
- Both tasks use known synthetic generators and closely related underlying
  order/difference relations.
- In-distribution validation is at ceiling at the endpoint.
- Weight decay changed parameter norms but did not act as an effective
  behavioral compression intervention.
- Mechanistic measurements exploit known synthetic ground truth and will not
  transfer automatically to natural-language settings.
- The release supports no claim about normative alignment.

## Recommended use

The data are suitable for reproducing the registered contrasts, checking
alternative hierarchical summaries, studying post-mastery trajectories,
analyzing the V8 objective factorial, and developing diagnostics for reusable
internal structure. Any exploratory reanalyses should be labeled as such and
should preserve the distinction among training, development, and locked
splits.
