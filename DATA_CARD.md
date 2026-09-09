# Released data card

## Summary

This repository contains checkpoint-level measurements from 486 small
transformer training runs across two tasks, each with a separately evaluated
locked deployment set. There are eleven checkpoints per run, producing 5,346
development rows and 5,346 locked rows.

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

JSONL files are keyed by `config_id` and `step`. Each locked file contains only
locked outcomes and joins to its development trajectory by those two fields.

## Statistical unit

The independently randomized training run is the observational unit. Paired
seed contrasts first average the crossed factorial cells within each seed:
weight decay, difficulty, and capacity for V2; difficulty and capacity for Task
2. Checkpoints and evaluation cases are not treated as independent replicates.

## Known limitations

- The structural objective receives privileged latent correspondence labels.
- Both tasks use known synthetic generators and closely related underlying
  order/difference relations.
- In-distribution validation is at ceiling at the endpoint.
- Weight decay changed parameter norms but did not act as an effective
  behavioral compression intervention.
- Mechanistic measurements exploit known synthetic ground truth and will not
  transfer automatically to natural-language settings.
- The release supports no claim about normative alignment.

## Recommended use

The data are suitable for reproducing both registered contrasts, checking
alternative hierarchical summaries, studying post-mastery trajectories, and
developing diagnostics for reusable internal structure. Any exploratory
reanalyzes should be labeled as such and should preserve the distinction among
training, development, and locked splits.
