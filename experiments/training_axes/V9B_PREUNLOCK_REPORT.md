# V9-B prospective endpoint freeze

V9-A remains stopped. No V9-B locked example was constructed or evaluated.

Decision: **unlock-fresh-lock**.

## Eligibility

| Gate | Pass |
|---|:---:|
| All 192 endpoints mastered | yes |
| Registered source equivalence | yes |
| V9-A stopped provenance verified | yes |

Endpoint failures: **0**.

## Source equivalence

### line

- `a25_confounded`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss -0.000 [-0.000, +0.000], source_log_loss_near -0.000 [-0.000, +0.000].
- `a50_confounded`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss -0.000 [-0.000, +0.000], source_log_loss_near -0.000 [-0.000, +0.000].
- `a00_diverse`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [+0.000, +0.000], source_log_loss_near +0.000 [+0.000, +0.000].
- `a25_diverse`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [+0.000, +0.000], source_log_loss_near +0.000 [+0.000, +0.000].
- `a50_diverse`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [-0.000, +0.000], source_log_loss_near +0.000 [-0.000, +0.000].
- `imposed`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [-0.000, +0.000], source_log_loss_near +0.000 [-0.000, +0.000].
- `eperm`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [+0.000, +0.000], source_log_loss_near +0.000 [+0.000, +0.000].

### circle

- `a25_confounded`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [-0.000, +0.000], source_log_loss_near +0.000 [-0.000, +0.000].
- `a50_confounded`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [-0.000, +0.000], source_log_loss_near +0.000 [-0.000, +0.000].
- `a00_diverse`: source_accuracy -0.002 [-0.005, +0.001], source_brier_skill -0.005 [-0.012, +0.002], source_log_loss +0.004 [-0.002, +0.010], source_log_loss_near +0.003 [-0.002, +0.009].
- `a25_diverse`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [+0.000, +0.000], source_log_loss_near +0.000 [+0.000, +0.000].
- `a50_diverse`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [+0.000, +0.000], source_log_loss_near +0.000 [+0.000, +0.000].
- `imposed`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [+0.000, +0.000], source_log_loss_near +0.000 [+0.000, +0.000].
- `eperm`: source_accuracy +0.000 [+0.000, +0.000], source_brier_skill +0.000 [+0.000, +0.000], source_log_loss +0.000 [+0.000, +0.000], source_log_loss_near +0.000 [+0.000, +0.000].

## Integrity

- v9b_protocol_sha256: `3df15ee3cf544de5da2009be6063a98710c59aff28c52ea649e65f14eb22be34`
- v9b_commitment_file_sha256: `c23328bdc937d4661fdc01e6af2e8b9253b22c6e30d4d943d22484651bcc1698`
- v9b_common_sha256: `855c7ecf279a57bae35d6569a9f94924c269c081f29953835aa6bd42c919cfa3`
- v9b_preunlock_analysis_sha256: `bd8e0c2a5211a6302a56961b1e028d365d7d235ef650bbc317638ea59a457f2b`
- v9b_locked_evaluator_sha256: `44d5f8794791c974f1f0c2ec247f2272153e2acb61d6e9ae9e901f2ed19bc94c`
- v9b_locked_all_evaluator_sha256: `859de06631737e25c371e6ec0b36072a8eb2f3086e9260701540ef7dc6503453`
- v9b_locked_analysis_sha256: `4d7ab0cfa038c58bf7261a6ac89931982b4149432b3666379927e13e05006714`
- v9a_freeze_sha256: `68bace8d3cfbd7be81d0c7b75925c118964bfa093cba574ba05c91f004026575`
- manifest_sha256: `1ef37d673bbcbea3ced44ad91d057a132c6161bbff2a7cda5b9f77eccb1cac29`
- task_sha256: `5423fea51baa596ef88d767bea8314d385c8dc8edc4204f2e193ae54f1a46a57`
- design_sha256: `e4f5b44b1e8ec0a2b5ee26011c2eefc5931f1a82af01ce273256f7bbf17490c7`
- metrics_sha256: `14d770bb641cf34055c1aaf6af1b3396739990c1b193a3ef2b58eb2236e7245e`
- model_sha256: `501621942a0f5658b0bac8352991d5cedd1e95d2a099b6b7f540435829d539f0`
- development_tree_sha256: `58f02993317c4983f4041f5e529b4dbfd6c460e642ea3e5b11ff7413d5ebf402`
- endpoint_checkpoint_tree_sha256: `d8f41a2ce7745d4700628a7e7ce5376f53da1643ca7792ae7ae660ccad367c1a`
- development_files: `192`
- endpoint_checkpoints: `192`

- Lock commitment: `32981c290d654269b5b364c608493d9948538ad9a14252846e4dc8f06aaded93`
