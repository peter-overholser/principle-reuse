# V9-A source-mastery repair protocol

Recorded after the registered 10,000-update V9-A feasibility pilot returned
`repair-source-mastery` and before any V9 locked role was constructed or
evaluated. This amendment preserves the original pilot report and does not
reinterpret its failed gate.

## Motivation

The 10k pilot produced the registered A50-plus-diverse feasibility signal on
both line and circle scaffolds, but at least one trajectory failed the
all-runs source-mastery rule. The circle imposed positive control also failed
its universal causal-validity requirement even though its behavioral
contrast, organization-score contrast, and all four behavioral signs passed.

The cheapest repair is a horizon extension. No architecture, data cell,
anchor set, loss, optimizer, learning rate, evaluation role, or threshold is
changed. Continuing the same trajectories also tests whether the emergent
advantage persists or merely reflects faster learning.

## Exact continuation

Continue all 64 pilot trajectories from their update-10,000 checkpoints to
update 20,000. Restore model parameters, auxiliary head, optimizer state,
NumPy RNG, CPU Torch RNG, and CUDA RNG exactly. The only permitted checkpoint
configuration difference is an increase of `steps` from 10,000 to 20,000.

Retain the original checkpoints and add evaluations at updates 12,000,
15,000, and 20,000. The original 10k results, report, and gate JSON are copied
to `v9_pilot_10k_archive/` before continuation. No locked dataset is
constructed during training or repair analysis.

## Repair gate

The repair endpoint is update 20,000. Main training is authorized only if all
of the following hold:

1. every run has source accuracy at least 0.95 at updates 15,000 and 20,000,
   and source Brier skill at least 0.80 at update 20,000;
2. on each scaffold, imposed minus `a00_confounded` H-dev-s non-anchor
   accuracy is at least +0.25 with all four paired differences positive,
   imposed organization exceeds baseline organization, and the imposed
   causal gate is valid in all four seeds;
3. on each scaffold, E-perm is at least 0.15 below imposed H-dev-s non-anchor
   accuracy;
4. on each scaffold, `a50_diverse` improves H-dev-s non-anchor accuracy by at
   least +0.10 with at least three of four paired differences positive and
   has higher organization than `a00_confounded`; and
5. exact-resume and deterministic implementation tests pass.

The thresholds are unchanged from the original pilot. If source mastery still
fails, stop and diagnose the individual trajectory. If the imposed causal
gate remains the only failure, do not weaken it on these seeds; register a
fresh-seed assay-reliability study. Any other failed gate follows the original
V9 decision map.

## Main study after a pass

If and only if the repair gate returns `run-main`, train the untouched main
seeds 512--523 from initialization for 20,000 updates. Main checkpoints are
0, 50, 100, 200, 400, 700, 1,000, 2,000, 4,000, 7,000, 10,000, 12,000,
15,000, and 20,000. Source mastery is required at 15,000 and 20,000, and the
registered source-equivalence and locked hierarchies are evaluated at 20,000.
All hypotheses, arms, locked roles, equivalence margins, and decision routes
otherwise remain those in `V9_PROTOCOL.md`.
