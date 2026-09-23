# V9-R2 fresh-seed causal-assay reliability protocol

Recorded after the registered V9 20k repair returned `stop-invalid-assay` and
before any V9 locked role was constructed or evaluated. The sole failed repair
condition was universal causal positive-control validity for `circle/imposed`:
three of four seeds passed. Source mastery, behavioral positive controls,
semantic specificity, and the A50-plus-diverse feasibility signal passed on
both scaffolds.

V9-R2 does not overturn either stopped gate. It estimates whether the causal
assay is sufficiently reliable on fresh circle models to support a
proportion-based validity rule in the untouched main study.

## Training screen

Train only the circle scaffold under three fixed arms:

- `a00_confounded`, the emergent baseline;
- `imposed`, the E+G positive control; and
- `a50_diverse`, the registered emergent treatment.

Use fresh training seeds 504--511, giving 24 runs. Architecture, data roles,
optimizer, batch size, learning rate, objective definitions, and the 20,000
update horizon are unchanged. Evaluate the same 14 development checkpoints as
the repaired V9 design. No locked role may be constructed or evaluated.

## Reliability assay

At the frozen update-20,000 checkpoint, run four independently seeded
development-only causal assays per model. Each replicate constructs fresh
source and H-dev-s evaluation batches, uses 4,096 examples per split, 1,024
opposite-label intervention pairs, and 16 random directions. Replicate seeds
are fixed by the evaluator before execution.

A replicate is valid under the original rule: target-to-target intervention
accuracy is at least 0.75 and exceeds its random-direction control by at least
0.05. A training seed is reliable when at least three of its four replicates
are valid. Training seed, not assay replicate, remains the unit of inference.

## Gate

V9-R2 authorizes the untouched main study only if:

1. every run has source accuracy at least 0.95 at updates 15,000 and 20,000,
   and source Brier skill at least 0.80 at update 20,000;
2. imposed minus baseline H-dev-s non-anchor accuracy has mean at least +0.25,
   a 95% paired interval above zero, and eight of eight positive seeds;
3. A50-diverse minus baseline H-dev-s non-anchor accuracy has mean at least
   +0.10, a 95% paired interval above zero, and at least seven of eight
   positive seeds;
4. A50-diverse minus baseline organization score has a 95% paired interval
   above zero;
5. no more than two of eight baseline seeds are reliable, at least six of
   eight imposed seeds are reliable, and at least seven of eight A50-diverse
   seeds are reliable;
6. for imposed and A50-diverse separately, mean target-to-target accuracy is
   at least 0.75, the paired-seed interval for target-to-target minus random
   control is above zero, and its mean is at least +0.10; and
7. for imposed and A50-diverse separately, the paired-seed interval for
   source-to-target minus random control is above zero and its mean is at
   least +0.10.

Failure of source or behavioral gates stops V9. Failure of the reliability
gate means the single-direction intervention is not a dependable confirmatory
measure for the circle scaffold; behavioral emergence may still be studied in
a separately registered design, but the present V9 main is not authorized.

## Main-study amendment after a pass

If and only if V9-R2 returns `run-main`, the untouched seeds 512--523 are
trained under the existing 192-run, two-scaffold V9 manifest. The behavioral,
source-equivalence, and locked hypotheses are unchanged. The organizational
hierarchy replaces the brittle requirement that all twelve primary seeds pass
the causal gate with the reliability-calibrated requirement that at least ten
of twelve pass. Invalid runs continue to contribute zero to the registered
organization score, and all probe, CKA, causal, and validity components are
reported separately.
