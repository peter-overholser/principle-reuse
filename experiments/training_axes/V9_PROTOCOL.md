# V9-A protocol: can reusable organization emerge from data structure?

Recorded after the V8 fresh lock and before any V9 model is trained. V8 showed
that correct semantic embedding correspondence is the dominant component of
the imposed structural objective. V9-A therefore asks whether two ordinary
properties of a training distribution--shared lexical anchors and reduced
lexicon--syntax confounding--can select the same kind of transferable solution
without a correspondence loss.

V9-A tests Selection and Emergence. It does not yet test Leverage,
Entrenchment, Compounding, or Use-dependence. V9-B will branch later updates
from the frozen V9-A trunks without changing this protocol's results.

## Scaffolds and model

Two latent geometries are trained from scratch:

- `line`: twelve ordered entities; the base label is the sign of a rank gap;
- `circle`: twelve phases; the base label is the sign of the shortest directed
  displacement, excluding zero and antipodal pairs.

Each scaffold uses eight lexicons and four grammars. Grammar 3 is marked only
near the beginning of the sequence while its operand interpretation occurs
later, making it the nonlocal composition test. Models are width-64,
two-block transformers trained for 10,000 updates with batch size 256,
learning rate 0.001, no weight decay, and no bottleneck.

## Data roles

The roles are fixed before training:

- common constrained source validation: the high-confound core cells;
- `H-dev-x`: exchangeable development combinations;
- `H-dev-s`: a development combination whose lexicon is seen under only one
  source grammar;
- `H-lock-x`: fresh exchangeable locked combinations;
- `H-lock-s`: fresh one-grammar locked combinations;
- `H-lat`: held latent pairs inside source cells;
- `H-joint`: the H-lock-s rendering shift combined with held latent pairs.

Every task generator returns the latent, rendering cell, held-pair status, and
anchor exposure. Development construction never emits a locked role.

## Manipulations

The emergent grid crosses anchor fraction with coverage diversity:

```text
anchor fraction alpha in {0, 0.25, 0.50}
coverage in {high-confound, diverse}
```

The high-confound source design has normalized lexicon--grammar mutual
information approximately 0.55. The diverse design expands grammar coverage
for the exchangeable lexicons and reduces it to approximately 0.19. Total
updates and examples are fixed, so diverse arms spread the same data budget
over more cells.

Three and six of the twelve entity tokens are shared across lexicons in the
0.25 and 0.50 anchor conditions. Anchor sets are nested and fixed. The primary
anchor outcome is evaluated on examples containing **none of the six possible
anchor entities**. Direct token overlap therefore cannot solve the primary
test.

All six emergent arms receive only ordinary binary task labels. Two controls
are added:

- `imposed`: the V8-derived `I+E+G` objective under high confounding and no
  anchors;
- `eperm`: the same objective with a different fixed incorrect item
  correspondence for every lexicon.

Hidden-state matching is omitted because V8 found no incremental contribution.
P-context is deferred because it changes information flow and requires a
separate view- and label-budget-matched design.

The resulting main allocation is:

```text
2 scaffolds x 8 arms x 12 paired seeds = 192 trunks.
```

Main seeds are 512--523. A feasibility pilot uses seeds 500--503 and cannot be
pooled with the main study.

## Measurements

Behavior is measured at updates 0, 50, 100, 200, 400, 700, 1,000, 2,000,
4,000, 7,000, and 10,000. Primary behavior is accuracy on the common
non-anchor subset of H-dev-s or H-lock-s. Accuracy, Brier skill, and log loss
on the full split, H-dev-x/H-lock-x, H-lat, and H-joint are secondary.

The organization battery excludes raw embedding correspondence, because that
is the imposed E loss, and excludes the exact G regression loss. It contains:

1. transferred probes of an assay-reserved midpoint/phase functional, fit on
   source renderings and evaluated without refitting on held renderings (not
   the signed gap optimized by G and not a later R1 outcome);
2. linear CKA of paired final-hidden representations across renderings;
3. guarded cross-rendering interchange, reported only when the target-domain
   direction beats its random-direction controls.

The bounded organization score is

```text
O = mean(clip(probe R2, 0, 1), CKA, clip(causal normalized effect, 0, 1)).
```

An invalid causal gate contributes zero to O and is flagged separately. A
confirmatory arm cannot support organizational emergence unless its own causal
gate is valid. The three components are always reported separately; O alone
is not treated as a mechanistic proof.

## Pilot gate

The four-seed pilot contains no locked evaluation and makes no confirmatory
claim. Proceed to the main study only if:

1. every run reaches source accuracy at least 0.95 at updates 7,000 and 10,000
   and source Brier skill at least 0.80 at update 10,000;
2. imposed minus baseline H-dev-s non-anchor accuracy is at least +0.25 on
   each scaffold, with all four paired differences positive;
3. imposed O exceeds baseline O on each scaffold and the imposed causal gate
   is valid in every seed;
4. E-perm is at least 0.15 below imposed H-dev-s non-anchor accuracy on each
   scaffold;
5. the fixed main arm `a50_diverse` improves H-dev-s non-anchor accuracy by at
   least +0.10 on each scaffold, with at least three of four paired
   differences positive, and has higher mean O than baseline; and
6. generator, role-disjointness, anchor-exclusion, and resume tests pass.

Failure routes are fixed: implementation failure is repaired and rerun;
failure of the imposed control stops V9-A as an invalid assay; success of the
imposed control but failure of every emergent arm is informative and routes
the next experiment to the separately matched P-context channel.

## Main validity and source fiber

Every main run must satisfy the pilot source-mastery thresholds. At update
10,000, each arm is paired against the no-anchor/high-confound baseline on the
common constrained source set. The 95% intervals must fit inside:

| Metric | Equivalence margin |
|---|---:|
| Accuracy | +/-0.02 |
| Brier skill | +/-0.03 |
| Log loss | +/-0.02 |
| Near-margin log loss | +/-0.05 |

Failure blocks locked release. A fixed source-log-loss-matched checkpoint
analysis uses the same target and tie rule as V8: among checkpoints with
source accuracy at least 0.95, minimize distance to source log loss 0.01 and
break ties toward the earlier checkpoint.

## Locked hierarchy

The primary arm is fixed in advance as `a50_diverse`; the baseline is
`a00_confounded`. Within each scaffold:

1. **Assay replication:** imposed minus baseline H-lock-s non-anchor accuracy
   has a 95% interval wholly above zero.
2. **Behavioral emergence:** `a50_diverse` minus baseline H-lock-s non-anchor
   accuracy has a 95% interval wholly above zero.
3. **Organizational emergence:** `a50_diverse` minus baseline O has a 95%
   interval wholly above zero, with at least two of the three component
   differences positive.
4. **Dose response:** within diverse coverage, the paired linear anchor trend
   over 0, 0.25, and 0.50 is positive.
5. **Structural replication:** steps 2 and 3 must hold on both scaffolds and
   H-lock-s. H-lock-x alone cannot support the claim.

The cross-scaffold route is `emergence-supported` only if all five steps pass
on both scaffolds. If step 1 passes but later steps fail, the route is
`imposed-only`. If step 1 fails, the route is `assay-failed`. Mixed scaffold
results are reported as `scaffold-specific`.

Anchor-only, diversity-only, interaction, E-perm, full-split, H-lat,
H-joint, calibration, trajectory, and matched-checkpoint analyses are fixed
secondary results and cannot rescue the primary hierarchy.

## Interpretation limits

Success would show that ordinary distributional structure can select a
transferable organization without explicit correspondence labels. It would
not show that anchors or diversity are necessary, nor that the organization
improves later learning; V9-B tests the latter from these frozen trunks.

Failure would be equally informative: together with V8, it would indicate
that privileged semantic correspondence can impose the organization at this
scale while the registered weakly emergent channels do not reliably discover
it.
