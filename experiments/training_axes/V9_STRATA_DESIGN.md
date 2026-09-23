# Strata / V9: an experimental corpus for the developmental thesis

Status: implementation design, revised after V8's locked route was
`embedding-tying-dominates`. V8 localized the imposed effect to correct
semantic embedding correspondence, with a shared latent readout providing the
small residual benefit on the order task. Strata asks the broader paper
question: whether ordinary data structure can select comparable reusable
organization, and whether that organization governs the reach, durability,
and later cost of learning.

The V8 result changes the build order. `I+E+G` is retained only as an imposed
positive control, E-perm+G is its semantic control, and hidden matching is
removed. The first V9 run is no longer a broad auxiliary-loss factorial. It is
a focused Selection/Emergence experiment over anchor and diversity channels.
Leverage branches are trained later from the frozen V9 trunks, so a failure of
emergence cannot be hidden by an omnibus endpoint.

## Claims and evidential separation

Each claim has its own manipulation and outcome. No result is allowed to stand
in for another claim.

| Claim | Manipulation | Primary outcome |
|---|---|---|
| Selection | Stage-0 curriculum within a matched validation fiber | Independent organization score `O` |
| Leverage | Fixed later update classes applied to matched trunks | Sample cost and locked transport |
| Compounding | Ordered sequence of new rule families | Stage-by-stage reach gap |
| Entrenchment | Time spent on a shortcut before organization pressure | Cost of acquiring `O` |
| Use-dependence | Fraction of intervening stages that read the latent | Retention and re-access cost |
| Emergence | Anchors/diversity/pairing without privileged labels | `O` and later leverage |

Organization must be measured with assay-reserved functions and examples that
are disjoint from the outcome rules it predicts.

## Generator contract

Every instance is generated from independently registered seeds as

```text
(scaffold, latent world W, renderer G, rule R,
 coverage design C, supervision Sigma, curriculum K).
```

The generator emits tokens plus the complete latent record: values, rendering
coordinates, rule parameters, minimal-pair IDs, donor IDs for interventions,
and held-cell role. Analysis never reconstructs latent metadata from model
outputs.

Generator code and hashes of unrevealed locked seeds are committed before a
run. Development seeds cannot generate locked records. Locked seeds are
revealed only after the analysis and composite organization score are frozen.

### Corrections required before implementation

Several attractive-looking manipulations are not yet scientifically valid:

1. Merely putting paired examples in the same ordinary minibatch does not give
   a transformer a cross-example information path. Under an additive
   per-example loss, it changes optimizer noise but does not expose the pairing.
   `P-data` is therefore a negative batching control. A real emergent channel,
   `P-context`, places two renderings in the same sequence or episode and gives
   only an ordinary task label whose solution can exploit their common latent;
   it supplies no correspondence label or alignment loss.
2. The organization composite cannot assume that a line, circle, and tree
   should have the same representational geometry. Each scaffold receives a
   preregistered, scaffold-specific score `O_s`. Cross-scaffold synthesis uses
   standardized effects and replication logic, not raw pooled assay values.
3. “Minimal update class” is a registered hypothesis, not an oracle label. A
   source-only capacity screen must show that every compared update class can
   solve the rule before transport differences are interpreted.
4. A rotation of one alphabet's embeddings is not automatically a selective
   edit of organization; it can damage the whole function. Break/make results
   are called causal only if source repair returns the model to the fiber and
   the edit changes the target O assays while leaving registered non-O
   controls inside equivalence margins.
5. Exact values of `rho` may be impossible for a finite cell grid. The coverage
   solver must register attainable designs and report achieved mutual
   information; it may not round a requested value after seeing outcomes.

## Latent scaffolds

| Scaffold | Primary latent | Expected geometry | Secondary latent |
|---|---|---|---|
| S1 line | Rank and signed gap over 12 entities | One-dimensional order | Parity or color |
| S2 circle | Phase in Z12 | Circular/Fourier code | Second modulus |
| S3 tree | Node in a depth-4 tree | Hierarchical code | Depth/subtree |
| S4 group/automaton | At least 12 states under registered actions | State-tracking code | Counter |

S1--S3 are used to develop hypotheses. S4 is sealed as an entire scaffold:
its generator is registered but no S4 sample is created until all hypotheses,
metrics, exclusions, and thresholds have been frozen on S1--S3.

Six-state S3 is too easy to distinguish state tracking from memorization at
this model size. The sealed scaffold will therefore use a 12--24-state group
action or automaton with held action-sequence lengths, while preserving a
small exhaustively enumerable latent world.

## Rendering layer

Each latent is rendered under independently controlled lexicon, syntax,
format, and surface variables.

- **Lexicon:** eight disjoint alphabets. An anchor fraction
  `alpha in {0, .1, .3}` controls shared entity tokens.
- **Syntax:** canonical; dual order/operator; a local negation wrapper; and a
  canonical grammar whose only marker appears at the sequence start. The last
  condition tests nonlocal grammar access.
- **Format:** infix versus key-value/function-call form.
- **Surface:** filler templates, distractors, length variation, and optionally
  Zipfian token frequencies.

The initial implementation uses lexicon and syntax only. Format and richer
surface variation enter only after the core generator and assay gates pass.

## Coverage design

The coverage generator controls:

- `delta`: fraction of lexicon-by-syntax cells observed, from .25 to .75;
- `rho`: normalized mutual information between lexicon and syntax, from a
  balanced Latin design at 0 to deterministic confounding at 1;
- `kappa`: minimum grammars observed per alphabet, from 1 to 4;
- `alpha`: shared-token anchor fraction, from 0 to .3.

For fixed `delta`, designs are chosen to hit the target `rho`, rather than
using a convenient but incomparable pattern. The current two-grammar studies
are a high-confound special case.

Held roles are explicit:

- `H-dev`: development cells;
- `H-lock-x`: exchangeable held cells;
- `H-lock-s`: structurally different holds, such as a held syntax-format pair
  or an alphabet observed under only one grammar;
- `H-lat`: held latent values/queries within observed cells;
- `H-joint`: combined structural and latent shift.

A confirmatory claim requires H-lock-s, not merely H-lock-x.

## Rule families and assay/outcome separation

Rules are parameterized families over each scaffold. Each outcome rule is
tagged in advance with the least update class expected to solve it given the
ideal organization.

| Family | Line examples | Expected least update |
|---|---|---|
| R0 base | sign of gap | Stage-0 task |
| R1 linear | threshold or sign flip | Readout; flip may be wrapper-solvable |
| R2 one-nonlinearity | absolute gap, gap parity | Low-rank adapter/readout MLP |
| R3 piecewise | interval membership, three-way argmax | Adapter |
| R4 compositional | threshold XOR secondary latent | Adapter plus new feature |
| R5 latent-free | template, length, token presence | Any; disuse control |
| R6 conflicting | context-marked reversal of R0 | Context gate |

Circle, tree, and group receive analogous families appropriate to their
geometry. Even and odd thresholds, disjoint bands, or equivalent parameter
partitions separate assay rules from outcome rules. A probe of the exact rule
later used as an outcome is prohibited.

## Supervision channels

Channels are independently switchable and classified before analysis.

| Channel | Class | Signal |
|---|---|---|
| Y | Base | Task labels |
| P-data | Null control | Paired renderings share a minibatch but have no cross-example path |
| P-context | Emergent | Paired renderings share an episode; only an ordinary episode label is supplied |
| P-loss | Imposed | Hidden alignment on paired renderings; omitted from V9-A after V8 |
| Z | Imposed | Privileged latent regression |
| E | Imposed | Cross-rendering embedding correspondence |
| E-perm | Control | Incorrect fixed correspondence |
| A | Emergent | Shared anchor tokens |
| D | Emergent | Low confound and high grammar diversity |
| WD | Implicit | Weight decay in {0, .1, 1} |

V8 determined that H should be dropped from the first Strata build. The
registered imposed control is `I+E+G`; E is the dominant component and G is
retained because it closes the small order-task deficit. Emergent arms receive
Y only: including I would make them auxiliary-supervised rather than a clean
test of emergence.

## Curricula

Each stage specifies `(rule, coverage, supervision, update class, budget)`.

### K1: induction / Selection

Train Stage 0 while varying supervision, seed, data order, diversity, anchors,
and weight decay. Select checkpoints using source-only log-loss and a hard
near-margin split. Compare `O` only after the validation fiber is established.

### K2: entrenchment

Train on a shortcut-favoring `rho=1`, Y-only design for
`T0 in {0, 1k, 4k, 16k, 64k}` updates, then apply either an imposed or emergent
organization pressure. Primary outcome: updates and examples needed to reach a
fixed `O`, with failure by a fixed budget treated as censoring rather than
arbitrarily large cost.

Entrenchment also includes equal-total-compute controls, a fresh optimizer at
the switch, and matched no-shortcut pretraining. Otherwise an exposure effect
could be generic loss of plasticity or optimizer aging rather than commitment
to a shortcut.

### K3: leverage and compounding

Starting from matched Stage-0 trunks, train R1 -> R2 -> R3 -> R4. At each
stage use disjoint outcome parameters and train on only part of the rendering
cells. Cross readout-only, registered-rank adapters at embedding/pre-block/
post-block locations, full fine-tuning, and full fine-tuning with replay.

Raw accuracy gaps are not the compounding estimand because ceiling effects can
make a persistent advantage look smaller. The primary compounding outcomes are
log sample-cost ratios and normalized excess log-loss at fixed budgets. Each
stage also has a branch beginning from performance-matched checkpoints, so a
later advantage is not merely inherited task accuracy.

### K4: use-dependence

Interleave K3 with R5 latent-free stages. Hold total updates fixed while the
fraction of stages that require the latent is
`lambda in {0, .25, .5, 1}`. Measure `O` during disuse and the cost of
re-accessing the latent afterward.

Include a no-update elapsed-time control and a latent-compatible task that
activates the same inputs without reading the registered principle. This
separates disuse from generic parameter drift and from interference caused by
the particular R5 objective.

### K5: interference

Train R6 with and without an explicit context gate, then return to an R1 rule.
Measure whether the old organization is stored but bypassed, destroyed, or
quickly re-accessed.

## Organization battery

The registered composite `O` combines standardized, direction-corrected
components from O1--O4 only. Weights, missing-value rules, and minimum valid
intervention coverage are frozen before any leverage outcome is opened.

| Assay | Measurement |
|---|---|
| O1 | Full transferred-probe matrix across renderings; mean and worst off-diagonal |
| O2 | Principal-angle and CKA alignment on registered minimal pairs |
| O3 | Guarded interchange intervention with target positive control and random directions |
| O4 | Dimension/rank of the shared latent subspace |
| O5 | Plasticity covariates: effective rank and dormant units; Hessian/gradient-noise estimates only in a prespecified subsample |

O5 is explanatory and is excluded from the organization composite. O1 alone
never establishes organization because it is too close to readout reach.

## Outcome battery

- Y1: updates/examples to a fixed source criterion;
- Y2: accuracy, Brier skill, and log-loss on H-lock-x and H-lock-s;
- Y3: retention of earlier stages in the complete model and protected core;
- Y4: re-access cost after disuse or interference;
- Y5: agreement with prespecified transforms of the core, including identity,
  negation, threshold shifts, and context gating;
- Y6: reach profile, the fraction of registered rules solved by each update
  class within a fixed budget.

## Stronger causal tests

### Organization dial

Sweep one imposed channel weight or one emergent parameter (`alpha`, `rho`) at
a fixed validation target. Estimate outcome as a function of independently
measured `O`. This is a dose-response/mediation analysis. The exclusion
assumption needed to call the dial an instrument is not guaranteed and will be
stated explicitly.

### Break/make edits

Apply a post-hoc edit, followed by a small fixed source-only repair budget that
returns the model to the validation fiber:

- **Break:** independently rotate selected rendering-specific embedding
  blocks in a high-O model;
- **Make:** Procrustes-align corresponding embedding blocks in a low-O model.

Run K3 afterward. A credible causal signature is symmetric: break reduces
leverage and make confers some leverage. Repair cost and the change in every O
component are reported; edits that do not selectively change O fail their
manipulation check.

Because both edits can alter ordinary task geometry, the preferred first break
test is layerwise projection or replacement of the independently identified
shared subspace, with a norm- and rank-matched random-subspace control. The
embedding rotations remain secondary unless they pass the stronger
selectivity checks above.

## Build order and gates

### V9-A: corpus and Selection/Emergence core

Implement S1 and S2; lexicon and syntax; K1; Y, `I+E+G`, E-perm+G, A, and D;
O1--O3; and Y2. Anchors are evaluated on a common subset containing no anchor
tokens, so success requires anchors to organize the unanchored vocabulary and
cannot be attributed to direct token overlap. P-data and P-context are deferred
until the clean A/D result is known: P-data is only a batching null, while
P-context changes sequence length, label count, and information flow and
therefore deserves its own matched protocol. Before scaling, require:

1. exact generator reproducibility and disjoint role tests;
2. source mastery and registered fiber equivalence;
3. valid O3 positive controls in every confirmatory arm;
4. no assay/outcome rule-parameter overlap;
5. the imposed positive control must improve development structural holds on
   both scaffolds, and E-perm must fail to reproduce it.

The first confirmatory allocation is the registered 3-by-2 anchor-by-diversity
grid plus the two V8-derived controls, with 12 paired seeds. A four-seed pilot
checks implementation and assay feasibility only. Do not launch later update
branches until this gate passes.

### V9-B: leverage from frozen V9-A trunks

Branch the baseline, strongest preregistered emergent arm, imposed control,
and semantic control into the R1 stage of K3. Use assay-reserved and
outcome-reserved parameters, a source-only update-capacity screen, and fixed
readout/adapter budgets. This is the first point at which V9 makes a Leverage
claim.

### V9-C: entrenchment and durability

Add K2, K4, K5, S3, and break/make edits. Claims remain independent: a failed
emergence contrast does not gate entrenchment, for example.

### V9-D: structural replication

Add format and H-lock-s, freeze all hypotheses, then reveal S4. The S4 outcome
is reported once, regardless of direction.

### V9-E: natural bridge

Use exact natural latents with multiple renderings:

- dates: ordinal day/weekday across formats and languages;
- numerals: digits, language-specific number words, Roman numerals;
- molecules: randomized SMILES with RDKit-computed properties.

Run first from scratch in the same small architecture, then in a small
pretrained language model with registered LoRA placements. The pretrained tier
asks whether pretraining already supplies O and whether information-flow
placement effects survive scale; it is not pooled statistically with the
from-scratch tier.

## Claim-specific falsifiers

| Claim | Supported if | Falsified by |
|---|---|---|
| Selection | Matched models differ reproducibly in independent O | Curricula converge to equivalent O within budget |
| Leverage | O predicts registered later reach within a fiber | No reach difference after valid manipulation checks |
| Compounding | Reach/sample-cost gap persists or grows across stages | Low-O models catch up by the registered stage |
| Entrenchment | Cost to acquire O rises with shortcut exposure | Flat exposure-cost relation |
| Use-dependence | Lower latent use reduces O or raises re-access cost | Flat lambda response |
| Emergence | A/D produce O without privileged loss in V9-A; later channels replicate it | Only imposed channels work |
| Causality | Break removes and make confers leverage | Selective O edits leave reach unchanged |
| Generality | Frozen predictions replicate on sealed S4 | S4 fails the registered pattern |

## Strategic interpretation

V8 is the mechanism audit and imposed positive control. V9-A asks the now
decisive question: whether the transferable organization can arise without
privileged correspondence. V9-B then tests whether independently measured O
predicts later reach. A publishable first Strata paper should claim only the
claims that pass separately across both geometries and H-lock-s. Compounding,
entrenchment, use-dependence, and natural/pretrained bridges remain valuable
precisely because they can fail separately; they should not be bundled into
one omnibus success criterion.
