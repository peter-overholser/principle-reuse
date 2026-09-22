# V8 locked assessment

This assessment was written after the registered V8 locked analysis. It does
not alter the registered hierarchy or replace the generated
`V8_LOCKED_REPORT.md`.

## Status

V8 is a clean confirmatory success. The registered decision route is
**`embedding-tying-dominates`**.

The transferred locked tree contains all 264 registered files and all 2,904
registered checkpoint rows. The transferred tree reproduces the locked-tree
hash in the Spark analysis. The development-freeze and locked-analysis hashes
also match the local frozen files.

## Registered result

The primary locked outcome is accuracy on fresh held alphabet--grammar
combinations. The three steps of the registered hierarchy give:

| Task | Full minus I | E-only minus full | HG minus I |
|---|---:|---:|---:|
| Order | +0.565 [+0.389, +0.741] | -0.057 [-0.087, -0.027] | -0.018 [-0.137, +0.101] |
| Differences | +0.497 [+0.275, +0.718] | +0.008 [+0.001, +0.015] | +0.091 [-0.218, +0.401] |

Thus:

1. the full structural objective replicates on both fresh locks;
2. E-only lies inside the preregistered +/-0.10 equivalence band around full
   on both tasks; and
3. the non-embedding combination, H+G, is not detectably better than the
   invariant baseline on either task.

The same conclusion appears in the factorial effects:

| Task | E main effect | H main effect | G main effect |
|---|---:|---:|---:|
| Order | +0.539 [+0.396, +0.682] | -0.036 [-0.087, +0.015] | +0.055 [-0.012, +0.123] |
| Differences | +0.468 [+0.307, +0.630] | -0.037 [-0.075, +0.002] | +0.079 [-0.045, +0.203] |

Correct embedding correspondence is the only component with a clear marginal
effect across both tasks. Hidden-state matching has no detectable incremental
benefit and is numerically negative. The gap objective alone is not reliably
sufficient.

## The reduced mechanism is E+G

The registered hierarchy appropriately calls E sufficient under its stated
10-point equivalence margin. A more precise substantive account is:

| Task | E-only | E+G | Full | E+G minus full |
|---|---:|---:|---:|---:|
| Order | 0.934 | 0.995 | 0.991 | +0.004 [-0.005, +0.013] |
| Differences | 0.996 | 0.997 | 0.988 | +0.009 [+0.001, +0.018] |

E-only is excellent and stable, but it is significantly 5.7 points below full
on order. Adding the shared signed-gap readout removes this residual deficit:
E+G is indistinguishable from full on order and slightly better on differences.
The parsimonious positive-control objective for the next study is therefore
E+G. H can be removed.

## Semantic specificity

The permuted-correspondence control is decisive:

| Task | E-perm minus valid E-only |
|---|---:|
| Order | -0.514 [-0.655, -0.373] |
| Differences | -0.314 [-0.479, -0.148] |

On order, E-perm is indistinguishable from the concrete control. On
differences it retains a partial advantage over concrete (+0.176 [+0.028,
+0.325]), indicating that generic tying or regularization can sometimes help,
but it does not reproduce the valid semantic correspondence effect. The large
valid-versus-permuted contrast establishes that the identity of the alignment,
not merely the existence of an auxiliary loss, matters.

## Generalization profile

The effect is specific to the shifted domain rather than a general improvement
on easy held queries:

| Task | Full minus I: combination | held latent/source cells | joint shift |
|---|---:|---:|---:|
| Order | +0.565 | +0.054 | +0.568 |
| Differences | +0.497 | +0.001 | +0.492 |

The differences task is especially clean: all arms are essentially perfect on
held latent queries within source cells, while they diverge by about 50 points
when the rendering combination changes. The effect survives the harder joint
combination-plus-latent shift.

The behavioral mechanism is also visible in the shortcut measurements. At the
endpoint, shortcut agreement falls from 0.574 to 0.009 on order and from 0.509
to 0.012 on differences when moving from I to full. The full models do not
merely become more accurate; they cease following the confounded rendering
shortcut on the fresh combinations.

## Fresh-lock replication and reliability

The full ranking of all eleven arms is extremely stable between development
and lock:

| Task | Development--lock arm correlation | Mean absolute arm shift |
|---|---:|---:|
| Order | 0.997 | 0.015 |
| Differences | 0.998 | 0.014 |

This is strong evidence that the result was not selected around the two
development cells. It remains an exchangeable within-scaffold replication,
not yet an out-of-scaffold demonstration.

Correct correspondence also changes reliability, not only the average:

| Task | I endpoint SD | E-only SD | E+G SD | Full SD |
|---|---:|---:|---:|---:|
| Order | 0.284 | 0.050 | 0.007 | 0.013 |
| Differences | 0.349 | 0.008 | 0.005 | 0.014 |

Every E-only, E+G, and full seed exceeds 0.80 on both locked tasks. The
invariant and other non-E objectives remain highly seed-dependent. Descriptively,
E selects the transferable solution reliably rather than merely increasing the
probability of an occasional successful seed.

## Source-fiber qualification

The primary endpoint comparison satisfies the preregistered validation-fiber
gate: source accuracy, Brier skill, overall log loss, and near-margin log loss
are equivalent at the endpoint. The large locked divergence is therefore not
explained by a source-domain performance gap.

The checkpoint analysis matched to source log loss 0.01 is more qualified:

| Task | Matched-checkpoint full minus I |
|---|---:|
| Order | +0.148 [-0.026, +0.321] |
| Differences | +0.262 [+0.039, +0.485] |

This robustness analysis is positive on differences but inconclusive on
order. It does not overturn the endpoint hierarchy, which was the registered
primary test. It does show that the order advantage is not fully installed at
the earliest checkpoint at which source loss reaches the matching target.
The locked trajectories clarify why: source performance is already saturated
while E-bearing models continue to improve on the shifted combinations over
thousands of later updates. That pattern is consistent with induced grokking
or post-mastery organizational development, but it should not be described as
an immediate consequence of source mastery.

## What V8 establishes

V8 supports the following statement:

> Among models that are equivalent on a constrained source domain, training
> with correct cross-rendering semantic correspondence reliably selects a
> solution that transfers to fresh rendering combinations and joint shifts.
> Neither output invariance, hidden-state matching, nor shared latent readout
> without that correspondence reliably selects the same solution.

It does not yet establish that reusable organization emerges without
privileged supervision. E supplies the correct item correspondence and is
therefore close to the semantic map needed for deployment, even though it does
not supply labels for the held combinations. The permuted control answers
whether arbitrary tying suffices; it does not answer whether ordinary data
structure can induce the correct map.

## Strategic consequence

V8 should serve as the paper's mechanistic localization and imposed positive
control, not as the whole developmental thesis. It considerably improves the
paper by replacing an underspecified claim about auxiliary supervision with a
specific result about semantic correspondence and by demonstrating a sharp
validation--deployment divergence under a fresh lock.

The next decisive experiment should ask whether the same organization and
leverage signature can be produced without E labels. The highest-value route
is the proposed Strata/V9 emergence manipulation: vary anchors and
lexicon--grammar diversity, use E+G as the imposed ceiling, and reserve a new
scaffold or structurally distinct held role for confirmation. Success there
would move the central claim from "privileged correspondence can impose a
transferable organization" to "developmental data structure can select it."
