# V9-B protocol: prospective confirmation with frozen V9-A trunks

Recorded after V9-A's development analysis returned `stop` and before any
V9 locked example was constructed or evaluated. V9-A remains stopped. This
protocol does not amend its gate or relabel its result.

V9-A failed exactly one interim source-mastery check:

```text
v9_sc-circle_a00_diverse_s-519, update 15,000, source accuracy 0.892
```

That run subsequently reached source accuracy 0.986, Brier skill 0.966, and
log loss 0.028 at the registered update-20,000 endpoint. Every one of the 192
main trunks satisfies the endpoint accuracy and Brier criteria, and the
registered between-arm source-equivalence test passes. V9-B therefore asks a
new prospective question: do the development-selected Selection/Emergence
hypotheses replicate on a concrete locked instance that remains untouched
until this protocol and the checkpoint tree are frozen?

The name V9-B is used for this confirmatory successor. The later-update
leverage study anticipated in the original V9-A protocol is deferred to V10.

## Fixed models and eligibility

No model is retrained, extended, replaced, or excluded. V9-B uses all 192
V9-A update-20,000 checkpoints: two scaffolds, eight arms, and paired training
seeds 512--523.

Before the reveal is transferred, the preunlock program must verify:

1. the original V9-A freeze has `unlock_allowed = false`, source equivalence
   passed, and its sole source failure is the registered update-15,000 event;
2. every update-20,000 checkpoint has source accuracy at least 0.95 and source
   Brier skill at least 0.80;
3. all registered arm-minus-baseline 95% source intervals at update 20,000
   remain within the original equivalence margins;
4. the complete development-result and checkpoint trees match the V9-A
   freeze; and
5. the protocol, manifest, generator, metrics, model, evaluators, analysis,
   and lock commitment match their registered hashes.

Any failure stops V9-B. Endpoint eligibility is a newly registered rule for
this untouched outcome; it is not applied retrospectively to V9-A.

## Locked instance and commit--reveal

The structural role definitions remain the pre-existing H-lock-x and
H-lock-s cells and the latent pairs withheld from all V9 training and
development construction. V9-B draws a fresh concrete sample from those
roles. Its salt is committed as:

```text
SHA-256(strip(UTF-8 reveal)) =
32981c290d654269b5b364c608493d9948538ad9a14252846e4dc8f06aaded93
```

The reveal is kept out of the initial code transfer. It is transferred only
after the outcome-free preunlock freeze authorizes evaluation. Domain-separated
hashes of the reveal determine:

- one common locked dataset seed per scaffold; and
- one assay seed per scaffold and paired training seed, shared across arms.

Thus paired arm comparisons use identical concrete examples and assay
randomness. Only update 20,000 is evaluated. The old V9-A unlock command
remains blocked.

## Primary locked hierarchy

The primary treatment remains `a50_diverse`; the baseline remains
`a00_confounded`. Within each scaffold, in order:

1. **Assay replication:** imposed minus baseline H-lock-s non-anchor accuracy
   has a 95% paired interval wholly above zero.
2. **Behavioral emergence:** `a50_diverse` minus baseline H-lock-s non-anchor
   accuracy has a 95% paired interval wholly above zero.
3. **Organizational emergence:** `a50_diverse` minus baseline organization
   score has a 95% paired interval wholly above zero; at least two of the
   probe, CKA, and causal component means are positive; and at least ten of
   twelve primary seeds pass the locked causal validity gate.
4. **Anchor dose:** within diverse coverage, `a50_diverse` minus
   `a00_diverse` H-lock-s non-anchor accuracy has a 95% paired interval wholly
   above zero.
5. **Structural replication:** steps 2--4 hold on both line and circle
   H-lock-s roles.

The route is `emergence-supported` only if the full hierarchy holds on both
scaffolds. Otherwise the fixed routes are `assay-failed`, `scaffold-specific`,
or `imposed-only`, as in V9-A.

## Fixed secondary analyses

The following are descriptive and cannot rescue the hierarchy:

- H-lock-x, H-joint, calibration, and all organization components;
- `a25_diverse` as the intermediate anchor dose;
- diversity contrasts at zero and 50% anchors; and
- the difference-in-differences between the anchor contrast under diverse
  and confounded coverage.

V9-B is an independent outcome-role confirmation of development-selected
hypotheses using frozen training seeds. It is not an independent retraining
replication and does not distinguish durable selection from faster emergence.
The registered post-mastery duration experiment is the next study for that
question.
