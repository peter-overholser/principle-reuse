"""Cross-domain comparison of composed differences.

The latent question is whether ``a - b > c - d``. Six alphabets rename the ten
integer values and two grammars express the same comparison after negating and
reversing both sides. Training, calibration, and locked domain combinations are
disjoint. A fixed quarter of latent quadruples is also held out.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


N_ALPH = 6
N_ITEMS = 10
N_SURF = 3
# Exclude the algebraic extremes (+/-18), which each have only one quadruple
# and therefore cannot populate both latent-query train and held-out splits.
MAX_MARGIN = 2 * (N_ITEMS - 1) - 1
ALL_DISTS = tuple(range(1, MAX_MARGIN + 1))

DIFFICULTY_LEVELS = {
    # Absolute decision margin. The mixed condition gives full coverage; the
    # restricted conditions isolate small- and large-margin evidence.
    "easy": tuple(range(7, MAX_MARGIN + 1)),
    "mixed": ALL_DISTS,
    "hard": (1, 2, 3),
}

CALIB_COMBOS = ((4, 1), (5, 1))
TEST_COMBOS = ((2, 1), (3, 1))
TRAIN_COMBOS = tuple(
    (alphabet, grammar)
    for alphabet in range(N_ALPH)
    for grammar in range(2)
    if (alphabet, grammar) not in CALIB_COMBOS + TEST_COMBOS
)

N_ITEM_TOKENS = N_ALPH * N_ITEMS
_SPECIAL_NAMES = ["BOS", "EOS", "Q", "MINUS", "LT", "GT", "PAD"] + [
    f"FILL{i}" for i in range(N_SURF)
]
SPECIAL = {
    name: N_ITEM_TOKENS + index for index, name in enumerate(_SPECIAL_NAMES)
}
VOCAB = N_ITEM_TOKENS + len(_SPECIAL_NAMES)
PAD = SPECIAL["PAD"]
SEQ_LEN = 11


def _token(alphabet: int, value: int) -> int:
    return alphabet * N_ITEMS + value


def _margin(query: tuple[int, int, int, int]) -> int:
    a, b, c, d = query
    return (a - b) - (c - d)


def _latent_buckets():
    buckets = defaultdict(list)
    for a in range(N_ITEMS):
        for b in range(N_ITEMS):
            for c in range(N_ITEMS):
                for d in range(N_ITEMS):
                    query = (a, b, c, d)
                    margin = _margin(query)
                    if margin and abs(margin) <= MAX_MARGIN:
                        buckets[margin].append(query)
    return dict(buckets)


LATENTS_BY_MARGIN = _latent_buckets()


def held_queries(seed: int = 29, fraction: float = 0.25):
    """Stratified holdout preserving every signed-margin category."""
    rng = np.random.default_rng(seed)
    held = set()
    for margin, candidates in sorted(LATENTS_BY_MARGIN.items()):
        count = min(len(candidates) - 1, max(1, round(fraction * len(candidates))))
        selected = rng.choice(len(candidates), size=count, replace=False)
        held.update(candidates[int(index)] for index in selected)
    return frozenset(held)


HELD_QUERIES = held_queries()
HELD_BY_MARGIN = {
    margin: tuple(query for query in candidates if query in HELD_QUERIES)
    for margin, candidates in LATENTS_BY_MARGIN.items()
}
TRAIN_BY_MARGIN = {
    margin: tuple(query for query in candidates if query not in HELD_QUERIES)
    for margin, candidates in LATENTS_BY_MARGIN.items()
}


def render(
    query: tuple[int, int, int, int],
    alphabet: int,
    grammar: int,
    surface: int,
):
    """Render two logically equivalent forms of a-b > c-d."""
    a, b, c, d = query
    if grammar == 0:
        values = (a, b, c, d)
        relation = SPECIAL["GT"]
    else:
        # a-b > c-d iff b-a < d-c.
        values = (b, a, d, c)
        relation = SPECIAL["LT"]
    av, bv, cv, dv = (_token(alphabet, value) for value in values)
    body = [av, SPECIAL["MINUS"], bv, relation,
            cv, SPECIAL["MINUS"], dv]
    tokens = [SPECIAL["BOS"]]
    if surface:
        tokens.append(SPECIAL[f"FILL{surface}"])
    tokens.extend([SPECIAL["Q"], *body, SPECIAL["EOS"]])
    margin = _margin(query)
    metadata = {
        "alphabet": alphabet,
        "grammar": grammar,
        "surface": surface,
        "rank_x": a,
        "rank_y": c,
        "distance": abs(margin),
        "signed_gap": margin / MAX_MARGIN,
        "pair": query,
    }
    return tokens, float(margin > 0), metadata


def pad(sequences):
    output = np.full((len(sequences), SEQ_LEN), PAD, dtype=np.int64)
    for index, sequence in enumerate(sequences):
        if len(sequence) > SEQ_LEN:
            raise ValueError(f"sequence length {len(sequence)} exceeds {SEQ_LEN}")
        output[index, :len(sequence)] = sequence
    return output


def _draw_latent(rng, dists, want_held):
    signed = int(rng.choice((-1, 1)))
    distance = int(rng.choice(tuple(dists)))
    candidates = (HELD_BY_MARGIN if want_held else TRAIN_BY_MARGIN)[
        signed * distance
    ]
    if not candidates:
        raise RuntimeError(
            f"no latent query for margin={signed * distance}, held={want_held}"
        )
    return candidates[int(rng.integers(0, len(candidates)))]


def sample(
    rng, n, combos, dists=ALL_DISTS, want_held=False, surfaces=range(N_SURF)
):
    sequences, labels, metadata = [], [], []
    combos, surfaces = tuple(combos), tuple(surfaces)
    for _ in range(n):
        query = _draw_latent(rng, dists, want_held)
        alphabet, grammar = combos[int(rng.integers(0, len(combos)))]
        surface = int(rng.choice(surfaces))
        sequence, label, meta = render(query, alphabet, grammar, surface)
        sequences.append(sequence)
        labels.append(label)
        metadata.append(meta)
    return pad(sequences), np.asarray(labels, dtype=np.float32), metadata


def paired_batch(rng, n_pairs, dists):
    first, second, labels, gaps, meta_first, meta_second = [], [], [], [], [], []
    for _ in range(n_pairs):
        query = _draw_latent(rng, dists, want_held=False)
        indices = rng.choice(len(TRAIN_COMBOS), size=2, replace=False)
        combo_one = TRAIN_COMBOS[int(indices[0])]
        combo_two = TRAIN_COMBOS[int(indices[1])]
        one = render(query, *combo_one, int(rng.integers(0, N_SURF)))
        two = render(query, *combo_two, int(rng.integers(0, N_SURF)))
        first.append(one[0])
        second.append(two[0])
        labels.append(one[1])
        gaps.append(one[2]["signed_gap"])
        meta_first.append(one[2])
        meta_second.append(two[2])
    return (
        pad(first), pad(second), np.asarray(labels, dtype=np.float32),
        np.asarray(gaps, dtype=np.float32), meta_first, meta_second,
    )


def paired_eval(seed, n_pairs, combos=CALIB_COMBOS):
    rng = np.random.default_rng(seed)
    first, second, labels, meta_first, meta_second = [], [], [], [], []
    combos = tuple(combos)
    for _ in range(n_pairs):
        query = _draw_latent(rng, ALL_DISTS, want_held=False)
        indices = rng.choice(len(combos), size=2, replace=len(combos) < 2)
        one = render(query, *combos[int(indices[0])], int(rng.integers(0, N_SURF)))
        two = render(query, *combos[int(indices[1])], int(rng.integers(0, N_SURF)))
        first.append(one[0])
        second.append(two[0])
        labels.append(one[1])
        meta_first.append(one[2])
        meta_second.append(two[2])
    return pad(first), pad(second), np.asarray(labels, dtype=np.float32), meta_first, meta_second


def evaluation_sets(seed=0, n=1000, train_dists=ALL_DISTS):
    rng = np.random.default_rng(200_000 + seed)
    return {
        "source": sample(rng, n, TRAIN_COMBOS, train_dists, want_held=False),
        "source_all": sample(rng, n, TRAIN_COMBOS, ALL_DISTS, want_held=False),
        "calib_combo": sample(rng, n, CALIB_COMBOS, ALL_DISTS, want_held=False),
        "test_combo": sample(rng, n, TEST_COMBOS, ALL_DISTS, want_held=False),
        "held_pair": sample(rng, n, TRAIN_COMBOS, ALL_DISTS, want_held=True),
        "test_both": sample(rng, n, TEST_COMBOS, ALL_DISTS, want_held=True),
    }


def validate_design():
    assert not set(TRAIN_COMBOS) & set(CALIB_COMBOS)
    assert not set(TRAIN_COMBOS) & set(TEST_COMBOS)
    assert not set(CALIB_COMBOS) & set(TEST_COMBOS)
    assert set(LATENTS_BY_MARGIN) == set(range(-MAX_MARGIN, 0)) | set(range(1, MAX_MARGIN + 1))
    rng = np.random.default_rng(0)
    x1, x2, labels, gaps, m1, m2 = paired_batch(rng, 256, ALL_DISTS)
    assert x1.shape == x2.shape == (256, SEQ_LEN)
    assert np.all((gaps > 0) == (labels > 0))
    assert all(a["pair"] == b["pair"] for a, b in zip(m1, m2))
    assert all(m["pair"] not in HELD_QUERIES for m in m1)


if __name__ == "__main__":
    validate_design()
    print(f"vocab={VOCAB} sequence_length={SEQ_LEN}")
    print(f"latent queries={sum(map(len, LATENTS_BY_MARGIN.values()))}")
    print(f"held latent queries={len(HELD_QUERIES)}")
    print("difference-comparison design checks passed")
