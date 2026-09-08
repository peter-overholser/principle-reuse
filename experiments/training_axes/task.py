"""Controlled relational task for the first training-dynamics experiment.

The task is deliberately simple enough to run hundreds of independent training
runs. Six alphabets render a shared ten-position total order, and two grammars
express the same comparison with reversed syntax. Every alphabet and grammar is
seen in training, but selected alphabet--grammar combinations and item pairs are
held out.

The training batch always contains paired renderings of the same latent query.
All abstraction conditions therefore see identical examples and token counts;
only the loss applied to the pair changes.
"""

from __future__ import annotations

import numpy as np


N_ALPH = 6
N_ITEMS = 10
N_SURF = 3
ALL_DISTS = tuple(range(1, N_ITEMS))

CALIB_COMBOS = ((4, 1), (5, 1))
TEST_COMBOS = ((2, 1), (3, 1))
TRAIN_COMBOS = tuple(
    (alphabet, grammar)
    for alphabet in range(N_ALPH)
    for grammar in range(2)
    if (alphabet, grammar) not in CALIB_COMBOS + TEST_COMBOS
)

assert {a for a, _ in TRAIN_COMBOS} == set(range(N_ALPH))
assert {g for _, g in TRAIN_COMBOS} == {0, 1}

ITEM0 = 0
N_ITEM_TOKENS = N_ALPH * N_ITEMS
_SPECIAL_NAMES = ["BOS", "EOS", "Q", "LT", "GT", "PAD"] + [
    f"FILL{i}" for i in range(N_SURF)
]
SPECIAL = {
    name: N_ITEM_TOKENS + index for index, name in enumerate(_SPECIAL_NAMES)
}
VOCAB = N_ITEM_TOKENS + len(_SPECIAL_NAMES)
PAD = SPECIAL["PAD"]
SEQ_LEN = 10


def _token(alphabet: int, rank: int) -> int:
    return ITEM0 + alphabet * N_ITEMS + rank


def held_pairs(seed: int = 17, fraction: float = 0.25) -> frozenset[tuple[int, int]]:
    """Return latent unordered rank pairs held out in every rendering."""
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(N_ITEMS) for b in range(a + 1, N_ITEMS)]
    count = round(fraction * len(pairs))
    chosen = rng.choice(len(pairs), size=count, replace=False)
    return frozenset(pairs[index] for index in chosen)


HELD_PAIRS = held_pairs()


def render(
    rank_x: int,
    rank_y: int,
    alphabet: int,
    grammar: int,
    surface: int,
) -> tuple[list[int], float, dict]:
    """Render the semantic query 'does x precede y?' in one domain."""
    token_x, token_y = _token(alphabet, rank_x), _token(alphabet, rank_y)
    body = (
        [token_x, SPECIAL["LT"], token_y]
        if grammar == 0
        else [token_y, SPECIAL["GT"], token_x]
    )
    tokens = [SPECIAL["BOS"]]
    if surface:
        tokens.append(SPECIAL[f"FILL{surface}"])
    tokens.extend([SPECIAL["Q"], *body, SPECIAL["EOS"]])
    label = float(rank_x < rank_y)
    signed_gap = (rank_y - rank_x) / (N_ITEMS - 1)
    metadata = {
        "alphabet": alphabet,
        "grammar": grammar,
        "surface": surface,
        "rank_x": rank_x,
        "rank_y": rank_y,
        "distance": abs(rank_y - rank_x),
        "signed_gap": signed_gap,
        "pair": tuple(sorted((rank_x, rank_y))),
    }
    return tokens, label, metadata


def pad(sequences: list[list[int]]) -> np.ndarray:
    output = np.full((len(sequences), SEQ_LEN), PAD, dtype=np.int64)
    for index, sequence in enumerate(sequences):
        if len(sequence) > SEQ_LEN:
            raise ValueError(f"sequence length {len(sequence)} exceeds {SEQ_LEN}")
        output[index, :len(sequence)] = sequence
    return output


def _draw_latent(rng: np.random.Generator, dists, want_held: bool) -> tuple[int, int]:
    dists = tuple(dists)
    for _ in range(10_000):
        distance = int(dists[rng.integers(0, len(dists))])
        low = int(rng.integers(0, N_ITEMS - distance))
        high = low + distance
        if ((low, high) in HELD_PAIRS) != want_held:
            continue
        return (low, high) if rng.integers(0, 2) else (high, low)
    raise RuntimeError("could not draw a latent pair for the requested split")


def sample(
    rng: np.random.Generator,
    n: int,
    combos,
    dists=ALL_DISTS,
    want_held: bool = False,
    surfaces=range(N_SURF),
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    sequences, labels, metadata = [], [], []
    combos, surfaces = tuple(combos), tuple(surfaces)
    for _ in range(n):
        x, y = _draw_latent(rng, dists, want_held)
        alphabet, grammar = combos[rng.integers(0, len(combos))]
        surface = int(surfaces[rng.integers(0, len(surfaces))])
        sequence, label, meta = render(x, y, alphabet, grammar, surface)
        sequences.append(sequence)
        labels.append(label)
        metadata.append(meta)
    return pad(sequences), np.asarray(labels, dtype=np.float32), metadata


def paired_batch(
    rng: np.random.Generator,
    n_pairs: int,
    dists,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Two distinct training renderings of each latent query."""
    first, second, labels, gaps, meta_first, meta_second = [], [], [], [], [], []
    for _ in range(n_pairs):
        rank_x, rank_y = _draw_latent(rng, dists, want_held=False)
        indices = rng.choice(len(TRAIN_COMBOS), size=2, replace=False)
        combo_first, combo_second = TRAIN_COMBOS[int(indices[0])], TRAIN_COMBOS[int(indices[1])]
        surface_first = int(rng.integers(0, N_SURF))
        surface_second = int(rng.integers(0, N_SURF))
        one = render(rank_x, rank_y, *combo_first, surface_first)
        two = render(rank_x, rank_y, *combo_second, surface_second)
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


def paired_eval(
    seed: int,
    n_pairs: int,
    combos=CALIB_COMBOS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Paired renderings for invariance and representation measurements."""
    rng = np.random.default_rng(seed)
    first, second, labels, meta_first, meta_second = [], [], [], [], []
    combos = tuple(combos)
    for _ in range(n_pairs):
        rank_x, rank_y = _draw_latent(rng, ALL_DISTS, want_held=False)
        idx = rng.choice(len(combos), size=2, replace=len(combos) < 2)
        one = render(rank_x, rank_y, *combos[int(idx[0])], int(rng.integers(0, N_SURF)))
        two = render(rank_x, rank_y, *combos[int(idx[1])], int(rng.integers(0, N_SURF)))
        first.append(one[0])
        second.append(two[0])
        labels.append(one[1])
        meta_first.append(one[2])
        meta_second.append(two[2])
    return pad(first), pad(second), np.asarray(labels, dtype=np.float32), meta_first, meta_second


def evaluation_sets(seed: int = 0, n: int = 1000, train_dists=ALL_DISTS) -> dict:
    rng = np.random.default_rng(100_000 + seed)
    return {
        # Matched source performance defines mastery. source_all separately
        # measures extrapolation beyond the difficulty distribution used to train.
        "source": sample(
            rng, n, TRAIN_COMBOS, dists=train_dists, want_held=False
        ),
        "source_all": sample(rng, n, TRAIN_COMBOS, want_held=False),
        "calib_combo": sample(rng, n, CALIB_COMBOS, want_held=False),
        "test_combo": sample(rng, n, TEST_COMBOS, want_held=False),
        "held_pair": sample(rng, n, TRAIN_COMBOS, want_held=True),
        "test_both": sample(rng, n, TEST_COMBOS, want_held=True),
    }


def validate_design() -> None:
    assert not set(TRAIN_COMBOS) & set(CALIB_COMBOS)
    assert not set(TRAIN_COMBOS) & set(TEST_COMBOS)
    assert not set(CALIB_COMBOS) & set(TEST_COMBOS)
    rng = np.random.default_rng(0)
    x1, x2, y, gaps, m1, m2 = paired_batch(rng, 256, ALL_DISTS)
    assert x1.shape == x2.shape == (256, SEQ_LEN)
    assert np.all((gaps > 0) == (y > 0))
    assert all(a["pair"] == b["pair"] for a, b in zip(m1, m2))
    assert all((m["pair"] in HELD_PAIRS) is False for m in m1)


if __name__ == "__main__":
    validate_design()
    print(f"vocab={VOCAB} sequence_length={SEQ_LEN}")
    print(f"train combinations: {TRAIN_COMBOS}")
    print(f"calibration combinations: {CALIB_COMBOS}")
    print(f"test combinations: {TEST_COMBOS}")
    print(f"held latent pairs: {sorted(HELD_PAIRS)}")
    print("design checks passed")
