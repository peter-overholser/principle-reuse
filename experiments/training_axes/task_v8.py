"""Fresh locked task instances for the V8 structural-component study.

V8 rotates alphabet roles, permutes item token IDs, uses new latent-holdout
seeds, and uses new evaluation-generator seeds.  Nothing here is imported by
the earlier experiments.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


N_ALPH = 6
N_ITEMS = 10
N_SURF = 3
N_ITEM_TOKENS = N_ALPH * N_ITEMS
CALIB_COMBOS = ((2, 1), (3, 1))
TEST_COMBOS = ((0, 1), (1, 1))
TRAIN_COMBOS = tuple(
    (alphabet, grammar)
    for alphabet in range(N_ALPH)
    for grammar in range(2)
    if (alphabet, grammar) not in CALIB_COMBOS + TEST_COMBOS
)


def _item_tokens(seed):
    rng = np.random.default_rng(seed)
    return rng.permutation(N_ITEM_TOKENS).reshape(N_ALPH, N_ITEMS)


def _mapping_permutations(seed):
    rng = np.random.default_rng(seed)
    values = np.stack([rng.permutation(N_ITEMS) for _ in range(N_ALPH)])
    if any(np.array_equal(values[0], row) for row in values[1:]):
        raise AssertionError("registered E-perm maps must differ across alphabets")
    return values


class FreshTaskBase:
    N_ALPH = N_ALPH
    N_ITEMS = N_ITEMS
    N_SURF = N_SURF
    N_ITEM_TOKENS = N_ITEM_TOKENS
    TRAIN_COMBOS = TRAIN_COMBOS
    CALIB_COMBOS = CALIB_COMBOS
    TEST_COMBOS = TEST_COMBOS

    def pad(self, sequences):
        output = np.full((len(sequences), self.SEQ_LEN), self.PAD, dtype=np.int64)
        for index, sequence in enumerate(sequences):
            if len(sequence) > self.SEQ_LEN:
                raise ValueError(f"sequence length {len(sequence)} exceeds {self.SEQ_LEN}")
            output[index, :len(sequence)] = sequence
        return output

    def sample(
        self, rng, n, combos, dists=None, want_held=False,
        surfaces=range(N_SURF), mark_shortcut=False,
    ):
        dists = self.ALL_DISTS if dists is None else tuple(dists)
        combos, surfaces = tuple(combos), tuple(surfaces)
        sequences, labels, metadata = [], [], []
        for _ in range(n):
            latent = self._draw_latent(rng, dists, want_held)
            alphabet, grammar = combos[int(rng.integers(0, len(combos)))]
            surface = int(rng.integers(0, len(surfaces)))
            sequence, label, meta = self.render(
                latent, alphabet, grammar, surfaces[surface]
            )
            if mark_shortcut:
                meta = {**meta, "shortcut_label": float(1.0 - label)}
            sequences.append(sequence)
            labels.append(label)
            metadata.append(meta)
        return (
            self.pad(sequences), np.asarray(labels, dtype=np.float32), metadata
        )

    def paired_batch(self, rng, n_pairs, dists):
        first, second, labels, gaps, meta_first, meta_second = [], [], [], [], [], []
        for _ in range(n_pairs):
            latent = self._draw_latent(rng, tuple(dists), want_held=False)
            indices = rng.choice(len(self.TRAIN_COMBOS), size=2, replace=False)
            combo_one = self.TRAIN_COMBOS[int(indices[0])]
            combo_two = self.TRAIN_COMBOS[int(indices[1])]
            one = self.render(
                latent, *combo_one, int(rng.integers(0, self.N_SURF))
            )
            two = self.render(
                latent, *combo_two, int(rng.integers(0, self.N_SURF))
            )
            first.append(one[0])
            second.append(two[0])
            labels.append(one[1])
            gaps.append(one[2]["signed_gap"])
            meta_first.append(one[2])
            meta_second.append(two[2])
        return (
            self.pad(first), self.pad(second),
            np.asarray(labels, dtype=np.float32),
            np.asarray(gaps, dtype=np.float32), meta_first, meta_second,
        )

    def paired_eval(self, seed, n_pairs, combos=None):
        rng = np.random.default_rng(self.EVAL_SEED + seed)
        combos = self.CALIB_COMBOS if combos is None else tuple(combos)
        first, second, labels, meta_first, meta_second = [], [], [], [], []
        for _ in range(n_pairs):
            latent = self._draw_latent(rng, self.ALL_DISTS, want_held=False)
            indices = rng.choice(len(combos), size=2, replace=len(combos) < 2)
            one = self.render(
                latent, *combos[int(indices[0])], int(rng.integers(0, self.N_SURF))
            )
            two = self.render(
                latent, *combos[int(indices[1])], int(rng.integers(0, self.N_SURF))
            )
            first.append(one[0])
            second.append(two[0])
            labels.append(one[1])
            meta_first.append(one[2])
            meta_second.append(two[2])
        return (
            self.pad(first), self.pad(second),
            np.asarray(labels, dtype=np.float32), meta_first, meta_second,
        )

    def evaluation_sets(
        self, seed=0, n=1000, train_dists=None, include_locked=False
    ):
        train_dists = self.ALL_DISTS if train_dists is None else tuple(train_dists)
        rng = np.random.default_rng(self.EVAL_SEED + 100_000 + seed)
        datasets = {
            "source": self.sample(
                rng, n, self.TRAIN_COMBOS, train_dists, want_held=False
            ),
            "source_all": self.sample(
                rng, n, self.TRAIN_COMBOS, self.ALL_DISTS, want_held=False
            ),
            "calib_combo": self.sample(
                rng, n, self.CALIB_COMBOS, self.ALL_DISTS,
                want_held=False, mark_shortcut=True,
            ),
        }
        if include_locked:
            datasets.update({
                "test_combo": self.sample(
                    rng, n, self.TEST_COMBOS, self.ALL_DISTS,
                    want_held=False, mark_shortcut=True,
                ),
                "held_pair": self.sample(
                    rng, n, self.TRAIN_COMBOS, self.ALL_DISTS, want_held=True
                ),
                "test_both": self.sample(
                    rng, n, self.TEST_COMBOS, self.ALL_DISTS,
                    want_held=True, mark_shortcut=True,
                ),
            })
        return datasets

    def validate_design(self):
        train, calibration, test = map(
            set, (self.TRAIN_COMBOS, self.CALIB_COMBOS, self.TEST_COMBOS)
        )
        assert not train & calibration and not train & test and not calibration & test
        assert len(train) == 8 and len(calibration) == len(test) == 2
        assert {alphabet for alphabet, _ in train} == set(range(self.N_ALPH))
        assert {grammar for _, grammar in train} == {0, 1}
        assert set(np.asarray(self.ITEM_TOKEN_IDS).reshape(-1)) == set(range(60))
        rng = np.random.default_rng(0)
        X1, X2, labels, gaps, first, second = self.paired_batch(
            rng, 128, self.ALL_DISTS
        )
        assert X1.shape == X2.shape == (128, self.SEQ_LEN)
        assert np.all((gaps > 0) == (labels > 0))
        assert all(one["pair"] == two["pair"] for one, two in zip(first, second))


class FreshOrderTask(FreshTaskBase):
    __name__ = "order_v8"
    ALL_DISTS = tuple(range(1, N_ITEMS))
    DIFFICULTY_LEVELS = {
        "easy": (5, 6, 7, 8, 9),
        "mixed": ALL_DISTS,
        "hard": (1, 2, 3),
    }
    _SPECIAL_NAMES = ["BOS", "EOS", "Q", "LT", "GT", "PAD"] + [
        f"FILL{i}" for i in range(N_SURF)
    ]
    SPECIAL = {
        name: N_ITEM_TOKENS + index for index, name in enumerate(_SPECIAL_NAMES)
    }
    VOCAB = N_ITEM_TOKENS + len(_SPECIAL_NAMES)
    PAD = SPECIAL["PAD"]
    SEQ_LEN = 10
    ITEM_TOKEN_IDS = _item_tokens(81_101)
    MAPPING_PERMUTATIONS = _mapping_permutations(81_201)
    EVAL_SEED = 810_000

    def __init__(self):
        rng = np.random.default_rng(8_117)
        pairs = [(a, b) for a in range(N_ITEMS) for b in range(a + 1, N_ITEMS)]
        chosen = rng.choice(len(pairs), size=round(0.25 * len(pairs)), replace=False)
        self.HELD_PAIRS = frozenset(pairs[int(index)] for index in chosen)

    def _draw_latent(self, rng, dists, want_held):
        for _ in range(10_000):
            distance = int(dists[int(rng.integers(0, len(dists)))])
            low = int(rng.integers(0, N_ITEMS - distance))
            high = low + distance
            if ((low, high) in self.HELD_PAIRS) != want_held:
                continue
            return (low, high) if rng.integers(0, 2) else (high, low)
        raise RuntimeError("could not draw fresh V8 order latent")

    def render(self, latent, alphabet, grammar, surface):
        rank_x, rank_y = latent
        token_x = int(self.ITEM_TOKEN_IDS[alphabet, rank_x])
        token_y = int(self.ITEM_TOKEN_IDS[alphabet, rank_y])
        body = (
            [token_x, self.SPECIAL["LT"], token_y]
            if grammar == 0
            else [token_y, self.SPECIAL["GT"], token_x]
        )
        tokens = [self.SPECIAL["BOS"]]
        if surface:
            tokens.append(self.SPECIAL[f"FILL{surface}"])
        tokens.extend([self.SPECIAL["Q"], *body, self.SPECIAL["EOS"]])
        pair = tuple(sorted((rank_x, rank_y)))
        return tokens, float(rank_x < rank_y), {
            "alphabet": alphabet, "grammar": grammar, "surface": surface,
            "rank_x": rank_x, "rank_y": rank_y,
            "distance": abs(rank_y - rank_x),
            "signed_gap": (rank_y - rank_x) / (N_ITEMS - 1),
            "pair": pair,
        }


class FreshDifferenceTask(FreshTaskBase):
    __name__ = "differences_v8"
    MAX_MARGIN = 2 * (N_ITEMS - 1) - 1
    ALL_DISTS = tuple(range(1, MAX_MARGIN + 1))
    DIFFICULTY_LEVELS = {
        "easy": tuple(range(7, MAX_MARGIN + 1)),
        "mixed": ALL_DISTS,
        "hard": (1, 2, 3),
    }
    _SPECIAL_NAMES = ["BOS", "EOS", "Q", "MINUS", "LT", "GT", "PAD"] + [
        f"FILL{i}" for i in range(N_SURF)
    ]
    SPECIAL = {
        name: N_ITEM_TOKENS + index for index, name in enumerate(_SPECIAL_NAMES)
    }
    VOCAB = N_ITEM_TOKENS + len(_SPECIAL_NAMES)
    PAD = SPECIAL["PAD"]
    SEQ_LEN = 11
    ITEM_TOKEN_IDS = _item_tokens(82_101)
    MAPPING_PERMUTATIONS = _mapping_permutations(82_201)
    EVAL_SEED = 820_000

    def __init__(self):
        buckets = defaultdict(list)
        for a in range(N_ITEMS):
            for b in range(N_ITEMS):
                for c in range(N_ITEMS):
                    for d in range(N_ITEMS):
                        query = (a, b, c, d)
                        margin = self._margin(query)
                        if margin and abs(margin) <= self.MAX_MARGIN:
                            buckets[margin].append(query)
        self.LATENTS_BY_MARGIN = dict(buckets)
        rng = np.random.default_rng(8_229)
        held = set()
        for margin, candidates in sorted(self.LATENTS_BY_MARGIN.items()):
            count = min(len(candidates) - 1, max(1, round(0.25 * len(candidates))))
            selected = rng.choice(len(candidates), size=count, replace=False)
            held.update(candidates[int(index)] for index in selected)
        self.HELD_QUERIES = frozenset(held)
        self.HELD_BY_MARGIN = {
            margin: tuple(query for query in candidates if query in self.HELD_QUERIES)
            for margin, candidates in self.LATENTS_BY_MARGIN.items()
        }
        self.TRAIN_BY_MARGIN = {
            margin: tuple(query for query in candidates if query not in self.HELD_QUERIES)
            for margin, candidates in self.LATENTS_BY_MARGIN.items()
        }

    @staticmethod
    def _margin(query):
        a, b, c, d = query
        return (a - b) - (c - d)

    def _draw_latent(self, rng, dists, want_held):
        signed = int(rng.choice((-1, 1)))
        distance = int(rng.choice(tuple(dists)))
        candidates = (self.HELD_BY_MARGIN if want_held else self.TRAIN_BY_MARGIN)[
            signed * distance
        ]
        return candidates[int(rng.integers(0, len(candidates)))]

    def render(self, query, alphabet, grammar, surface):
        a, b, c, d = query
        values = (a, b, c, d) if grammar == 0 else (b, a, d, c)
        relation = self.SPECIAL["GT"] if grammar == 0 else self.SPECIAL["LT"]
        av, bv, cv, dv = (
            int(self.ITEM_TOKEN_IDS[alphabet, value]) for value in values
        )
        body = [
            av, self.SPECIAL["MINUS"], bv, relation,
            cv, self.SPECIAL["MINUS"], dv,
        ]
        tokens = [self.SPECIAL["BOS"]]
        if surface:
            tokens.append(self.SPECIAL[f"FILL{surface}"])
        tokens.extend([self.SPECIAL["Q"], *body, self.SPECIAL["EOS"]])
        margin = self._margin(query)
        return tokens, float(margin > 0), {
            "alphabet": alphabet, "grammar": grammar, "surface": surface,
            "rank_x": a, "rank_y": c, "distance": abs(margin),
            "signed_gap": margin / self.MAX_MARGIN, "pair": query,
        }


ORDER = FreshOrderTask()
DIFFERENCES = FreshDifferenceTask()


if __name__ == "__main__":
    for task in (ORDER, DIFFERENCES):
        task.validate_design()
        print(
            task.__name__, f"train={task.TRAIN_COMBOS}",
            f"calib={task.CALIB_COMBOS}", f"test={task.TEST_COMBOS}",
        )
