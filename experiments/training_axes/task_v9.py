"""V9 line/circle corpus with anchor and coverage interventions.

Development construction cannot emit locked roles.  The caller must pass
``include_locked=True`` explicitly after the registered source gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


N_ALPH = 8
N_ITEMS = 12
N_GRAMMARS = 4
N_SURF = 3
N_ITEM_TOKENS = N_ALPH * N_ITEMS
ANCHOR_RANKS = {
    0: (),
    3: (0, 4, 9),
    6: (0, 1, 4, 6, 9, 11),
}

# The carrier alphabet 4 sees every grammar.  Development/locked structural
# lexicons 5--7 see only grammar 0.  Exchangeable lexicons 0--3 see one
# grammar under high confounding and three under diverse coverage.
CORE_COMBOS = (
    (0, 0), (1, 0), (2, 1), (3, 1),
    (4, 0), (4, 1), (4, 2), (4, 3),
    (5, 0), (6, 0), (7, 0),
)
DIVERSE_COMBOS = tuple(sorted(
    {(alphabet, grammar) for alphabet in range(4) for grammar in range(3)}
    | {(4, grammar) for grammar in range(4)}
    | {(5, 0), (6, 0), (7, 0)}
))
DEV_X_COMBOS = ((2, 3), (3, 3))
DEV_S_COMBOS = ((5, 3),)
LOCK_X_COMBOS = ((0, 3), (1, 3))
LOCK_S_COMBOS = ((6, 3), (7, 3))


def normalized_mutual_information(cells):
    """NMI between lexicon and grammar under uniform cell sampling."""
    cells = tuple(cells)
    joint = {(a, g): 1.0 / len(cells) for a, g in cells}
    pa = {a: sum(v for (aa, _), v in joint.items() if aa == a)
          for a in range(N_ALPH)}
    pg = {g: sum(v for (_, gg), v in joint.items() if gg == g)
          for g in range(N_GRAMMARS)}
    mi = sum(
        value * math.log(value / (pa[a] * pg[g]))
        for (a, g), value in joint.items()
    )
    ha = -sum(value * math.log(value) for value in pa.values() if value)
    hg = -sum(value * math.log(value) for value in pg.values() if value)
    return mi / min(ha, hg)


@dataclass(frozen=True)
class V9Batch:
    tokens: np.ndarray
    labels: np.ndarray
    gaps: np.ndarray
    metadata: list[dict]


class V9Task:
    N_ALPH = N_ALPH
    N_ITEMS = N_ITEMS
    N_ITEM_TOKENS = N_ITEM_TOKENS
    CORE_COMBOS = CORE_COMBOS
    DIVERSE_COMBOS = DIVERSE_COMBOS
    DEV_X_COMBOS = DEV_X_COMBOS
    DEV_S_COMBOS = DEV_S_COMBOS
    LOCK_X_COMBOS = LOCK_X_COMBOS
    LOCK_S_COMBOS = LOCK_S_COMBOS
    MAX_ANCHOR_RANKS = frozenset(ANCHOR_RANKS[6])

    def __init__(self, scaffold: str):
        if scaffold not in {"line", "circle"}:
            raise ValueError(f"unknown V9 scaffold: {scaffold}")
        self.scaffold = scaffold
        self.__name__ = f"v9_{scaffold}"
        scaffold_offset = 0 if scaffold == "line" else 10_000
        rng = np.random.default_rng(91_101 + scaffold_offset)
        self._base_item_tokens = rng.permutation(N_ITEM_TOKENS).reshape(
            N_ALPH, N_ITEMS
        )
        mapping_rng = np.random.default_rng(91_201 + scaffold_offset)
        self.MAPPING_PERMUTATIONS = np.stack([
            mapping_rng.permutation(N_ITEMS) for _ in range(N_ALPH)
        ])
        self._special_names = (
            "BOS", "EOS", "Q", "FWD", "REV", "NEG", "G3", "PAD",
            "FILL1", "FILL2",
        )
        self.SPECIAL = {
            name: N_ITEM_TOKENS + index
            for index, name in enumerate(self._special_names)
        }
        self.VOCAB = N_ITEM_TOKENS + len(self._special_names)
        self.PAD = self.SPECIAL["PAD"]
        self.SEQ_LEN = 10
        self.EVAL_SEED = 9_100_000 + scaffold_offset
        self.DEV_HELD_PAIRS, self.LOCK_HELD_PAIRS = self._make_holdouts(
            91_301 + scaffold_offset
        )

    def _make_holdouts(self, seed):
        rng = np.random.default_rng(seed)
        buckets = {}
        canonical = []
        seen = set()
        for latent in self._all_latents():
            key = tuple(sorted(latent))
            if key in seen:
                continue
            seen.add(key)
            canonical.append(latent)
        for latent in canonical:
            bucket = abs(self._signed_gap_raw(latent))
            buckets.setdefault(bucket, []).append(latent)
        development, locked = set(), set()
        for _, values in sorted(buckets.items()):
            ordering = rng.permutation(len(values))
            count = max(1, round(0.15 * len(values)))
            if 2 * count >= len(values):
                count = max(1, (len(values) - 1) // 2)
            for target, indices in (
                (development, ordering[:count]),
                (locked, ordering[count:2 * count]),
            ):
                for index in indices:
                    x, y = values[int(index)]
                    target.add((x, y)); target.add((y, x))
        if not development.isdisjoint(locked):
            raise AssertionError("V9 development and locked latent roles overlap")
        return frozenset(development), frozenset(locked)

    def _all_latents(self):
        if self.scaffold == "line":
            return tuple(
                (x, y) for x in range(N_ITEMS) for y in range(N_ITEMS) if x != y
            )
        values = []
        for x in range(N_ITEMS):
            for signed_gap in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
                values.append((x, (x + signed_gap) % N_ITEMS))
        return tuple(values)

    def _signed_gap_raw(self, latent):
        x, y = latent
        if self.scaffold == "line":
            return y - x
        raw = (y - x) % N_ITEMS
        if raw > N_ITEMS // 2:
            raw -= N_ITEMS
        if raw in (0, N_ITEMS // 2):
            raise ValueError("V9 circle excludes zero and antipodal pairs")
        return raw

    def item_token_ids(self, anchor_count: int):
        if anchor_count not in ANCHOR_RANKS:
            raise ValueError(f"unsupported anchor count: {anchor_count}")
        values = self._base_item_tokens.copy()
        for rank in ANCHOR_RANKS[anchor_count]:
            values[:, rank] = values[0, rank]
        return values

    def pad(self, sequences):
        output = np.full((len(sequences), self.SEQ_LEN), self.PAD, dtype=np.int64)
        for index, sequence in enumerate(sequences):
            if len(sequence) > self.SEQ_LEN:
                raise ValueError(f"sequence length {len(sequence)} exceeds {self.SEQ_LEN}")
            output[index, :len(sequence)] = sequence
        return output

    def _draw_latent(self, rng, held_role="train", nonanchor_only=False):
        if held_role not in {"train", "dev", "lock"}:
            raise ValueError(f"unknown V9 latent role: {held_role}")
        for _ in range(20_000):
            if self.scaffold == "line":
                x, y = rng.choice(N_ITEMS, size=2, replace=False)
                latent = int(x), int(y)
            else:
                x = int(rng.integers(0, N_ITEMS))
                gap = int(rng.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)))
                latent = x, (x + gap) % N_ITEMS
            in_dev = latent in self.DEV_HELD_PAIRS
            in_lock = latent in self.LOCK_HELD_PAIRS
            if held_role == "train" and (in_dev or in_lock):
                continue
            if held_role == "dev" and not in_dev:
                continue
            if held_role == "lock" and not in_lock:
                continue
            if nonanchor_only and any(v in self.MAX_ANCHOR_RANKS for v in latent):
                continue
            return latent
        raise RuntimeError("could not draw V9 latent under registered constraints")

    def render(self, latent, alphabet, grammar, surface, anchor_count):
        x, y = latent
        token_ids = self.item_token_ids(anchor_count)
        tx, ty = int(token_ids[alphabet, x]), int(token_ids[alphabet, y])
        if grammar == 0:
            body = [self.SPECIAL["Q"], tx, self.SPECIAL["FWD"], ty]
        elif grammar == 1:
            body = [self.SPECIAL["Q"], ty, self.SPECIAL["REV"], tx]
        elif grammar == 2:
            body = [
                self.SPECIAL["Q"], self.SPECIAL["NEG"], ty,
                self.SPECIAL["FWD"], tx,
            ]
        elif grammar == 3:
            body = [
                self.SPECIAL["G3"], self.SPECIAL["Q"], ty,
                self.SPECIAL["FWD"], tx,
            ]
        else:
            raise ValueError(f"unknown grammar: {grammar}")
        tokens = [self.SPECIAL["BOS"]]
        if surface:
            tokens.append(self.SPECIAL[f"FILL{surface}"])
        tokens.extend(body)
        tokens.append(self.SPECIAL["EOS"])
        raw_gap = self._signed_gap_raw(latent)
        scale = (N_ITEMS - 1) if self.scaffold == "line" else 5
        assay_latent = (
            (x + y - (N_ITEMS - 1)) / (N_ITEMS - 1)
            if self.scaffold == "line"
            else 0.5 * (
                math.cos(2 * math.pi * x / N_ITEMS)
                + math.cos(2 * math.pi * y / N_ITEMS)
            )
        )
        exposure = sum(value in self.MAX_ANCHOR_RANKS for value in latent)
        return tokens, float(raw_gap > 0), {
            "scaffold": self.scaffold,
            "alphabet": int(alphabet),
            "grammar": int(grammar),
            "surface": int(surface),
            "x": int(x), "y": int(y),
            "pair": (int(x), int(y)),
            "distance": abs(int(raw_gap)),
            "signed_gap": float(raw_gap / scale),
            "assay_latent": float(assay_latent),
            "held_role": (
                "dev" if latent in self.DEV_HELD_PAIRS else
                "lock" if latent in self.LOCK_HELD_PAIRS else "train"
            ),
            "max_anchor_exposure": int(exposure),
            "active_anchor_exposure": int(sum(
                value in ANCHOR_RANKS[anchor_count] for value in latent
            )),
        }

    def sample(
        self, rng, n, combos, anchor_count, *, held_role="train",
        nonanchor_only=False,
    ):
        combos = tuple(combos)
        sequences, labels, gaps, metadata = [], [], [], []
        for _ in range(n):
            latent = self._draw_latent(
                rng, held_role=held_role, nonanchor_only=nonanchor_only
            )
            alphabet, grammar = combos[int(rng.integers(0, len(combos)))]
            surface = int(rng.integers(0, N_SURF))
            tokens, label, meta = self.render(
                latent, alphabet, grammar, surface, anchor_count
            )
            sequences.append(tokens)
            labels.append(label)
            gaps.append(meta["signed_gap"])
            metadata.append(meta)
        return V9Batch(
            self.pad(sequences), np.asarray(labels, dtype=np.float32),
            np.asarray(gaps, dtype=np.float32), metadata,
        )

    def training_batch(self, rng, n, coverage, anchor_count):
        combos = CORE_COMBOS if coverage == "confounded" else DIVERSE_COMBOS
        return self.sample(rng, n, combos, anchor_count, held_role="train")

    def paired_training_batch(self, rng, n_pairs, anchor_count):
        first, second, labels, gaps, first_meta, second_meta = [], [], [], [], [], []
        for _ in range(n_pairs):
            latent = self._draw_latent(rng, held_role="train")
            indices = rng.choice(len(CORE_COMBOS), size=2, replace=False)
            examples = []
            for index in indices:
                alphabet, grammar = CORE_COMBOS[int(index)]
                examples.append(self.render(
                    latent, alphabet, grammar, int(rng.integers(0, N_SURF)),
                    anchor_count,
                ))
            first.append(examples[0][0]); second.append(examples[1][0])
            labels.append(examples[0][1]); gaps.append(examples[0][2]["signed_gap"])
            first_meta.append(examples[0][2]); second_meta.append(examples[1][2])
        return (
            self.pad(first), self.pad(second),
            np.asarray(labels, dtype=np.float32),
            np.asarray(gaps, dtype=np.float32), first_meta, second_meta,
        )

    def paired_cross_render(self, seed, n_pairs, anchor_count, target="dev_s"):
        targets = {
            "dev_x": DEV_X_COMBOS, "dev_s": DEV_S_COMBOS,
            "lock_x": LOCK_X_COMBOS, "lock_s": LOCK_S_COMBOS,
        }
        rng = np.random.default_rng(self.EVAL_SEED + 700_000 + int(seed))
        source, target_rows = [], []
        for _ in range(n_pairs):
            latent = self._draw_latent(rng, held_role="train")
            source_combo = CORE_COMBOS[int(rng.integers(0, len(CORE_COMBOS)))]
            target_cells = targets[target]
            target_combo = target_cells[int(rng.integers(0, len(target_cells)))]
            source.append(self.render(
                latent, *source_combo, int(rng.integers(0, N_SURF)), anchor_count
            ))
            target_rows.append(self.render(
                latent, *target_combo, int(rng.integers(0, N_SURF)), anchor_count
            ))
        return (
            self.pad([row[0] for row in source]),
            self.pad([row[0] for row in target_rows]),
            np.asarray([row[1] for row in source], dtype=np.float32),
            [row[2] for row in source], [row[2] for row in target_rows],
        )

    def evaluation_sets(
        self, seed, n, anchor_count, *, include_locked=False
    ):
        rng = np.random.default_rng(self.EVAL_SEED + 100_000 + int(seed))
        datasets = {
            "source": self.sample(rng, n, CORE_COMBOS, anchor_count),
            "source_extended": self.sample(
                rng, n, DIVERSE_COMBOS, anchor_count
            ),
            "dev_x": self.sample(rng, n, DEV_X_COMBOS, anchor_count),
            "dev_s": self.sample(rng, n, DEV_S_COMBOS, anchor_count),
            "dev_held_source": self.sample(
                rng, n, CORE_COMBOS, anchor_count, held_role="dev"
            ),
            "dev_joint": self.sample(
                rng, n, DEV_S_COMBOS, anchor_count, held_role="dev"
            ),
        }
        if include_locked:
            datasets.update({
                "lock_x": self.sample(rng, n, LOCK_X_COMBOS, anchor_count),
                "lock_s": self.sample(rng, n, LOCK_S_COMBOS, anchor_count),
                "lock_held_source": self.sample(
                    rng, n, CORE_COMBOS, anchor_count, held_role="lock"
                ),
                "lock_joint": self.sample(
                    rng, n, LOCK_S_COMBOS, anchor_count, held_role="lock"
                ),
            })
        return datasets

    def validate_design(self):
        roles = [
            set(CORE_COMBOS), set(DEV_X_COMBOS), set(DEV_S_COMBOS),
            set(LOCK_X_COMBOS), set(LOCK_S_COMBOS),
        ]
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                assert left.isdisjoint(right)
        assert set(CORE_COMBOS).issubset(DIVERSE_COMBOS)
        assert normalized_mutual_information(CORE_COMBOS) > 0.5
        assert normalized_mutual_information(DIVERSE_COMBOS) < 0.2
        for anchors in ANCHOR_RANKS:
            tokens = self.item_token_ids(anchors)
            for rank in range(N_ITEMS):
                unique = len(set(tokens[:, rank]))
                assert unique == (1 if rank in ANCHOR_RANKS[anchors] else N_ALPH)
        rng = np.random.default_rng(99)
        for role in ("train", "dev", "lock"):
            batch = self.sample(rng, 64, CORE_COMBOS, 6, held_role=role)
            assert batch.tokens.shape == (64, self.SEQ_LEN)
            assert np.all((batch.gaps > 0) == (batch.labels > 0))
            assert all(row["held_role"] == role for row in batch.metadata)
        nonanchor = self.sample(
            rng, 64, DEV_S_COMBOS, 6, nonanchor_only=True
        )
        assert all(row["max_anchor_exposure"] == 0 for row in nonanchor.metadata)


LINE = V9Task("line")
CIRCLE = V9Task("circle")
TASKS = {"line": LINE, "circle": CIRCLE}


if __name__ == "__main__":
    for name, task in TASKS.items():
        task.validate_design()
        print(
            name,
            f"rho_high={normalized_mutual_information(CORE_COMBOS):.3f}",
            f"rho_diverse={normalized_mutual_information(DIVERSE_COMBOS):.3f}",
            f"dev_held={len(task.DEV_HELD_PAIRS)}",
            f"lock_held={len(task.LOCK_HELD_PAIRS)}",
        )
