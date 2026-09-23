"""Deterministic implementation tests for V9-B."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.training_axes.v9b_common import derive_seed, verify_reveal


class V9BTests(unittest.TestCase):
    def test_commitment_accepts_only_matching_reveal(self):
        reveal = "registered-secret"
        digest = hashlib.sha256(reveal.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commitment = root / "commitment.json"
            commitment.write_text(
                '{"scheme":"sha256-strip-utf8-v1","commitment":"'
                + digest + '"}\n'
            )
            good = root / "good.txt"
            good.write_text(reveal + "\n")
            bad = root / "bad.txt"
            bad.write_text("wrong\n")
            self.assertEqual(verify_reveal(commitment, good), reveal)
            with self.assertRaises(ValueError):
                verify_reveal(commitment, bad)

    def test_seed_domains_are_deterministic_and_separate(self):
        reveal = "registered-secret"
        first = derive_seed(reveal, "dataset", "line")
        self.assertEqual(first, derive_seed(reveal, "dataset", "line"))
        self.assertNotEqual(first, derive_seed(reveal, "dataset", "circle"))
        self.assertNotEqual(
            derive_seed(reveal, "assay", "line", 512, 20_000),
            derive_seed(reveal, "assay", "line", 513, 20_000),
        )


if __name__ == "__main__":
    unittest.main()
