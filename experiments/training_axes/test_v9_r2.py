"""Deterministic implementation tests for V9-R2."""

from __future__ import annotations

import unittest

from experiments.training_axes.analyze_v9_r2 import reliability_summary
from experiments.training_axes.design_v9_r2 import R2_ARMS, R2_SEEDS, make_configs


class V9R2Tests(unittest.TestCase):
    def test_registered_allocation(self):
        configs = make_configs()
        self.assertEqual(len(configs), 24)
        self.assertEqual({c.scaffold for c in configs}, {"circle"})
        self.assertEqual({c.arm for c in configs}, set(R2_ARMS))
        self.assertEqual({c.seed for c in configs}, set(R2_SEEDS))
        self.assertTrue(all(c.steps == 20_000 for c in configs))

    def test_reliable_seed_requires_three_replicates(self):
        ids = {
            (arm, seed): f"{arm}-{seed}"
            for arm in R2_ARMS for seed in R2_SEEDS
        }
        by_model = {}
        for arm in R2_ARMS:
            for seed in R2_SEEDS:
                valid_count = 3 if arm != "a00_confounded" else 2
                by_model[ids[(arm, seed)]] = [
                    {
                        "causal_valid": float(index < valid_count),
                        "causal_target_to_target": 0.9,
                        "causal_source_to_target": 0.88,
                        "causal_random": 0.5,
                    }
                    for index in range(4)
                ]
        arms, details = reliability_summary(ids, by_model)
        self.assertEqual(arms["a00_confounded"]["reliable_seeds"], 0)
        self.assertEqual(arms["imposed"]["reliable_seeds"], 8)
        self.assertEqual(arms["a50_diverse"]["reliable_seeds"], 8)
        self.assertFalse(details["a00_confounded"][str(R2_SEEDS[0])]["reliable"])
        self.assertTrue(details["imposed"][str(R2_SEEDS[0])]["reliable"])


if __name__ == "__main__":
    unittest.main()
