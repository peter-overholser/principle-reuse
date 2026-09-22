"""Deterministic implementation tests for the V8 component ablation."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from experiments.panel import model as model_library
from experiments.training_axes import task_v8
from experiments.training_axes.design_v8 import ARMS, FACTOR_ARMS, make_configs
from experiments.training_axes.run import (
    gradient_norm_diagnostics, training_loss,
)


class V8Tests(unittest.TestCase):
    def test_registered_factorial_and_identifiers(self):
        configs = make_configs()
        self.assertEqual(len(configs), 264)
        self.assertEqual(len({config.config_id for config in configs}), 264)
        self.assertEqual(len(FACTOR_ARMS), 8)
        self.assertEqual(
            set(FACTOR_ARMS),
            {f"e{e}h{h}g{g}" for e in (0, 1) for h in (0, 1) for g in (0, 1)},
        )
        self.assertEqual(set(ARMS) - set(FACTOR_ARMS), {"concrete", "eperm", "gsep"})
        self.assertEqual({config.seed for config in configs}, set(range(400, 412)))

    def test_fresh_roles_and_locked_construction(self):
        for task_module in (task_v8.ORDER, task_v8.DIFFERENCES):
            task_module.validate_design()
            development = task_module.evaluation_sets(
                seed=9, n=32, include_locked=False
            )
            unlocked = task_module.evaluation_sets(
                seed=9, n=32, include_locked=True
            )
            self.assertEqual(
                set(development), {"source", "source_all", "calib_combo"}
            )
            self.assertEqual(
                set(unlocked),
                {
                    "source", "source_all", "calib_combo", "test_combo",
                    "held_pair", "test_both",
                },
            )
            for split in development:
                for left, right in zip(development[split][:2], unlocked[split][:2]):
                    self.assertTrue(np.array_equal(left, right))
            self.assertTrue(set(task_module.TRAIN_COMBOS).isdisjoint(
                task_module.CALIB_COMBOS
            ))
            self.assertTrue(set(task_module.TRAIN_COMBOS).isdisjoint(
                task_module.TEST_COMBOS
            ))
            self.assertFalse(np.array_equal(
                np.asarray(task_module.ITEM_TOKEN_IDS),
                np.arange(task_module.N_ITEM_TOKENS).reshape(
                    task_module.N_ALPH, task_module.N_ITEMS
                ),
            ))

    @staticmethod
    def _loss_for_arm(task_module, arm):
        config = next(
            config for config in make_configs(seeds=(400,), smoke=True)
            if config.v8_split == "order" and config.v8_arm == arm
        )
        if task_module is task_v8.DIFFERENCES:
            config = next(
                config for config in make_configs(seeds=(400,), smoke=True)
                if config.v8_split == "differences" and config.v8_arm == arm
            )
        config = replace(config, width=32)
        torch.manual_seed(17)
        model = model_library.build(
            config.family, task_module.VOCAB, task_module.SEQ_LEN,
            task_module.PAD, d=config.width, layers=config.layers,
            bottleneck_dim=0,
        )
        head = torch.nn.Linear(
            config.width,
            task_module.N_ALPH if config.gap_head_mode == "separate" else 1,
        )
        batch = task_module.paired_batch(
            np.random.default_rng(23), 8, config.train_dists
        )
        loss, parts, components = training_loss(
            model, head, batch, config, torch.device("cpu"), step=0,
            task_module=task_module, return_components=True,
        )
        return config, model, head, batch, loss, parts, components

    def test_component_switches_and_gradient_accounting(self):
        expected = {
            "concrete": (0, 0, 0, 0),
            "e0h0g0": (1, 0, 0, 0),
            "e1h0g0": (1, 0, 0, 1),
            "e0h1g0": (1, 1, 0, 0),
            "e0h0g1": (1, 0, 1, 0),
            "e1h1g1": (1, 1, 1, 1),
            "eperm": (1, 0, 0, 1),
            "gsep": (1, 0, 1, 0),
        }
        for arm, active in expected.items():
            _, model, head, _, loss, parts, components = self._loss_for_arm(
                task_v8.ORDER, arm
            )
            self.assertTrue(torch.isfinite(loss))
            observed = tuple(int(parts[name]) for name in (
                "output_invariant_active", "representation_active",
                "rule_active", "mapping_active",
            ))
            self.assertEqual(observed, active, arm)
            gradients = gradient_norm_diagnostics(
                components, list(model.parameters()) + list(head.parameters())
            )
            self.assertAlmostEqual(
                sum(
                    gradients[f"gradient_share_{name}"]
                    for name in ("binary", "invariant", "representation", "latent", "mapping")
                ),
                1.0, places=6,
            )

    def test_controls_change_only_registered_mechanism(self):
        _, _, _, _, _, valid, _ = self._loss_for_arm(task_v8.ORDER, "e1h0g0")
        _, _, _, _, _, permuted, _ = self._loss_for_arm(task_v8.ORDER, "eperm")
        self.assertNotAlmostEqual(
            valid["loss_mapping"], permuted["loss_mapping"], places=8
        )
        shared_config, _, shared_head, _, _, shared, _ = self._loss_for_arm(
            task_v8.DIFFERENCES, "e0h0g1"
        )
        separate_config, _, separate_head, _, _, separate, _ = self._loss_for_arm(
            task_v8.DIFFERENCES, "gsep"
        )
        self.assertEqual(shared_config.gap_head_mode, "shared")
        self.assertEqual(separate_config.gap_head_mode, "separate")
        self.assertEqual(shared_head.out_features, 1)
        self.assertEqual(separate_head.out_features, task_v8.N_ALPH)
        self.assertGreater(shared["loss_latent"], 0)
        self.assertGreater(separate["loss_latent"], 0)


if __name__ == "__main__":
    unittest.main()
