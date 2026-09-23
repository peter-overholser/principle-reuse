"""Deterministic implementation tests for V9-A."""

from __future__ import annotations

import unittest
import json
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import torch

from experiments.panel import model as model_library
from experiments.training_axes.design_v9 import (
    ARMS, MAIN_SEEDS, MAIN_STEPS, PILOT_SEEDS, PILOT_STEPS, REPAIR_STEPS,
    arm_fields, make_configs,
)
from experiments.training_axes.run_v9 import (
    optimizer_for, restore, save_checkpoint, training_loss,
    v9_checkpoint_steps,
)
from experiments.training_axes.task_v9 import (
    ANCHOR_RANKS, CIRCLE, CORE_COMBOS, DEV_S_COMBOS, DIVERSE_COMBOS,
    LINE, LOCK_S_COMBOS, normalized_mutual_information,
)


class V9Tests(unittest.TestCase):
    def test_registered_allocations(self):
        self.assertEqual(len(make_configs(PILOT_SEEDS)), 64)
        self.assertEqual(len(make_configs(MAIN_SEEDS)), 192)
        self.assertEqual(len(ARMS), 8)
        self.assertEqual(set(PILOT_SEEDS), set(range(500, 504)))
        self.assertEqual(set(MAIN_SEEDS), set(range(512, 524)))
        self.assertEqual(make_configs(PILOT_SEEDS)[0].steps, PILOT_STEPS)
        self.assertEqual(
            make_configs(MAIN_SEEDS, steps=MAIN_STEPS)[0].steps, MAIN_STEPS
        )

    def test_registered_repair_checkpoints(self):
        self.assertEqual(
            v9_checkpoint_steps(REPAIR_STEPS),
            [0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 7000,
             10000, 12000, 15000, 20000],
        )

    def test_roles_and_holdouts_are_disjoint(self):
        for task in (LINE, CIRCLE):
            task.validate_design()
            self.assertTrue(set(CORE_COMBOS).isdisjoint(DEV_S_COMBOS))
            self.assertTrue(set(CORE_COMBOS).isdisjoint(LOCK_S_COMBOS))
            self.assertTrue(task.DEV_HELD_PAIRS.isdisjoint(task.LOCK_HELD_PAIRS))
            development = task.evaluation_sets(0, 32, 6, include_locked=False)
            unlocked = task.evaluation_sets(0, 32, 6, include_locked=True)
            self.assertFalse(any(key.startswith("lock_") for key in development))
            self.assertTrue({"lock_x", "lock_s", "lock_held_source", "lock_joint"}.issubset(unlocked))
            for name, batch in development.items():
                self.assertTrue(np.array_equal(batch.tokens, unlocked[name].tokens))
                self.assertTrue(np.array_equal(batch.labels, unlocked[name].labels))

    def test_anchor_sets_nested_and_primary_subset_has_no_direct_anchor(self):
        self.assertTrue(set(ANCHOR_RANKS[3]).issubset(ANCHOR_RANKS[6]))
        for task in (LINE, CIRCLE):
            batch = task.sample(
                np.random.default_rng(7), 128, DEV_S_COMBOS, 6,
                nonanchor_only=True,
            )
            self.assertTrue(all(row["max_anchor_exposure"] == 0 for row in batch.metadata))
            for row in batch.metadata:
                self.assertFalse(set(row["pair"]) & set(ANCHOR_RANKS[6]))

    def test_coverage_intervention_hits_registered_nmi(self):
        self.assertGreater(normalized_mutual_information(CORE_COMBOS), 0.5)
        self.assertLess(normalized_mutual_information(DIVERSE_COMBOS), 0.2)

    def test_imposed_and_emergent_losses(self):
        torch.manual_seed(2)
        rng = np.random.default_rng(3)
        for arm in ("a50_diverse", "imposed", "eperm"):
            config = next(
                item for item in make_configs((500,), smoke=True)
                if item.scaffold == "line" and item.arm == arm
            )
            model = model_library.build(
                "transformer", LINE.VOCAB, LINE.SEQ_LEN, LINE.PAD,
                d=config.width, layers=config.layers,
            )
            gap_head = torch.nn.Linear(config.width, 1)
            loss, parts = training_loss(
                model, gap_head, LINE, config, rng, torch.device("cpu")
            )
            self.assertTrue(torch.isfinite(loss))
            if arm.startswith("a"):
                self.assertEqual(parts["loss_invariant"], 0.0)
                self.assertEqual(parts["loss_latent"], 0.0)
                self.assertEqual(parts["loss_mapping"], 0.0)
            else:
                self.assertGreater(parts["loss_latent"], 0.0)
                self.assertGreater(parts["loss_mapping"], 0.0)

    def test_arm_fields(self):
        self.assertEqual(arm_fields("a00_confounded"), (0, "confounded", 0, 0))
        self.assertEqual(arm_fields("a50_diverse"), (6, "diverse", 0, 0))
        self.assertEqual(arm_fields("imposed"), (0, "confounded", 1, 0))
        self.assertEqual(arm_fields("eperm"), (0, "confounded", 1, 1))

    def test_checkpoint_resume_restores_model_and_rng(self):
        config = next(
            item for item in make_configs((500,), smoke=True)
            if item.scaffold == "line" and item.arm == "a00_confounded"
        )
        torch.manual_seed(11)
        model = model_library.build(
            "transformer", LINE.VOCAB, LINE.SEQ_LEN, LINE.PAD,
            d=config.width, layers=config.layers,
        )
        head = torch.nn.Linear(config.width, 1)
        optimizer = optimizer_for(model, head, config)
        rng = np.random.default_rng(12)
        expected_model = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoints"
            save_checkpoint(
                checkpoint_dir / "step-000000.pt", model, head, optimizer,
                rng, config, 0,
            )
            (root / "result.jsonl").write_text(json.dumps({"step": 0}) + "\n")
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)
            rng.random(5)
            step, mode = restore(
                root / "result.jsonl", checkpoint_dir, model, head,
                optimizer, rng, config,
            )
            self.assertEqual((step, mode), (0, "a"))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[key]))
            expected_rng = np.random.default_rng(12)
            self.assertAlmostEqual(rng.random(), expected_rng.random())

    def test_step_extension_is_explicit_and_exact(self):
        original = next(
            item for item in make_configs((500,), smoke=False)
            if item.scaffold == "line" and item.arm == "a00_confounded"
        )
        extended = replace(original, steps=REPAIR_STEPS)
        torch.manual_seed(21)
        model = model_library.build(
            "transformer", LINE.VOCAB, LINE.SEQ_LEN, LINE.PAD,
            d=original.width, layers=original.layers,
        )
        head = torch.nn.Linear(original.width, 1)
        optimizer = optimizer_for(model, head, original)
        rng = np.random.default_rng(22)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoints"
            save_checkpoint(
                checkpoint_dir / "step-010000.pt", model, head, optimizer,
                rng, original, 10_000,
            )
            (root / "result.jsonl").write_text(
                json.dumps({"step": 10_000}) + "\n"
            )
            with self.assertRaises(ValueError):
                restore(
                    root / "result.jsonl", checkpoint_dir, model, head,
                    optimizer, rng, extended,
                )
            step, mode = restore(
                root / "result.jsonl", checkpoint_dir, model, head,
                optimizer, rng, extended, allow_step_extension=True,
            )
            self.assertEqual((step, mode), (10_000, "a"))


if __name__ == "__main__":
    unittest.main()
