"""Train one cell of the compression x difficulty x abstraction experiment.

Run this module from the repository root, for example:

    python -m experiments.training_axes.run --smoke --device cpu
    python -m experiments.training_axes.run \
        --compression medium --difficulty mixed --abstraction invariant --seed 0

The normal run never evaluates the locked test sets. It saves model checkpoints
that are evaluated later by ``evaluate_locked.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from experiments.panel import model as model_library
from experiments.training_axes import metrics, task
from experiments.training_axes import task_differences
from experiments.training_axes.design import (
    ABSTRACTION_LEVELS,
    CAPACITY_LEVELS,
    COMPRESSION_LEVELS,
    DIFFICULTY_LEVELS,
    ExperimentConfig,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints"

TASKS = {"order": task, "differences": task_differences}


def make_config(args) -> ExperimentConfig:
    task_module = TASKS[args.task]
    difficulty_levels = getattr(task_module, "DIFFICULTY_LEVELS", DIFFICULTY_LEVELS)
    width = args.width if args.width is not None else CAPACITY_LEVELS[args.capacity]
    prefix = f"{args.run_prefix}_" if args.run_prefix else ""
    task_tag = "" if args.task == "order" else f"t-{args.task}_"
    capacity_tag = f"_p-{args.capacity}" if args.capacity != "standard" else ""
    config_id = (
        f"{prefix}{task_tag}c-{args.compression}_d-{args.difficulty}_"
        f"a-{args.abstraction}{capacity_tag}_s-{args.seed:02d}"
    )
    config = ExperimentConfig(
        config_id=config_id,
        seed=args.seed,
        compression=args.compression,
        weight_decay=COMPRESSION_LEVELS[args.compression],
        difficulty=args.difficulty,
        train_dists=difficulty_levels[args.difficulty],
        abstraction=args.abstraction,
        task_name=args.task,
        capacity=args.capacity,
        family=args.family,
        width=width,
        layers=args.layers,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        invariant_weight=args.invariant_weight,
        representation_weight=args.representation_weight,
        latent_weight=args.latent_weight,
        mapping_weight=args.mapping_weight,
    )
    if args.smoke:
        config = replace(config, width=48, layers=2, batch_size=32, steps=30)
    return config


def optimizer_for(model, auxiliary_head, config: ExperimentConfig):
    """AdamW with decay restricted to matrix-valued parameters."""
    decay, no_decay = [], []
    named_parameters = list(model.named_parameters()) + [
        (f"auxiliary.{name}", parameter)
        for name, parameter in auxiliary_head.named_parameters()
    ]
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "ln" in name.lower() or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def training_loss(model, auxiliary_head, batch, config, device, task_module=task):
    X1, X2, labels, signed_gap, _, _ = batch
    pair_count = len(labels)
    X = torch.as_tensor(np.concatenate([X1, X2]), device=device)
    y = torch.as_tensor(np.tile(labels, 2), device=device)
    gap = torch.as_tensor(np.tile(signed_gap, 2), device=device)

    residuals, mask = model.residuals(X)
    last = (~mask).sum(1) - 1
    hidden = residuals[-1][torch.arange(len(X), device=device), last]
    output = model.readout(residuals[-1], mask)
    binary = torch.nn.functional.binary_cross_entropy_with_logits(output, y)
    total = binary
    invariant = torch.zeros((), device=device)
    representation = torch.zeros((), device=device)
    latent = torch.zeros((), device=device)
    mapping = torch.zeros((), device=device)

    if config.abstraction in {"invariant", "latent"}:
        probability = torch.sigmoid(output)
        invariant = torch.mean((probability[:pair_count] - probability[pair_count:]) ** 2)
        total = total + config.invariant_weight * invariant

    if config.abstraction == "latent":
        hidden_one = torch.nn.functional.normalize(hidden[:pair_count], dim=1)
        hidden_two = torch.nn.functional.normalize(hidden[pair_count:], dim=1)
        representation = torch.mean(1.0 - torch.sum(hidden_one * hidden_two, dim=1))
        latent_prediction = auxiliary_head(hidden).squeeze(1)
        latent = torch.nn.functional.mse_loss(latent_prediction, gap)
        # The strongest abstraction condition is given the certified rendering
        # map: tokens at the same latent rank in different alphabets should use
        # a common centered embedding. This is the explicit-abstraction
        # analogue of supplying unit-conversion or schema correspondences.
        item_embedding = model.emb.weight[:task_module.N_ITEM_TOKENS].reshape(
            task_module.N_ALPH, task_module.N_ITEMS, -1
        )
        centered = item_embedding - item_embedding.mean(1, keepdim=True)
        shared = centered.mean(0, keepdim=True)
        mapping = torch.nn.functional.mse_loss(centered, shared.expand_as(centered))
        total = (
            total
            + config.representation_weight * representation
            + config.latent_weight * latent
            + config.mapping_weight * mapping
        )

    parts = {
        "loss": float(total.detach().cpu()),
        "loss_binary": float(binary.detach().cpu()),
        "loss_invariant": float(invariant.detach().cpu()),
        "loss_representation": float(representation.detach().cpu()),
        "loss_latent": float(latent.detach().cpu()),
        "loss_mapping": float(mapping.detach().cpu()),
    }
    return total, parts


def checkpoint_steps(total: int, smoke: bool) -> list[int]:
    if smoke:
        return [0, 10, 30]
    proposed = [0, 50, 100, 200, 400, 700, 1_000, 2_000, 4_000, 7_000, total]
    return sorted({step for step in proposed if step <= total} | {total})


def save_checkpoint(
    path: Path, model, auxiliary_head, optimizer, rng, config, step: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "step": step,
        "model": model.state_dict(),
        "auxiliary_head": auxiliary_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def restore_if_available(
    results_path, checkpoint_dir, model, auxiliary_head, optimizer, rng, config
) -> int:
    """Resume exactly from the last measured checkpoint, or skip a finished run."""
    if not results_path.exists():
        return 0
    with results_path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        return 0
    last_step = int(rows[-1]["step"])
    if last_step >= config.steps:
        print(f"already complete at step {last_step}; skipping {config.config_id}")
        return -1
    checkpoint_path = checkpoint_dir / f"step-{last_step:06d}.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"partial result stops at {last_step}, but {checkpoint_path} is missing; "
            "move the partial result aside or pass --overwrite"
        )
    saved = torch.load(checkpoint_path, map_location=next(model.parameters()).device,
                       weights_only=True)
    model.load_state_dict(saved["model"])
    auxiliary_head.load_state_dict(saved["auxiliary_head"])
    optimizer.load_state_dict(saved["optimizer"])
    rng.bit_generator.state = saved["numpy_rng"]
    torch.set_rng_state(saved["torch_rng"].cpu())
    if torch.cuda.is_available() and "cuda_rng" in saved:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    print(f"resuming {config.config_id} from step {last_step}")
    return last_step


def train(
    config, device, results_path: Path, checkpoint_dir: Path, args, task_module=task
) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = model_library.build(
        config.family, task_module.VOCAB, task_module.SEQ_LEN, task_module.PAD,
        d=config.width, layers=config.layers,
    ).to(device)
    auxiliary_head = torch.nn.Linear(config.width, 1).to(device)
    optimizer = optimizer_for(model, auxiliary_head, config)
    datasets = task_module.evaluation_sets(
        seed=0, n=128 if args.smoke else args.eval_size,
        train_dists=config.train_dists,
    )
    steps_to_measure = checkpoint_steps(config.steps, args.smoke)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        start_step = 0
        mode = "w"
    else:
        start_step = restore_if_available(
            results_path, checkpoint_dir, model, auxiliary_head, optimizer, rng, config
        )
        if start_step < 0:
            return False
        mode = "a" if results_path.exists() else "w"
    started = time.time()
    latest_loss = {
        "loss": float("nan"), "loss_binary": float("nan"),
        "loss_invariant": 0.0, "loss_representation": 0.0, "loss_latent": 0.0,
        "loss_mapping": 0.0,
    }

    for step in range(start_step, config.steps + 1):
        # A resumed checkpoint already has a result row. Continue with the
        # update following that checkpoint rather than measuring it twice.
        if step in steps_to_measure and not (step == start_step and mode == "a"):
            measurements = metrics.evaluate_checkpoint(
                model, datasets, device, seed=config.seed + step,
                include_locked=args.unlock_test,
                structural_pairs=64 if args.smoke else args.structural_pairs,
                task_module=task_module,
            )
            row = {
                **asdict(config), "train_dists": list(config.train_dists),
                "step": step, "elapsed_seconds": time.time() - started,
                **latest_loss, **measurements,
            }
            with results_path.open(mode) as handle:
                handle.write(json.dumps(row, allow_nan=True) + "\n")
            mode = "a"
            if not args.no_save_checkpoints:
                save_checkpoint(
                    checkpoint_dir / f"step-{step:06d}.pt",
                    model, auxiliary_head, optimizer, rng, config, step,
                )
            print(
                f"step={step:>6} loss={latest_loss['loss_binary']:.4f} "
                f"source={measurements['source_accuracy']:.3f} "
                f"calib={measurements['calib_combo_accuracy']:.3f} "
                f"probe={measurements['probe_target_r2']:+.3f} "
                f"IIA={measurements['causal_source_to_target']:.3f}"
            )
        if step == config.steps:
            break

        model.train()
        auxiliary_head.train()
        optimizer.zero_grad(set_to_none=True)
        batch = task_module.paired_batch(
            rng, max(1, config.batch_size // 2), config.train_dists
        )
        loss, latest_loss = training_loss(
            model, auxiliary_head, batch, config, device, task_module=task_module
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(auxiliary_head.parameters()), 1.0
        )
        warmup = min(200, max(1, config.steps // 10))
        learning_rate = config.learning_rate * min(1.0, (step + 1) / warmup)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", choices=COMPRESSION_LEVELS, default="medium")
    parser.add_argument("--difficulty", choices=DIFFICULTY_LEVELS, default="mixed")
    parser.add_argument("--task", choices=TASKS, default="order")
    parser.add_argument("--abstraction", choices=ABSTRACTION_LEVELS, default="latent")
    parser.add_argument("--capacity", choices=CAPACITY_LEVELS, default="standard")
    parser.add_argument("--run-prefix", choices=("v2", "v3"), default=None,
                        help="namespace results for a registered experiment revision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", choices=("transformer", "lstm"), default="transformer")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--invariant-weight", type=float, default=1.0)
    parser.add_argument("--representation-weight", type=float, default=0.1)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--mapping-weight", type=float, default=1.0)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--structural-pairs", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_module = TASKS[args.task]
    task_module.validate_design()
    config = make_config(args)
    device = model_library.get_device(args.device)
    suffix = "_smoke" if args.smoke else ""
    results_path = args.results_dir / f"{config.config_id}{suffix}.jsonl"
    checkpoint_dir = args.checkpoints_dir / f"{config.config_id}{suffix}"
    print(json.dumps(asdict(config), indent=2))
    print(f"device={device} test_sets={'UNLOCKED' if args.unlock_test else 'locked'}")
    changed = train(
        config, device, results_path, checkpoint_dir, args,
        task_module=task_module,
    )
    if changed:
        print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
