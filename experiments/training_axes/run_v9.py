"""Train one V9-A Selection/Emergence trunk."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.panel import model as model_library
from experiments.training_axes.design_v9 import V9Config
from experiments.training_axes.metrics_v9 import evaluate_checkpoint
from experiments.training_axes.run import checkpoint_steps, cpu_byte_rng_states
from experiments.training_axes.task_v9 import TASKS


def v9_checkpoint_steps(total, smoke=False):
    """V9 checkpoints, including the registered 20k repair horizon."""
    if smoke:
        return checkpoint_steps(total, True)
    proposed = list(checkpoint_steps(min(total, 10_000), False))
    if total > 10_000:
        proposed.extend((10_000, 12_000, 15_000, total))
    return sorted({int(step) for step in proposed if step <= total} | {int(total)})


def logical_embeddings(model, task, anchor_count):
    weights = model.effective_embedding_weights()
    indices = torch.as_tensor(
        task.item_token_ids(anchor_count), device=weights.device, dtype=torch.long
    )
    return weights[indices]


def optimizer_for(model, gap_head, config):
    decay, no_decay = [], []
    named = list(model.named_parameters()) + [
        (f"gap_head.{name}", value) for name, value in gap_head.named_parameters()
    ]
    for name, parameter in named:
        if parameter.ndim < 2 or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.learning_rate)


def final_hidden(model, tokens):
    residuals, mask = model.residuals(tokens)
    last = (~mask).sum(1) - 1
    hidden = residuals[-1][torch.arange(len(tokens), device=tokens.device), last]
    logits = model.readout(residuals[-1], mask, tokens)
    return hidden, logits


def training_loss(model, gap_head, task, config, rng, device):
    zero = torch.zeros((), device=device)
    invariant = mapping = latent = zero
    if config.imposed:
        pair_count = max(1, config.batch_size // 2)
        X1, X2, labels, gaps, _, _ = task.paired_training_batch(
            rng, pair_count, config.anchor_count
        )
        tokens = torch.as_tensor(np.concatenate([X1, X2]), device=device)
        target = torch.as_tensor(np.tile(labels, 2), device=device)
        gap = torch.as_tensor(np.tile(gaps, 2), device=device)
        hidden, logits = final_hidden(model, tokens)
        binary = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits)
        invariant = torch.mean(
            (probability[:pair_count] - probability[pair_count:]) ** 2
        )
        latent = torch.nn.functional.mse_loss(gap_head(hidden).squeeze(1), gap)
        blocks = logical_embeddings(model, task, config.anchor_count)
        blocks = blocks - blocks.mean(1, keepdim=True)
        if config.permuted:
            permutations = torch.as_tensor(
                task.MAPPING_PERMUTATIONS, device=device, dtype=torch.long
            )
            blocks = torch.gather(
                blocks, 1, permutations.unsqueeze(-1).expand_as(blocks)
            )
        shared = blocks.mean(0, keepdim=True)
        mapping = torch.nn.functional.mse_loss(blocks, shared.expand_as(blocks))
        total = binary + invariant + latent + mapping
    else:
        batch = task.training_batch(
            rng, config.batch_size, config.coverage, config.anchor_count
        )
        tokens = torch.as_tensor(batch.tokens, device=device)
        target = torch.as_tensor(batch.labels, device=device)
        _, logits = final_hidden(model, tokens)
        binary = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        total = binary
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_binary": float(binary.detach().cpu()),
        "loss_invariant": float(invariant.detach().cpu()),
        "loss_latent": float(latent.detach().cpu()),
        "loss_mapping": float(mapping.detach().cpu()),
    }


def save_checkpoint(path, model, gap_head, optimizer, rng, config, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config), "step": int(step),
        "model": model.state_dict(), "gap_head": gap_head.state_dict(),
        "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = cpu_byte_rng_states(torch.cuda.get_rng_state_all())
    torch.save(payload, path)


def restore(
    results_path, checkpoint_dir, model, gap_head, optimizer, rng, config,
    *, allow_step_extension=False,
):
    if not results_path.exists():
        return 0, "w"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    if not rows:
        return 0, "w"
    last = int(rows[-1]["step"])
    if last >= config.steps:
        print(f"complete; skipping {config.config_id}")
        return -1, "a"
    path = checkpoint_dir / f"step-{last:06d}.pt"
    if not path.exists():
        raise RuntimeError(f"partial V9 result has no checkpoint: {path}")
    saved = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    expected = asdict(config)
    if saved["config"] != expected:
        saved_config = dict(saved["config"])
        saved_steps = int(saved_config.pop("steps"))
        expected_without_steps = dict(expected)
        expected_steps = int(expected_without_steps.pop("steps"))
        valid_extension = (
            allow_step_extension
            and saved_config == expected_without_steps
            and saved_steps < expected_steps
            and last == saved_steps
        )
        if not valid_extension:
            raise ValueError(f"V9 checkpoint config mismatch: {path}")
    model.load_state_dict(saved["model"])
    gap_head.load_state_dict(saved["gap_head"])
    optimizer.load_state_dict(saved["optimizer"])
    rng.bit_generator.state = saved["numpy_rng"]
    torch.set_rng_state(saved["torch_rng"].cpu())
    if torch.cuda.is_available() and "cuda_rng" in saved:
        torch.cuda.set_rng_state_all(cpu_byte_rng_states(saved["cuda_rng"]))
    print(f"resuming {config.config_id} from step {last}")
    return last, "a"


def train(config, device, results_path, checkpoint_dir, *, smoke=False,
          overwrite=False, no_save_checkpoints=False, eval_size=1000,
          structural_pairs=256, allow_step_extension=False):
    task = TASKS[config.scaffold]
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = model_library.build(
        "transformer", task.VOCAB, task.SEQ_LEN, task.PAD,
        d=config.width, layers=config.layers, bottleneck_dim=0,
    ).to(device)
    gap_head = torch.nn.Linear(config.width, 1).to(device)
    optimizer = optimizer_for(model, gap_head, config)
    datasets = task.evaluation_sets(
        seed=0, n=128 if smoke else eval_size,
        anchor_count=config.anchor_count, include_locked=False,
    )
    steps = v9_checkpoint_steps(config.steps, smoke)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        start, mode = 0, "w"
    else:
        start, mode = restore(
            results_path, checkpoint_dir, model, gap_head, optimizer, rng, config,
            allow_step_extension=allow_step_extension,
        )
        if start < 0:
            return
    latest = {
        "loss": float("nan"), "loss_binary": float("nan"),
        "loss_invariant": 0.0, "loss_latent": 0.0, "loss_mapping": 0.0,
    }
    started = time.time()
    for step in range(start, config.steps + 1):
        if step in steps and not (step == start and mode == "a"):
            measured = evaluate_checkpoint(
                model, task, datasets, device, config.seed + step,
                config.anchor_count, include_locked=False,
                structural_pairs=64 if smoke else structural_pairs,
            )
            row = {
                **asdict(config), "step": int(step),
                "elapsed_seconds": time.time() - started,
                **latest, **measured,
            }
            with results_path.open(mode) as handle:
                handle.write(json.dumps(row, allow_nan=True) + "\n")
            mode = "a"
            if not no_save_checkpoints:
                save_checkpoint(
                    checkpoint_dir / f"step-{step:06d}.pt",
                    model, gap_head, optimizer, rng, config, step,
                )
            print(
                f"{config.config_id} step={step:>6} "
                f"source={measured['source_accuracy']:.3f} "
                f"dev_x={measured['dev_x_nonanchor_accuracy']:.3f} "
                f"dev_s={measured['dev_s_nonanchor_accuracy']:.3f} "
                f"O={measured['organization_dev_s']:.3f}"
            )
        if step == config.steps:
            break
        model.train(); gap_head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, latest = training_loss(model, gap_head, task, config, rng, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(gap_head.parameters()), 1.0
        )
        warmup = min(200, max(1, config.steps // 10))
        learning_rate = config.learning_rate * min(1.0, (step + 1) / warmup)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--structural-pairs", type=int, default=256)
    parser.add_argument("--allow-step-extension", action="store_true")
    args = parser.parse_args()
    config = V9Config(**json.loads(args.config_json))
    device = model_library.get_device(args.device)
    train(
        config, device,
        args.results_dir / f"{config.config_id}.jsonl",
        args.checkpoints_dir / config.config_id,
        smoke=args.smoke, overwrite=args.overwrite,
        no_save_checkpoints=args.no_save_checkpoints,
        eval_size=args.eval_size, structural_pairs=args.structural_pairs,
        allow_step_extension=args.allow_step_extension,
    )


if __name__ == "__main__":
    main()
