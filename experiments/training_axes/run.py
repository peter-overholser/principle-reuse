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
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from experiments.panel import model as model_library
from experiments.training_axes import metrics, task
from experiments.training_axes import task_differences
from experiments.training_axes import task_v8
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

TASKS = {
    "order": task,
    "differences": task_differences,
    "order_v8": task_v8.ORDER,
    "differences_v8": task_v8.DIFFERENCES,
}


def make_config(args) -> ExperimentConfig:
    task_module = TASKS[args.task]
    difficulty_levels = getattr(task_module, "DIFFICULTY_LEVELS", DIFFICULTY_LEVELS)
    width = args.width if args.width is not None else CAPACITY_LEVELS[args.capacity]
    prefix = f"{args.run_prefix}_" if args.run_prefix else ""
    task_tag = "" if args.task == "order" else f"t-{args.task}_"
    capacity_tag = f"_p-{args.capacity}" if args.capacity != "standard" else ""
    generated_id = (
        f"{prefix}{task_tag}c-{args.compression}_d-{args.difficulty}_"
        f"a-{args.abstraction}{capacity_tag}_s-{args.seed:02d}"
    )
    config_id = args.config_id or generated_id
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
        bottleneck_dim=args.bottleneck_dim,
        aux_schedule=args.aux_schedule,
        aux_fraction=args.aux_fraction,
        aux_updates=args.aux_updates,
        rule_signal=args.rule_signal,
        bridge_signal=args.bridge_signal,
        reset_step=args.reset_step,
        v3_arm=args.v3_arm,
        source_only_eval=args.source_only_eval,
        aux_mode=args.aux_mode,
        aux_bank_size=args.aux_bank_size,
        aux_batch_size=args.aux_batch_size,
        v4_arm=args.v4_arm,
        v5_arm=args.v5_arm,
        representation_signal=args.representation_signal,
        mapping_signal=args.mapping_signal,
        output_invariant_signal=args.output_invariant_signal,
        mapping_control=args.mapping_control,
        gap_head_mode=args.gap_head_mode,
        v8_split=args.v8_split,
        v8_arm=args.v8_arm,
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


def auxiliary_active(config: ExperimentConfig, step: int) -> bool:
    """Return whether the registered V3 auxiliary loss is active this update."""
    schedule = config.aux_schedule
    if schedule == "legacy":
        raise ValueError("legacy objectives are selected by abstraction, not a schedule")
    if schedule == "none":
        return False
    if schedule == "early":
        return step < config.aux_updates
    if schedule == "late":
        return step >= config.steps - config.aux_updates
    if schedule == "throughout":
        count = int(round(config.steps * config.aux_fraction))
        if count <= 0:
            return False
        if count >= config.steps:
            return True
        # This Bresenham-style rule places exactly ``count`` active updates as
        # evenly as possible without sampling another RNG stream.
        return ((step + 1) * count) // config.steps > (step * count) // config.steps
    raise ValueError(f"unknown auxiliary schedule: {schedule}")


def auxiliary_updates_before(config: ExperimentConfig, step: int) -> int:
    """Number of active auxiliary updates strictly before ``step``."""
    if config.aux_schedule == "none":
        return 0
    if config.aux_schedule == "early":
        return min(int(step), int(config.aux_updates))
    if config.aux_schedule == "late":
        return max(0, int(step) - (int(config.steps) - int(config.aux_updates)))
    if config.aux_schedule == "throughout":
        count = int(round(config.steps * config.aux_fraction))
        return min(count, (int(step) * count) // int(config.steps))
    if config.aux_schedule == "legacy":
        return 0
    raise ValueError(f"unknown auxiliary schedule: {config.aux_schedule}")


def validate_v3_config(config: ExperimentConfig) -> None:
    if config.aux_schedule == "legacy":
        return
    if config.aux_schedule not in {"none", "throughout", "early", "late"}:
        raise ValueError(f"unknown auxiliary schedule: {config.aux_schedule}")
    if not 0.0 <= config.aux_fraction <= 1.0:
        raise ValueError("aux_fraction must be in [0, 1]")
    if not 0 <= config.aux_updates <= config.steps:
        raise ValueError("aux_updates must be in [0, steps]")
    if config.rule_signal not in {0, 1} or config.bridge_signal not in {0, 1}:
        raise ValueError("rule_signal and bridge_signal must be 0 or 1")
    if config.representation_signal not in {-1, 0, 1}:
        raise ValueError("representation_signal must be -1, 0, or 1")
    if config.mapping_signal not in {-1, 0, 1}:
        raise ValueError("mapping_signal must be -1, 0, or 1")
    if config.output_invariant_signal not in {0, 1}:
        raise ValueError("output_invariant_signal must be 0 or 1")
    if config.mapping_control not in {"valid", "permuted"}:
        raise ValueError("mapping_control must be valid or permuted")
    if config.gap_head_mode not in {"shared", "separate"}:
        raise ValueError("gap_head_mode must be shared or separate")
    if config.bottleneck_dim < 0 or config.bottleneck_dim > config.width:
        raise ValueError("bottleneck_dim must be in [0, width]")
    if config.bottleneck_dim == 0 and not config.v8_arm:
        raise ValueError("focused V3--V5 runs require bottleneck_dim in [1, width]")
    if config.reset_step and not 0 < config.reset_step < config.steps:
        raise ValueError("reset_step must be strictly between 0 and steps")
    if config.aux_mode not in {
        "online_global", "bank", "permuted", "unbridged_test"
    }:
        raise ValueError(f"unknown auxiliary mode: {config.aux_mode}")
    if config.aux_bank_size < 0 or config.aux_batch_size < 0:
        raise ValueError("auxiliary bank and batch sizes must be non-negative")
    if config.aux_mode != "online_global" and config.aux_schedule != "none":
        if config.aux_bank_size <= 0 or config.aux_batch_size <= 0:
            raise ValueError("banked auxiliary schedules require a non-empty bank/batch")


def resolved_structural_signals(config: ExperimentConfig) -> tuple[bool, bool]:
    """Resolve hidden and embedding signals without changing old studies."""
    representation = (
        bool(config.bridge_signal)
        if config.representation_signal == -1
        else bool(config.representation_signal)
    )
    mapping = (
        bool(config.bridge_signal)
        if config.mapping_signal == -1
        else bool(config.mapping_signal)
    )
    return representation, mapping


def logical_item_embeddings(model, task_module):
    weights = (
        model.effective_embedding_weights()
        if hasattr(model, "effective_embedding_weights") else model.emb.weight
    )
    token_ids = getattr(task_module, "ITEM_TOKEN_IDS", None)
    if token_ids is None:
        return weights[:task_module.N_ITEM_TOKENS].reshape(
            task_module.N_ALPH, task_module.N_ITEMS, -1
        )
    indices = torch.as_tensor(token_ids, device=weights.device, dtype=torch.long)
    return weights[indices]


def mapping_blocks(item_embedding, config, task_module):
    centered = item_embedding - item_embedding.mean(1, keepdim=True)
    if config.mapping_control == "valid":
        return centered
    permutations = torch.as_tensor(
        task_module.MAPPING_PERMUTATIONS,
        device=centered.device, dtype=torch.long,
    )
    gather = permutations.unsqueeze(-1).expand_as(centered)
    return torch.gather(centered, 1, gather)


def gap_prediction(auxiliary_head, hidden, meta_first, meta_second, config, device):
    prediction = auxiliary_head(hidden)
    if config.gap_head_mode == "shared":
        return prediction.squeeze(1)
    alphabets = torch.as_tensor(
        [row["alphabet"] for row in meta_first]
        + [row["alphabet"] for row in meta_second],
        device=device, dtype=torch.long,
    )
    return prediction.gather(1, alphabets.unsqueeze(1)).squeeze(1)


def _bank_auxiliary_loss(
    model, auxiliary_head, batch, config, device, task_module
):
    """V4 loss on explicitly annotated examples and observed token anchors only."""
    X1, X2, _, signed_gap, meta_first, meta_second = batch
    pair_count = len(signed_gap)
    X = torch.as_tensor(np.concatenate([X1, X2]), device=device)
    gap = torch.as_tensor(np.tile(signed_gap, 2), device=device)
    residuals, mask = model.residuals(X)
    last = (~mask).sum(1) - 1
    hidden = residuals[-1][torch.arange(len(X), device=device), last]

    zero = torch.zeros((), device=device)
    representation, latent, mapping = zero, zero, zero
    total = zero
    representation_signal, mapping_signal = resolved_structural_signals(config)
    if representation_signal:
        hidden_one = torch.nn.functional.normalize(hidden[:pair_count], dim=1)
        hidden_two = torch.nn.functional.normalize(hidden[pair_count:], dim=1)
        representation = torch.mean(1.0 - torch.sum(hidden_one * hidden_two, dim=1))
        total = total + config.representation_weight * representation
    if mapping_signal:
        item_embedding = logical_item_embeddings(model, task_module)
        # Centering is a normalization, not an all-token correspondence target;
        # detaching the means prevents unannotated tokens receiving gradients.
        means = item_embedding.mean(1, keepdim=True).detach()
        left, right = [], []
        for index, (one, two) in enumerate(zip(meta_first, meta_second)):
            for value in sorted(set(one["pair"])):
                target_value = int(value)
                if config.aux_mode == "permuted":
                    target_value = (target_value + 1 + index % 8) % task_module.N_ITEMS
                    if target_value == value:
                        target_value = (target_value + 1) % task_module.N_ITEMS
                left.append(item_embedding[one["alphabet"], value] - means[one["alphabet"], 0])
                right.append(
                    item_embedding[two["alphabet"], target_value]
                    - means[two["alphabet"], 0]
                )
        mapping = torch.nn.functional.mse_loss(torch.stack(left), torch.stack(right))
        total = total + config.mapping_weight * mapping
    if config.rule_signal:
        latent_prediction = gap_prediction(
            auxiliary_head, hidden, meta_first, meta_second, config, device
        )
        latent = torch.nn.functional.mse_loss(latent_prediction, gap)
        total = total + config.latent_weight * latent
    return total, representation, latent, mapping


def training_loss(
    model, auxiliary_head, batch, config, device, step=0, task_module=task,
    auxiliary_batch=None, return_components=False,
):
    X1, X2, labels, signed_gap, meta_first, meta_second = batch
    pair_count = len(labels)
    X = torch.as_tensor(np.concatenate([X1, X2]), device=device)
    y = torch.as_tensor(np.tile(labels, 2), device=device)
    gap = torch.as_tensor(np.tile(signed_gap, 2), device=device)

    residuals, mask = model.residuals(X)
    last = (~mask).sum(1) - 1
    hidden = residuals[-1][torch.arange(len(X), device=device), last]
    output = model.readout(residuals[-1], mask, X)
    binary = torch.nn.functional.binary_cross_entropy_with_logits(output, y)
    total = binary
    invariant = torch.zeros((), device=device)
    representation = torch.zeros((), device=device)
    latent = torch.zeros((), device=device)
    mapping = torch.zeros((), device=device)

    legacy = config.aux_schedule == "legacy"
    active = False if legacy else auxiliary_active(config, step)
    use_rule = False
    use_bridge = False
    use_representation = False
    use_mapping = False

    # Keep the old objective in its own branch so V1/V2/Task-2 retain the
    # same operations and floating-point addition order.
    if legacy:
        if config.abstraction in {"invariant", "latent"}:
            probability = torch.sigmoid(output)
            invariant = torch.mean(
                (probability[:pair_count] - probability[pair_count:]) ** 2
            )
            total = total + config.invariant_weight * invariant
        if config.abstraction == "latent":
            use_rule = True
            use_bridge = True
            use_representation = True
            use_mapping = True
            hidden_one = torch.nn.functional.normalize(hidden[:pair_count], dim=1)
            hidden_two = torch.nn.functional.normalize(hidden[pair_count:], dim=1)
            representation = torch.mean(
                1.0 - torch.sum(hidden_one * hidden_two, dim=1)
            )
            latent_prediction = gap_prediction(
                auxiliary_head, hidden, meta_first, meta_second, config, device
            )
            latent = torch.nn.functional.mse_loss(latent_prediction, gap)
            item_embedding = logical_item_embeddings(model, task_module)
            centered = mapping_blocks(item_embedding, config, task_module)
            shared = centered.mean(0, keepdim=True)
            mapping = torch.nn.functional.mse_loss(
                centered, shared.expand_as(centered)
            )
            total = (
                total
                + config.representation_weight * representation
                + config.latent_weight * latent
                + config.mapping_weight * mapping
            )
    elif config.aux_mode == "online_global":
        use_rule = active and bool(config.rule_signal)
        resolved_representation, resolved_mapping = resolved_structural_signals(config)
        use_representation = active and resolved_representation
        use_mapping = active and resolved_mapping
        use_bridge = use_representation or use_mapping

    if not legacy and active and bool(config.output_invariant_signal):
        probability = torch.sigmoid(output)
        invariant = torch.mean(
            (probability[:pair_count] - probability[pair_count:]) ** 2
        )
        total = total + config.invariant_weight * invariant

    if not legacy and use_representation:
        hidden_one = torch.nn.functional.normalize(hidden[:pair_count], dim=1)
        hidden_two = torch.nn.functional.normalize(hidden[pair_count:], dim=1)
        representation = torch.mean(1.0 - torch.sum(hidden_one * hidden_two, dim=1))
        total = total + config.representation_weight * representation
    if not legacy and use_mapping:
        item_embedding = logical_item_embeddings(model, task_module)
        centered = mapping_blocks(item_embedding, config, task_module)
        shared = centered.mean(0, keepdim=True)
        mapping = torch.nn.functional.mse_loss(centered, shared.expand_as(centered))
        total = total + config.mapping_weight * mapping

    if not legacy and use_rule:
        latent_prediction = gap_prediction(
            auxiliary_head, hidden, meta_first, meta_second, config, device
        )
        latent = torch.nn.functional.mse_loss(latent_prediction, gap)
        total = total + config.latent_weight * latent

    if not legacy and config.aux_mode != "online_global" and active:
        if auxiliary_batch is None:
            raise ValueError("active banked auxiliary loss requires an auxiliary batch")
        use_rule = bool(config.rule_signal)
        use_representation, use_mapping = resolved_structural_signals(config)
        use_bridge = use_representation or use_mapping
        auxiliary, representation, latent, mapping = _bank_auxiliary_loss(
            model, auxiliary_head, auxiliary_batch, config, device, task_module
        )
        total = total + auxiliary

    parts = {
        "loss": float(total.detach().cpu()),
        "loss_binary": float(binary.detach().cpu()),
        "loss_invariant": float(invariant.detach().cpu()),
        "loss_representation": float(representation.detach().cpu()),
        "loss_latent": float(latent.detach().cpu()),
        "loss_mapping": float(mapping.detach().cpu()),
        "aux_active": float(active),
        "rule_active": float(use_rule),
        "bridge_active": float(use_bridge),
        "representation_active": float(use_representation),
        "mapping_active": float(use_mapping),
        "output_invariant_active": float(
            active and bool(config.output_invariant_signal)
        ),
        "loss_auxiliary": float((total - binary).detach().cpu()),
        "core_reset": 0.0,
    }
    if return_components:
        components = {
            "binary": binary,
            "invariant": config.invariant_weight * invariant,
            "representation": config.representation_weight * representation,
            "latent": config.latent_weight * latent,
            "mapping": config.mapping_weight * mapping,
        }
        return total, parts, components
    return total, parts


def gradient_norm_diagnostics(components, parameters):
    """Norm and share of each weighted loss component's parameter gradient."""
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    norms = {}
    for name, component in components.items():
        if not component.requires_grad:
            norms[name] = 0.0
            continue
        gradients = torch.autograd.grad(
            component, parameters, retain_graph=True, allow_unused=True
        )
        squared = torch.zeros((), device=component.device)
        for gradient in gradients:
            if gradient is not None:
                squared = squared + torch.sum(gradient.detach().float().square())
        norms[name] = float(torch.sqrt(squared).cpu())
    denominator = sum(norms.values())
    result = {}
    for name, value in norms.items():
        result[f"gradient_norm_{name}"] = value
        result[f"gradient_share_{name}"] = (
            value / denominator if denominator > 0 else 0.0
        )
    result["gradient_norm_component_sum"] = denominator
    return result


def prepare_auxiliary_bank(config, task_module):
    if config.aux_mode == "online_global" or config.aux_bank_size == 0:
        return None, {
            "aux_bank_hash": "none", "aux_unique_queries": 0,
            "aux_anchor_alphabets": 0, "aux_anchor_items": 0,
        }
    if task_module is not task_differences:
        raise ValueError("banked auxiliary supervision is implemented for differences")
    scope = "unbridged_test" if config.aux_mode == "unbridged_test" else "all"
    control = "permuted" if config.aux_mode == "permuted" else "valid"
    bank = task_module.auxiliary_bank(
        config.seed, config.aux_bank_size, config.train_dists,
        scope=scope, control=control,
    )
    digest = hashlib.sha256()
    for array in bank[:4]:
        digest.update(np.asarray(array).tobytes())
    for first, second in zip(bank[4], bank[5]):
        digest.update(repr((
            first["alphabet"], first["grammar"], first["pair"],
            second["alphabet"], second["grammar"], second["pair"],
        )).encode())
    alphabets = {
        meta["alphabet"] for side in (bank[4], bank[5]) for meta in side
    }
    anchors = {
        (meta["alphabet"], value)
        for side in (bank[4], bank[5]) for meta in side
        for value in meta["pair"]
    }
    return bank, {
        "aux_bank_hash": digest.hexdigest(),
        "aux_unique_queries": int(config.aux_bank_size),
        "aux_anchor_alphabets": len(alphabets),
        "aux_anchor_items": len(anchors),
    }


def checkpoint_steps(
    total: int, smoke: bool, config: ExperimentConfig | None = None
) -> list[int]:
    if smoke:
        proposed = [0, min(10, total), total]
    else:
        proposed = [0, 50, 100, 200, 400, 700, 1_000, 2_000, 4_000, 7_000, total]
    if config is not None and config.aux_schedule != "legacy":
        boundaries = []
        if config.aux_schedule == "early" and config.aux_updates:
            boundaries.append(config.aux_updates)
        if config.aux_schedule == "late" and config.aux_updates:
            boundaries.append(total - config.aux_updates)
        if config.reset_step:
            boundaries.append(config.reset_step)
        for boundary in boundaries:
            proposed.extend([boundary, boundary + 1])
    return sorted({step for step in proposed if step <= total} | {total})


def reset_bottleneck_core(model, optimizer, seed: int) -> None:
    """Reset only the learned bottleneck basis and its Adam moments."""
    model.reset_bottleneck(seed)
    for parameter in model.bottleneck.parameters():
        optimizer.state.pop(parameter, None)


def cpu_byte_rng_states(states):
    """Return CUDA RNG states in the format required by PyTorch generators.

    Checkpoints loaded with ``map_location=device`` also move RNG-state tensors
    to that device.  ``torch.cuda.set_rng_state_all`` nevertheless requires CPU
    ByteTensors, so normalize both old and newly written checkpoints here.
    """
    return [
        (
            state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            if isinstance(state, torch.Tensor)
            else torch.as_tensor(
                state, dtype=torch.uint8, device="cpu"
            ).contiguous()
        )
        for state in states
    ]


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
        payload["cuda_rng"] = cpu_byte_rng_states(
            torch.cuda.get_rng_state_all()
        )
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
        torch.cuda.set_rng_state_all(cpu_byte_rng_states(saved["cuda_rng"]))
    print(f"resuming {config.config_id} from step {last_step}")
    return last_step


def train(
    config, device, results_path: Path, checkpoint_dir: Path, args, task_module=task
) -> None:
    validate_v3_config(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = model_library.build(
        config.family, task_module.VOCAB, task_module.SEQ_LEN, task_module.PAD,
        d=config.width, layers=config.layers,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)
    auxiliary_head = torch.nn.Linear(
        config.width,
        task_module.N_ALPH if config.gap_head_mode == "separate" else 1,
    ).to(device)
    optimizer = optimizer_for(model, auxiliary_head, config)
    auxiliary_bank, auxiliary_bank_stats = prepare_auxiliary_bank(
        config, task_module
    )
    dataset_kwargs = {
        "seed": 0, "n": 128 if args.smoke else args.eval_size,
        "train_dists": config.train_dists,
    }
    if config.v8_arm:
        dataset_kwargs["include_locked"] = bool(args.unlock_test)
    datasets = task_module.evaluation_sets(**dataset_kwargs)
    steps_to_measure = checkpoint_steps(config.steps, args.smoke, config=config)
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
    diagnostic_steps = set()
    diagnostics_path = None
    if config.v8_arm:
        diagnostic_steps = set(range(50, min(2_000, config.steps) + 1, 50))
        diagnostic_steps.update(
            step for step in (4_000, 7_000, 10_000) if step <= config.steps
        )
        if args.smoke:
            diagnostic_steps.add(config.steps)
        diagnostics_path = results_path.parent / "_diagnostics" / results_path.name
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            diagnostics_path.write_text("")
        elif start_step == 0 and diagnostics_path.exists():
            raise RuntimeError(
                f"diagnostics exist without resumable results: {diagnostics_path}; "
                "move them aside or pass --overwrite"
            )
        elif diagnostics_path.exists():
            retained = [
                line for line in diagnostics_path.read_text().splitlines()
                if line and int(json.loads(line)["step"]) <= start_step
            ]
            diagnostics_path.write_text(
                "".join(f"{line}\n" for line in retained)
            )
    latest_loss = {
        "loss": float("nan"), "loss_binary": float("nan"),
        "loss_invariant": 0.0, "loss_representation": 0.0, "loss_latent": 0.0,
        "loss_mapping": 0.0, "loss_auxiliary": 0.0,
        "aux_active": 0.0, "rule_active": 0.0,
        "bridge_active": 0.0, "representation_active": 0.0,
        "mapping_active": 0.0, "output_invariant_active": 0.0,
        "core_reset": 0.0,
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
                source_only=bool(config.source_only_eval),
            )
            row = {
                **asdict(config), "train_dists": list(config.train_dists),
                "step": step, "elapsed_seconds": time.time() - started,
                **auxiliary_bank_stats,
                "aux_updates_completed": auxiliary_updates_before(config, step),
                "aux_presentations_completed": (
                    auxiliary_updates_before(config, step) * config.aux_batch_size
                    if config.aux_mode != "online_global" else 0
                ),
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
                f"calib={measurements.get('calib_combo_accuracy', float('nan')):.3f} "
                f"probe={measurements.get('probe_target_r2', float('nan')):+.3f} "
                f"IIA={measurements.get('causal_source_to_target', float('nan')):.3f}"
            )
        if step == config.steps:
            break

        reset_now = bool(config.reset_step and step == config.reset_step)
        if reset_now:
            reset_bottleneck_core(
                model, optimizer, seed=4_000_003 + 10_007 * config.seed + step
            )

        model.train()
        auxiliary_head.train()
        optimizer.zero_grad(set_to_none=True)
        batch = task_module.paired_batch(
            rng, max(1, config.batch_size // 2), config.train_dists
        )
        bank_batch = None
        if (
            auxiliary_bank is not None
            and auxiliary_active(config, step)
        ):
            ordinal = auxiliary_updates_before(config, step)
            bank_batch = task_module.auxiliary_bank_batch(
                auxiliary_bank, ordinal, config.aux_batch_size
            )
        completed_step = step + 1
        diagnose = completed_step in diagnostic_steps
        loss_result = training_loss(
            model, auxiliary_head, batch, config, device, step=step,
            task_module=task_module, auxiliary_batch=bank_batch,
            return_components=diagnose,
        )
        if diagnose:
            loss, latest_loss, components = loss_result
            gradient_diagnostics = gradient_norm_diagnostics(
                components,
                list(model.parameters()) + list(auxiliary_head.parameters()),
            )
            diagnostic_row = {
                "config_id": config.config_id,
                "seed": config.seed,
                "v8_split": config.v8_split,
                "v8_arm": config.v8_arm,
                "step": completed_step,
                **latest_loss,
                **gradient_diagnostics,
            }
            with diagnostics_path.open("a") as handle:
                handle.write(json.dumps(diagnostic_row, allow_nan=True) + "\n")
        else:
            loss, latest_loss = loss_result
        latest_loss["core_reset"] = float(reset_now)
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
    parser.add_argument("--config-id", default=None,
                        help="exact registered ID (used by focused V3 manifests)")
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
    parser.add_argument("--bottleneck-dim", type=int, default=0)
    parser.add_argument(
        "--aux-schedule",
        choices=("legacy", "none", "throughout", "early", "late"),
        default="legacy",
    )
    parser.add_argument("--aux-fraction", type=float, default=1.0)
    parser.add_argument("--aux-updates", type=int, default=0)
    parser.add_argument("--rule-signal", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bridge-signal", type=int, choices=(0, 1), default=0)
    parser.add_argument("--reset-step", type=int, default=0)
    parser.add_argument("--v3-arm", default="")
    parser.add_argument("--source-only-eval", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--aux-mode",
        choices=("online_global", "bank", "permuted", "unbridged_test"),
        default="online_global",
    )
    parser.add_argument("--aux-bank-size", type=int, default=0)
    parser.add_argument("--aux-batch-size", type=int, default=0)
    parser.add_argument("--v4-arm", default="")
    parser.add_argument("--v5-arm", default="")
    parser.add_argument(
        "--representation-signal", type=int, choices=(-1, 0, 1), default=-1
    )
    parser.add_argument(
        "--mapping-signal", type=int, choices=(-1, 0, 1), default=-1
    )
    parser.add_argument(
        "--output-invariant-signal", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--mapping-control", choices=("valid", "permuted"), default="valid"
    )
    parser.add_argument(
        "--gap-head-mode", choices=("shared", "separate"), default="shared"
    )
    parser.add_argument("--v8-split", default="")
    parser.add_argument("--v8-arm", default="")
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
