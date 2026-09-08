"""Behavioral and structural measurements for training-axis checkpoints."""

from __future__ import annotations

import math

import numpy as np
import torch

from experiments.training_axes import task


@torch.no_grad()
def logits(model, X: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], device=device)
        values.append(model(xb).float().cpu().numpy())
    return np.concatenate(values)


def final_hidden(model, xb: torch.Tensor) -> torch.Tensor:
    """Differentiable representation at the query's final token."""
    residuals, mask = model.residuals(xb)
    last = (~mask).sum(1) - 1
    return residuals[-1][torch.arange(len(xb), device=xb.device), last]


@torch.no_grad()
def representations(model, X: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], device=device)
        values.append(final_hidden(model, xb).float().cpu().numpy())
    return np.concatenate(values)


def brier_skill(probability: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Brier skill relative to the balanced 0.5 forecast."""
    return 1.0 - (probability - label) ** 2 / 0.25


def behavior_metrics(model, dataset, device, prefix: str) -> dict[str, float]:
    X, y, metadata = dataset
    logit = logits(model, X, device)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    prediction = probability >= 0.5
    result = {
        f"{prefix}_accuracy": float(np.mean(prediction == (y > 0.5))),
        f"{prefix}_brier_skill": float(np.mean(brier_skill(probability, y))),
        f"{prefix}_confidence": float(np.mean(np.maximum(probability, 1 - probability))),
    }
    distance = np.asarray([row["distance"] for row in metadata])
    for name, selected in {
        "near": distance <= 3,
        "middle": (distance >= 4) & (distance <= 6),
        "far": distance >= 7,
    }.items():
        if selected.any():
            result[f"{prefix}_accuracy_{name}"] = float(
                np.mean(prediction[selected] == (y[selected] > 0.5))
            )
            result[f"{prefix}_brier_skill_{name}"] = float(
                np.mean(brier_skill(probability[selected], y[selected]))
            )
    return result


def _ridge_fit(X: np.ndarray, y: np.ndarray, penalty: float = 1e-3) -> np.ndarray:
    # Use float64 for the normal equations. Some BLAS builds accumulate the
    # float32 residual-stream matrices unreliably even at this small scale.
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(X)), X])
    regularizer = penalty * np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    # NumPy 2.0 with Apple's Accelerate backend can emit spurious floating-point
    # warnings for finite matrix products. Verify finiteness explicitly instead.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gram = design.T @ design
        rhs = design.T @ y
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("non-finite values in latent-probe normal equations")
    return np.linalg.solve(gram + regularizer, rhs)


def _ridge_predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        prediction = np.column_stack([np.ones(len(X)), X]) @ coefficients
    if not np.isfinite(prediction).all():
        raise FloatingPointError("non-finite latent-probe predictions")
    return prediction


def latent_probe_transfer(model, source, target, device) -> dict[str, float]:
    """Fit signed-gap decoding on sources and transfer it without refitting."""
    X_source, _, meta_source = source
    X_target, _, meta_target = target
    H_source = representations(model, X_source, device)
    H_target = representations(model, X_target, device)
    gap_source = np.asarray([row["signed_gap"] for row in meta_source])
    gap_target = np.asarray([row["signed_gap"] for row in meta_target])
    coefficients = _ridge_fit(H_source, gap_source)

    result = {}
    for name, H, gap in (
        ("source", H_source, gap_source), ("target", H_target, gap_target)
    ):
        prediction = _ridge_predict(H, coefficients)
        denominator = np.sum((gap - gap.mean()) ** 2)
        r2 = 1.0 - np.sum((gap - prediction) ** 2) / max(denominator, 1e-12)
        result[f"probe_{name}_r2"] = float(r2)
        result[f"probe_{name}_sign_accuracy"] = float(
            np.mean((prediction > 0) == (gap > 0))
        )
    return result


def paired_invariance(
    model, device, seed: int, n_pairs: int = 512, task_module=task
) -> dict[str, float]:
    X1, X2, _, _, _ = task_module.paired_eval(seed, n_pairs)
    logit1, logit2 = logits(model, X1, device), logits(model, X2, device)
    probability1 = 1.0 / (1.0 + np.exp(-np.clip(logit1, -30, 30)))
    probability2 = 1.0 / (1.0 + np.exp(-np.clip(logit2, -30, 30)))
    H1, H2 = representations(model, X1, device), representations(model, X2, device)
    denominator = np.linalg.norm(H1, axis=1) * np.linalg.norm(H2, axis=1)
    cosine = np.sum(H1 * H2, axis=1) / np.maximum(denominator, 1e-12)
    return {
        "paired_probability_gap": float(np.mean(np.abs(probability1 - probability2))),
        "paired_prediction_disagreement": float(np.mean((logit1 > 0) != (logit2 > 0))),
        "paired_hidden_cosine": float(np.mean(cosine)),
    }


def embedding_structure(model, task_module=task) -> dict[str, float]:
    """Shared rank-axis alignment and effective rank of item embeddings."""
    embedding = model.emb.weight[:task_module.N_ITEM_TOKENS].detach().float().cpu().numpy()
    embedding = embedding.reshape(task_module.N_ALPH, task_module.N_ITEMS, -1)
    rank = np.arange(task_module.N_ITEMS, dtype=float)
    rank -= rank.mean()
    directions = []
    centered_blocks = []
    for alphabet in range(task_module.N_ALPH):
        block = embedding[alphabet] - embedding[alphabet].mean(0, keepdims=True)
        direction = rank @ block / max(rank @ rank, 1e-12)
        direction /= max(np.linalg.norm(direction), 1e-12)
        directions.append(direction)
        centered_blocks.append(block)
    pair_cosines = [
        float(directions[i] @ directions[j])
        for i in range(len(directions))
        for j in range(i + 1, len(directions))
    ]
    singular_values = np.linalg.svd(np.concatenate(centered_blocks), compute_uv=False)
    energy = singular_values ** 2
    participation_rank = energy.sum() ** 2 / max(np.sum(energy ** 2), 1e-12)
    return {
        "embedding_rank_axis_cosine": float(np.mean(pair_cosines)),
        "embedding_rank_axis_cosine_min": float(np.min(pair_cosines)),
        "embedding_participation_rank": float(participation_rank),
    }


def parameter_structure(model) -> dict[str, float]:
    squared_norm = 0.0
    count = 0
    for parameter in model.parameters():
        values = parameter.detach().float()
        squared_norm += float(torch.sum(values * values).cpu())
        count += parameter.numel()
    return {
        "parameter_l2": math.sqrt(squared_norm),
        "parameter_rms": math.sqrt(squared_norm / max(count, 1)),
        "parameter_count": float(count),
    }


def _write_direction(H: np.ndarray, variable: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    variable = np.asarray(variable, dtype=np.float64)
    centered_H = H - H.mean(0, keepdims=True)
    centered_variable = variable - variable.mean()
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        numerator = centered_variable @ centered_H
        denominator = centered_variable @ centered_variable
    if not np.isfinite(numerator).all() or not np.isfinite(denominator):
        raise FloatingPointError("non-finite values while fitting write direction")
    direction = numerator / max(denominator, 1e-12)
    return direction / max(np.linalg.norm(direction), 1e-12)


@torch.no_grad()
def causal_reuse(
    model,
    source,
    target,
    device,
    seed: int = 0,
    n_pairs: int = 256,
    n_random: int = 4,
    floor: float = 0.75,
    margin: float = 0.05,
) -> dict[str, float]:
    """Swap a signed-gap direction learned on sources into target examples."""
    model.eval()
    rng = np.random.default_rng(seed)
    X_source, _, meta_source = source
    X_target, y_target, meta_target = target
    H_source = representations(model, X_source, device)
    H_target = representations(model, X_target, device)
    gap_source = np.asarray([row["signed_gap"] for row in meta_source])
    gap_target = np.asarray([row["signed_gap"] for row in meta_target])
    source_direction = _write_direction(H_source, gap_source)
    target_direction = _write_direction(H_target, gap_target)

    bases, donors = [], []
    positive = np.flatnonzero(y_target > 0.5)
    negative = np.flatnonzero(y_target <= 0.5)
    for _ in range(n_pairs):
        if rng.integers(0, 2):
            bases.append(int(rng.choice(positive)))
            donors.append(int(rng.choice(negative)))
        else:
            bases.append(int(rng.choice(negative)))
            donors.append(int(rng.choice(positive)))
    bases, donors = np.asarray(bases), np.asarray(donors)
    counterfactual = y_target[donors]
    X_base = torch.as_tensor(X_target[bases], device=device)
    residuals, mask = model.residuals(X_base)
    layer = model.n_resid - 1
    last = (~mask).sum(1) - 1
    base_hidden = torch.as_tensor(H_target[bases], device=device, dtype=torch.float32)
    donor_hidden = torch.as_tensor(H_target[donors], device=device, dtype=torch.float32)

    def intervene(direction: np.ndarray) -> float:
        unit = torch.as_tensor(direction, device=device, dtype=torch.float32)
        delta = donor_hidden - base_hidden
        replacement = base_hidden + (delta @ unit).unsqueeze(1) * unit.unsqueeze(0)
        modified = residuals[layer].clone()
        modified[torch.arange(len(X_base), device=device), last] = replacement.to(modified.dtype)
        changed_logits = model.forward_from(X_base, layer, modified).float().cpu().numpy()
        return float(np.mean((changed_logits > 0) == (counterfactual > 0.5)))

    source_to_target = intervene(source_direction)
    target_to_target = intervene(target_direction)
    random_values = []
    for _ in range(n_random):
        random_direction = rng.normal(size=H_target.shape[1])
        random_direction /= np.linalg.norm(random_direction)
        random_values.append(intervene(random_direction))
    random_control = float(np.mean(random_values))
    valid = target_to_target >= floor and target_to_target - random_control >= margin
    normalized = (
        (source_to_target - random_control) / (target_to_target - random_control)
        if valid else np.nan
    )
    return {
        "causal_source_to_target": source_to_target,
        "causal_target_to_target": target_to_target,
        "causal_random": random_control,
        "causal_normalized": float(normalized),
        "causal_valid": float(valid),
    }


def evaluate_checkpoint(
    model,
    datasets: dict,
    device,
    seed: int,
    include_locked: bool = False,
    structural_pairs: int = 512,
    task_module=task,
) -> dict[str, float]:
    result = {}
    allowed = ["source", "source_all", "calib_combo"]
    if include_locked:
        allowed += ["test_combo", "held_pair", "test_both"]
    for name in allowed:
        result.update(behavior_metrics(model, datasets[name], device, name))
    result.update(latent_probe_transfer(
        model, datasets["source"], datasets["calib_combo"], device
    ))
    result.update(paired_invariance(
        model, device, seed, structural_pairs, task_module=task_module
    ))
    result.update(embedding_structure(model, task_module=task_module))
    result.update(parameter_structure(model))
    result.update(causal_reuse(
        model, datasets["source"], datasets["calib_combo"], device,
        seed=seed, n_pairs=min(structural_pairs, 256),
    ))
    return result
