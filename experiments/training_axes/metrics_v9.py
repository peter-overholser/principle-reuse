"""Behavioral and organization assays for V9-A."""

from __future__ import annotations

import numpy as np
import torch

from experiments.training_axes import metrics


def as_tuple(batch):
    return batch.tokens, batch.labels, batch.metadata


def subset(batch, selected):
    selected = np.asarray(selected, dtype=bool)
    return (
        batch.tokens[selected], batch.labels[selected],
        [row for row, keep in zip(batch.metadata, selected) if keep],
    )


def linear_cka(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left -= left.mean(0, keepdims=True)
    right -= right.mean(0, keepdims=True)
    # Apple's Accelerate backend can warn spuriously on finite float64 matrix
    # products.  Suppress the warning and check the result explicitly.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cross = left.T @ right
        left_gram = left.T @ left
        right_gram = right.T @ right
    if not all(np.isfinite(value).all() for value in (cross, left_gram, right_gram)):
        raise FloatingPointError("non-finite V9 CKA matrix")
    numerator = np.sum(cross * cross)
    left_norm = np.sqrt(np.sum(left_gram ** 2))
    right_norm = np.sqrt(np.sum(right_gram ** 2))
    return float(numerator / max(left_norm * right_norm, 1e-12))


def assay_probe_transfer(model, source, target, device):
    """Transfer an assay-reserved latent probe without refitting."""
    h_source = metrics.representations(model, source.tokens, device)
    h_target = metrics.representations(model, target.tokens, device)
    y_source = np.asarray([row["assay_latent"] for row in source.metadata])
    y_target = np.asarray([row["assay_latent"] for row in target.metadata])
    coefficients = metrics._ridge_fit(h_source, y_source)
    result = {}
    for name, hidden, values in (
        ("source", h_source, y_source), ("target", h_target, y_target)
    ):
        prediction = metrics._ridge_predict(hidden, coefficients)
        denominator = np.sum((values - values.mean()) ** 2)
        result[f"probe_{name}_r2"] = float(
            1.0 - np.sum((values - prediction) ** 2) / max(denominator, 1e-12)
        )
        result[f"probe_{name}_sign_accuracy"] = float(
            np.mean((prediction > 0) == (values > 0))
        )
    return result


@torch.no_grad()
def paired_hidden_assay(model, task, device, seed, anchor_count, target, n_pairs):
    source, held, _, _, _ = task.paired_cross_render(
        seed, n_pairs, anchor_count, target=target
    )
    h_source = metrics.representations(model, source, device)
    h_held = metrics.representations(model, held, device)
    denominator = np.linalg.norm(h_source, axis=1) * np.linalg.norm(h_held, axis=1)
    cosine = np.sum(h_source * h_held, axis=1) / np.maximum(denominator, 1e-12)
    return {
        f"paired_{target}_hidden_cka": linear_cka(h_source, h_held),
        f"paired_{target}_hidden_cosine": float(np.mean(cosine)),
    }


@torch.no_grad()
def embedding_correspondence(model, task, anchor_count):
    weights = model.effective_embedding_weights().detach().float().cpu().numpy()
    blocks = weights[task.item_token_ids(anchor_count)]
    pairwise = []
    for left in range(task.N_ALPH):
        for right in range(left + 1, task.N_ALPH):
            pairwise.append(linear_cka(blocks[left], blocks[right]))
    return {
        "embedding_correspondence_cka": float(np.mean(pairwise)),
        "embedding_correspondence_cka_min": float(np.min(pairwise)),
    }


def organization_score(result, target):
    causal = result.get(f"causal_{target}_normalized", float("nan"))
    causal_valid = bool(result.get(f"causal_{target}_valid", 0.0))
    values = (
        np.clip(result[f"probe_{target}_r2"], 0.0, 1.0),
        np.clip(result[f"paired_{target}_hidden_cka"], 0.0, 1.0),
        np.clip(causal, 0.0, 1.0) if causal_valid and np.isfinite(causal) else 0.0,
    )
    return float(np.mean(values))


def evaluate_checkpoint(
    model, task, datasets, device, seed, anchor_count, *, include_locked=False,
    structural_pairs=256,
):
    result = {}
    development = (
        "source", "source_extended", "dev_x", "dev_s", "held_source",
        "dev_joint",
    )
    development = tuple(
        "dev_held_source" if name == "held_source" else name
        for name in development
    )
    locked = (
        "lock_x", "lock_s", "lock_held_source", "lock_joint"
    ) if include_locked else ()
    for name in development + locked:
        batch = datasets[name]
        result.update(metrics.behavior_metrics(
            model, as_tuple(batch), device, name
        ))
        selected = [row["max_anchor_exposure"] == 0 for row in batch.metadata]
        if any(selected):
            result.update(metrics.behavior_metrics(
                model, subset(batch, selected), device, f"{name}_nonanchor"
            ))

    for target in (("dev_x", "dev_s") + (("lock_x", "lock_s") if include_locked else ())):
        probe = assay_probe_transfer(
            model, datasets["source"], datasets[target], device
        )
        result[f"probe_{target}_r2"] = probe["probe_target_r2"]
        result[f"probe_{target}_sign_accuracy"] = probe["probe_target_sign_accuracy"]
        result.update(paired_hidden_assay(
            model, task, device, seed, anchor_count, target,
            n_pairs=structural_pairs,
        ))
        causal = metrics.causal_reuse(
            model, as_tuple(datasets["source"]), as_tuple(datasets[target]),
            device, seed=seed + (0 if target.endswith("x") else 100_000),
            n_pairs=min(256, structural_pairs),
        )
        for key, value in causal.items():
            suffix = key.removeprefix("causal_")
            result[f"causal_{target}_{suffix}"] = value
        result[f"organization_{target}"] = organization_score(result, target)
        result[f"organization_{target}_causal_valid"] = causal["causal_valid"]

    result.update(embedding_correspondence(model, task, anchor_count))
    result.update(metrics.parameter_structure(model))
    return result
