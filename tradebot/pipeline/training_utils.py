"""Training, evaluation, and gate-selection helpers."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


log = logging.getLogger("nn")


def make_class_weights(labels: np.ndarray, class_count: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    weights = np.ones(class_count, dtype=np.float32)
    total = counts.sum()
    for cls in range(class_count):
        if counts[cls] > 0:
            weights[cls] = total / (float(class_count) * counts[cls])
    return torch.tensor(weights, dtype=torch.float32)


def make_sample_weights(labels: np.ndarray, class_count: int = 3) -> torch.Tensor:
    class_weights = make_class_weights(labels, class_count=class_count).to(torch.float64).numpy()
    sample_weights = class_weights[labels.astype(np.int64)]
    sample_weights /= max(sample_weights.mean(), 1e-12)
    return torch.tensor(sample_weights, dtype=torch.double)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.to(torch.float32))
        else:
            self.register_buffer("alpha", None)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1.0 - pt).pow(self.gamma)
        alpha_t = 1.0 if self.alpha is None else self.alpha[targets]
        loss = -alpha_t * focal_term * log_pt
        return loss.mean()


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    sample_weights: torch.Tensor | None = None,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
    )


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    log.info("evaluate_model: starting - %d batches", len(loader))
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if batch_idx % 100 == 0:
                log.info("evaluate_model: batch %d/%d", batch_idx, len(loader))
            logits_list.append(model(xb.to(device)).cpu().numpy())
            labels_list.append(yb.numpy())
    log.info("evaluate_model: done - concatenating %d arrays", len(logits_list))
    result = np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)
    log.info("evaluate_model: result shapes - logits=%s, labels=%s", result[0].shape, result[1].shape)
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def gate_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    selected = confidences >= threshold if probs.shape[1] == 2 else (preds > 0) & (confidences >= threshold)
    selected_trades = int(selected.sum())
    precision = float((preds[selected] == labels[selected]).mean()) if selected_trades else float("nan")
    selected_mean_confidence = float(confidences[selected].mean()) if selected_trades else float("nan")
    return {
        "selected_trades": selected_trades,
        "trade_coverage": float(selected.mean()),
        "precision": precision,
        "mean_confidence": float(confidences.mean()),
        "selected_mean_confidence": selected_mean_confidence,
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def choose_confidence_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    min_selected: int,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> float:
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    is_binary = probs.shape[1] == 2
    candidate_mask = np.ones(len(preds), dtype=bool) if is_binary else preds > 0
    threshold_min = min(max(0.0, float(threshold_min)), 0.999999)
    threshold_max = min(max(threshold_min, float(threshold_max)), 0.999999)
    threshold_steps = max(2, int(threshold_steps))
    candidate_count = len(preds) if is_binary else int(candidate_mask.sum())
    if candidate_count == 0:
        log.warning(
            "Confidence gate selection: model produced no BUY/SELL predictions; falling back to threshold %.2f.",
            threshold_min,
        )
        return threshold_min

    min_selected = max(1, min_selected)
    best_threshold = threshold_min
    best_precision = -1.0
    best_selected = -1
    best_coverage = -1.0
    relaxed_threshold = threshold_min
    relaxed_precision = -1.0
    relaxed_selected = -1
    relaxed_coverage = -1.0
    found_candidate = False
    found_relaxed_candidate = False

    for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
        selected = candidate_mask & (confidences >= threshold)
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue

        precision = float((preds[selected] == labels[selected]).mean())
        coverage = float(selected.mean())
        found_relaxed_candidate = True
        if precision > relaxed_precision + 1e-12 or (
            abs(precision - relaxed_precision) <= 1e-12
            and (selected_count > relaxed_selected or (selected_count == relaxed_selected and coverage > relaxed_coverage))
        ):
            relaxed_threshold = float(threshold)
            relaxed_precision = precision
            relaxed_selected = selected_count
            relaxed_coverage = coverage

        if selected_count < min_selected:
            continue
        found_candidate = True
        if precision > best_precision + 1e-12 or (
            abs(precision - best_precision) <= 1e-12
            and (selected_count > best_selected or (selected_count == best_selected and coverage > best_coverage))
        ):
            best_threshold = float(threshold)
            best_precision = precision
            best_selected = selected_count
            best_coverage = coverage

    if found_candidate:
        print("Chosen confidence threshold: %.2f with precision %.4f and coverage %.4f" % (best_threshold, best_precision, best_coverage))
        return best_threshold
    if found_relaxed_candidate:
        log.warning(
            "Confidence gate selection: no threshold produced at least %d BUY/SELL trades; "
            "falling back to threshold %.2f with %d trades and precision %.4f.",
            min_selected,
            relaxed_threshold,
            relaxed_selected,
            relaxed_precision,
        )
        print(
            "Chosen confidence threshold: %.2f with precision %.4f and coverage %.4f"
            % (relaxed_threshold, relaxed_precision, relaxed_coverage)
        )
        return relaxed_threshold

    log.warning(
        "Confidence gate selection: no threshold selected any BUY/SELL trades; falling back to threshold %.2f.",
        threshold_min,
    )
    return threshold_min


def summarize_gate(name: str, probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = gate_metrics(labels, probs, threshold)
    if metrics["selected_trades"]:
        log.info(
            "%s: threshold=%.2f precision=%.4f coverage=%.4f trades=%d mean_selected_conf=%.4f",
            name,
            threshold,
            float(metrics["precision"]),
            float(metrics["trade_coverage"]),
            int(metrics["selected_trades"]),
            float(metrics["selected_mean_confidence"]),
        )
    else:
        log.warning("%s: threshold=%.2f produced no trades.", name, threshold)
    return metrics
