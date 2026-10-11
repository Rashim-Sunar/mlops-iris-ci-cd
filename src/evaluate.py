"""Evaluation metrics for multiclass Iris predictions."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Calculate standard multiclass classification metrics."""
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predictions must have the same length.")
    if len(y_true) == 0:
        raise ValueError("True labels and predictions cannot be empty.")
    if any(value is None for value in y_true) or any(value is None for value in y_pred):
        raise ValueError("True labels and predictions must contain usable values.")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
