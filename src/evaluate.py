"""Evaluation metrics for multiclass Iris predictions."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def _validate_prediction_inputs(y_true: Sequence, y_pred: Sequence) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predictions must have the same length.")
    if len(y_true) == 0:
        raise ValueError("True labels and predictions cannot be empty.")
    if any(value is None for value in y_true) or any(value is None for value in y_pred):
        raise ValueError("True labels and predictions must contain usable values.")


def calculate_detailed_metrics(
    y_true: Sequence,
    y_pred: Sequence,
) -> dict[str, object]:
    """Calculate scalar, confusion-matrix, and per-class metrics."""
    _validate_prediction_inputs(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        str(label): {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for label, class_precision, class_recall, class_f1, class_support in zip(
            labels, precision, recall, f1, support, strict=True
        )
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": [str(label) for label in labels],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": per_class,
    }


def calculate_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Calculate standard multiclass classification metrics."""
    detailed_metrics = calculate_detailed_metrics(y_true, y_pred)
    return {
        metric_name: float(detailed_metrics[metric_name])
        for metric_name in ("accuracy", "precision", "recall", "f1")
    }
