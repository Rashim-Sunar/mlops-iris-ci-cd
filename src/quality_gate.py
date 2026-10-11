"""End-to-end ML quality gate and JSON report generation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_validation import VALID_TARGET_LABELS, load_and_validate_csv
from .evaluate import calculate_detailed_metrics
from .preprocessing import RANDOM_STATE, DatasetSplit, split_dataset
from .train import train_from_split

# Educational acceptance criteria. These are deliberately kept in one place.
QUALITY_GATE_CONFIG = {
    "accuracy_threshold": 0.90,
    "macro_f1_threshold": 0.90,
    "test_size": 0.20,
    "random_state": RANDOM_STATE,
}


@dataclass(frozen=True)
class QualityGateResult:
    """Machine-readable result of the mandatory quality checks."""

    metrics: dict[str, float]
    criteria: list[dict[str, object]]
    passed: bool
    failure_reasons: list[str]
    labels: list[str]
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float | int]]


def _metric_is_valid(value: object) -> bool:
    return (
        isinstance(value, int | float | np.number)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def _coerce_metric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def assess_quality_gate(
    metrics: Mapping[str, object],
    labels: Sequence[str] | None = None,
    valid_target_labels: set[str] = VALID_TARGET_LABELS,
    confusion_matrix: list[list[int]] | None = None,
    per_class: dict[str, dict[str, float | int]] | None = None,
) -> QualityGateResult:
    """Assess metrics and predictions against the educational criteria."""
    metric_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    normalized_metrics = {
        name: _coerce_metric(metrics[name]) if name in metrics else float("nan")
        for name in metric_names
    }
    failure_reasons: list[str] = []
    metrics_valid = all(
        _metric_is_valid(value) for value in normalized_metrics.values()
    )
    criteria: list[dict[str, object]] = [
        {
            "name": "metrics_finite_and_bounded",
            "observed": metrics_valid,
            "threshold": "all metrics finite and between 0 and 1",
            "passed": metrics_valid,
        }
    ]
    if not metrics_valid:
        failure_reasons.append(
            "One or more evaluation metrics are non-finite or out of range."
        )

    observed_labels = set(labels or [])
    labels_valid = observed_labels.issubset(valid_target_labels)
    criteria.append(
        {
            "name": "predictions_are_valid_labels",
            "observed": sorted(observed_labels),
            "threshold": sorted(valid_target_labels),
            "passed": labels_valid,
        }
    )
    if not labels_valid:
        failure_reasons.append(
            "Predictions contain labels outside the allowed target labels."
        )

    threshold_checks = (
        ("accuracy", QUALITY_GATE_CONFIG["accuracy_threshold"]),
        ("macro_f1", QUALITY_GATE_CONFIG["macro_f1_threshold"]),
    )
    for metric_name, threshold in threshold_checks:
        observed = normalized_metrics[metric_name]
        passed = _metric_is_valid(observed) and observed >= threshold
        criteria.append(
            {
                "name": f"{metric_name}_threshold",
                "observed": observed,
                "threshold": threshold,
                "passed": passed,
            }
        )
        if not passed:
            failure_reasons.append(
                f"{metric_name} {observed:.4f} is below the required {threshold:.2f}."
            )

    return QualityGateResult(
        metrics=normalized_metrics,
        criteria=criteria,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        labels=sorted(observed_labels),
        confusion_matrix=confusion_matrix or [],
        per_class=per_class or {},
    )


def evaluate_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    valid_target_labels: set[str] = VALID_TARGET_LABELS,
) -> QualityGateResult:
    """Calculate detailed metrics and assess held-out predictions."""
    detailed = calculate_detailed_metrics(y_true, y_pred)
    scalar_metrics = {
        "accuracy": detailed["accuracy"],
        "macro_precision": detailed["precision"],
        "macro_recall": detailed["recall"],
        "macro_f1": detailed["f1"],
    }
    return assess_quality_gate(
        scalar_metrics,
        labels=[*y_true, *y_pred],
        valid_target_labels=valid_target_labels,
        confusion_matrix=detailed["confusion_matrix"],
        per_class=detailed["per_class"],
    )


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_quality_report(
    dataset_identifier: str,
    dataframe_size: int,
    feature_names: list[str],
    dataset_split: DatasetSplit,
    model: object,
    result: QualityGateResult,
) -> dict[str, object]:
    """Build a JSON-serializable report for CI artifacts."""
    classifier = model.named_steps["classifier"]
    return {
        "dataset": {
            "identifier": dataset_identifier,
            "sample_count": dataframe_size,
            "feature_names": feature_names,
        },
        "split": {
            "training_samples": len(dataset_split.X_train),
            "test_samples": len(dataset_split.X_test),
            "test_size": QUALITY_GATE_CONFIG["test_size"],
            "random_state": QUALITY_GATE_CONFIG["random_state"],
        },
        "model": {
            "name": type(classifier).__name__,
            "parameters": classifier.get_params(),
        },
        "metrics": result.metrics,
        "labels": result.labels,
        "confusion_matrix": result.confusion_matrix,
        "per_class": result.per_class,
        "thresholds": {
            "accuracy": QUALITY_GATE_CONFIG["accuracy_threshold"],
            "macro_f1": QUALITY_GATE_CONFIG["macro_f1_threshold"],
        },
        "criteria": result.criteria,
        "overall_result": "PASS" if result.passed else "FAIL",
        "failure_reasons": result.failure_reasons,
    }


def write_quality_report(report: Mapping[str, object], report_path: str | Path) -> None:
    """Write a valid JSON report, converting non-finite values to null."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report_file:
        json.dump(_json_safe(report), report_file, indent=2, allow_nan=False)
        report_file.write("\n")


def run_quality_gate(
    csv_path: str | Path = "data/raw/Iris.csv",
    report_path: str | Path = "reports/quality_report.json",
) -> QualityGateResult:
    """Validate, train, evaluate, write the report, and return the result."""
    dataframe = load_and_validate_csv(csv_path)
    dataset_split = split_dataset(
        dataframe,
        test_size=QUALITY_GATE_CONFIG["test_size"],
        random_state=QUALITY_GATE_CONFIG["random_state"],
    )
    model, _ = train_from_split(dataset_split)
    result = evaluate_predictions(
        dataset_split.y_test,
        model.predict(dataset_split.X_test),
    )
    report = build_quality_report(
        str(csv_path),
        len(dataframe),
        list(dataset_split.X_train.columns),
        dataset_split,
        model,
        result,
    )
    write_quality_report(report, report_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Iris ML quality gate.")
    parser.add_argument("csv_path", nargs="?", default="data/raw/Iris.csv")
    parser.add_argument("--report", default="reports/quality_report.json")
    args = parser.parse_args(argv)
    try:
        result = run_quality_gate(args.csv_path, args.report)
    except (OSError, TypeError, ValueError) as error:
        print(f"ML quality gate: FAIL ({error})", file=sys.stderr)
        return 1

    print("ML quality gate: PASS" if result.passed else "ML quality gate: FAIL")
    for criterion in result.criteria:
        print(f"- {criterion['name']}: {'PASS' if criterion['passed'] else 'FAIL'}")
    for reason in result.failure_reasons:
        print(f"Reason: {reason}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
