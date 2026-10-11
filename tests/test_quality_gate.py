import json

import joblib
import pytest
from src.data_validation import VALID_TARGET_LABELS
from src.preprocessing import split_dataset
from src.quality_gate import (
    QualityGateResult,
    assess_quality_gate,
    build_quality_report,
    evaluate_predictions,
    main,
    run_quality_gate,
    write_quality_report,
)
from src.train import train_from_split


def passing_metrics():
    return {
        "accuracy": 0.95,
        "macro_precision": 0.94,
        "macro_recall": 0.93,
        "macro_f1": 0.92,
    }


def test_quality_gate_passes_all_thresholds():
    result = assess_quality_gate(passing_metrics(), labels=list(VALID_TARGET_LABELS))
    assert result.passed
    assert not result.failure_reasons


def test_quality_gate_fails_when_accuracy_is_below_threshold():
    metrics = passing_metrics()
    metrics["accuracy"] = 0.89
    result = assess_quality_gate(metrics, labels=list(VALID_TARGET_LABELS))
    assert not result.passed
    assert any("accuracy" in reason for reason in result.failure_reasons)


def test_quality_gate_fails_when_macro_f1_is_below_threshold():
    metrics = passing_metrics()
    metrics["macro_f1"] = 0.89
    result = assess_quality_gate(metrics, labels=list(VALID_TARGET_LABELS))
    assert not result.passed
    assert any("macro_f1" in reason for reason in result.failure_reasons)


def test_quality_gate_reports_both_threshold_failures():
    result = assess_quality_gate(
        {**passing_metrics(), "accuracy": 0.5, "macro_f1": 0.5},
        labels=list(VALID_TARGET_LABELS),
    )
    assert not result.passed
    assert len(result.failure_reasons) == 2


def test_quality_gate_rejects_non_finite_metrics():
    result = assess_quality_gate(
        {**passing_metrics(), "macro_f1": float("nan")},
        labels=list(VALID_TARGET_LABELS),
    )
    assert not result.passed
    assert any("non-finite" in reason for reason in result.failure_reasons)


def test_quality_gate_rejects_out_of_range_metrics():
    result = assess_quality_gate(
        {**passing_metrics(), "macro_precision": 1.1},
        labels=list(VALID_TARGET_LABELS),
    )
    assert not result.passed
    assert any("non-finite" in reason for reason in result.failure_reasons)


def test_quality_gate_rejects_invalid_prediction_labels():
    result = evaluate_predictions(
        ["Iris-setosa", "Iris-versicolor"],
        ["Iris-setosa", "not-an-iris"],
    )
    assert not result.passed
    assert any("allowed target labels" in reason for reason in result.failure_reasons)


def test_quality_gate_rejects_incompatible_predictions():
    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions(["Iris-setosa"], [])


def test_quality_report_has_expected_structure(tmp_path, iris_dataframe):
    dataset_split = split_dataset(iris_dataframe)
    model, _ = train_from_split(dataset_split)
    result = evaluate_predictions(
        dataset_split.y_test,
        model.predict(dataset_split.X_test),
    )
    report = build_quality_report(
        "data/raw/Iris.csv",
        len(iris_dataframe),
        list(dataset_split.X_train.columns),
        dataset_split,
        model,
        result,
    )
    report_path = tmp_path / "reports" / "quality_report.json"
    write_quality_report(report, report_path)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["overall_result"] == "PASS"
    assert loaded["dataset"]["sample_count"] == 150
    assert loaded["split"]["test_samples"] == 30
    assert loaded["thresholds"] == {"accuracy": 0.9, "macro_f1": 0.9}
    assert len(loaded["confusion_matrix"]) == 3
    assert set(loaded["per_class"]) == VALID_TARGET_LABELS


def test_report_converts_non_finite_values_to_null(tmp_path):
    report_path = tmp_path / "quality_report.json"
    write_quality_report({"metric": float("nan")}, report_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"metric": None}


def test_failing_quality_gate_returns_nonzero_exit_status(monkeypatch, tmp_path):
    failing_result = QualityGateResult(
        metrics={
            "accuracy": 0.5,
            "macro_precision": 0.5,
            "macro_recall": 0.5,
            "macro_f1": 0.5,
        },
        criteria=[],
        passed=False,
        failure_reasons=["accuracy below threshold"],
        labels=[],
        confusion_matrix=[],
        per_class={},
    )
    monkeypatch.setattr(
        "src.quality_gate.run_quality_gate", lambda *_args: failing_result
    )
    assert main(["data/raw/Iris.csv", "--report", str(tmp_path / "report.json")]) == 1


def test_quality_gate_entry_point_uses_real_dataset(tmp_path):
    report_path = tmp_path / "quality_report.json"
    assert main(["data/raw/Iris.csv", "--report", str(report_path)]) == 0
    assert report_path.is_file()


def test_passing_quality_gate_packages_evaluated_pipeline(tmp_path):
    result = run_quality_gate(
        "data/raw/Iris.csv",
        tmp_path / "quality_report.json",
        tmp_path / "artifacts",
    )
    assert result.passed
    model = joblib.load(tmp_path / "artifacts" / "iris_model.joblib")
    metadata = json.loads((tmp_path / "artifacts" / "model_metadata.json").read_text())
    assert model.predict([[5.1, 3.5, 1.4, 0.2]])[0] in VALID_TARGET_LABELS
    assert metadata["overall_result"] == "PASS"
    assert metadata["feature_names"] == [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]
