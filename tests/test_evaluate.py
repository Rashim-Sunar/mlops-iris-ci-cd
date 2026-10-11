import pytest
from src.evaluate import calculate_metrics
from src.preprocessing import split_dataset
from src.train import train_classifier


def test_metrics_are_calculated_for_multiclass_predictions():
    metrics = calculate_metrics(
        ["a", "b", "c", "a"],
        ["a", "b", "a", "a"],
    )
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert metrics["accuracy"] == 0.75
    assert all(0 <= value <= 1 for value in metrics.values())


def test_evaluation_rejects_incompatible_lengths():
    with pytest.raises(ValueError, match="same length"):
        calculate_metrics(["a", "b"], ["a"])


def test_evaluation_rejects_empty_inputs():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_metrics([], [])


def test_end_to_end_training_and_evaluation(iris_dataframe):
    dataset_split = split_dataset(iris_dataframe)
    model, _ = train_classifier(dataset_split.X_train, dataset_split.y_train)
    metrics = calculate_metrics(
        dataset_split.y_test,
        model.predict(dataset_split.X_test),
    )
    assert metrics["accuracy"] >= 0.9
