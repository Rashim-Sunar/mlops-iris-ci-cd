"""Train the Iris classifier."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_validation import load_and_validate_csv
from .evaluate import calculate_metrics
from .preprocessing import DatasetSplit, split_dataset

RANDOM_STATE = 42


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, int]]:
    """Fit a reusable scaling and logistic-regression pipeline."""
    if len(X_train) != len(y_train) or len(X_train) == 0:
        raise ValueError(
            "Training features and labels must have equal, non-zero lengths."
        )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=200, random_state=random_state),
            ),
        ]
    )
    model.fit(X_train, y_train)
    metadata = {"training_rows": len(X_train), "feature_count": X_train.shape[1]}
    return model, metadata


def train_from_split(dataset_split: DatasetSplit) -> tuple[Pipeline, dict[str, int]]:
    """Train from a prepared dataset split."""
    return train_classifier(dataset_split.X_train, dataset_split.y_train)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Iris classifier."
    )
    parser.add_argument("csv_path", nargs="?", default="data/raw/Iris.csv")
    args = parser.parse_args()

    dataframe = load_and_validate_csv(args.csv_path)
    dataset_split = split_dataset(dataframe)
    model, metadata = train_from_split(dataset_split)
    metrics = calculate_metrics(
        dataset_split.y_test, model.predict(dataset_split.X_test)
    )
    print(f"Training metadata: {metadata}")
    print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
