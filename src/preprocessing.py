"""Feature selection and reproducible train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_validation import FEATURE_COLUMNS, TARGET_COLUMN

RANDOM_STATE = 42


@dataclass
class DatasetSplit:
    """Training and testing data after feature/target separation."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def separate_features_and_target(
    dataframe: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictive features and target without identifier columns."""
    selected_features = feature_columns or FEATURE_COLUMNS
    missing_columns = [
        column
        for column in [*selected_features, target_column]
        if column not in dataframe
    ]
    if missing_columns:
        raise ValueError(
            f"Missing columns for preprocessing: {', '.join(missing_columns)}"
        )
    return dataframe[selected_features].copy(), dataframe[target_column].copy()


def split_dataset(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Separate data and create a deterministic stratified split."""
    features, target = separate_features_and_target(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)
