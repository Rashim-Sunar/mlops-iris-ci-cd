"""Validation helpers for the Kaggle Iris Species CSV."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
TARGET_COLUMN = "Species"
IDENTIFIER_COLUMNS = ["Id"]
VALID_TARGET_LABELS = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


class DataValidationError(ValueError):
    """Raised when the input data cannot be used for training."""


def validate_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    valid_target_labels: set[str] | None = None,
) -> pd.DataFrame:
    """Validate an Iris-like dataframe and return it unchanged.

    Duplicate rows are reported as non-blocking warnings because duplicate
    observations are valid records for this educational dataset.
    """
    if dataframe.empty:
        raise DataValidationError("The dataset contains no usable records.")

    feature_columns = feature_columns or FEATURE_COLUMNS
    valid_target_labels = valid_target_labels or VALID_TARGET_LABELS
    required_columns = [*feature_columns, target_column]
    missing_columns = [column for column in required_columns if column not in dataframe]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {', '.join(missing_columns)}."
        )

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric:
        raise DataValidationError(
            f"Feature columns must be numeric: {', '.join(non_numeric)}."
        )

    missing_counts = dataframe[required_columns].isna().sum()
    missing_columns_with_values = missing_counts[missing_counts > 0]
    if not missing_columns_with_values.empty:
        details = ", ".join(
            f"{column} ({count})"
            for column, count in missing_columns_with_values.items()
        )
        raise DataValidationError(f"Missing values detected: {details}.")

    infinite_columns = [
        column
        for column in feature_columns
        if not np.isfinite(dataframe[column].to_numpy()).all()
    ]
    if infinite_columns:
        raise DataValidationError(
            "Infinite values detected in numerical features: "
            f"{', '.join(infinite_columns)}."
        )

    observed_labels = set(dataframe[target_column].astype(str))
    invalid_labels = observed_labels - valid_target_labels
    if invalid_labels:
        raise DataValidationError(
            f"Invalid target labels: {', '.join(sorted(invalid_labels))}."
        )
    if dataframe[target_column].nunique() < 2:
        raise DataValidationError("The target must contain at least two classes.")

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count:
        warnings.warn(
            f"Found {duplicate_count} duplicate row(s); they will be retained.",
            UserWarning,
            stacklevel=2,
        )
    return dataframe


def load_and_validate_csv(path: str | Path) -> pd.DataFrame:
    """Load and strictly validate a CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataValidationError(f"CSV file does not exist: {csv_path}")
    try:
        dataframe = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as error:
        raise DataValidationError(f"Could not load CSV file: {csv_path}") from error
    return validate_dataframe(dataframe)
