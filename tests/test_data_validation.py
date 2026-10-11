import warnings

import pandas as pd
import pytest
from src.data_validation import (
    DataValidationError,
    load_and_validate_csv,
    validate_dataframe,
)


def test_valid_dataset_is_accepted(iris_dataframe):
    validated = validate_dataframe(iris_dataframe)
    assert validated.shape == (150, 6)


def test_actual_csv_can_be_loaded():
    dataframe = load_and_validate_csv("data/raw/Iris.csv")
    assert set(dataframe["Species"]) == {
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    }


def test_missing_required_column_is_rejected(iris_dataframe):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(iris_dataframe.drop(columns=["PetalWidthCm"]))


def test_missing_values_are_rejected(iris_dataframe):
    iris_dataframe.loc[0, "SepalLengthCm"] = pd.NA
    with pytest.raises(DataValidationError, match="Missing values"):
        validate_dataframe(iris_dataframe)


def test_incorrect_feature_type_is_rejected(iris_dataframe):
    iris_dataframe["PetalLengthCm"] = "not numeric"
    with pytest.raises(DataValidationError, match="must be numeric"):
        validate_dataframe(iris_dataframe)


def test_infinite_feature_value_is_rejected(iris_dataframe):
    iris_dataframe.loc[0, "SepalWidthCm"] = float("inf")
    with pytest.raises(DataValidationError, match="Infinite values"):
        validate_dataframe(iris_dataframe)


def test_invalid_target_label_is_rejected(iris_dataframe):
    iris_dataframe.loc[0, "Species"] = "unknown"
    with pytest.raises(DataValidationError, match="Invalid target labels"):
        validate_dataframe(iris_dataframe)


def test_duplicate_rows_are_reported_and_retained(iris_dataframe):
    duplicated = pd.concat(
        [iris_dataframe, iris_dataframe.iloc[[0]]], ignore_index=True
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_dataframe(duplicated)
    assert any("duplicate row" in str(item.message) for item in caught)


def test_empty_dataset_is_rejected(iris_dataframe):
    with pytest.raises(DataValidationError, match="no usable records"):
        validate_dataframe(iris_dataframe.iloc[0:0])
