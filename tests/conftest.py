from pathlib import Path

import pandas as pd
import pytest

DATASET_PATH = Path(__file__).parents[1] / "data" / "raw" / "Iris.csv"


@pytest.fixture
def iris_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)
