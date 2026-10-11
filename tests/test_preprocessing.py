from src.preprocessing import separate_features_and_target, split_dataset


def test_features_and_target_are_separated(iris_dataframe):
    features, target = separate_features_and_target(iris_dataframe)
    assert list(features.columns) == [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]
    assert "Id" not in features.columns
    assert target.name == "Species"
    assert len(features) == len(target) == 150


def test_split_dimensions_are_correct(iris_dataframe):
    dataset_split = split_dataset(iris_dataframe)
    assert dataset_split.X_train.shape == (120, 4)
    assert dataset_split.X_test.shape == (30, 4)
    assert len(dataset_split.y_train) == 120
    assert len(dataset_split.y_test) == 30


def test_split_is_reproducible(iris_dataframe):
    first = split_dataset(iris_dataframe)
    second = split_dataset(iris_dataframe)
    assert first.X_train.equals(second.X_train)
    assert first.y_test.equals(second.y_test)
