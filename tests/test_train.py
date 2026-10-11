from sklearn.pipeline import Pipeline
from src.preprocessing import split_dataset
from src.train import train_classifier


def test_model_training_returns_fitted_pipeline(iris_dataframe):
    dataset_split = split_dataset(iris_dataframe)
    model, metadata = train_classifier(dataset_split.X_train, dataset_split.y_train)
    predictions = model.predict(dataset_split.X_test)
    assert isinstance(model, Pipeline)
    assert len(predictions) == len(dataset_split.y_test)
    assert metadata == {"training_rows": 120, "feature_count": 4}
