import json

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.api import create_app

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
LABELS = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def _app(tmp_path):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", DummyClassifier(strategy="constant", constant=LABELS[0])),
        ]
    )
    model.fit([[5.1, 3.5, 1.4, 0.2]], [LABELS[0]])
    model_path = tmp_path / "iris_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "Test Iris classifier",
                "model_version": "test",
                "feature_names": FEATURES,
                "valid_target_labels": LABELS,
                "metrics": {"accuracy": 1.0, "macro_f1": 1.0},
            }
        ),
        encoding="utf-8",
    )
    return create_app(model_path, metadata_path).test_client()


def _valid_payload():
    return dict(zip(FEATURES, [5.1, 3.5, 1.4, 0.2], strict=True))


def test_health_and_model_info(tmp_path):
    client = _app(tmp_path)
    assert client.get("/health").json == {"status": "healthy"}
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json["feature_names"] == FEATURES
    assert info.json["supported_classes"] == LABELS


def test_valid_prediction_has_expected_schema_and_class(tmp_path):
    response = _app(tmp_path).post("/predict", json=_valid_payload())
    assert response.status_code == 200
    assert set(response.json) == {"prediction"}
    assert response.json["prediction"] in LABELS


def test_prediction_rejects_invalid_inputs(tmp_path):
    client = _app(tmp_path)
    assert client.post("/predict", json={}).status_code == 400
    invalid_type = _valid_payload()
    invalid_type[FEATURES[0]] = "5.1"
    assert client.post("/predict", json=invalid_type).status_code == 400
    non_finite = _valid_payload()
    non_finite[FEATURES[0]] = float("inf")
    assert client.post("/predict", json=non_finite).status_code == 400
    assert client.post("/predict", data="not-json").status_code == 400


def test_missing_model_reports_not_ready(tmp_path):
    client = create_app(
        tmp_path / "missing.joblib", tmp_path / "missing.json"
    ).test_client()
    assert client.get("/health").status_code == 503
    assert client.get("/model-info").status_code == 503
    assert client.post("/predict", json=_valid_payload()).status_code == 503
