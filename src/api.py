"""Flask inference API for the validated Iris model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, request

DEFAULT_MODEL_PATH = Path("artifacts/iris_model.joblib")
DEFAULT_METADATA_PATH = Path("artifacts/model_metadata.json")


def _error(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def _load_artifacts(
    model_path: Path, metadata_path: Path
) -> tuple[Any, dict[str, Any]]:
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Required model artifacts are missing.")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")
    return model, metadata


def create_app(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> Flask:
    """Create an app and load model artifacts exactly once."""
    app = Flask(__name__)
    model = None
    metadata: dict[str, Any] = {}
    load_error: Exception | None = None
    try:
        model, metadata = _load_artifacts(Path(model_path), Path(metadata_path))
    except Exception as error:
        load_error = error

    app.config["MODEL"] = model
    app.config["MODEL_METADATA"] = metadata
    app.config["MODEL_LOAD_ERROR"] = load_error

    @app.get("/health")
    def health():
        if app.config["MODEL"] is None:
            return _error("Model is not ready.", 503)
        return jsonify({"status": "healthy hurray!!!!"})

    @app.get("/model-info")
    def model_info():
        if app.config["MODEL"] is None:
            return _error("Model is not ready.", 503)
        model_metadata = app.config["MODEL_METADATA"]
        return jsonify(
            {
                "model_name": model_metadata.get("model_name"),
                "model_version": model_metadata.get(
                    "model_version", model_metadata.get("source_commit_sha")
                ),
                "feature_names": model_metadata.get("feature_names", []),
                "supported_classes": model_metadata.get("valid_target_labels", []),
                "metrics": model_metadata.get("metrics", {}),
            }
        )

    @app.post("/predict")
    def predict():
        if app.config["MODEL"] is None:
            return _error("Model is not ready.", 503)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error("Request body must be valid JSON object.", 400)
        feature_names = app.config["MODEL_METADATA"].get("feature_names", [])
        if not isinstance(feature_names, list) or not feature_names:
            return _error("Model feature metadata is invalid.", 500)
        missing = [name for name in feature_names if name not in payload]
        if missing:
            return _error(f"Missing required features: {', '.join(missing)}.", 400)
        values = [payload[name] for name in feature_names]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in values
        ):
            return _error("All feature values must be finite numbers.", 400)
        try:
            prediction = app.config["MODEL"].predict([values])[0]
        except (ValueError, TypeError, RuntimeError):
            return _error("Could not generate a prediction.", 422)
        prediction = str(prediction)
        valid_labels = set(app.config["MODEL_METADATA"].get("valid_target_labels", []))
        if prediction not in valid_labels:
            return _error("Model returned an unsupported prediction.", 500)
        return jsonify({"prediction": prediction})

    return app


app = create_app()
