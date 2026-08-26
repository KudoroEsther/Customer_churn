"""Flask API serving the churn prediction model."""
import logging

from flask import Flask, jsonify, request

from src.config import MODEL_PATH
from src.predict import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict_endpoint():
    """Accepts a single customer record or a list of records as JSON.

    Example: {"tenure": 5, "MonthlyCharges": 70.35, "TotalCharges": 350.0, ...}
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify(error="Request body must be JSON."), 400

    is_batch = isinstance(payload, list)
    records = payload if is_batch else [payload]

    try:
        labels, proba = predict(model, records)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 400
        logger.exception("Prediction failed")
        return jsonify(error=str(exc)), 400

    results = [
        {"churn_prediction": label, "churn_probability": prob}
        for label, prob in zip(labels, proba)
    ]
    return jsonify(results if is_batch else results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
