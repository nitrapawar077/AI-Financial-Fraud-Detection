import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fraud_detection_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

loaded_model = joblib.load(MODEL_PATH)
loaded_scaler = joblib.load(SCALER_PATH)
loaded_features = joblib.load(FEATURE_PATH)


def predict_transaction(transaction):

    transaction_df = pd.DataFrame(
        [transaction]
    )

    transaction_df = transaction_df[
        loaded_features
    ]

    scaled_transaction = loaded_scaler.transform(
        transaction_df
    )

    probability = loaded_model.predict_proba(
        scaled_transaction
    )[0, 1]

    prediction = (
        "FRAUD"
        if probability >= 0.5
        else "LEGITIMATE"
    )

    if probability >= 0.75:
        risk = "HIGH RISK"

    elif probability >= 0.40:
        risk = "MEDIUM RISK"

    else:
        risk = "LOW RISK"

    return {
        "fraud_probability":
            round(float(probability), 4),

        "prediction":
            prediction,

        "risk_tier":
            risk
    }


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "service":
            "AI-Powered Financial Fraud Detection System",

        "status":
            "running",

        "endpoint":
            "/predict"
    })


@app.route("/predict", methods=["POST"])
def predict_api():

    try:

        transaction = request.get_json(
            force=True
        )

        missing_fields = [
            feature
            for feature in loaded_features
            if feature not in transaction
        ]

        if missing_fields:

            return jsonify({
                "error":
                    f"Missing fields: {missing_fields}"
            }), 400

        result = predict_transaction(
            transaction
        )

        return jsonify(result), 200

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
