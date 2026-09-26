from flask import Flask, request, jsonify
import pandas as pd
import joblib

superkart_api = Flask(__name__)

model = joblib.load("superkart_model.joblib")

EXPECTED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Age",
    "Product_Category",
]


@superkart_api.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SuperKart Sales Forecasting API is running."})


@superkart_api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@superkart_api.post("/v1/predict")
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

        input_df = input_df[EXPECTED_COLUMNS]
        prediction = model.predict(input_df)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    try:
        data = request.get_json()
        input_df = pd.DataFrame(data)

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

        input_df = input_df[EXPECTED_COLUMNS]
        predictions = model.predict(input_df)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)