from flask import Flask, request, jsonify
import pandas as pd
import joblib
import io

superkart_api = Flask(__name__)

model = joblib.load("superkart_model.joblib")

EXPECTED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
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
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Expected a 'file' field."}), 400

        uploaded_file = request.files["file"]
        input_df = pd.read_csv(io.StringIO(uploaded_file.read().decode("utf-8")))

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

        input_df_for_prediction = input_df[EXPECTED_COLUMNS]
        predictions = model.predict(input_df_for_prediction)

        input_df["Predicted_Sales_Total"] = predictions

        return input_df.to_csv(index=False), 200, {"Content-Type": "text/csv"}

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)