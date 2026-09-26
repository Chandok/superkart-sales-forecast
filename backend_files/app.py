from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

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


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SuperKart Sales Forecasting API is running."})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format. Provide a JSON object or list of objects."}), 400

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

        input_df = input_df[EXPECTED_COLUMNS]

        predictions = model.predict(input_df)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)