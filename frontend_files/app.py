import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="SuperKart Sales Forecasting", layout="centered")

st.title("SuperKart Sales Forecasting")
st.write("Predict product-store sales revenue using the trained model.")

BACKEND_URL = "http://localhost:7860/v1/predict"

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------- Single Prediction ----------------
with tab1:
    st.subheader("Enter Product & Store Details")

    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05)
        product_type = st.selectbox("Product Type", [
            "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
            "Snack Foods", "Meat", "Household", "Hard Drinks", "Fruits and Vegetables",
            "Breads", "Soft Drinks", "Breakfast", "Others", "Starchy Foods", "Seafood"
        ])
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
        product_category = st.selectbox("Product Category", ["Food", "Drinks", "Non-Consumable"])

    with col2:
        store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", [
            "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
        ])
        store_age = st.number_input("Store Age (years)", min_value=0, value=15)

    if st.button("Predict Sales"):
        st.write("Button clicked!")
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Store_Age": store_age,
            "Product_Category": product_category,
        }

        try:
            st.write(f"Calling backend at: {BACKEND_URL}")
            response = requests.post(BACKEND_URL, json=payload, timeout=10)
            st.write(f"Response status: {response.status_code}")
            if response.status_code == 200:
                prediction = response.json()["predictions"][0]
                st.success(f"Predicted Sales Total: {prediction:.2f}")
            else:
                st.error(f"Error: {response.json()}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ---------------- Batch Prediction ----------------
with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    st.write("CSV must contain the following columns:")
    st.code(
        "Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_Type, "
        "Product_MRP, Store_Id, Store_Size, Store_Location_City_Type, Store_Type, "
        "Store_Age, Product_Category"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("Run Batch Prediction"):
            try:
                records = batch_df.to_dict(orient="records")
                response = requests.post(BACKEND_URL, json=records, timeout=30)

                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    batch_df["Predicted_Sales_Total"] = predictions
                    st.success("Batch prediction complete!")
                    st.dataframe(batch_df)

                    csv_output = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions as CSV",
                        data=csv_output,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Error: {response.json()}")
            except Exception as e:
                st.error(f"Request failed: {e}")