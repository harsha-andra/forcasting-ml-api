import streamlit as st
import requests

import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLEXSRS"] = "false"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"








API_URL = "https://forcasting-ml-api.onrender.com/predict"  # Change to your deployed URL when ready

st.title("📈 Forecasting ML API Demo")

st.markdown("Enter the following input features (defaults provided for quick testing):")

# Categorical features
item_id = st.number_input("Item ID", value=1)
dept_id = st.number_input("Dept ID", value=1)
cat_id = st.number_input("Cat ID", value=1)
store_id = st.number_input("Store ID", value=1)
state_id = st.number_input("State ID", value=1)

# Numerical features
lag_1 = st.number_input("Lag 1", value=10.5)
lag_7 = st.number_input("Lag 7", value=11.2)
lag_28 = st.number_input("Lag 28", value=9.8)
rolling_mean_7 = st.number_input("Rolling Mean 7", value=10.0)
rolling_mean_28 = st.number_input("Rolling Mean 28", value=10.1)

if st.button("🔍 Predict"):
    payload = {
        "item_id": item_id,
        "dept_id": dept_id,
        "cat_id": cat_id,
        "store_id": store_id,
        "state_id": state_id,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_28": lag_28,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_28": rolling_mean_28,
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            forecast = response.json().get("forecasted_sales")
            st.success(f"✅ Forecasted Sales: {forecast}")
        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
