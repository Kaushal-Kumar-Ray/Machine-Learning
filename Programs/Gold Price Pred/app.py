import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("gold_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="📈",
    layout="centered"
)

# App title
st.title("💰 Gold Price Prediction Dashboard")
st.write("Predict gold price using macroeconomic indicators")

st.markdown("---")

# Input fields
spx = st.number_input("📊 S&P 500 Index (SPX)", value=1500.0)
uso = st.number_input("🛢️ Oil Price (USO)", value=70.0)
slv = st.number_input("🥈 Silver Price (SLV)", value=18.0)
eurusd = st.number_input("💱 EUR / USD Exchange Rate", value=1.12)

# Predict button
if st.button("🔮 Predict Gold Price"):
    # Prepare input
    input_data = np.array([[spx, uso, slv, eurusd]])
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Output
    st.success(f"💰 Predicted Gold Price: **${prediction[0]:.2f}**")
