import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("clarification_model.pkl")

model = load_model()
st.set_page_config(page_title="Clarification Prediction", layout="centered")
st.title("Palm Oil Clarification ML Prediction")

st.write("Enter process parameters:")

# ------------------ Inputs ------------------------
clarifier_temp = st.number_input("Clarifier Temperature (°C)", 80.0, 100.0, 92.0)
feed_flow = st.number_input("Feed Flow", 0.0, 100.0, 46.0)
dilution_water = st.number_input("Dilution Water", 0.0, 50.0, 10.0)
sludge_density = st.number_input("Sludge Density", 0.5, 2.0, 1.15)
retention_time = st.number_input("Retention Time (min)", 10, 120, 50)
skimmer_speed = st.number_input("Skimmer Speed", 0, 50, 18)

temp_drop = st.number_input("Temperature Drop", 0.0, 5.0, 1.1)
flow_ratio = st.number_input("Flow Ratio", 0.0, 1.0, 0.4)
water_ratio = st.number_input("Water Ratio", 0.0, 1.0, 0.22)

# temp_drop = clarifier_temp - (clarifier_temp - temp_drop)
# flow_ratio = feed_flow / (feed_flow + dilution_water) if (feed_flow + dilution_water) != 0 else 0.0
# water_ratio = dilution_water / (feed_flow + dilution_water) if (feed_flow + dilution_water) != 0 else 0.0
# ---------------- Prediction ----------------------
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "clarifier_temp": clarifier_temp,
        "feed_flow": feed_flow,
        "dilution_water": dilution_water,
        "sludge_density": sludge_density,
        "retention_time": retention_time,
        "skimmer_speed": skimmer_speed,
        "temp_drop": temp_drop,
        "flow_ratio": flow_ratio,
        "water_ratio": water_ratio
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Value: {prediction:.3f}")

# --------------------------------------------------
# Optional: Show raw input
# --------------------------------------------------
with st.expander("Show input data"):
    st.dataframe(input_df if 'input_df' in locals() else pd.DataFrame())
