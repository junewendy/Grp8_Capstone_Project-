import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
model = joblib.load('coffee_disease_model_v1.pkl')

st.set_page_config(page_title="Coffee Risk AI", page_icon="☕")
st.title("☕ Coffee Leaf Rust: Early Warning System")
st.markdown("---")

# 2. User Inputs
st.sidebar.header("Input Weather Conditions")
temp = st.sidebar.number_input("Current Temp (Avg) °C", value=22.0)
hum = st.sidebar.slider("Current Humidity (%)", 0, 100, 60)
rain = st.sidebar.number_input("Current Rainfall (mm)", value=5.0)
wind = st.sidebar.number_input("Wind Speed (m/s)", value=2.0)

# 3. Analyze Button
if st.button("Analyze Risk Level"):
    # Create the DataFrame the model expects
    data = {
        'Temp (Avg)': [temp], 'Humidity (%)': [hum], 'Rainfall (mm)': [rain], 'Wind Speed (m/s)': [wind],
        'Month': [1], 'Crop Stage': ['Flowering'],
        'Temp (Avg)_Lag14': [temp], 'Humidity (%)_Lag14': [hum], 'Rainfall (mm)_Lag14': [rain], 'Wind Speed (m/s)_Lag14': [wind],
        'Temp (Avg)_Avg_Last14Days': [temp], 'Humidity (%)_Avg_Last14Days': [hum], 'Rainfall (mm)_Avg_Last14Days': [rain], 'Wind Speed (m/s)_Avg_Last14Days': [wind]
    }
    input_df = pd.DataFrame(data)
    
    # 4. Prediction & Mapping
    prediction_numeric = model.predict(input_df)[0]
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    prediction_text = risk_map.get(prediction_numeric, str(prediction_numeric))
    
    # 5. The Professional Output
    if prediction_text == 'High':
        st.error(f"### 🚨 Predicted Risk: {prediction_text}")
        st.warning("**Recommendation:** Initiate preventive fungicide application immediately and alert regional officers.")
    elif prediction_text == 'Medium':
        st.warning(f"### 🔔 Predicted Risk: {prediction_text}")
        st.info("**Recommendation:** Increase monitoring frequency. Inspect lower leaf surfaces for yellow-orange spores.")
    else:
        st.success(f"### ✅ Predicted Risk: {prediction_text}")
        st.write("**Recommendation:** No immediate action required. Continue standard maintenance.")