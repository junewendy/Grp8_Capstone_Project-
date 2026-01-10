import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
# Ensure this .pkl file is in the same folder as this app.py
model = joblib.load('coffee_disease_model_v1.pkl')

st.set_page_config(page_title="Coffee Risk AI", page_icon="☕")
st.title("☕ Coffee Leaf Rust: Early Warning System")
st.markdown("---")

st.sidebar.header("Input Weather Conditions")
st.sidebar.write("Provide the current weather metrics to calculate the risk score.")

# 2. User Inputs
temp = st.sidebar.number_input("Current Temp (Avg) °C", value=22.0)
hum = st.sidebar.slider("Current Humidity (%)", 0, 100, 60)
rain = st.sidebar.number_input("Current Rainfall (mm)", value=5.0)
wind = st.sidebar.number_input("Wind Speed (m/s)", value=2.0)

# 3. The "Auto-Fill" Logic for Model Compatibility
if st.button("Analyze Risk Level"):
    # We create a dataframe with all columns the model expects
    data = {
        'Temp (Avg)': [temp],
        'Humidity (%)': [hum],
        'Rainfall (mm)': [rain],
        'Wind Speed (m/s)': [wind],
        'Month': [1], 
        'Crop Stage': ['Flowering'], 
        'Temp (Avg)_Lag14': [temp], 
        'Humidity (%)_Lag14': [hum],
        'Rainfall (mm)_Lag14': [rain],
        'Wind Speed (m/s)_Lag14': [wind],
        'Temp (Avg)_Avg_Last14Days': [temp],
        'Humidity (%)_Avg_Last14Days': [hum],
        'Rainfall (mm)_Avg_Last14Days': [rain],
        'Wind Speed (m/s)_Avg_Last14Days': [wind]
    }
    
    input_df = pd.DataFrame(data)
    
    # 4. Prediction
    prediction_numeric = model.predict(input_df)[0]
    
    # 5. Mapping & Labels (This fixes the "2" issue)
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    prediction_text = risk_map.get(prediction_numeric, str(prediction_numeric))
    
    # 6. Professional UI with Warning Messages
    st.subheader("Results:")
    
    if prediction_text == 'High':
        st.error(f"### 🚨 Predicted Risk: {prediction_text}")
        st.markdown("""
        **Recommendation for Farm Management:**
        * **Immediate Action:** Initiate preventive fungicide application.
        * **Scouting:** Increase daily inspections of lower leaf surfaces.
        * **Alert:** Notify regional agricultural officers of high-risk conditions.
        """)
    elif prediction_text == 'Medium':
        st.warning(f"### 🔔 Predicted Risk: {prediction_text}")
        st.markdown("""
        **Recommendation for Farm Management:**
        * **Monitoring:** Increase scouting frequency to twice a week.
        * **Preparation:** Ensure fungicide stocks are available if conditions worsen.
        """)
    else:
        st.success(f"### ✅ Predicted Risk: {prediction_text}")
        st.markdown("""
        **Recommendation for Farm Management:**
        * **Routine:** Continue standard nutritional and pruning cycles.
        * **Status:** No immediate intervention required.
        """)

    # Show the raw model output for technical transparency
    st.caption(f"Raw Model Output: {prediction_numeric}")