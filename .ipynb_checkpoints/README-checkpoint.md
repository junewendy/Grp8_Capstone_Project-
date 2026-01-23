# Grp8_Capstone_Project-
# ☕ Coffee Leaf Rust: Early Warning System
![Coffee Leaf Rust](URL_OF_THE_IMAGE_HERE)

## 📌 Project Overview
**Protect your harvest. Reduce costs. Grow better coffee.**

This project uses Machine Learning to predict the risk of Coffee Leaf Rust (CLR) based on weather patterns, helping farmers in Kenya move from reactive to proactive farm management.

### Executive Summary
Coffee is a cornerstone of Kenya's agricultural economy. Coffee Leaf Rust (CLR) can reduce yields by up to 50% if not managed proactively. This project leverages Machine Learning to predict disease risk levels (Low, Medium, High) based on environmental factors like Temperature, Humidity, and Rainfall.

By providing a 14-day predictive window, this tool allows for Precision Agriculture, reducing unnecessary fungicide costs and stabilizing farm income.

### Technical Stack
Language: Python 3.10

Libraries: Pandas, Scikit-Learn, Joblib, Matplotlib

Deployment: Streamlit (Web Interface)

Model: Random Forest Classifier (within a Preprocessing Pipeline)

### Key Features
Real-time Risk Assessment: Interactive dashboard for farmers and extension officers.

Time-Series Integration: The model accounts for 14-day weather lags and rolling averages.

Actionable Insights: Rather than just a score, the app provides specific management recommendations for each risk level.

### Why use this tool?
Coffee Leaf Rust (CLR) can destroy a farm's profit in a single season. Most farmers spray fungicide after they see the yellow spots, but by then, the damage is already done.

This AI tool acts as a weather-based scout. It analyzes your local temperature and rainfall to tell you the risk level before the disease spreads, helping you:

Save Money: Only spray when the risk is truly high.

Protect Yield: Stop the rust before it causes leaves to fall.

Plan Better: Know when to increase scouting in the fields.

### How to Run the App
Clone the repository:

Bash

git clone https://github.com/YourUsername/Phase-5-Project.git
Navigate to the directory:

Bash

cd "phase 5 project"
Run the Streamlit app:

Bash

python -m streamlit run app.py

### How the App Works
This app uses a Machine Learning model trained on historical weather and disease data.

Input: You enter today's Temperature, Humidity, and Rainfall.

Analysis: The app calculates how these conditions affect the rust fungus.

Output: You get a simple Low, Medium, or High risk rating with clear instructions on what to do next.

## Recommendations Provided
✅ Low Risk: Your trees are safe. Continue with regular weeding and nutrition.

🔔 Medium Risk: Warning. The weather is favoring the fungus. Start checking the undersides of leaves more frequently.

🚨 High Risk: Danger. Conditions are perfect for an outbreak. Prepare for preventive spraying immediately.

### Model Performance
Accuracy: 99.50%

F1-Score: 99.53%

Recall: 99.06%

**Primary Metric:** I prioritized **Recall** for the 'High Risk' class to ensure no potential outbreaks are missed by the system.
