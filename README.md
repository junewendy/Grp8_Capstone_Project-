# Model to Predict Coffee Disease Risk for Proactive Farm Management  
### Data Science Team (Group 8)

**Project Date:** June 2026  

## Team Members
- June Masolo  
- Catherine Kaino  
- Joram Lemaiyan  
- Kennedy Omoro  
- Kigen Tuwei  
- Hellen Khisa  
- Alvin Ngeno  

---

## Project Overview

This project develops an **AI-powered Predictive Early Warning System** for **Coffee Leaf Rust (CLR)** using weather data.  
The system supports farmers with **data-driven, proactive spraying decisions**, reducing unnecessary chemical use while protecting crop yield and income.

![Project Banner](projectbanner.png)

---

## 1. Business Problem

Coffee production in Kenya is increasingly threatened by unpredictable disease outbreaks such as **Coffee Leaf Rust**, which can reduce smallholder yields by up to **70%** and lead to significant financial instability.

Current disease management practices are largely **reactive**:
- Farmers wait for visible symptoms when it is often too late  
- Others apply fungicides indiscriminately as a precaution  
- This leads to:
  - High production costs  
  - Reduced profitability  
  - Environmental degradation  

There is therefore a **critical need for a data-driven early warning system** that leverages environmental data to provide **timely, accurate, and actionable risk predictions**, enabling farmers to spray only when necessary.

---

## 2. Data Understanding

This project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** framework to ensure a structured and rigorous data science workflow.

### Data Source  
Data was obtained from the **NASA POWER API**, which provides high-quality agro-meteorological datasets suitable for climate-sensitive applications.

- **Data Period:** 01-01-2010 to 31-12-2020  
- **Geographical Focus:** Nyeri County, Kenya (a major coffee-growing zone)  
- **Data Type:** Daily weather observations relevant to plant disease development  

The dataset was selected because environmental variables such as **temperature, humidity, rainfall, and wind speed** are scientifically linked to fungal disease outbreaks.

---

## 2.1 Imports & Environment Setup

The project environment was configured using standard Python data science libraries for:
- Data manipulation  
- Visualization  
- Exploratory data analysis  
- Machine learning modeling  

These tools supported a reproducible and rigorous analytical workflow throughout the project.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to understand the structure, quality, and behavior of the data before modeling.

### 3.1 Data Quality Validation

Logical validation checks were performed to ensure the data values were physically realistic for the Nyeri region:

- Maximum temperature detected: **21.02°C**  
- Minimum temperature detected: **11.91°C**  
- Maximum humidity detected: **93.19%**  

These values fall within realistic environmental ranges, confirming that the dataset is **credible and suitable for modeling**.

---

## 3.2 Univariate Analysis (Distributions)

Univariate analysis was conducted on the four core weather variables used in the project:
- Temperature (T2M)  
- Relative Humidity (RH2M)  
- Rainfall (PRECTOTCORR)  
- Wind Speed (WS2M)  

Histograms were used to assess:
- Distribution shapes  
- Presence of outliers  
- Environmental patterns  
- Normality characteristics  

### Visualizing Weather Patterns in Nyeri (2010–2020)

The plots below illustrate the distributions of the four climate variables used in the model.

![Univariate Weather Analysis](disributions.jpeg)

*Figure: Distribution of temperature, humidity, rainfall, and wind speed in Nyeri (2010–2020).*

---

## Key Insights from the Analysis

**Temperature (T2M)**
- Temperatures are relatively stable throughout the period.
- Most observations fall between **14°C and 19°C**.
- This range is optimal for coffee growth but also favorable for fungal development.

**Humidity (RH2M)**
- The distribution is **left-skewed**, indicating consistently high humidity.
- The majority of days fall between **75% and 90%**.
- This confirms that Nyeri provides an environment highly conducive to disease outbreaks.

**Rainfall (PRECTOTCORR)**
- The rainfall distribution exhibits strong **zero-inflation** (many days with no rain).
- Heavy rainfall events are rare but significant, reaching up to **70mm per day**.
- This confirms that **moisture availability is episodic but critical** in triggering disease risk.

**Wind Speed (WS2M)**
- Wind speeds are generally moderate, centered around **2.0–2.5 m/s**.
- Very few extreme wind events (>4.0 m/s) are observed.
- This supports later modeling findings that wind contributes minimally to disease prediction.

---

## Dataset Summary

The final dataset used for modeling consisted of:

- **Rows:** 4,018  
- **Columns:** 4  
- **Missing Values:** None  

This confirms that the dataset is **clean, complete, and suitable for machine learning modeling**.

---
## 4. Data Preparation & Feature Engineering

A rigorous data preparation process was undertaken to transform the raw weather data into a format suitable for predictive modeling. This step is critical in ensuring that the model reflects **real agronomic behavior** rather than purely statistical patterns.

### 4.1 Domain-Driven Feature Design

Rather than relying solely on statistical transformations, this project incorporated **agronomic knowledge** to guide feature construction. The preparation process involved:

- Column renaming for clarity and interpretability  
- Date-based mapping of coffee crop stages  
- Rule-based construction of the **target variable (Risk Label)**  
- Time-aware feature engineering using **lagged variables** and **rolling averages**  

This ensured that the model remained **scientifically grounded and contextually meaningful**.

---

## 4.2 Target Variable Construction (Risk Label)

To build a supervised machine learning model, a target variable was required. Since labeled disease outbreak data was unavailable, the dataset was labeled using **agronomic rules commonly used by plant pathologists**.

### Agronomic Basis for Risk Labeling

Research on **Coffee Leaf Rust (Hemileia vastatrix)** indicates that outbreaks are most likely when:

- **Temperature:** between **15°C and 30°C**  
  - Optimal range: **21°C – 25°C**  
- **Relative Humidity:** sustained above **90%** for 24–48 hours  
- **Rainfall:** present (to facilitate spore dispersal), but not excessively heavy  

Additionally, disease susceptibility varies by **crop growth stage**. In Kenya, coffee plants are most vulnerable during:
- **Flowering stage** (March–April and October–November)  
- **Early cherry development stage**  

Using these scientifically grounded conditions, a rule-based logic was applied to generate the target column:

> **Risk Label (Target):** Low, Medium, or High risk of disease outbreak

This approach ensures that the model predictions are **biologically meaningful and practically interpretable**.

---

## 4.3 Temporal Feature Engineering (Lagging and Rolling Averages)

A key limitation of naive models is that they treat weather and disease risk as occurring simultaneously. In reality, there is a **biological delay (incubation period)** between conducive weather conditions and visible disease symptoms.

### Scientific Motivation

For Coffee Leaf Rust in East Africa:
- Reported incubation lag ranges between **15–30 days**  
- Early warning systems often use a shorter operational lag of **8–15 days**  

Based on this evidence, this project implemented a **14-day predictive lag window**, enabling the model to function as a true **forecasting system** rather than a reactive classifier.

---

## 4.4 Creation of Lagged Features

Four core weather variables were transformed into time-aware predictors:

- Temperature  
- Humidity  
- Rainfall  
- Wind Speed  

Two types of temporal features were created:

### a) 14-Day Lag Features
These represent the weather conditions **14 days prior**, allowing the model to learn delayed disease responses.

Example:
- `Humidity (%)_Lag14` = humidity value recorded 14 days earlier  

### b) 14-Day Rolling Averages
Sustained environmental conditions are often more biologically relevant than isolated events. Therefore, rolling averages were computed to capture **cumulative exposure**.

Example:
- `Humidity (%)_Avg_Last14Days` = average humidity over the previous 14 days  

This allows the model to distinguish between:
- One unusually humid day  
vs  
- Two weeks of consistently high humidity (which is much more dangerous biologically)

---

## 4.5 Final Dataset Structure After Feature Engineering

After generating lagged features and rolling averages, the initial rows containing missing values (created by shifting) were removed.

A sample of the final engineered dataset is shown below:

| Date       | Risk Label (Target) | Humidity (%)_Lag14 | Humidity (%)_Avg_Last14Days |
|------------|----------------------|---------------------|------------------------------|
| 2010-01-15 | Medium               | 86.57              | 78.30                        |
| 2010-01-16 | Medium               | 89.25              | 77.43                        |
| 2010-01-17 | Medium               | 85.61              | 76.35                        |
| 2010-01-18 | Medium               | 85.22              | 75.62                        |
| 2010-01-19 | Medium               | 78.20              | 74.39                        |

This confirms that the dataset now reflects **real-world temporal causality**, allowing the model to generate **practical early warnings** rather than merely explaining historical patterns.

---

