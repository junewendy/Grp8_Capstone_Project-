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

![Univariate Weather Analysis](histogram.png)

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

# 5. Modelling

This section sets up the supervised learning pipeline, defines the train–test split with stratification, and compares multiple algorithms (Logistic Regression, Random Forest, XGBoost) using a consistent preprocessing framework.

The main goal is to predict the multi-class target `RiskLabel(Target)` (Low, Medium, High) while handling class imbalance and avoiding data leakage through careful preprocessing and pipeline design.

## 5.1 Preparing X and y

The first step is to select the input features (weather and crop stage) and encode the target into numeric form using the mapping **{"Low": 0, "Medium": 1, "High": 2}**.

This preserves the ordinal meaning of the classes while making them compatible with scikit-learn models and metrics.

```python
# Define features and target
feature_cols = ["Temp(Avg)", "Humidity(%)", "Rainfall(mm)", "WindSpeed(m/s)", "CropStage"]
target_col = "RiskLabel(Target)"

X = df_final[feature_cols].copy()
y = df_final[target_col].copy()

# Encode target: Low=0, Medium=1, High=2
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
y_encoded = y.map(risk_mapping)

print("Class distribution (encoded):")
print(y_encoded.value_counts())

## 5.2 Train–Test Split with Stratification

The dataset is split into training and testing sets using train_test_split, with stratify=y_encoded to preserve the class distribution across Low, Medium, and High in both splits.

This is important because disease risk classes are imbalanced and a non-stratified split could produce unrealistic performance estimates.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train target distribution:")
print(y_train.value_counts(normalize=True))
print("Test target distribution:")
print(y_test.value_counts(normalize=True))

## 5.3 Handling Class Imbalance with Class Weights

Because some risk categories are less frequent, the notebook computes balanced class weights and uses them in models that support class_weight, especially Logistic Regression and Random Forest.

This up-weights minority classes, reducing the tendency of the model to focus only on the majority (e.g., Low risk) class.

```python
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))

print("Computed class weights:", class_weight_dict)

# Optional: save class weights for later reuse
with open("class_weights.pkl", "wb") as f:
    pickle.dump(class_weight_dict, f)

## 5.4 Preprocessing and Pipeline Setup

The notebook uses a ColumnTransformer to apply different transformations to numeric and categorical features before feeding them to the models.

Numeric features are standardised using StandardScaler, and the categorical CropStage is encoded with OrdinalEncoder, then all steps are wrapped inside a scikit-learn Pipeline.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

numeric_features = ["Temp(Avg)", "Humidity(%)", "Rainfall(mm)", "WindSpeed(m/s)"]
categorical_features = ["CropStage"]

numeric_transformer = StandardScaler()
categorical_transformer = OrdinalEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

## 5.5 Model Definitions

Multiple models are defined in a models dictionary so they can be trained and evaluated in a uniform loop.

The dictionary includes a baseline Logistic Regression, a RandomForestClassifier, and an XGBClassifier, each with its own hyperparameters and the use of class_weight="balanced" where supported.

```python
from xgboost import XGBClassifier  # used for tree boosting

models = {
    "Baseline (LogReg)": LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        scale_pos_weight=1
    )
}

## 5.6 Building the Training Pipeline

Each model is wrapped inside a full training pipeline that includes the shared preprocessor. This ensures that all models receive identically transformed data during training and prediction.

```python

from sklearn.pipeline import Pipeline

# Create pipelines for each model
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

## 5.7 Model Training and Cross-Validation

To assess generalization performance before looking at the test set, each pipeline is evaluated using 5‑fold StratifiedKFold cross‑validation on the training data.

```python

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipeline in pipelines.items():
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1
    )
    cv_results[name] = scores
    print(f"{name}: CV F1‑weighted = {scores.mean():.3f} (±{scores.std():.3f})")

## 5.8 Final Model Training on Full Training Set

After cross‑validation, each pipeline is refitted on the entire training set to produce the final models ready for evaluation on the held‑out test set.

```python
fitted_models = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    fitted_models[name] = pipeline
    print(f"Trained {name}")

# 6. Model Evaluation

## 6.1 Performance Metrics
Because the business cost of missing a High‑risk day is much higher than a false alarm, the evaluation uses a suite of metrics that highlight different aspects of model quality:

Accuracy: Overall percentage of correct predictions

F1‑Score (weighted): Harmonic mean of precision and recall, weighted by support

Precision per class: How many predicted High‑risk days truly were High‑risk

Recall per class: How many actual High‑risk days were correctly flagged

Confusion Matrix: Visual breakdown of prediction vs. actual

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"Evaluation for {model_name}")
    print(f"{'='*50}")
    
    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    
    return y_pred

## 6.2 Model Comparison Results

The three models are evaluated on the held‑out test set, producing the following performance summary:


Model	| Accuracy	| Weighted F1	| High‑Risk Recall	| High‑Risk Precision|
Baseline (LogReg) |	0.713	| 0.704	| 0.67	| 0.65 |
Random Forest	| 0.742	| 0.731	| 0.73	| 0.71|
XGBoost	| 0.759	| 0.748	| 0.76	|

```python
# Evaluate all models
results = {}
for name, model in fitted_models.items():
    y_pred = evaluate_model(model, X_test, y_test, name)
    results[name] = {
        "predictions": y_pred,
        "accuracy": accuracy_score(y_test, y_pred)
    }

## 6.3 Feature Importance Analysis (Random Forest & XGBoost)

