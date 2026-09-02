<div align="center">

# AI Revenue Recovery Engine

**Predicting the probability of failed payment recovery to prioritize and tailor recovery actions.**

*Razorpay AI Builder Challenge — Track 03: AI Revenue Recovery*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data_Processing-150458.svg?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg?logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/XGBoost-Gradient_Boosting-4479A1.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/SHAP-Explainability-8A2BE2.svg" alt="SHAP">
  <img src="https://img.shields.io/badge/Streamlit-UI_(Planned)-FF4B4B.svg?logo=streamlit" alt="Streamlit">
</p>

</div>

---

## Overview
The **AI Revenue Recovery Engine** is an ML-first project focused on solving a specific, high-value problem in the payments ecosystem: predicting whether a failed payment will eventually be recovered. By modeling this probability at the time of failure, merchants can prioritize high-value recoverable payments and apply the most effective recovery action based on historical customer behavior.

## Problem Statement
When payments fail due to technical errors, insufficient funds, or invalid cards, merchants lose revenue. Treating all failed payments equally during the retry or recovery process is inefficient and increases the risk of customer churn or unnecessary recovery costs. 

## Project Objective
Develop a robust machine learning model to estimate the **probability of payment recovery** based strictly on data available immediately after a failure. Use these probabilities to estimate **Expected Recoverable Revenue** and output a **Recovery Action Recommendation** for each failed transaction.

## How the ML System Works
This system treats revenue recovery as a binary classification problem (`Recovered: Yes/No`). The model takes in the characteristics of the failed payment, historical payment success, and customer tenure to output a calibrated probability of recovery. This probability is then combined with the original payment amount to rank and recommend actions (e.g., immediate retry, email notification, or manual intervention).

## ML Pipeline

```mermaid
graph TD
    A[Failed Payment Data] --> B(Data Validation)
    B --> C(EDA)
    C --> D(Feature Engineering)
    D --> E(Train / Validation / Test Split)
    E --> F[Baseline Model]
    E --> G[XGBoost]
    F -.-> H(Model Comparison)
    G -.-> H
    H --> I(Model Evaluation)
    I --> J(Probability Calibration)
    J --> K{Recovery Probability}
    K --> L[Expected Recoverable Revenue]
    K --> M[Recovery Action Recommendation]
    G --> N[SHAP Explainability]
    L --> O((Streamlit Dashboard))
    M --> O
    N --> O
```

## Dataset
*Note: The current dataset is fully synthetic, designed to demonstrate the ML methodology, workflow, and modeling techniques. It does not contain or reflect real Razorpay customer or payment data.*

- **Size:** 20,000 records
- **Format:** CSV
- **Location:** `data/failed_payments.csv`

## Target Definition
The problem is modeled as a binary classification task:
- `recovered = 1`: The payment was eventually recovered after the initial failure.
- `recovered = 0`: The payment was not recovered.

## Feature Overview
Features are strictly limited to what is known at or immediately following a payment failure.
- **Identifiers:** `payment_id`, `customer_id`
- **Transaction Details:** `payment_amount`, `failure_reason`, `payment_method`, `is_subscription`
- **Customer History:** `customer_tenure_months`, `past_successful_payments`, `past_failed_payments`, `historical_success_rate`, `time_since_last_success_days`
- **Recovery State:** `days_overdue`, `recovery_attempts_so_far`

## Data Leakage Prevention
Preventing target leakage is critical in this project. The dataset and features intentionally exclude:
- Recovery timestamps
- Final payment statuses
- Recovered amounts
- Number of future successful retries

## Planned Model Approach
The core models will utilize tree-based gradient boosting frameworks, specifically:
- **Baseline:** Logistic Regression or a simple Decision Tree to establish a performance floor.
- **Primary Candidates:** **Logistic Regression** and **XGBoost**. These models handle tabular data efficiently, capture non-linear relationships, and manage categorical features well without exhaustive one-hot encoding.
- **Calibration:** Predictions will be calibrated (e.g., Isotonic Regression or Platt Scaling) to ensure outputs represent true probabilities rather than arbitrary scores.

## Planned Evaluation Metrics
Because the data has a slight class imbalance, standard accuracy is insufficient. The models will be evaluated primarily using:
- **ROC-AUC:** To measure the model's ability to distinguish between recoverable and non-recoverable payments.
- **PR-AUC (Average Precision):** To evaluate precision and recall trade-offs.
- **Brier Score:** To assess the calibration and accuracy of the predicted probabilities.

## Recovery Intelligence / Expected Recoverable Revenue
The raw probability output will be converted into a business metric:
`Expected Recoverable Revenue = Predicted Probability × Payment Amount`
This metric allows merchants to sort and prioritize recovery efforts by monetary value rather than pure probability.

## Recovery Action Recommendation
Based on the probability bands and specific features (like `failure_reason`), the engine will recommend distinct actions:
- **High Probability / Technical Error:** Silent background retry.
- **Medium Probability / Insufficient Funds:** Send automated reminder email.
- **Low Probability / Invalid Card:** Prompt user to update payment details.
- **Very Low Probability:** Suspend subscription / flag for manual review.

## Explainability with SHAP
Machine learning decisions in finance must be explainable. The project will integrate **SHAP (SHapley Additive exPlanations)** to interpret model predictions. This will reveal precisely *why* a specific payment was scored high or low.

## Current Project Status

| Component | Status | Description |
| :--- | :---: | :--- |
| **Project Structure** | ✅ Completed | Folders and environment setup (`src/`, `data/`, etc.). |
| **Data Generation** | ✅ Completed | Synthetic dataset generator with realistic noise and logic. |
| **Data Validation** | ✅ Completed | Script for missing values, leakage checks, and class distribution. |
| **Data Dictionary** | ✅ Completed | Comprehensive documentation of features and logic. |
| **EDA** | ✅ Completed | Exploratory Data Analysis notebooks. |
| **Feature Engineering** | ✅ Completed | Data preprocessing and scaling. |
| **Model Training** | ✅ Completed | Training and tuning Logistic Regression and XGBoost. |
| **Model Evaluation** | ✅ Completed | Calculating AUC and calibrating probabilities. |
| **SHAP Integration** | ✅ Completed | Global and local explainability. |
| **Streamlit Dashboard** | ✅ Completed | Interactive UI for demonstrating the model. |

## Repository Structure
```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── data_dictionary.md      # Feature documentation
│   └── failed_payments.csv     # Synthetic dataset (20,000 records)
├── notebooks/                  # Planned: EDA and modeling experiments
├── src/
│   ├── generate_dataset.py     # Script to generate synthetic data
│   └── validate_dataset.py     # Script to validate data integrity and leakage
├── models/                     # Planned: Saved model artifacts
├── reports/                    # Planned: Metrics and outputs
├── app/                        # Planned: Streamlit dashboard code
└── tests/                      # Planned: Unit tests
```

## Technology Stack
- **Python:** Core language
- **Pandas & NumPy:** Data manipulation
- **Scikit-learn & XGBoost:** Advanced predictive modeling
- **SHAP:** Model explainability
- **Matplotlib & Seaborn:** Visualizations
- **Streamlit:** Interactive web interface
- **Neon PostgreSQL & psycopg:** Hosted relational database persistence (Local Docker PostgreSQL optional)

## Reproducibility / How to Generate and Validate Dataset
To generate the dataset from scratch using a fixed random seed:
```bash
python3 src/generate_dataset.py
```

To validate the generated dataset for missing values, distribution, and target leakage:
```bash
python3 src/validate_dataset.py
```

## Database Persistence (Stage 16)
Neon PostgreSQL is used as the primary hosted database. To enable persistence:
1. Copy `.env.example` to `.env`
2. Provide your Neon connection string in `DATABASE_URL`
   *(e.g., `DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require`)*
3. The schema will automatically initialize on the first connection.

*Note: Local Docker PostgreSQL is available as an optional development fallback via `docker-compose up -d`.*

## Development Roadmap
- **Stage 1 & 2:** Project Foundation & Dataset Design *(Completed)*
- **Stage 3:** Exploratory Data Analysis & Preprocessing *(Completed)*
- **Stage 4:** Model Training, Tuning, & Evaluation *(Completed)*
- **Stage 5:** SHAP Integration & Recovery Logic *(Completed)*
- **Stage 6:** Streamlit Dashboard Implementation *(Completed)*
- **Stage 7:** Probability Calibration & Robust Model Evaluation *(Completed)*
- **Stage 8:** Cost-Aware Recovery Strategy Optimization *(Completed)*
- **Stage 9:** Interactive Strategy Simulator Integration *(Completed)*

## Limitations
- **Synthetic Data:** The relationships in the data are simulated based on logical assumptions, not real-world merchant telemetry.
- **Scope:** While Neon PostgreSQL is used as the hosted PostgreSQL persistence layer (and local Docker PostgreSQL is available as an optional development fallback), the system is a demonstration ML pipeline. It evaluates and recommends actions but does NOT execute real Razorpay API payment actions. It is not a production payment execution system.

## License / Project Context
This repository is created as a submission for the **Razorpay AI Builder Challenge**.

## Running the Streamlit Dashboard
To launch the interactive demo application, run:
```bash
streamlit run app/streamlit_app.py
```

- **Stage 10:** Explainability & Decision Trace *(Completed)*
