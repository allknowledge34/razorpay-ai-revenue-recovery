# AI Revenue Recovery Predictor

## Project Name
AI Revenue Recovery Predictor

## Problem Statement
Predict whether a failed payment is likely to be recovered, and use that prediction to recommend a simple recovery action.

## Track
Razorpay AI Builder challenge, Track 3: AI Revenue Recovery

## Current Scope
The current scope is strictly focused on establishing the project foundation. This includes creating the initial repository structure, defining essential dependencies, and setting up the environment for the upcoming ML pipeline development.

## Planned ML Pipeline
1. **Data Generation/Ingestion:** Create or load a dataset simulating failed payments, their characteristics, and ultimate recovery status.
2. **Data Preprocessing & EDA:** Clean data, perform exploratory data analysis, and engineer features.
3. **Model Training:** Train a LightGBM model to predict the probability of a successful payment recovery.
4. **Model Evaluation:** Assess performance using appropriate metrics (e.g., ROC AUC, Precision, Recall).
5. **Model Explainability:** Utilize SHAP values to explain model predictions and identify key drivers.
6. **Action Recommendation:** Map prediction probabilities and key features to actionable recovery strategies.

## Technology Stack
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning utilities and preprocessing
- **LightGBM**: Gradient boosting framework for the core predictive model
- **SHAP**: Explainable AI (XAI) for model interpretability
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Interactive web dashboard for presentation (planned for later stage)

## Development Stages
1. **Stage 1: Project Foundation (Current)** - Setting up directory structure, environment, and documentation.
2. **Stage 2: Data Preparation** - Preparing the dataset and conducting EDA.
3. **Stage 3: Model Training & Evaluation** - Building, tuning, and evaluating the core LightGBM model.
4. **Stage 4: Explainability & Recommendations** - Integrating SHAP and defining business logic for recovery actions.
5. **Stage 5: Dashboard Development** - Building the Streamlit UI to present the model and recommendations.
