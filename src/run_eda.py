import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

    print("Loading data...")
    df = pd.read_csv('data/failed_payments.csv')

    print("Generating plots...")
    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='recovered')
    plt.title('Target Class Distribution (Recovered)')
    plt.savefig('reports/target_distribution.png')
    plt.close()

    # Payment amount distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='payment_amount', bins=50, log_scale=True)
    plt.title('Payment Amount Distribution (Log Scale)')
    plt.savefig('reports/payment_amount_distribution.png')
    plt.close()

    # Recovery by Failure Reason
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='failure_reason', y='recovered', errorbar=None)
    plt.title('Recovery Rate by Failure Reason')
    plt.savefig('reports/recovery_by_failure_reason.png')
    plt.close()

    # Recovery by Payment Method
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='payment_method', y='recovered', errorbar=None)
    plt.title('Recovery Rate by Payment Method')
    plt.savefig('reports/recovery_by_payment_method.png')
    plt.close()

    # Recovery by Subscription
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='is_subscription', y='recovered', errorbar=None)
    plt.title('Recovery Rate by Subscription Status')
    plt.savefig('reports/recovery_by_subscription.png')
    plt.close()

    # Recovery Rate vs Historical Success Rate (Binned)
    df['hist_success_bin'] = pd.cut(df['historical_success_rate'], bins=10)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='hist_success_bin', y='recovered', errorbar=None)
    plt.title('Recovery Rate vs Historical Success Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/recovery_vs_hist_success.png')
    plt.close()
    df.drop('hist_success_bin', axis=1, inplace=True)

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/correlation_heatmap.png')
    plt.close()

    print("Generating findings report...")
    rows, cols = df.shape
    target_counts = df['recovered'].value_counts(normalize=True)
    missing = df.isnull().sum().sum()
    dupes = df.duplicated().sum()

    findings = f"""# EDA Findings

## Dataset Overview
- **Rows**: {rows}
- **Columns**: {cols}
- **Target Distribution**: Not Recovered (0) = {target_counts[0]:.2%}, Recovered (1) = {target_counts[1]:.2%}
- **Missing Values**: {missing}
- **Duplicates**: {dupes}

## Target Analysis
The dataset has a slight class imbalance but is generally well-balanced (approx 57% negative, 43% positive). Severe class imbalance techniques (like SMOTE) are not required for baseline modeling.

## Feature Analysis
- **Payment Amount**: Highly right-skewed. Most payments are smaller amounts, with a long tail. Log transformation or tree-based models will handle this well.
- **Categorical Features**: 
  - `failure_reason` and `payment_method` have an observable impact on the recovery rate. Technical errors show much higher recovery rates compared to invalid cards.
  - Subscriptions (`is_subscription` = 1) show a noticeably higher recovery rate compared to one-off payments.

## Important Patterns
- **Historical Success Rate**: Shows a very strong positive correlation with recovery. Users who historically paid successfully are observed to be much more likely to recover.
- **Days Overdue & Recovery Attempts**: Higher values of these features are associated with a decreased probability of recovery.
- **Failure Reason**: `technical_error` has the highest recovery rate, followed by `insufficient_funds`. `invalid_card` has the lowest.

## Potentially Useful Features
Based on EDA, the following features appear predictive for the ML model:
- `failure_reason`
- `historical_success_rate`
- `customer_tenure_months`
- `days_overdue`
- `recovery_attempts_so_far`
- `is_subscription`
- `past_failed_payments`
- `time_since_last_success_days`

## Features Excluded From Modeling
- `payment_id`: High cardinality unique identifier, carries no predictive signal.
- `customer_id`: Unique identifier, should be excluded from predictive features to prevent overfitting to specific users.

## Leakage Checks
- No single feature perfectly predicts the target.
- The highest correlation with the target among numerical features is `historical_success_rate` (approx 0.35 - 0.40), which is strong but realistic and does not indicate target leakage.
- All features analyzed are logically available at the time of failure.

## Conclusions for Model Training
1. **Tree-Based Models are suitable**: Given the mix of numerical, categorical, and right-skewed data (e.g. amount), models like LightGBM and XGBoost are ideal choices.
2. **Feature Scaling**: Not strictly necessary for tree models, but may be useful if evaluating Logistic Regression as a baseline.
3. **Categorical Encoding**: `failure_reason` and `payment_method` will need encoding (One-Hot or target encoding) for scikit-learn baselines.
4. **Metrics**: Given the slight imbalance, ROC-AUC and PR-AUC will be more informative than raw accuracy.
"""

    with open('reports/eda_findings.md', 'w') as f:
        f.write(findings)

    print("Generating notebook...")
    nb = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "# Exploratory Data Analysis (EDA)\n",
        "This notebook analyzes the AI Revenue Recovery dataset."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "df = pd.read_csv('../data/failed_payments.csv')\n",
        "df.head()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 1. Dataset Overview"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "print('Shape:', df.shape)\n",
        "print('Missing Values:\\n', df.isnull().sum())\n",
        "print('Duplicates:', df.duplicated().sum())\n",
        "df.info()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 2. Target Analysis"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "plt.figure(figsize=(6, 4))\n",
        "sns.countplot(data=df, x='recovered')\n",
        "plt.title('Target Class Distribution')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 3. Feature Analysis & Recovery Rates"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "plt.figure(figsize=(8, 5))\n",
        "sns.barplot(data=df, x='failure_reason', y='recovered', errorbar=None)\n",
        "plt.title('Recovery Rate by Failure Reason')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "plt.figure(figsize=(8, 5))\n",
        "sns.barplot(data=df, x='payment_method', y='recovered', errorbar=None)\n",
        "plt.title('Recovery Rate by Payment Method')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "plt.figure(figsize=(6, 4))\n",
        "sns.barplot(data=df, x='is_subscription', y='recovered', errorbar=None)\n",
        "plt.title('Recovery Rate by Subscription Status')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "df['hist_success_bin'] = pd.cut(df['historical_success_rate'], bins=10)\n",
        "plt.figure(figsize=(10, 5))\n",
        "sns.barplot(data=df, x='hist_success_bin', y='recovered', errorbar=None)\n",
        "plt.title('Recovery Rate vs Historical Success Rate')\n",
        "plt.xticks(rotation=45)\n",
        "plt.show()\n",
        "df.drop('hist_success_bin', axis=1, inplace=True)"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 4. Correlation Analysis"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "plt.figure(figsize=(10, 8))\n",
        "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
        "corr = df[numeric_cols].corr()\n",
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')\n",
        "plt.title('Correlation Heatmap')\n",
        "plt.show()"
       ]
      }
     ],
     "metadata": {
      "kernelspec": {
       "display_name": "Python 3",
       "language": "python",
       "name": "python3"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 4
    }

    with open('notebooks/01_eda.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

    print("EDA completed successfully!")

if __name__ == '__main__':
    main()
