# E-Commerce-Churn-Prediction
# E-Commerce Customer Churn Prediction

A binary classification project to predict customer churn for an e-commerce platform using LightGBM, with hyperparameter tuning and feature-driven business insights.

## Problem Statement

Customer churn directly impacts revenue for subscription and e-commerce businesses. This project builds a predictive model to identify customers likely to churn, enabling proactive retention strategies rather than reactive loss.

## Dataset

- Source: [E-Commerce Customer Churn Analysis and Prediction (Kaggle)](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
- 5,630 customer records, 16 features after preprocessing
- Target variable: `Churn` (binary — 0 = retained, 1 = churned)
- Class distribution: ~83% retained, ~17% churned (moderate imbalance)

Features include tenure, preferred login device, city tier, payment mode, app usage hours, order category, satisfaction score, marital status, complaint history, coupon usage, order count, days since last order, and cashback amount.

## Approach

### 1. Data Cleaning
- Dropped non-predictive/leaky columns: `CustomerID`, `WarehouseToHome`, `NumberOfAddress`
- Missing values handled via mean imputation (numerical) and mode imputation (categorical)
- Categorical variables label-encoded for compatibility with LightGBM

### 2. Exploratory Data Analysis
- Correlation analysis of all features against `Churn`
- Churn rate breakdown by login device, city tier, gender, and marital status
- Distribution comparisons (app usage hours, cashback amount, satisfaction score) between churned and retained customers

**Key findings:**
- `Tenure` shows the strongest negative correlation with churn (-0.34) — newer customers churn more
- `Complain` shows the strongest positive correlation (+0.25) — customers who raised complaints are significantly more likely to churn
- `MaritalStatus` and `SatisfactionScore` also show moderate correlation with churn

### 3. Modeling
- Train/Validation/Test split: 60% / 20% / 20% (stratified on `Churn`)
- Baseline model: `LGBMClassifier` with default parameters
- Hyperparameter tuning: `RandomizedSearchCV` across 7 parameters (`num_leaves`, `learning_rate`, `boosting_type`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`) — 30 candidate combinations × 5-fold CV = 150 model fits
- Best parameters selected based on F1-score

### 4. Evaluation

| Metric | Baseline (Val) | Tuned (Val) | Tuned (Test) |
|---|---|---|---|
| F1-score | 0.747 | 0.780 | 0.853 |
| ROC-AUC | — | 0.939 | 0.966 |
| Accuracy | — | 0.93 | 0.95 |

Test set confusion matrix:
[[908  29]
[ 27 162]]



Correctly identified 162 of 189 actual churners (86% recall) with only 27 false positives.

**Note:** Test performance was slightly higher than validation performance. Given the relatively small number of churn cases (~190 per split), this gap is most likely attributable to sample variance rather than data leakage — verified via correlation and feature importance checks, with no single feature showing implausibly high correlation with the target.

### 5. Feature Importance

Top predictors identified by the tuned model align with the correlation analysis, with `Tenure`, `Complain`, and `CashbackAmount` ranking among the most influential features — consistent, interpretable signals rather than artifacts of the encoding process.

## Business Implications

- **Tenure-based targeting:** Newer customers are highest-risk; onboarding and early-engagement retention offers may reduce churn more than broad, later-stage interventions.
- **Complaint resolution as a retention lever:** Complaint history is one of the strongest predictors — timely, effective complaint resolution could materially reduce churn.
- **Model applicability:** With 86% recall on churners at the current threshold, the model can support targeted retention campaigns by flagging at-risk customers before they leave.

## Tech Stack

- Python, pandas, NumPy
- LightGBM (`LGBMClassifier`)
- scikit-learn (`RandomizedSearchCV`, preprocessing, metrics)
- Matplotlib, Seaborn (visualization)

