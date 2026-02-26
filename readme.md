# CreditGuard: Credit Risk Assessment Pipeline

> CreditGuard is just a name to make this project name's short :p

This project was inspired by my intern experience at a local bank company in a risk management division. There I learned different types of risk in banking, including credit risk, which drove me to make this project.

## Project Description

CreditGuard is a machine learning pipeline engineered to predict loan defaults (Charge-Offs) in a highly imbalanced credit dataset. The project focuses on building a robust, production-ready architecture that rigorously separates structural data cleaning from statistical feature engineering to prevent data leakage.

A core emphasis of CreditGuard is Explainable AI (XAI). Financial risk models cannot be black boxes; therefore, this pipeline utilizes Permutation Feature Importance for statistical validation on unseen data, and SHAP (SHapley Additive exPlanations) to provide directional, domain-aligned interpretability for regulatory and business alignment.

## Dependencies

The environment relies on the following core libraries:

* **Core Data & Computation:** pandas, numpy
* **Machine Learning & Pipelines:** scikit-learn, xgboost, lightgbm
* **Experiment Tracking:** mlflow
* **Interpretability:** shap
* **Visualization:** matplotlib, seaborn

## Data Source

The dataset is sourced from the [Kaggle Lending Club Loan Data repository](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv)

The dataset consists of historical loan records containing borrower financial history, credit line utilization, and loan characteristics (e.g., `annual_inc`, `dti`, `open_acc`, `sub_grade`, `int_rate`). The target variable is `loan_status`, which presents an approximately 80/20 class imbalance between **"Fully Paid"** (majority) and **"Charged Off"** (minority) loans.

---

## Experiments Tracking

All model training, parameter logging, and artifact generation (such as feature importance plots) are strictly version-controlled using MLflow.

| Exp. | Data Size | Modeling Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Key Metrics (Focus: Recall & PR-AUC) |
|---|---|---|---|---|---|---|---|---|
| 01 | (2436,36) | Random Forest (Baseline Exploration) | 0.992 | 0.000 | 0.000 | 0.000 | 0.548 | 0 Recall (Majority Class Collapse) |
| 02a | (4994,28) | Random Forest (Bagging) | 0.745 | 0.378 | 0.420 | 0.398 | 0.681 | Highest Precision & ROC-AUC, Balanced |
| 02b | (4994,28) | XGBoost (Boosting) | 0.734 | 0.358 | 0.410 | 0.382 | 0.676 | Moderate Balance |
| 02c | (4994,28) | LightGBM (Boosting) | 0.713 | 0.341 | 0.465 | 0.394 | 0.663 | Highest Recall, Aggressive Risk Flagging |
---

## Architecture & Workflow Principles



1. **Strict Phase Separation & Zero-Leakage Pipeline:** Structural data wrangling (handling raw text, basic standardizations) is isolated from statistical Feature Engineering. A robust `scikit-learn` pipeline utilizes `ColumnTransformer` to enforce a strict Train/Test vault. Operations requiring group statistics (Median Imputation, Target Encoding, dynamic One-Hot Encoding) are strictly fitted on the training set and subsequently applied to the test set, mathematically preventing time-travel data leakage.
2. **Imbalance Handling:** Algorithmic penalties are prioritized over synthetic data generation in initial experiments, utilizing `scale_pos_weight` and explicit `class_weight` dictionaries to mathematically force the models to hunt for minority class signals and prioritize Recall.
3. **Dual Interpretability Validation:**
   - **Permutation Importance:** Used to validate which features genuinely drive generalizable accuracy on the holdout test set, identifying overfitted features (e.g., negative importance scores).
   - **SHAP Values:** Used to map the directional impact of features (e.g., higher interest rates equating to higher risk) to ensure the mathematical model logic aligns with real-world credit underwriting principles.