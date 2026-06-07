# Detecting Financial Fraud at Scale: Gradient Boosting and Ensemble Methods on the IEEE-CIS Transaction Dataset

**DSC 148 — Introduction to Data Mining | UCSD Spring 2026**

## Overview

This project applies machine learning to the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) provided by Vesta Corporation. The task is binary classification: predict whether a credit card transaction is fraudulent (`isFraud = 1`) based on transaction and identity features. The dataset contains 590,540 transactions with 434 features after merging the transaction and identity tables.

The fraud rate is approximately 3.5%, making this a heavily imbalanced classification problem. Evaluation focuses on **AUC-ROC** and **F1-score on the fraud class** — raw accuracy is misleading at this imbalance ratio.

## Dataset

| Table | Rows | Columns |
|---|---|---|
| `train_transaction.csv` | 590,540 | 394 |
| `train_identity.csv` | 144,233 | 41 |
| Merged | 590,540 | 434 |


Data files are not included in this repository. Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place in `data/`.

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis — class imbalance, missing values, temporal patterns, adversarial validation |
| `02_baseline_lr.ipynb` | Baseline 1: Logistic Regression |
| `03_baseline_nb.ipynb` | Baseline 2: Gaussian Naive Bayes |
| `04_lgbm.ipynb` | Proposed model: LightGBM |
| `05_xgb.ipynb` | XGBoost |
| `06_catboost.ipynb` | CatBoost with native categorical encoding |
| `07_rf.ipynb` | Random Forest |
| `08_ensemble.ipynb` | Ensemble: AUC-weighted + Nelder-Mead optimized blending |

Run notebooks in order (02 through 07 before 08). Each model notebook saves results to `results/`.


## Project Structure

```
├── data/                    # Raw CSVs (not tracked)
├── results/                 # Saved model outputs (not tracked)
├── 01_eda.ipynb
├── 02_baseline_lr.ipynb
├── 03_baseline_nb.ipynb
├── 04_lgbm.ipynb
├── 05_xgb.ipynb
├── 06_catboost.ipynb
├── 07_rf.ipynb
└── 08_ensemble.ipynb
```