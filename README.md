# Credit Risk Assessment

A machine learning system for predicting credit default risk,
built on the "Give Me Some Credit" dataset from Kaggle.

## Problem

Financial institutions face a critical challenge: determining
whether a loan applicant will default within the next two years,
defined as falling 90 or more days past due on any debt obligation.
Approving a high-risk applicant leads to direct financial loss,
while rejecting a low-risk applicant means lost revenue and
potential reputational damage. The cost of these two errors
is asymmetric — which makes this problem particularly interesting
from a machine learning perspective.

## Why This Problem Matters

According to the World Bank, access to credit is one of the
strongest predictors of economic mobility. A well-calibrated
credit risk model does not just protect the bank — it also
ensures that creditworthy individuals are not unfairly rejected.
This dual objective (protecting the lender, serving the borrower)
shapes every modeling decision in this project.

## Existing Approaches & Their Trade-offs

**Traditional Scorecard (WoE/IV)**
The industry standard for decades. Highly interpretable and
regulatorily compliant, but requires manual binning of features,
assumes linear relationships, and often underperforms on
complex, non-linear data patterns.

**Logistic Regression**
The statistical baseline. Fast, interpretable, and well-understood
mathematically. Performs surprisingly well in credit risk contexts,
but struggles when feature interactions are important.

**Ensemble Methods (Random Forest, XGBoost)**
State-of-the-art predictive performance. Captures non-linear
relationships and feature interactions automatically. The trade-off
is reduced interpretability — which this project addresses
explicitly using SHAP values.

## Our Approach

This project compares four models (Logistic Regression, Decision
Tree, Random Forest, XGBoost) on a common evaluation framework,
selects the best performer, and wraps it in an interactive
application that provides not just a prediction, but an
explanation — making the system suitable for real-world use
where regulatory interpretability is required.

## Primary Metric: ROC-AUC

Given the severe class imbalance in the dataset (~6.7% defaults),
accuracy is a misleading metric. ROC-AUC measures the model's
ability to discriminate between defaulters and non-defaulters
regardless of threshold, making it the appropriate primary metric.
The KS statistic serves as a secondary metric, standard in the
banking industry for model validation.

## Project Structure

- `notebooks/` — Jupyter notebooks (EDA → Preprocessing → Modeling → Evaluation → Recommendation)
- `src/utils.py` — shared helper functions
- `models/` — saved trained models
- `app.py` — interactive CLI for new customer assessment
- `data/` — dataset (not tracked in git)

## Models Compared

Logistic Regression · Decision Tree · Random Forest · XGBoost

## Results

*(to be filled after modeling)*

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```