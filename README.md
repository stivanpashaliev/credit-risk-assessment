# Credit Risk Assessment

A machine learning system for predicting credit default risk,
built on the "Give Me Some Credit" dataset from Kaggle.

## Problem

Banks need to assess whether a loan applicant will default
within the next 2 years (90+ days past due). This project
builds and evaluates multiple ML models to support that decision,
and provides a CLI application for assessing new customers.

## Project Structure

- `notebooks/` — Jupyter notebooks (EDA → Preprocessing → Modeling → Evaluation → App)
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