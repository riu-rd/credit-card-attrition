---
title: Credit Card Attrition
emoji: 💳
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: addresses customer credit card churn in a bank
license: mit
---

# Credit Card Attrition Analysis

## Overview
This project addresses customer churn in a bank's credit card division by building predictive models to identify customers likely to close their accounts. The solution helps design retention strategies and improve customer satisfaction.

## Project Components

### 📊 Data Analysis & Preprocessing
- Comprehensive exploratory data analysis of customer information, transaction history, and demographic details
- Handling of missing values, outliers, and duplicate records
- Management of class imbalance in the target variable (AttritionFlag)
- Feature engineering to create meaningful predictors

### 🤖 Machine Learning Models
- Implementation of multiple classification algorithms:
  - LightGBM
  - XGBoost
- Hyperparameter tuning and cross-validation

### 📈 Model Evaluation
- Comprehensive evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC curves
  - Confusion matrices
- Feature importance analysis to identify key attrition drivers

### 🖥️ Interactive Dashboard
- Streamlit application for visualizing attrition risk
- Real-time predictions and insights
- User-friendly interface for business stakeholders

## Files
- `src/credit_card_attrition.ipynb`: Complete analysis and model development
- `src/app.py`: Streamlit dashboard application
- `src/datasets/`: Dataset files
- `src/models/`: Saved model artifacts

## Key Insights
The analysis identifies critical factors driving customer attrition and provides actionable recommendations for the bank's retention strategies.

## Usage
Run the Streamlit app:
```bash
streamlit run src/app.py
```