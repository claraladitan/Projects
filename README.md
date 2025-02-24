**Stroke Risk Prediction Project**

This project uses machine learning to predict stroke risk based on patient symptoms and clinical indicators. The goal is to build models that can accurately tell whether someone is at risk of stroke, which can help medical professionals and patients make better-informed decisions.

**Project Details**

- [Kaggle Notebook](https://www.kaggle.com/code/claraladitan/stroke-risk-prediction-model-comparison) – View the interactive version of this project on Kaggle.
- [Kaggle Dataset](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset/data) – Access the stroke dataset used in this project.


**Project Overview**

In this project, I explored a stroke dataset that contains various numeric predictors—mostly binary symptom indicators—and a binary target variable (at_risk_binary) that tells whether a patient is at risk of stroke. I focused on predicting this binary outcome using different machine learning techniques.

**Process**

1. Data Exploration & Preparation

- Understanding the Data: I started by exploring the dataset and examining the features to understand their distributions and relationships with the target.
- Cleaning & Formatting: I converted the target variable to a factor and ensured that all predictors were correctly formatted. I also made sure that no feature was directly “leaking” information about the target.

2. Model Building
I built several models to compare performance

- Logistic Regression (GLM): A basic model that gives a probability of being at risk.
- Decision Tree: A simple, interpretable model that splits data based on feature values.
- Regularized Logistic Regression (GLMNET): This model uses a mix of LASSO and Ridge regularization to handle complex data and avoid overfitting.
- XGBoost: An ensemble method that combines many decision trees to capture complex patterns in the data.
  
3. Evaluation & Cross-Validation

- I split the data into training and test sets and used cross-validation to ensure that the results are robust and not just due to a lucky split.
- I generated confusion matrices, ROC curves, and computed metrics like accuracy, sensitivity, specificity, and AUC for each model.
  
**Results**

- The Decision Tree model achieved around 81% accuracy, with moderate sensitivity and specificity.
- The GLMNET and XGBoost models showed near-perfect scores in cross-validation and on the test set, suggesting they captured the complex relationships in the data very well.
- While near-perfect performance is rare in real-world data, I double-checked the splits, correlations, and cross-validation procedures to rule out data leakage. The results seem to reflect the strength of these advanced models.

**Insights & Real-World Application**

- More advanced models like GLMNET and XGBoost can capture subtle interactions in the data that simpler models might miss.
- A single decision tree, while easy to interpret, may underfit the data if the underlying relationships are complex.
- It is crucial to validate that the models are not inadvertently “cheating” by using leaked data.

**Real-World Impact:**

- In a healthcare setting, accurate stroke risk prediction can help doctors identify high-risk patients early and tailor preventive measures accordingly.
- With further validation and clinical trials, such predictive models could be integrated into decision support systems in hospitals or health apps, contributing to improved patient outcomes and efficient resource allocation.
