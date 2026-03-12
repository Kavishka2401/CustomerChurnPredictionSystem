📊 Telco Customer Churn Prediction System
A Machine Learning project using Neural Networks and Decision Trees

Overview
This project builds a Churn Prediction System using the Telco Customer Churn Dataset (Kaggle).
The goal is to predict whether a customer will churn, enabling telecom companies to take proactive retention actions.
Two ML models were developed:
  1. Neural Network (NN) — tuned manually with class weights
  2. Decision Tree (DT) — tuned using GridSearchCV + SMOTE oversampling

This repository includes data preprocessing, model building, evaluation metrics, visualizations, and ethical considerations.

Project Structure

├── EDA                   
├── Preprocessing for Neural Networks             
├── Neural Network Baseline model                 
├── Comparative Analysis of Imbalance Handling             
├── Final Neural Network Model   
├── Final Decision Tree Model (includes preprocessing)  
└── README.md     

Dataset
Telco Customer Churn Dataset
Source: Kaggle
URL: https://www.kaggle.com/blastchar/telco-customer-churn

Data Preprocessing
1. Handling Missing & Categorical Values
  Replaced " " and invalid values
  Applied One-Hot Encoding
  Removed irrelevant identifier: customerID
2. Imbalance Handling
  Neural Network: Class weights
  Decision Tree: SMOTE oversampling
3. Feature Selection
  Chi-Square test used to select top 12 features (NN)
  DT trained on both full and reduced feature sets

Models Used
1. Neural Network (Keras Sequential Model)
  Architecture: 32 → 16 → 1
  Activation: ReLU + Sigmoid
  Optimizer: Adam
  Loss: Binary Crossentropy
  Regularization: Dropout
  Metrics: F1, Recall, Precision, Accuracy, ROC-AUC
  Class imbalance handled using class weights
2. Decision Tree Classifier
  Hyperparameter tuning using GridSearchCV
  Best params:
    ccp_alpha = 0.001
    criterion = 'entropy'
    max_depth = 10
    min_samples_split = 10
    min_samples_leaf = 20
  Feature importance visualized
  Tree structure plotted
  SMOTE applied for training

Model Evaluation
Both models were evaluated using:
  Accuracy
  Precision
  Recall
  F1 Score
  ROC-AUC
  Confusion Matrix
  Precision-Recall Curve
  Cross-Validation (DT)

Key Findings
1. Both models achieved moderate performance due to the dataset’s imbalance.
2. NN showed good recall but limited interpretability.
3. DT improved after feature reduction.
4. Both models had high false positives, predicting many non-churners as churners

System Limitations
1. Moderate model performance due to data size & imbalance
2. Grid search unavailable for NN due to Keras wrapper limitations
3. NN lacks interpretability (no feature importance)
4. High false positive rate in both models

Future Enhancements
1. Use ensemble models (XGBoost, Random Forest, LightGBM)
2. Apply advanced imbalance techniques (SMOTE-Tomek, ADASYN)
3. Integrate SHAP/LIME for NN explainability
4. Deploy the model as a REST API
5. Expand dataset using real customer activity logs
