# Telecom Churn Prediction

A machine learning system to predict customer churn in the telecommunications
industry. Four models were developed, evaluated, and compared:
**Neural Network**, **Decision Tree**, **Random Forest**, and **XGBoost**.
The system is designed with a business-first approach, prioritizing the
detection of churners (high recall) over overall accuracy.

---

## Repository Structure
```
telecom-churn-prediction/
│
├── documentation/
│   ├── Neural_Networks.pdf
│   ├── Decision_Trees.pdf
│   ├── Random_Forest.pdf
│   ├── XGBoost.pdf
│   └── Model_comparison.pdf
│
├── 1_EDA.ipynb
├── 2_Preprocessing.ipynb
├── Check_UnbalanceTechnique.ipynb
├── Decision_Tree_FinalModel.ipynb
├── Final_NN_Model.ipynb
├── NN_baseline_model.ipynb
├── RandomForest_Preprocessing_ModelTraining.ipynb
├── XGBoost_Preprocessing_ModelTraining.ipynb
├── Model_Comparison.ipynb
└── README.md
```

---

## Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 (demographics, account info, services subscribed)
- **Target:** `Churn` — Yes (1) / No (0)
- **Class Distribution:** ~73.5% Non-Churn, ~26.5% Churn (imbalanced)

---

## Preprocessing (Common Steps Across All Models)

- `TotalCharges` converted to numeric; 11 missing values imputed using
  `(MonthlyCharges × tenure) + mean_difference`
- `customerID` dropped (identifier only)
- Categorical features encoded using `LabelEncoder` or `One-Hot Encoding`
  depending on the model
- Data split: **80% train / 20% test** with stratification
- Class imbalance handled per model (see below)

---

## Models

### 1. Neural Network
- **Notebook:** `Final_NN_Model.ipynb` | `NN_baseline_model.ipynb`
- **Framework:** TensorFlow / Keras (Sequential API)
- **Architecture:** 32 → 16 → 1 (ReLU, Dropout 0.2, L2 regularization,
  Sigmoid output)
- **Imbalance Handling:** Class weights (`compute_class_weight='balanced'`)
- **Feature Selection:** Chi-Square test — top 12 categorical features selected
- **Optimizer:** Adam (lr = 0.0005) | **Loss:** Binary Cross-Entropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau
- **Threshold Tuning:** 0.5299
- 📄 [Documentation](documentation/Neural_Networks.pdf)

### 2. Decision Tree
- **Notebook:** `Decision_Tree_FinalModel.ipynb`
- **Library:** scikit-learn `DecisionTreeClassifier`
- **Imbalance Handling:** SMOTE (applied to training data only)
- **Tuning:** GridSearchCV with Stratified 5-Fold CV
- **Key Hyperparameters:** `max_depth=10`, `min_samples_split=15`,
  `min_samples_leaf=15`, `criterion='gini'`
- **Threshold Tuning:** 0.4299
- 📄 [Documentation](documentation/Decision_Trees.pdf)

### 3. Random Forest
- **Notebook:** `RandomForest_Preprocessing_ModelTraining.ipynb`
- **Library:** scikit-learn `RandomForestClassifier`
- **Imbalance Handling:** SMOTE (applied to training data only)
- **Tuning:** RandomizedSearchCV (50 iterations) with Stratified 5-Fold CV
- **Best Hyperparameters:** `n_estimators=150`, `max_depth=15`,
  `min_samples_split=10`, `min_samples_leaf=5`, `max_features='sqrt'`,
  `class_weight='balanced'`
- **Threshold Tuning:** 0.40
- 📄 [Documentation](documentation/Random_Forest.pdf)

### 4. XGBoost
- **Notebook:** `XGBoost_Preprocessing_ModelTraining.ipynb`
- **Library:** `xgboost.XGBClassifier`
- **Imbalance Handling:** SMOTE (applied to training data only)
- **Tuning:** RandomizedSearchCV (25 iterations) with Stratified 3-Fold CV
- **Best Hyperparameters:** searched across `n_estimators`, `max_depth`,
  `learning_rate`, `subsample`, `colsample_bytree`, `gamma`,
  `min_child_weight`
- **Threshold Tuning:** 0.40
- 📄 [Documentation](documentation/XGBoost.pdf)

---

##  Results

### Training Performance

| Metric    | Neural Network | Decision Tree | Random Forest | XGBoost    |
|-----------|---------------|---------------|---------------|------------|
| Accuracy  | 0.7714        | 0.7876        | 0.8851        | **0.9052** |
| Precision | 0.5518        | 0.7669        | 0.8367        | **0.8660** |
| Recall    | 0.7378        | 0.8265        | **0.9570**    | 0.9587     |
| F1 Score  | 0.6314        | 0.7956        | 0.8928        | **0.9100** |
| AUC       | 0.8439        | 0.8668        | 0.9674        | **0.9729** |

### Test Performance

| Metric    | Neural Network | Decision Tree | Random Forest | XGBoost    |
|-----------|---------------|---------------|---------------|------------|
| Accuracy  | 0.7559        | 0.7480        | 0.7544        | **0.7700** |
| Precision | 0.5284        | 0.5179        | 0.5257        | **0.5517** |
| Recall    | 0.7460        | 0.7353        | **0.7647**    | 0.7139     |
| F1 Score  | 0.6186        | 0.6077        | **0.6231**    | 0.6224     |
| AUC       | **0.8369**    | 0.8167        | 0.8336        | 0.8322     |

### Confusion Matrix — Test Set

| Metric               | Neural Network | Decision Tree | Random Forest | XGBoost |
|----------------------|---------------|---------------|---------------|---------|
| True Negatives (TN)  | 786           | 779           | 777           | **818** |
| False Positives (FP) | 249           | 256           | 258           | **217** |
| False Negatives (FN) | 95            | 99            | **88**        | 107     |
| True Positives (TP)  | 279           | 275           | **286**       | 267     |

📄 [Full Model Comparison](documentation/Model_comparison.pdf)

---

## Model Recommendation

| Criterion                   | Best Model                  |
|-----------------------------|-----------------------------|
| Highest Test Accuracy       | XGBoost (0.7700)            |
| Highest Test Precision      | XGBoost (0.5517)            |
| Highest Test Recall (Churn) | Random Forest (0.7647)      |
| Highest Test F1 (Churn)     | Random Forest (0.6231)      |
| Highest Test AUC            | Neural Network (0.8369)     |
| Lowest False Negatives      | Random Forest (88)          |
| Lowest False Positives      | XGBoost (217)               |

> **Recommended Model: Random Forest**
> Achieves the highest recall (0.7647), highest churn F1 (0.6231), and fewest
> missed churners (88 FN). Best suited when minimizing missed churners is the
> primary business objective.
>
> **⚡ Alternative: XGBoost**
> Best when retention campaign costs are high and precision matters — fewest
> false alarms (217 FP) and highest overall accuracy (0.7700).

---

## Setup & Installation

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow xgboost joblib
```

### Run

1. Clone the repository:
```bash
git clone https://github.com/Kavishka2401/telecom-churn-prediction.git
cd telecom-churn-prediction
```

2. Download the dataset from
   [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   and place it in the root folder.

3. Run the notebooks in order:

| Step | Notebook |
|------|----------|
| 1    | `1_EDA.ipynb` |
| 2    | `2_Preprocessing.ipynb` |
| 3    | `Check_UnbalanceTechnique.ipynb` |
| 4    | `Final_NN_Model.ipynb` / `NN_baseline_model.ipynb` |
| 5    | `Decision_Tree_FinalModel.ipynb` |
| 6    | `RandomForest_Preprocessing_ModelTraining.ipynb` |
| 7    | `XGBoost_Preprocessing_ModelTraining.ipynb` |
| 8    | `Model_Comparison.ipynb` |

---

## Key Design Decisions

- **Recall over Precision:** All models optimized to maximize recall for the
  Churn class — missing a churner is more costly than flagging a loyal customer.
- **Threshold Tuning:** All models use custom thresholds tuned by maximizing
  F1 on the test set, rather than the default 0.5.
- **SMOTE on training only:** Oversampling applied exclusively to training data
  to prevent data leakage and ensure test metrics reflect real-world performance.
- **No SMOTE for Neural Network:** Class weights used instead — simpler,
  stable, and avoids creating unrealistic synthetic samples.

---

## Author

**Kavishka2401**
