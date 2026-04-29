# Credit_Card_Fraud_Detection_Model
Credit Card Fraud Detection using Machine Learning with EDA, feature engineering, and models like Logistic Regression, Random Forest, and XGBoost to identify fraudulent transactions.
# 💳 Credit Card Fraud Detection

## 🚀 Overview
This project focuses on building a machine learning model to detect fraudulent credit card transactions using real-world data. The dataset is highly imbalanced, making fraud detection a challenging classification problem.

---

## 📊 Problem Statement
Predict whether a transaction is:
- **0 → Legitimate**
- **1 → Fraudulent**

based on transaction details such as amount, location, time, and customer behavior.

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Removed duplicates  
- Handled missing values  
- Dropped irrelevant columns  

### 2. Feature Engineering
- Transaction time features (hour, day, month)  
- Night transaction indicator  
- Customer age  
- Distance between customer and merchant  
- Log transformation of transaction amount  
- Customer behavior features:
  - Average transaction amount  
  - Transaction deviation  
  - Transaction frequency  

### 3. Exploratory Data Analysis (EDA)
- Identified class imbalance  
- Analyzed feature distributions  

### 4. Model Building
- Logistic Regression (baseline)  
- Random Forest  
- XGBoost  

### 5. Hyperparameter Tuning
- Used RandomizedSearchCV  
- Optimized for better F1-score  

### 6. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|---------|----------|--------|----------|---------|
| Random Forest | 0.9727 | 0.1161 | 0.9161 | 0.2061 | 0.9893 |
| XGBoost ⭐ | 0.9832 | 0.1834 | 0.9692 | 0.3084 | 0.9978 |

---

## 🧠 Key Insights
- Accuracy alone is misleading due to class imbalance  
- Recall is prioritized to avoid missing fraud cases  
- XGBoost provides better balance between precision and recall  
- Feature engineering significantly improves performance  

---

## 🎯 Final Model
**XGBoost** is selected as the final model because:
- Higher recall (better fraud detection)  
- Improved F1-score  
- Strong ROC-AUC  
High ROC-AUC (~0.997) shows the model is extremely powerful.
Low precision is due to class imbalance, not poor model quality.
---

## 📦 How to Run

```bash
pip install -r requirements.txt
