# 📉 Telco Customer Churn Prediction  
### Demo of a Supervised Learning System Application (Decision Tree)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Model-Decision%20Tree-orange)
![Dataset](https://img.shields.io/badge/Dataset-Telco%20Customer%20Churn-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

# 🚀 Overview

This project demonstrates a supervised learning application that predicts whether a telecommunications customer is likely to churn. The model is trained using the **Telco Customer Churn dataset** and applies a **Decision Tree classifier** to learn patterns associated with customer behavior and service usage.

The goal of this demo is to show how labeled historical data can be used to build predictive models that identify customers at risk of leaving a service provider. Such insights can help organizations take proactive steps to improve customer retention and reduce revenue loss.

---

# 🏗 System Architecture

```
              +-----------------------------+
              | Telco Customer Dataset     |
              | telco-customer-churn.csv   |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              | Data Preprocessing          |
              | - Data Cleaning             |
              | - Feature Selection         |
              | - Encoding Categorical Data |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              | Train-Test Split (80/20)    |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              | Decision Tree Classifier    |
              | Model Training              |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              | Model Predictions           |
              | Churn Probability           |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              | Model Evaluation            |
              | Confusion Matrix & Metrics  |
              +-----------------------------+
```

---

# 📂 Dataset

Dataset: **Telco Customer Churn Dataset**

Source:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains **7,043 customer records** with demographic information, service subscriptions, billing data, and a labeled churn indicator.

Target variable:

```
Churn
0 → Customer stayed
1 → Customer churned
```

Example features include:

- Gender  
- SeniorCitizen  
- Partner / Dependents  
- Tenure  
- InternetService  
- Contract type  
- MonthlyCharges  
- TotalCharges  
- PaymentMethod  

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- jupyter  

---

# ▶ How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook:

```
telco-churn-prediction.ipynb
```

Run all cells to:

- load the dataset  
- train the model  
- evaluate predictions  
- analyze churn drivers  

---

# 🧠 Model Details

The project uses a **Decision Tree Classifier** implemented with **scikit-learn**.

```python
from sklearn.tree import DecisionTreeClassifier
```

Decision Trees classify data by learning a sequence of decision rules that split the dataset based on feature values.

Example rule:

```
IF Contract = Month-to-month
   AND MonthlyCharges > threshold
   → Higher churn probability
```

---

# 📊 Model Evaluation

The model is evaluated using an **80/20 train-test split**.

Metrics used include:

- Accuracy  
- Balanced Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC

Example evaluation outputs:

```
Accuracy ≈ 0.79
Balanced Accuracy ≈ 0.74
Precision ≈ 0.61
Recall ≈ 0.55
F1-score ≈ 0.58
ROC-AUC ≈ 0.76
```

These metrics help assess how well the model distinguishes between churn and non-churn customers.

---

# 🔍 Feature Importance

Feature importance analysis reveals which customer attributes most strongly influence churn predictions.

Key drivers identified by the model include:

- Contract type (Month-to-month)  
- TotalCharges  
- InternetService (Fiber optic)  
- Tenure  
- MonthlyCharges  
- TechSupport availability  
- PaymentMethod (Electronic check)  

These features help explain why certain customer profiles are more likely to churn.

---

# 🧪 Example Prediction

Example customer input:

```json
{
  "tenure": 3,
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "MonthlyCharges": 94.85,
  "PaymentMethod": "Electronic check"
}
```

Prediction output:

```
Predicted Churn: Yes
Probability: 0.823
```

The model identifies this profile as high risk due to short tenure, high charges, and a month-to-month contract.

---

# 📜 License

This project is open source and available under the [MIT License](LICENSE).
