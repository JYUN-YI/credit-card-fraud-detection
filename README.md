# 💳 Credit Card Fraud Detection

## 📌 Project Objective
This project aims to build an end-to-end **credit card fraud detection system** that identifies fraudulent transactions from highly imbalanced data.  
The focus is not only on model performance, but also on **interpretability, reproducibility, and real-world usability** through an interactive web application.

Key objectives:
- Handle extreme class imbalance with 492 frauds out of 284,807 records
- Explore transaction behavior patterns
- Train and evaluate machine learning models
- Interpret model decisions using feature importance techniques
- Deploy an interactive Streamlit application

## 🏗️ Project Structure
```bash
📦credit-card-fraud-detection/
│
├── data/
│ └── creditcard.csv
│
├── models/
│ ├── fraud_model.pkl
│ ├── X_test.csv
│ └── y_test.csv
│
├── notebooks/
│ ├── 01_data_processing.py
│ ├── 02_EDA_visualization.py
│ ├── 03_modeling_preparation.py
│ ├── 04_model_training.py
│ ├── 05_model_explainability.py
│ └── Credit-Card-Fraud-Detection_Full.py
│
├── pages/
│ ├── 1_dataset_overview.py
│ ├── 2_model_performance.py
│ └── 3_transaction_simulator.py
│
├── utils/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_inference.py
│ └── shap_explain.py
│
├── streamlit_app.py
├── create_model.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔍 Exploratory Data Analysis (EDA)
Key findings from the EDA include:
- The dataset is **extremely imbalanced** (fraud transactions < 0.2%)
- Fraudulent transactions tend to:
  - Have **lower median transaction amounts**
  - Appear as **small transactions** or **extreme outliers**
- Time-based patterns reveal non-uniform fraud activity distribution

Visualizations include:
- Class imbalance analysis
- Amount distribution (log-scale)
- Time-Based transaction patterns
- Feature distribution analysis

## 🤖 Machine Learning Models
The project experiments with supervised learning models for binary classification:
- Logistic Regression (baseline)
- Tree-based models (e.g., Random Forest, XGBoost)
- Threshold tuning for business-oriented decision making

Evaluation metrics:
- ROC-AUC
- Precision / Recall
- Confusion Matrix
- Threshold-sensitive performance analysis

Special attention is given to **Recall and Precision trade-offs**, which are critical in fraud detection.

## 🔑 Feature Importance Analysis
To improve model transparency:
- Feature importance from tree-based models
- SHAP (SHapley Additive exPlanations) values
- Global and local interpretability analysis

This helps explain **why a transaction is classified as fraudulent**, which is crucial for real-world financial systems.

## 🚀 Interactive Application
An interactive **Streamlit web application** is deployed to demonstrate:
- Dataset overview and statistics
- Model performance visualization (ROC, confusion matrix)
- Adjustable decision threshold
- Transaction simulation for real-time fraud prediction

This transforms the project from a notebook-based analysis into a **deployable ML product**.

## ⚙️ Technologies Used
- Programming: Python
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Logistic Regression, Random Forest, XGBoost
- Model Interpretability: SHAP
- Deployment: Streamlit, Docker, Hugging Face Spaces
- Version Control: Git, GitHub
- Development Environment: Databricks Free Edition

## 📎 Dataset
**Kaggle Credit Card Fraud Detection Dataset**  
> European cardholders transactions (anonymized features)

- Total transactions: ~284,000  
- Fraud cases: ~492  
- Highly imbalanced binary classification problem

## 🎨 Portfolio Showcase
Please view the report and interact with the live application here:

- [Kimberly Lin | Portfolio | 2013 EU Credit Card Fraud Detection](https://kimberlylin.webflow.io/resources/2013-eu-credit-card-fraud-detection)
- [Streamlit App on Hugging Face Spaces](https://huggingface.co/spaces/jyunyilin/credit-card-fraud-detection)

## ✨ Future Improvements
- Experiment with advanced imbalance techniques (SMOTE, focal loss)
- Add cross-validation and hyperparameter tuning
- Incorporate cost-sensitive learning
- Add model monitoring and drift detection
- Expand deployment with CI/CD pipeline
