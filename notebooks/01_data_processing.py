# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Data Processing 
# MAGIC - Environment setup and libraries
# MAGIC - Data loading 
# MAGIC - Data preview and schema inspection
# MAGIC - Data formatting and standardization
# MAGIC
# MAGIC In addition, understand the dataset structure:
# MAGIC - Class column: 0 = legitimate, 1 = fraud
# MAGIC - Features V1-V28 are PCA-transformed (anonymized)
# MAGIC - Amount and Time are raw features

# COMMAND ----------

# Importing pandas and numpy for data analysis and manipulation 
import pandas as pd
import numpy as np

# Importing matplotlib for creating visualizations
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import seaborn for statistical data visualization
import seaborn as sns

import umap

import shap 

# Handling the class imbalance
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Model selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# XGBoost
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import classification_report, roc_auc_score

# To suppress warnings
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# Read dataset from Databricks Catalog
spark_df = spark.read.table("workspace.default.creditcard")
df = spark_df.toPandas()

# Preview the dataset and check any missing values
print("Raw data:")
display(df.head())
print("Missing data:")
print(df.isnull().sum().sum())
print("\n")

# Understnad the dataset structure
print("Data Info: ")
display(df.info())
print("\n")
print("Data Description: ")
print(df.describe())
print("\n")
df['Class'].value_counts()