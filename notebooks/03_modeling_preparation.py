# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Modeling Preparation and Strategy
# MAGIC ### 3-1. Problem Definition & Evaluation Metrics
# MAGIC The goal is to identify transactions (Class = 1) from legitimate ones (Class = 0). Given the dataset's severe class imbalance, accuracy alone is insufficient. Therefore, evaluation focuses on metrics such as precision, recall, F1-score, and ROC-AUC to ensure that fraudulent transactions are correctly detected.
# MAGIC
# MAGIC ### 3-2. Data Preparation
# MAGIC 3-2-1. Handling imbalance
# MAGIC
# MAGIC The dataset exhibits extreme class imbalance, with fraudulent transactions representing only 0.17% of all records. To prevent the model from being biased toward the majority class, we applied oversampling of the minority class (SMOTE) and/or used class-weight adjustments in tree-based models. This ensures that the classifier gives sufficient attention to the rare fraudulent cases while training.

# COMMAND ----------

X = df[top_features.index]
y = df['Class']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC 3-2-2. Feature Selection & Scaling
# MAGIC
# MAGIC Based on the EDA, we selected the top features that show the strongest correlation with the target variable (Class). These features capture meaningful distributional differences between fraudulent and legitimate transactions.
# MAGIC For linear models, features were standardized to have zero mean and unit variance, whereas tree-based models were trained on raw features, as they are insensitive to feature scaling.

# COMMAND ----------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-3. Model Selection (Random Forest vs. Logistic Regression vs. XGBoost)
# MAGIC
# MAGIC We evaluated several classification algorithms suitable for imbalanced datasets.
# MAGIC
# MAGIC - Random Forest and XGBoost were chosen as tree-based models capable of capturing non-linear relationships and complex interactions.
# MAGIC - Logistic Regression was included as a baseline linear model for comparison.
# MAGIC
# MAGIC Class weights were adjusted to balance the contribution of minority class samples, which improves the model’s ability to detect fraudulent transactions.

# COMMAND ----------

# Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train_res, y_train_res)


# COMMAND ----------

# XGBoost

# Calculate the proportion between Class 0 and Class 1
count_0 = (y_train == 0).sum()
count_1 = (y_train == 1).sum()

scale_pos_weight = count_0 / count_1
print("scale_pos_weight =", scale_pos_weight)

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train_res, y_train_res)

# COMMAND ----------

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)