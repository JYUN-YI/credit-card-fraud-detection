# Databricks notebook source
# MAGIC %pip install shap
# MAGIC

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. EDA and Visualization 
# MAGIC ### 2-1. Class Imbalance Visualization 
# MAGIC Since fraudulent transactions (Class = 1) are extremely rare, the dataset exhibits a severe class imbalance. Fraud cases account for only 0.17% of all transactions, with 492 frauds out of 284,807 records.
# MAGIC
# MAGIC This imbalance has important implications for model evaluation, as accuracy alone would be misleading. Therefore, in the subsequent analysis, we apply balancing techniques to create comparable samples of legitimate and fraudulent transactions for exploratory comparison purposes.

# COMMAND ----------

# MAGIC %md
# MAGIC - Check for class imbalance 

# COMMAND ----------

sns.countplot(data=df, x='Class')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-2. Transaction Amounts Distribution
# MAGIC Using a boxplot helps compare the median, spread, and outlier between fraud and non-fraud. Fraudulent transactions sometimes have a narrower or unusual range.

# COMMAND ----------

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]


legit['Amount'].describe()

# COMMAND ----------

fraud['Amount'].describe()

# COMMAND ----------

plt.figure(figsize=(8, 5))
color = ["lightblue", "orange"]
sns.boxplot(x='Class', y='Amount', data=df, palette=color)
plt.title("Transaction Amount Distribution by Class (Log Scale)")
plt.xlabel("Class (0 = Legit, 1 = Fraud)")
plt.ylabel("Amount ($)")
plt.ylim(0, 500)  # Optional: limit y-axis to focus on most values
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The boxplot shows that fraudulent transactions have a lower median amount than legitimate ones, indicating that fraud often involves smaller transaction values.
# MAGIC
# MAGIC However, the fraud class exhibits a larger interquartile range (IQR), suggesting greater variability in transaction amounts among typical fraud cases. This indicates that fraudulent activity does not follow a single consistent amount pattern, but rather spans a wider range of transaction values compared to legitimate transactions.

# COMMAND ----------

# MAGIC %md
# MAGIC The histograms display the distribution of transaction amounts.
# MAGIC
# MAGIC - The data is divided into 50 intervals along the x-axis. (bins=50)
# MAGIC - The y-axis represents the number of transactions that fall within each amount/time range.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

sns.histplot(df['Amount'], bins=50, ax=axes[0])
axes[0].set_title('Amount Distribution')

sns.histplot(df['Time'], bins=50, ax=axes[1])
axes[1].set_title('Time Distribution')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The histogram illustrates the time elapsed since the first transaction, measured in seconds.
# MAGIC
# MAGIC - This is useful for identifying when transactions occur and whether fraudulent activity is more common during certain times of the day.
# MAGIC
# MAGIC The distribution of transaction amounts separately for fraudulent and legitimate transactions.
# MAGIC
# MAGIC The output explanation for Amount column:
# MAGIC
# MAGIC - count 284315.000000 → Total number of transactions analyzed
# MAGIC - mean 88.291022 → Average transaction amount ≈ 88.29
# MAGIC - std 250.105092 → Standard deviation: lots of variation
# MAGIC - min 0.000000 → Smallest transaction was 0.00
# MAGIC - 25% 5.650000 → 25% of transactions were less than 5.65
# MAGIC - 50% 22.000000 → Median amount = 22.00
# MAGIC - 75% 77.050000 → 75% were less than 77.05
# MAGIC - max 25691.160000 → Largest transaction = $25,691.16
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Using a histogram shows the frequency distribution of transaction amounts clearly. The fraud cases cluster at specific values, like under $100 or around even dollar amounts

# COMMAND ----------

plt.figure(figsize=(12, 5))

# Legitimate transactions
plt.subplot(1, 2, 1)
sns.histplot(legit['Amount'], bins=50, color='green')
plt.title("Legit Transactions")
plt.xlabel("Amount ($)")
plt.ylabel("Count")
plt.xlim(0, 1000)  # Optional: focus on normal range

# Fraudulent transactions
plt.subplot(1, 2, 2)
sns.histplot(fraud['Amount'], bins=50, color='red')
plt.title("Fraudulent Transactions")
plt.xlabel("Amount ($)")
plt.ylabel("Count")
plt.xlim(0, 1000)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Key insights
# MAGIC - Most legitimate transactions are small (median = $22), but there's a long tail — some are very large.
# MAGIC
# MAGIC - The high standard deviation and large max show that the distribution is right-skewed — meaning most transactions are small, but a few big ones pull the average up.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-3. Time-Based Transaction Patterns
# MAGIC 2-3-1. Transaction Frequency Over Time
# MAGIC
# MAGIC When visualized using density-based distributions, fraudulent transactions exhibit a more uneven temporal distribution, with higher relative density concentrated in specific time windows. In contrast, legitimate transactions display a more uniform temporal density across the observation period.

# COMMAND ----------

# Convert seconds to hours
df['Hour'] = df['Time'] / 3600

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x='Hour',
    hue='Class',
    bins=48,
    stat='density',
    common_norm=False,
    element='step'
)

plt.xlabel("Time (Hours)")
plt.ylabel("Density")
plt.title("Transaction Time Distribution by Class (Density)")
plt.legend(title="Class", labels=["Legit", "Fraud"])
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 2-3-2. Fraud Ratio Over Time
# MAGIC
# MAGIC Certain time windows exhibit elevated fraud ratios, suggesting that fraudulent activity is more concentrated during specific periods rather than occurring uniformly over time.

# COMMAND ----------

# Create hourly bins
df['Hour_bin'] = df['Hour'].astype(int)

# Calculate fraud ratio per hour
hourly_stats = (
    df.groupby('Hour_bin')['Class']
      .agg(['count', 'sum'])
      .reset_index()
)

hourly_stats['fraud_ratio'] = hourly_stats['sum'] / hourly_stats['count']

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=hourly_stats,
    x='Hour_bin',
    y='fraud_ratio',
    marker='o'
)

plt.xlabel("Hour")
plt.ylabel("Fraud Ratio")
plt.title("Hourly Fraud Ratio Over Time")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 2-3-3. Time Gap Between Transactions
# MAGIC
# MAGIC Fraudulent transactions tend to occur in shorter time intervals, indicating burst-like activity patterns compared to legitimate transactions.

# COMMAND ----------

# Sort by time
df_sorted = df.sort_values('Time')

# Compute time difference between consecutive transactions
df_sorted['time_diff'] = df_sorted['Time'].diff()

plt.figure(figsize=(8, 5))
sns.boxplot(
    x='Class',
    y='time_diff',
    data=df_sorted
)

plt.yscale('log')
plt.xlabel("Class (0 = Legit, 1 = Fraud)")
plt.ylabel("Time Between Transactions (seconds, log scale)")
plt.title("Time Gap Between Transactions by Class")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-4. Feature Distribution Analysis (V1–V28)
# MAGIC 2-4-0. Selection of Strong Features via Pearson Correlation
# MAGIC
# MAGIC Using the Pearson correlation between all numeric features and the target variable (Class), sorts them by absolute strength, and selects the top 5 features most strongly associated with fraudulent transactions. These features are then used for detailed distributional and multivariate analysis.

# COMMAND ----------

# Compute correlation with Class
corr = df.corr(numeric_only=True)['Class'].sort_values(key=abs, ascending=False)

# Remove Class itself
corr_features = corr.drop('Class')

# Select top features
top_features = corr_features.head(5)
top_features


# COMMAND ----------

# MAGIC %md
# MAGIC 2-4-1 Selected Feature Distributions

# COMMAND ----------

features = top_features.index[:6]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.kdeplot(
        data=df,
        x=feature,
        hue='Class',
        common_norm=False,
        fill=True,
        ax=axes[i]
    )
    
    axes[i].set_title(f"Distribution of {feature} by Class")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Density")

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 2-4-2 Correlation Analysis
# MAGIC
# MAGIC Correlation analysis highlights a subset of features that are strongly associated with fraudulent transactions, supporting the use of feature-based classification models.

# COMMAND ----------

plt.figure(figsize=(8, 5))

top_features.plot(
    kind='barh',
    color='steelblue'
)

plt.xlabel("Correlation with Fraud (Class)")
plt.title("Top Features Correlated with Fraud")
plt.gca().invert_yaxis()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 2-4-3. Multivariate Insight
# MAGIC
# MAGIC Legitimate transactions (Class 0) tend to fall within a relatively narrow feature range, exhibiting small variations and consistent behavioral patterns. In contrast, fraudulent transactions (Class 1) are more dispersed across the feature space, with some observations lying at extreme values. This deviation from typical patterns highlights anomalous behavior that can be exploited as a predictive signal in classification models.

# COMMAND ----------

sample_df = df.sample(5000, random_state=42)

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=sample_df,
    x=top_features.index[0],
    y=top_features.index[1],
    hue='Class',
    alpha=0.6
)

plt.title("Multivariate Feature Relationship")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 2-4-4. Non-linear Feature Structure Visualization (UMAP)
# MAGIC
# MAGIC UMAP was applied to project high-dimensional transaction features into a two-dimensional space. Legitimate transactions form a dense and compact cluster, while fraudulent transactions appear more scattered and partially separated. This suggests that fraud cases deviate from normal transaction patterns across multiple features, providing useful signals for downstream classification models.

# COMMAND ----------

# Only numerical features are considered (excluding Class).
features = df.drop(columns=['Class'])

# Sampling (for visualization only)
X_umap = features.sample(5000, random_state=42)
y_umap = df.loc[X_umap.index, 'Class']

# Building up the UMAP model
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

# Dimensionality reduction
embedding = reducer.fit_transform(X_umap)

# Plotting
plt.figure(figsize=(7, 5))

sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=y_umap,
    alpha=0.6,
    palette={0: "steelblue", 1: "orange"}
)

plt.title("UMAP Projection of Transaction Features")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Class", labels=["Legit", "Fraud"])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-5. Summary of EDA Findings
# MAGIC
# MAGIC Exploratory data analysis reveals that the dataset is extremely imbalanced, with fraudulent transactions accounting for only 0.17% of all records. Transaction amount analysis shows that fraud cases tend to involve smaller amounts, with occasional extreme values, distinguishing them from legitimate transactions.
# MAGIC
# MAGIC Time-based analysis indicates that fraudulent transactions are unevenly distributed over the observation period, exhibiting localized temporal patterns rather than a uniform distribution.
# MAGIC
# MAGIC Feature distribution and correlation analysis identify several variables (e.g., V14, V17) that exhibit strong associations with fraudulent activity. Multivariate visualization further shows that legitimate transactions cluster within a narrow feature range, while fraud cases are more dispersed and often lie outside typical regions.
# MAGIC
# MAGIC Overall, these findings suggest that fraudulent behavior deviates from normal transaction patterns and provides exploitable signals for classification models.

# COMMAND ----------

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


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-4. Model Evaluation
# MAGIC
# MAGIC Models were evaluated on a held-out test set using metrics appropriate for highly imbalanced data. Since accuracy can be misleading due to the low prevalence of fraud, the following metrics were emphasized:
# MAGIC - Recall: the proportion of fraudulent transactions correctly identified
# MAGIC - Precision: the proportion of predicted fraud cases that are truly fraudulent
# MAGIC - F1-score: the harmonic mean of precision and recall
# MAGIC - ROC-AUC: overall separability between legitimate and fraudulent classes
# MAGIC
# MAGIC Confusion matrices, ROC curves, and Precision–Recall (PR) curves were used to assess model performance. While ROC curves provide a general measure of a model’s discriminative ability, PR curves are more informative in highly imbalanced settings, as they focus on the trade-off between precision and recall for the minority (fraud) class. This evaluation framework enables a more realistic assessment of both fraud detection capability and prediction reliability.

# COMMAND ----------

# MAGIC %md
# MAGIC 3-4-1. Precision–Recall Curve Analysis
# MAGIC
# MAGIC Given the extreme class imbalance, the Precision–Recall (PR) curve provides a more informative evaluation than ROC-AUC.
# MAGIC
# MAGIC The PR curve highlights the trade-off between detecting fraudulent transactions (recall) and minimizing false alarms (precision).
# MAGIC
# MAGIC A high ROC-AUC may still correspond to poor precision for the minority class, making PR analysis critical for fraud detection tasks.

# COMMAND ----------

y_prob = model.predict_proba(X_test)[:,1]

precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (AP = {ap:.3f})")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The Precision–Recall curve yields an Average Precision (AP) score of 0.813, indicating strong performance in identifying fraudulent transactions under severe class imbalance. Since AP summarizes precision across all recall levels, this result suggests that the model maintains high prediction reliability even as recall increases. Compared to the extremely low baseline precision determined by fraud prevalence, the achieved AP demonstrates substantial improvement over random guessing and highlights the model’s effectiveness in practical fraud detection.

# COMMAND ----------

# MAGIC %md
# MAGIC 3-4-2. Cost-based Evaluation (Business Perspective)
# MAGIC
# MAGIC To reflect real-world impact, we introduced a cost-sensitive evaluation assuming a false negative cost of $500 and a false positive cost of $5.
# MAGIC
# MAGIC This analysis demonstrates that models with higher recall for fraud can significantly reduce overall financial loss, even at the expense of lower precision.

# COMMAND ----------

FN_COST = 500
FP_COST = 5

models = {
    'Random Forest': clf,
    'Logistic Regression': lr,
    'XGBoost': xgb
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Cost-based evaluation
    total_cost = fn * FN_COST + fp * FP_COST

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"Total Cost: ${total_cost:,}")
    print()


# COMMAND ----------

# MAGIC %md
# MAGIC Among the three evaluated models, Random Forest achieved the best balance between recall (0.85) and precision (0.47) for fraud detection, successfully identifying most fraudulent transactions while limiting false positives.
# MAGIC
# MAGIC Logistic Regression and XGBoost achieved slightly higher recall (0.90 and 0.88, respectively), but their extremely low precision led to a large number of false positives, substantially increasing operational cost.
# MAGIC
# MAGIC When incorporating cost-based evaluation, Random Forest resulted in the lowest total cost, despite not having the highest ROC-AUC. This indicates that Random Forest provides the most practical and cost-effective performance for fraud detection in this highly imbalanced dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-5. Insights from Modeling
# MAGIC
# MAGIC Confusion matrix analysis provides insight into the types of errors made by each model. In fraud detection, false negatives represent missed fraudulent transactions with direct financial loss, while false positives incur operational costs due to unnecessary transaction reviews.
# MAGIC
# MAGIC Among the evaluated models, Logistic Regression and XGBoost demonstrated strong overall separability and high recall; however, their extremely low precision resulted in a large number of false positives, significantly increasing operational cost and limiting practical usability. In contrast, Random Forest achieved the lowest total cost by maintaining high fraud recall while limiting false positives, making it the most practical model for deployment.
# MAGIC
# MAGIC These results are consistent with EDA findings, confirming that fraudulent transactions tend to deviate from typical feature ranges rather than follow simple linear patterns. Consequently, non-linear, tree-based models are better suited to capture these complex behaviors.
# MAGIC
# MAGIC Overall, these insights emphasize the importance of balancing recall and precision, leveraging EDA-driven feature understanding, and incorporating cost-based evaluation to guide practical model selection in real-world fraud detection.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Model Explainability (SHAP)
# MAGIC While predictive performance is critical in fraud detection, model transparency is equally important for trust, regulatory compliance, and human-in-the-loop decision making. Therefore, SHAP was applied to interpret the predictions of the selected Random Forest model at both global and transaction levels.
# MAGIC
# MAGIC ### 4-1. Motivation for Model Interpretability
# MAGIC Although predictive performance is critical in fraud detection, model interpretability is equally important for trust, regulatory compliance, and operational decision-making. Fraud detection systems are often used in high-stakes environments where incorrect predictions may lead to financial loss or poor customer experience. Therefore, understanding which features drive model predictions is essential before deployment.
# MAGIC
# MAGIC ### 4-2. Global Feature Importance (SHAP Summary)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# Create a TreeExplainer 
explainer = shap.TreeExplainer(clf)

# Compute SHAP values (only the test set)
shap_values = explainer.shap_values(X_test)

# Plot
shap.summary_plot(
  shap_values[:, :, 1], # fraud class
  X_test,
  show=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC From the SHAP summary plot, the top features influencing fraud prediction are V14, V12, V10, V17, and V16.
# MAGIC
# MAGIC Most high feature values (red points) have negative SHAP values, indicating they decrease the predicted fraud probability, whereas most low feature values (blue points) have positive SHAP values, indicating they increase the predicted fraud probability.
# MAGIC
# MAGIC This suggests that, for these features, transactions with unusually low values are more likely to be classified as fraud by the model, while high values are associated with normal transactions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4-3. Transaction-level Explanation (Single Case SHAP)
# MAGIC

# COMMAND ----------

# Pick the highest predicted fraud probability
y_prob = clf.predict_proba(X_test)[:, 1]
fraud_idx = np.argmax(y_prob)
fraud_instance = X_test.iloc[[fraud_idx]]

# Compute SHAP values (The latest SHAP)
shap_values = explainer(fraud_instance)  # <- new API, returns Explanation object

# For binary classifier, pick positive/fraud class
# shap_values.values has shape (n_samples, n_features, n_classes)
if shap_values.values.ndim == 3:
    values_1d = shap_values.values[0, :, 1]  # pick first instance, class=1
elif shap_values.values.ndim == 2:
    values_1d = shap_values.values[0]        # already single output
else:
    raise ValueError("Unsupported SHAP values shape")

# Base value (scalar)
base_value = shap_values.base_values
if isinstance(base_value, (np.ndarray, list)):
    base_value = np.array(base_value).flatten()[0]

# Waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=values_1d,
        base_values=base_value,
        data=fraud_instance.iloc[0].values,
        feature_names=fraud_instance.columns.tolist()
    )
)


# COMMAND ----------

# MAGIC %md
# MAGIC In the SHAP waterfall plot for this single transaction, V14, V12, V10, and V17 contribute positively to the predicted fraud probability, with SHAP values of 0.23, 0.13, 0.09, and 0.07, respectively. V16 has a slight negative contribution of -0.01, indicating it slightly reduces the fraud probability. Overall, the model classifies this transaction as high-risk primarily due to the strong positive contributions from V14, V12, V10, and V17.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4-4. Consistency with EDA Findings
# MAGIC The SHAP analysis reinforces insights obtained from the EDA. Features identified as highly discriminative during exploratory analysis also exhibit strong contributions in the trained Random Forest model. This consistency increases confidence that the model is learning meaningful behavioral patterns rather than spurious correlations.
# MAGIC
# MAGIC The feature ranking based on SHAP values differs from the simple correlation analysis because correlation measures only the linear relationship between each feature and the target, independently of other features. SHAP, on the other hand, captures the contribution of each feature within the trained Random Forest model, accounting for non-linear interactions and dependencies among features. As a result, some features with moderate correlation, such as V14, may appear more important to the model than features with higher linear correlation, such as V17.
# MAGIC
# MAGIC This observation also aligns with the non-linear structures revealed by UMAP and multivariate scatter plots in the EDA, further confirming that the Random Forest model effectively captures complex feature patterns.
# MAGIC