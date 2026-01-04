# Databricks notebook source
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