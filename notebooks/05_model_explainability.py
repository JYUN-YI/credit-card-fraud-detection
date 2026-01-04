# Databricks notebook source
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