# Databricks notebook source
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