import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns
import numpy as np
import joblib

st.title("Model Performance")

# Load model and predictions
model = joblib.load("models/fraud_model.pkl")
X_test = pd.read_csv("models/X_test.csv")
y_test = pd.read_csv("models/y_test.csv")

# Threshold slider
threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

# Predict
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob > threshold).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, int(val), ha="center", va="center")
st.pyplot(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0, 1], [0, 1], linestyle="--")
ax2.legend()
st.pyplot(fig2)
