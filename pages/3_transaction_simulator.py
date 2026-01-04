import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

from utils.model_inference import predict_transaction
from utils.shap_explain import explain_transaction

st.title("Transaction Simulator")

# User input
amount = st.number_input("Transaction Amount", min_value=0.0, value=10.0)
time = st.number_input("Transaction Time", min_value=0.0, value=10000.0)

transaction = pd.DataFrame({"Amount":[amount], "Time":[time]})

# Predict
prob, decision = predict_transaction(transaction)
st.metric("Fraud Probability", f"{prob[0]*100:.2f}%")
st.metric("Decision", decision[0])

# SHAP explanation
st.subheader("SHAP Feature Importance")
shap_fig = explain_transaction(transaction)
st.pyplot(shap_fig)
