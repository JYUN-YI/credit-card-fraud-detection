import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_preprocessing import load_data  # 假設你有一個 load_data function

st.title("Dataset Overview")

# Load data
df = load_data()

# Fraud rate
fraud_rate = df['Class'].mean()
st.metric("Fraud Rate", f"{fraud_rate*100:.2f}%")

# Amount distribution
st.subheader("Transaction Amount Distribution")
fig, ax = plt.subplots()
sns.histplot(df, x='Amount', hue='Class', bins=50, log_scale=(False, True), ax=ax)
st.pyplot(fig)

# Time-based chart
st.subheader("Transactions over Time")
time_df = df.groupby('Time')['Class'].mean().reset_index()
st.line_chart(time_df.rename(columns={'Time':'index'}).set_index('index'))
