import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Fraud Detection App")
st.write(
    """
    This app let users explore credit card transaction details, review model performance, and simulate the fraud risk of a single transaction.
    """
)
