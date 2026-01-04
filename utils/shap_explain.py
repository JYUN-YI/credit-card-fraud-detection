# utils/shap_explain.py
import shap
import matplotlib
matplotlib.use("Agg")  # Avoid Docker crash
import matplotlib.pyplot as plt
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

def explain_transaction(transaction_df):
    """
    transaction_df: pd.DataFrame
    Returns: matplotlib figure of top 5 SHAP features
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transaction_df)

    fig, ax = plt.subplots(figsize=(6,4))
    shap.summary_plot(shap_values, transaction_df, plot_type="bar", max_display=5, show=False)
    return fig
