# utils/model_inference.py
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

def predict_transaction(transaction_df):
    """
    transaction_df: pd.DataFrame, columns align with models
    Returns: probabilities, decision
    """
    probs = model.predict_proba(transaction_df[["Time","Amount"]])[:,1]
    decisions = ["Fraud" if p>0.5 else "Legit" for p in probs]
    return probs, decisions
