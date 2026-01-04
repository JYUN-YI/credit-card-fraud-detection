import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

MODEL_PATH = "models/fraud_model.pkl"

if os.path.exists(MODEL_PATH):
    print("Model already exists. Skip training.")
    exit(0)

os.makedirs("models", exist_ok=True)

# Load data
DATA_PATH = "data/creditcard.csv"
df = pd.read_csv(DATA_PATH, nrows=50000)

# Features & target
X = df[["Time", "Amount"]]
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train RandomForest
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Save model + test data
joblib.dump(model, "models/fraud_model.pkl")
X_test.to_csv("models/X_test.csv", index=False)
y_test.to_csv("models/y_test.csv", index=False)

print("✅ Model training completed")
