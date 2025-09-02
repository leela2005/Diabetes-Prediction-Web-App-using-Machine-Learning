import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Load dataset
df = pd.read_csv("D:/ml_assignment/diabetes_dataset.csv")
# Drop unnecessary columns
columns_to_remove = ["LDL", "Hip Circumference", "WHR", "Diet Type"]
existing_columns = [col for col in columns_to_remove if col in df.columns]
X = df.drop(columns=["Outcome"] + existing_columns)
y = df["Outcome"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# Metrics
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
# Save model and scaler
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
