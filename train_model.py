import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv("sample_credit_data.csv")

X = df[['Age', 'Income', 'CreditScore', 'LoanAmount']]
y = df['Approved']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Save model + scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Training complete! Model saved.")
