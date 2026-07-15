from preprocess import load_data, preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# Load and preprocess data
# -----------------------------
df = load_data()

X_train, X_test, y_train, y_test = preprocess_data(df)

# -----------------------------
# Create Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# -----------------------------
# Train the model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "../models/placement_model.pkl")

print("\nModel saved successfully!")