import joblib

# Load trained model
model = joblib.load("../models/placement_model.pkl")

print("✅ Model Loaded Successfully!")

# Display model information
print(model)