import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("../data/student_placement_prediction_dataset_2026.csv")

# Remove unnecessary columns
df = df.drop(["student_id", "salary_package_lpa"], axis=1)

# Target column
y = df["placement_status"]

# Feature columns
X = df.drop("placement_status", axis=1)

# Categorical columns
categorical_features = [
    "gender",
    "branch",
    "college_tier",
    "volunteer_experience"
]

# Numerical columns
numerical_features = [col for col in X.columns if col not in categorical_features]

# -----------------------
# Preprocessing
# -----------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

# -----------------------
# Pipeline
# -----------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# -----------------------
# Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# Train
# -----------------------
pipeline.fit(X_train, y_train)

# -----------------------
# Predict
# -----------------------
pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

# -----------------------
# Save Pipeline
# -----------------------
joblib.dump(
    pipeline,
    "../models/placement_pipeline.pkl"
)

print("Pipeline Saved Successfully!")