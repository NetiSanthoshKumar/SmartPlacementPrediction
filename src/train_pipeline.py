from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "student_placement_prediction_dataset_2026.csv"
MODEL_DIR = BASE_DIR / "models"
CATEGORICAL_FEATURES = ["gender", "branch", "college_tier", "volunteer_experience"]


def build_preprocessor(numerical_features):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), numerical_features),
    ])


def main():
    df = pd.read_csv(DATA_PATH).drop(columns=["student_id", "salary_package_lpa"])
    X = df.drop(columns="placement_status")
    y = df["placement_status"]
    numerical_features = [column for column in X.columns if column not in CATEGORICAL_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    }
    trained = {}
    rows = []
    for name, factory in factories.items():
        candidate = Pipeline([
            ("preprocessor", build_preprocessor(numerical_features)),
            ("model", factory()),
        ])
        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_test)
        probabilities = candidate.predict_proba(X_test)[:, list(candidate.classes_).index("Placed")]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, pos_label="Placed"),
            "recall": recall_score(y_test, predictions, pos_label="Placed"),
            "f1": f1_score(y_test, predictions, pos_label="Placed"),
            "roc_auc": roc_auc_score((y_test == "Placed").astype(int), probabilities),
        })
        trained[name] = candidate

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best_name = comparison.iloc[0]["model"]
    best_pipeline = trained[best_name]
    best_predictions = best_pipeline.predict(X_test)
    print(comparison.to_string(index=False))
    print("\nSelected model:", best_name)
    print("\nClassification report:\n", classification_report(y_test, best_predictions))
    print("Confusion matrix:\n", confusion_matrix(y_test, best_predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODEL_DIR / "placement_pipeline.pkl")
    comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = best_pipeline.named_steps["model"]
    importances = model.feature_importances_ if hasattr(model, "feature_importances_") else abs(model.coef_[0])
    pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(MODEL_DIR / "feature_importance.csv", index=False)


if __name__ == "__main__":
    main()