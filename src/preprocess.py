import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    # Load dataset from the project data directory relative to this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "student_placement_prediction_dataset_2026.csv")
    df = pd.read_csv(csv_path)

    # Remove unnecessary columns
    df = df.drop(["student_id", "salary_package_lpa"], axis=1)

    return df


def preprocess_data(df):
    # Separate Features and Target
    X = df.drop("placement_status", axis=1)
    y = df["placement_status"]

    # One-Hot Encode categorical features
    X = pd.get_dummies(
        X,
        columns=["gender", "branch", "college_tier", "volunteer_experience"],
        drop_first=True
    )

    # Convert target to numbers
    y = y.map({
        "Not Placed": 0,
        "Placed": 1
    })

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Data :", X_train.shape)
    print("Testing Data  :", X_test.shape)