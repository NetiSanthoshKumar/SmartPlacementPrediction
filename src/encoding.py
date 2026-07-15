import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("../data/student_placement_prediction_dataset_2026.csv")

# Remove unnecessary columns
df = df.drop(["student_id", "salary_package_lpa"], axis=1)

# Create LabelEncoder
encoder = LabelEncoder()

# Convert text columns into numbers
df["gender"] = encoder.fit_transform(df["gender"])
df["branch"] = encoder.fit_transform(df["branch"])
df["college_tier"] = encoder.fit_transform(df["college_tier"])
df["volunteer_experience"] = encoder.fit_transform(df["volunteer_experience"])
df["placement_status"] = encoder.fit_transform(df["placement_status"])

# Display first 5 rows
print(df.head())