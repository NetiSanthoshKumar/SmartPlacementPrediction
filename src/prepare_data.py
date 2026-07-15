import pandas as pd

# Load the dataset
df = pd.read_csv("../data/student_placement_prediction_dataset_2026.csv")

# Remove unnecessary columns
df = df.drop(["student_id", "salary_package_lpa"], axis=1)

# Display the first 5 rows
print(df.head())

# Display column names
print(df.columns)

# Display the shape
print(df.shape)