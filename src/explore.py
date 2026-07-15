import pandas as pd

# Read the dataset
df = pd.read_csv("../data/student_placement_prediction_dataset_2026.csv")

# Show first 5 rows
print(df.head())

# Number of rows and columns
print("\nShape of Dataset:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())