import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/student_placement_prediction_dataset_2026.csv")

# -----------------------------
# Basic Information
# -----------------------------
print("========== DATASET INFORMATION ==========\n")

print("Shape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Placement Distribution
# -----------------------------
print("\n========== PLACEMENT STATUS ==========\n")
print(df["placement_status"].value_counts())

# -----------------------------
# CGPA Analysis
# -----------------------------
print("\n========== CGPA ANALYSIS ==========\n")

print("CGPA Statistics:")
print(df["cgpa"].describe())

print("\nAverage CGPA by Placement:")
print(df.groupby("placement_status")["cgpa"].mean())

# -----------------------------
# Coding Skill Analysis
# -----------------------------
print("\n========== CODING SKILL ==========\n")

print(df.groupby("placement_status")["coding_skill_score"].mean())

# -----------------------------
# Aptitude Score Analysis
# -----------------------------
print("\n========== APTITUDE SCORE ==========\n")

print(df.groupby("placement_status")["aptitude_score"].mean())

# -----------------------------
# Communication Skill Analysis
# -----------------------------
print("\n========== COMMUNICATION SKILL ==========\n")

print(df.groupby("placement_status")["communication_skill_score"].mean())

# -----------------------------
# Internship Analysis
# -----------------------------
print("\n========== INTERNSHIPS ==========\n")

print(df.groupby("placement_status")["internships_count"].mean())

# -----------------------------
# Projects Analysis
# -----------------------------
print("\n========== PROJECTS ==========\n")

print(df.groupby("placement_status")["projects_count"].mean())

# ==========================================================
# VISUALIZATIONS
# ==========================================================

# Placement Status Bar Chart
plt.figure(figsize=(6,4))
df["placement_status"].value_counts().plot(kind="bar")
plt.title("Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.show()

# CGPA Box Plot
plt.figure(figsize=(6,4))
df.boxplot(column="cgpa", by="placement_status")
plt.title("CGPA vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")
plt.show()

# Coding Skill Box Plot
plt.figure(figsize=(6,4))
df.boxplot(column="coding_skill_score", by="placement_status")
plt.title("Coding Skill Score vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Coding Skill Score")
plt.show()

# Aptitude Score Box Plot
plt.figure(figsize=(6,4))
df.boxplot(column="aptitude_score", by="placement_status")
plt.title("Aptitude Score vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Aptitude Score")
plt.show()

# Communication Skill Box Plot
plt.figure(figsize=(6,4))
df.boxplot(column="communication_skill_score", by="placement_status")
plt.title("Communication Skill vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Communication Skill Score")
plt.show()

# Internship Box Plot
plt.figure(figsize=(6,4))
df.boxplot(column="internships_count", by="placement_status")
plt.title("Internships vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Internships Count")
plt.show()

# Projects Box Plot
plt.figure(figsize=(6,4))
df.boxplot(column="projects_count", by="placement_status")
plt.title("Projects vs Placement")
plt.suptitle("")
plt.xlabel("Placement Status")
plt.ylabel("Projects Count")
plt.show()