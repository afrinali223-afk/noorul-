import pandas as pd

# Load the original dataset
df = pd.read_csv("HR_Job_Placement_Dataset.csv")

# 1. Display the number of rows and columns
print("\n--- DATASET SHAPE ---")
print(df.shape)

# 2. Display the first five records
print("\n--- FIRST FIVE RECORDS ---")
print(df.head())

# 3. Display all column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 4. Check column data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# 5. Check missing values in each column
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 6. Check duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 7. Display basic numerical statistics
print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())

# 8. Display the target variable distribution
print("\n--- PLACEMENT STATUS ---")
print(df["status"].value_counts(dropna=False))

