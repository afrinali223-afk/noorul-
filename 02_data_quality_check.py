
import pandas as pd

df = pd.read_csv("HR_Job_Placement_Dataset.csv")

# 1. Inspect categorical values and their frequencies
categorical_columns = df.select_dtypes(
    include=["object", "category", "string"]
).columns

print("\n--- CATEGORICAL VALUE CHECK ---")

for col in categorical_columns:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False).head(20))

# 2. Inspect duplicate records
print("\n--- DUPLICATE RECORDS ---")
duplicates = df[df.duplicated(keep=False)]
print(duplicates.head(10))

# 3. Check academic percentage ranges
percentage_columns = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "skills_match_percentage"
]

print("\n--- SCORE RANGE CHECK ---")
print(df[percentage_columns].agg(["min", "max"]).T)

print("\n--- INVALID PERCENTAGE VALUES ---")

for col in percentage_columns:
    invalid = df[(df[col] < 0) | (df[col] > 100)]
    print(f"{col}: {len(invalid)} invalid values")
# 4. Check numerical columns for unusual values
print("\n--- NUMERICAL COLUMN SUMMARY ---")
print(df.describe().T)

print("\nQuality check completed.")
