import pandas as pd

# Load final placement probability dataset
df = pd.read_csv("placement_probability_scores.csv")

# Save as master dataset
df.to_csv("master_job_placement_dataset.csv", index=False)

print("=" * 60)
print("MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 records:")
print(df.head())

print("\nSaved as master_job_placement_dataset.csv")