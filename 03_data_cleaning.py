
import pandas as pd

# ==========================================
# 1. LOAD ORIGINAL DATASET
# ==========================================
df = pd.read_csv("HR_Job_Placement_Dataset.csv")

print("Original dataset shape:", df.shape)

# ==========================================
# 2. REMOVE EXACT DUPLICATE ROWS
# ==========================================
duplicates_removed = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicates_removed)

# ==========================================
# 3. STANDARDIZE TEXT VALUES
# ==========================================
categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for col in categorical_columns:
    df[col] = df[col].str.strip()

# Standardize capitalization
df["gender"] = df["gender"].str.title()
df["internship_experience"] = (
    df["internship_experience"].str.title()
)

# ==========================================
# 4. ANALYZE MISSING VALUES
# ==========================================
print("\n--- MISSING VALUE ANALYSIS ---")

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

for col in df.columns:
    if missing_count[col] > 0:
        print(
            f"{col}: {missing_count[col]} missing "
            f"({missing_percentage[col]:.2f}%)"
        )

# ==========================================
# 5. IMPUTE MISSING VALUES
# Based on professor's guidelines
# ==========================================
imputation_log = []

for col in df.columns[df.isnull().any()]:

    missing_pct = missing_percentage[col]

    if missing_pct > 60:
        print(f"Review column for removal: {col}")
        continue

    elif missing_pct >= 40:
        print(f"Review column before imputation: {col}")
        continue

    elif missing_pct >= 20:
        print(f"Domain-based review required: {col}")
        continue

    if pd.api.types.is_numeric_dtype(df[col]):

        # Numeric: median imputation
        fill_value = df[col].median()
        df[col] = df[col].fillna(fill_value)
        method = "Median"

    else:
        # Categorical: mode imputation
        modes = df[col].mode(dropna=True)

        if modes.empty:
            raise ValueError(
                f"Cannot impute column {col}: no known values"
            )

        fill_value = modes.iloc[0]
        df[col] = df[col].fillna(fill_value)
        method = "Mode"

    imputation_log.append({
        "Column": col,
        "Missing %": round(missing_pct, 2),
        "Method": method,
        "Fill value": fill_value
    })

    print(f"Filled {col} using {method}: {fill_value}")

# Remove duplicates created after missing-value imputation
duplicates_after_imputation = df.duplicated().sum()
df = df.drop_duplicates().copy()

print(
    "Additional duplicate rows removed after imputation:",
    duplicates_after_imputation
)

# ==========================================
# 6. FINAL VALIDATION
# ==========================================
print("\n--- FINAL VALIDATION ---")

print("Dataset shape:", df.shape)
print("Duplicate rows remaining:", df.duplicated().sum())
print("\n--- REMAINING DUPLICATE ROWS ---")
print(df[df.duplicated(keep=False)].sort_values(
    by=list(df.columns)
).head(10).to_string(index=False))
print("Total missing values:", df.isnull().sum().sum())

remaining_missing = df.columns[df.isnull().any()].tolist()
print("Columns still containing missing values:", remaining_missing)

print("\n--- IMPUTATION SUMMARY ---")
print(pd.DataFrame(imputation_log).to_string(index=False))

# Fail if any missing values remain
assert df.isnull().sum().sum() == 0, (
    "ERROR: Some missing values remain. Review the output."
)



print("\n--- LOGICAL CONSISTENCY CHECKS ---")

# Check negative values in columns where negatives are invalid
non_negative_columns = [
    "age_years",
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "skills_match_percentage",
    "certifications_count",
    "years_of_experience",
    "previous_ctc_lpa",
    "expected_ctc_lpa",
    "notice_period_days",
    "employment_gap_months"
]

for col in non_negative_columns:
    invalid_count = (df[col] < 0).sum()
    print(f"{col}: {invalid_count} negative values")

# Check whether experience exceeds age
experience_issues = (
    df["years_of_experience"] > df["age_years"]
).sum()

print(
    "\nRecords where experience exceeds age:",
    experience_issues
)

# Check expected categorical values
expected_categories = {
    "gender": ["Male", "Female"],
    "internship_experience": ["Yes", "No"],
    "career_switch_willingness": ["Willing", "Not Willing"],
    "relevant_experience": ["Relevant", "Not Relevant"],
    "job_role_match": ["Matched", "Not Matched"],
    "company_tier": ["Tier 1", "Tier 2", "Tier 3"],
    "competition_level": ["Low", "Medium", "High"],
    "bond_requirement": ["Required", "Not Required"],
    "layoff_history": ["Yes", "No"],
    "relocation_willingness": ["Willing", "Not Willing"],
    "status": ["Placed", "Not Placed"]
}

print("\n--- UNEXPECTED CATEGORICAL VALUES ---")

for col, allowed_values in expected_categories.items():
    unexpected = sorted(
        set(df[col].dropna().unique()) - set(allowed_values)
    )
    print(f"{col}: {unexpected}")
    
# 8. NUMERIC RANGE AND OUTLIER CHECKS

print("\n--- NUMERIC RANGE ANALYSIS ---")

numeric_columns = df.select_dtypes(include="number").columns

# Display key statistics for all numeric columns
print(
    df[numeric_columns]
    .describe()
    .T
    .round(2)
    .to_string()
)

# Check whether percentages and scores are within 0–100
score_columns = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "skills_match_percentage"
]

print("\n--- SCORES OUTSIDE 0–100 ---")

for col in score_columns:
    invalid = df[(df[col] < 0) | (df[col] > 100)]
    print(f"{col}: {len(invalid)} records outside 0–100")

# Flag potentially unusual ages for review; do not delete them
print("\n--- AGE REVIEW ---")
print("Minimum age:", df["age_years"].min())
print("Maximum age:", df["age_years"].max())
print("Records below age 18:", (df["age_years"] < 18).sum())
print("Records above age 70:", (df["age_years"] > 70).sum())

# Flag experience that may be unusually high for a person's age
print("\n--- EXPERIENCE PLAUSIBILITY REVIEW ---")
experience_flags = (
    df["years_of_experience"] > (df["age_years"] - 14)
)
print(
    "Records with experience exceeding age minus 14:",
    experience_flags.sum()
)

# 9. STATISTICAL OUTLIER DETECTION USING IQR

print("\n--- IQR OUTLIER ANALYSIS ---")

outlier_columns = [
    "age_years",
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "skills_match_percentage",
    "certifications_count",
    "years_of_experience",
    "previous_ctc_lpa",
    "expected_ctc_lpa",
    "notice_period_days",
    "employment_gap_months"
]

for col in outlier_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outlier_count = (
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ).sum()

    print(
        f"{col}: {outlier_count} potential outliers "
        f"(limits: {lower_limit:.2f} to {upper_limit:.2f})"
    )

# 10. INSPECT POTENTIAL OUTLIER VALUES

print("\n--- CERTIFICATION COUNTS ---")
print(df["certifications_count"].value_counts().sort_index().to_string())

print("\n--- NOTICE PERIOD VALUES ---")
print(df["notice_period_days"].value_counts().sort_index().to_string())

print("\n--- LARGEST EMPLOYMENT GAPS ---")
print(
    df["employment_gap_months"]
    .value_counts()
    .sort_index()
    .tail(10)
    .to_string()
)

print("\n--- HIGHEST PREVIOUS SALARIES ---")
print(
    df["previous_ctc_lpa"]
    .sort_values(ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n--- HIGHEST EXPECTED SALARIES ---")
print(
    df["expected_ctc_lpa"]
    .sort_values(ascending=False)
    .head(10)
    .to_string(index=False)
)

# 11. SALARY CONSISTENCY CHECK

print("\n--- SALARY CONSISTENCY CHECK ---")

# Count candidates whose expected CTC is lower than previous CTC
lower_expected = df[
    df["expected_ctc_lpa"] < df["previous_ctc_lpa"]
]

print(
    "Candidates with expected CTC below previous CTC:",
    len(lower_expected)
)

print(
    "Percentage of dataset:",
    round(len(lower_expected) / len(df) * 100, 2),
    "%"
)

# Check how many have equal expected and previous CTC
equal_salary = (
    df["expected_ctc_lpa"] == df["previous_ctc_lpa"]
).sum()

print("Candidates with equal expected and previous CTC:", equal_salary)

# Display a few examples for investigation only
print("\nSample records with lower expected CTC:")
print(
    lower_expected[
        ["age_years", "years_of_experience",
         "previous_ctc_lpa", "expected_ctc_lpa", "status"]
    ].head(10).to_string(index=False)
)

# 12. FINAL CATEGORY AND TARGET CHECKS

print("\n--- DEGREE SPECIALIZATION VALUES ---")
print(df["degree_specialization"].value_counts(dropna=False).to_string())

print("\n--- TARGET VARIABLE DISTRIBUTION ---")
print(df["status"].value_counts(dropna=False).to_string())

print("\n--- TARGET VARIABLE PERCENTAGES ---")
print(
    (df["status"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
    .to_string()
)

print("\n--- FINAL DATA TYPES ---")
print(df.dtypes.to_string())

df.to_csv("cleaned_step2.csv", index=False)

print("\nSUCCESS: No missing values remain.")
print("Saved cleaned dataset: cleaned_step2.csv")