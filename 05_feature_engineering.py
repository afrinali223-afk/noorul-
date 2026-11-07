
import pandas as pd

# ============================================================
# STEP 1: LOAD THE CLEANED DATASET
# ============================================================

df = pd.read_csv("cleaned_step2.csv")

print("=" * 60)
print("JOB ACCEPTANCE / PLACEMENT FEATURE ENGINEERING")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# FEATURE 1: EXPERIENCE CATEGORY
# Fresher = 0 years
# Junior  = 1-2 years
# Senior  = 3+ years
# ============================================================

def categorize_experience(years):
    if pd.isna(years) or years < 0:
        return "Unknown"
    elif years == 0:
        return "Fresher"
    elif years <= 2:
        return "Junior"
    else:
        return "Senior"


df["experience_category"] = (
    df["years_of_experience"].apply(categorize_experience)
)


# ============================================================
# FEATURE 2: ACADEMIC PERFORMANCE BAND
# Uses the average of SSC, HSC and degree percentages.
# Below 60 = Low; 60-74.99 = Medium; 75+ = High
# These are project-defined thresholds.
# ============================================================

df["academic_average"] = df[
    ["ssc_percentage", "hsc_percentage", "degree_percentage"]
].mean(axis=1)


def categorize_academics(score):
    if pd.isna(score):
        return "Unknown"
    elif score < 60:
        return "Low"
    elif score < 75:
        return "Medium"
    else:
        return "High"


df["academic_performance_band"] = (
    df["academic_average"].apply(categorize_academics)
)


# ============================================================
# FEATURE 3: SKILLS MATCH LEVEL
# Below 60 = Low; 60-79.99 = Medium; 80+ = High
# These are project-defined thresholds.
# ============================================================

def categorize_skills(score):
    if pd.isna(score):
        return "Unknown"
    elif score < 60:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"


df["skills_match_level"] = (
    df["skills_match_percentage"].apply(categorize_skills)
)


# ============================================================
# FEATURE 4: INTERVIEW PERFORMANCE CATEGORY
# The dataset has no single interview_score column.
# Use the average of technical, aptitude and communication scores.
# Below 60 = Low; 60-74.99 = Medium; 75+ = High
# These are project-defined thresholds.
# ============================================================

assessment_columns = [
    "technical_score",
    "aptitude_score",
    "communication_score"
]

df["combined_assessment_score"] = df[assessment_columns].mean(axis=1)


def categorize_interview(score):
    if pd.isna(score):
        return "Unknown"
    elif score < 60:
        return "Low"
    elif score < 75:
        return "Medium"
    else:
        return "High"


df["interview_performance_category"] = (
    df["combined_assessment_score"].apply(categorize_interview)
)


# ============================================================
# FEATURE 5: PLACEMENT PROBABILITY SCORE
# This must come from a trained ML model.
# Do NOT create an artificial probability here.
# It will be added after model training.
# ============================================================


# ============================================================
# DISPLAY RESULTS
# ============================================================

# ============================================================
# DISPLAY RESULTS
# ============================================================
new_features = [
    "experience_category",
    "academic_average",
    "academic_performance_band",
    "skills_match_level",
    "combined_assessment_score",
    "interview_performance_category"
]
print("\nNew feature columns created:")

for column in new_features:
    print(f"\n{column}:")

    if pd.api.types.is_numeric_dtype(df[column]):
        print(df[column].describe().round(2))
    else:
        print(df[column].value_counts())

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print("Dataset shape after feature engineering:", df.shape)
print("Missing values in new features:")
print(df[new_features].isnull().sum())

print("\nSample records:")
print(
    df[
        [
            "years_of_experience",
            "experience_category",
            "academic_average",
            "academic_performance_band",
            "skills_match_percentage",
            "skills_match_level",
            "combined_assessment_score",
            "interview_performance_category"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# SAVE OUTPUT
# ============================================================

df.to_csv("feature_engineered.csv", index=False)

print("\nSaved successfully as feature_engineered.csv")
print(
    "\nNote: placement probability will be generated "
    "after training the classification model."
)
print("Feature engineering completed.")
