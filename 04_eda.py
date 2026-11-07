
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_step2.csv")

print("--- DATASET OVERVIEW ---")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n--- FIRST 5 RECORDS ---")
print(df.head())

print("\n--- COLUMN INFORMATION ---")
df.info()

print("\n--- TARGET DISTRIBUTION ---")
print(df["status"].value_counts())

print("\n--- TARGET PERCENTAGE ---")
print(
    (df["status"].value_counts(normalize=True) * 100).round(2)
)

# --- NUMERICAL FEATURE SUMMARY ---
print("\n--- NUMERICAL FEATURE SUMMARY ---")

numerical_columns = df.select_dtypes(include="number").columns

print(
    df[numerical_columns]
    .describe()
    .round(2)
    .T
)

# --- CATEGORICAL FEATURE ANALYSIS ---
print("\n--- CATEGORICAL FEATURE ANALYSIS ---")

categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for column in categorical_columns:
    if column == "status":
        continue

    print(f"\n{column}")
    print(df[column].value_counts())
    print(
        "Percentage distribution:"
    )
    print(
        (df[column].value_counts(normalize=True) * 100)
        .round(2)
    )

# --- NUMERICAL FEATURES BY PLACEMENT STATUS ---
print("\n--- NUMERICAL FEATURES BY PLACEMENT STATUS ---")

numerical_columns = df.select_dtypes(
    include="number"
).columns

status_summary = df.groupby("status")[numerical_columns].mean()

print(status_summary.round(2).T)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# JOB ACCEPTANCE PREDICTION SYSTEM - EXPLORATORY DATA ANALYSIS
# ==========================================================

# Load cleaned dataset
df = pd.read_csv("cleaned_step2.csv")

print("=" * 65)
print("JOB ACCEPTANCE PREDICTION SYSTEM - EDA")
print("=" * 65)

# ----------------------------------------------------------
# 1. DATASET OVERVIEW
# ----------------------------------------------------------
print("\n1. DATASET OVERVIEW")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Total missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nColumn information:")
df.info()

# ----------------------------------------------------------
# 2. TARGET DISTRIBUTION
# ----------------------------------------------------------
print("\n2. TARGET DISTRIBUTION")

print(df["status"].value_counts())
print("\nTarget percentages:")
print((df["status"].value_counts(normalize=True) * 100).round(2))

# Create binary target for calculating placement rates
# Placed = 1, Not Placed = 0
df["placed_binary"] = (df["status"] == "Placed").astype(int)

# ----------------------------------------------------------
# 3. NUMERICAL FEATURE SUMMARY
# ----------------------------------------------------------
print("\n3. NUMERICAL FEATURE SUMMARY")

numerical_columns = df.select_dtypes(include="number").columns
numerical_columns = numerical_columns.drop(
    "placed_binary", errors="ignore"
)

print(df[numerical_columns].describe().round(2).T)

# ----------------------------------------------------------
# 4. CATEGORICAL FEATURE DISTRIBUTION
# ----------------------------------------------------------
print("\n4. CATEGORICAL FEATURE DISTRIBUTION")

categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for column in categorical_columns:
    if column == "status":
        continue

    print(f"\n{column}")
    print(df[column].value_counts())
    print("\nPercentage:")
    print(
        (df[column].value_counts(normalize=True) * 100)
        .round(2)
    )

# ----------------------------------------------------------
# REQUIREMENT 1:
# ASSESSMENT SCORES VS PLACEMENT STATUS
# ----------------------------------------------------------
print("\n5. ASSESSMENT SCORES VS PLACEMENT")

score_columns = [
    "technical_score",
    "aptitude_score",
    "communication_score"
]

score_summary = df.groupby("status")[score_columns].mean()
print("\nAverage scores by status:")
print(score_summary.round(2).T)

# Optional combined assessment score.
# This is NOT an actual interview score.
# It assumes all three scores have equal importance.
df["combined_assessment_score"] = df[score_columns].mean(axis=1)

print("\nAverage combined assessment score by status:")
print(
    df.groupby("status")["combined_assessment_score"]
    .mean()
    .round(2)
)

fig, ax = plt.subplots(figsize=(10, 6))

score_summary.T.plot(kind="bar", ax=ax)

ax.set_title("Assessment Scores by Placement Status")
ax.set_xlabel("Assessment Component")
ax.set_ylabel("Average Score")
ax.legend(title="Status")
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()

# Combined score distribution by status
fig, ax = plt.subplots(figsize=(8, 5))

for status in ["Not Placed", "Placed"]:
    values = df.loc[
        df["status"] == status,
        "combined_assessment_score"
    ]
    ax.hist(values, bins=25, alpha=0.5, label=status)

ax.set_title("Combined Assessment Score Distribution by Status")
ax.set_xlabel("Combined Assessment Score")
ax.set_ylabel("Number of Candidates")
ax.legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# REQUIREMENT 2:
# SKILLS MATCH PERCENTAGE VS PLACEMENT RATE
# ----------------------------------------------------------
print("\n6. SKILLS MATCH VS PLACEMENT RATE")

df["skills_match_group"] = pd.cut(
    df["skills_match_percentage"],
    bins=[0, 50, 60, 70, 80, 90, 100],
    labels=[
        "Up to 50",
        "50-60",
        "60-70",
        "70-80",
        "80-90",
        "90-100"
    ],
    include_lowest=True
)

skills_summary = df.groupby(
    "skills_match_group",
    observed=False
).agg(
    candidate_count=("placed_binary", "size"),
    placement_rate=("placed_binary", "mean")
)

skills_summary["placement_rate"] *= 100

print(skills_summary.round(2))

fig, ax = plt.subplots(figsize=(9, 5))

skills_summary["placement_rate"].plot(
    kind="bar",
    ax=ax
)

ax.set_title("Placement Rate by Skills Match Percentage")
ax.set_xlabel("Skills Match Group (%)")
ax.set_ylabel("Placement Rate (%)")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# REQUIREMENT 3:
# COMPANY TIER VS PLACEMENT RATE
# ----------------------------------------------------------
print("\n7. COMPANY TIER VS PLACEMENT RATE")

company_summary = df.groupby("company_tier").agg(
    candidate_count=("placed_binary", "size"),
    placed_count=("placed_binary", "sum"),
    placement_rate=("placed_binary", "mean")
)

company_summary["placement_rate"] *= 100

print(company_summary.round(2))

# Maintain logical tier order
tier_order = ["Tier 1", "Tier 2", "Tier 3"]
company_summary = company_summary.reindex(tier_order)

fig, ax = plt.subplots(figsize=(8, 5))

company_summary["placement_rate"].plot(
    kind="bar",
    ax=ax
)

ax.set_title("Placement Rate by Company Tier")
ax.set_xlabel("Company Tier")
ax.set_ylabel("Placement Rate (%)")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# REQUIREMENT 4:
# EXPERIENCE VS PLACEMENT PROBABILITY
# ----------------------------------------------------------
print("\n8. EXPERIENCE VS PLACEMENT RATE")

experience_summary = df.groupby("years_of_experience").agg(
    candidate_count=("placed_binary", "size"),
    placed_count=("placed_binary", "sum"),
    placement_rate=("placed_binary", "mean")
)

experience_summary["placement_rate"] *= 100

print(experience_summary.round(2))

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    experience_summary.index,
    experience_summary["placement_rate"],
    marker="o"
)

ax.set_title("Placement Rate by Years of Experience")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Placement Rate (%)")
ax.set_xticks(experience_summary.index)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# REQUIREMENT 5:
# COMPETITION LEVEL VS PLACEMENT RATE
# ----------------------------------------------------------
print("\n9. COMPETITION LEVEL VS PLACEMENT RATE")

competition_summary = df.groupby("competition_level").agg(
    candidate_count=("placed_binary", "size"),
    placed_count=("placed_binary", "sum"),
    placement_rate=("placed_binary", "mean")
)

competition_summary["placement_rate"] *= 100

competition_order = ["Low", "Medium", "High"]
competition_summary = competition_summary.reindex(
    competition_order
)

print(competition_summary.round(2))

fig, ax = plt.subplots(figsize=(8, 5))

competition_summary["placement_rate"].plot(
    kind="bar",
    ax=ax
)

ax.set_title("Placement Rate by Competition Level")
ax.set_xlabel("Competition Level")
ax.set_ylabel("Placement Rate (%)")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# REQUIREMENT 6:
# CORRELATION ANALYSIS AMONG NUMERICAL FEATURES
# ----------------------------------------------------------
print("\n10. CORRELATION ANALYSIS")

correlation_columns = [
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

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(2))

fig, ax = plt.subplots(figsize=(13, 10))

image = ax.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(range(len(correlation_columns)))
ax.set_yticks(range(len(correlation_columns)))

ax.set_xticklabels(
    correlation_columns,
    rotation=90
)
ax.set_yticklabels(correlation_columns)

# Display correlation values inside the heatmap
for i in range(len(correlation_columns)):
    for j in range(len(correlation_columns)):
        ax.text(
            j, i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7
        )

fig.colorbar(image, ax=ax, label="Pearson Correlation")
ax.set_title("Correlation Heatmap of Numerical Features")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# ADDITIONAL ANALYSIS:
# PLACEMENT RATE BY RELEVANT EXPERIENCE
# ----------------------------------------------------------
print("\n11. PLACEMENT RATE BY RELEVANT EXPERIENCE")

relevant_summary = df.groupby("relevant_experience").agg(
    candidate_count=("placed_binary", "size"),
    placed_count=("placed_binary", "sum"),
    placement_rate=("placed_binary", "mean")
)

relevant_summary["placement_rate"] *= 100

print(relevant_summary.round(2))

# ----------------------------------------------------------
# ADDITIONAL ANALYSIS:
# PLACEMENT RATE BY INTERNSHIP EXPERIENCE
# ----------------------------------------------------------
print("\n12. PLACEMENT RATE BY INTERNSHIP EXPERIENCE")

internship_summary = df.groupby("internship_experience").agg(
    candidate_count=("placed_binary", "size"),
    placed_count=("placed_binary", "sum"),
    placement_rate=("placed_binary", "mean")
)

internship_summary["placement_rate"] *= 100

print(internship_summary.round(2))

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)
