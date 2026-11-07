import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# JOB PLACEMENT PREDICTION SYSTEM
# ALL ANALYST TASKS IN ONE PYTHON FILE
# ==========================================================

# 1. LOAD DATASET

df = pd.read_csv("master_job_placement_dataset.csv")

output_folder = Path("analyst_analysis_outputs")
output_folder.mkdir(exist_ok=True)

df["placed_binary"] = (df["status"] == "Placed").astype(int)

print("=" * 60)
print("JOB PLACEMENT ANALYST TASKS")
print("=" * 60)

print("Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())


# ==========================================================
# TASK 1: ACADEMIC SCORES VS PLACEMENT OUTCOME
# ==========================================================

print("\nTASK 1: ACADEMIC SCORES VS PLACEMENT")

academic_columns = [
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "academic_average"
]

academic_summary = df.groupby("status")[academic_columns].mean()

print(academic_summary.round(2))

academic_summary.to_csv(
    output_folder / "01_academic_analysis.csv"
)

academic_summary.plot(kind="bar", figsize=(9, 5))

plt.title("Academic Scores vs Placement Outcome")
plt.xlabel("Placement Status")
plt.ylabel("Average Academic Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "01_academic_analysis.png")
plt.close()


# ==========================================================
# TASK 2: SKILLS MATCH VS INTERVIEW PERFORMANCE
# ==========================================================

print("\nTASK 2: SKILLS MATCH VS ASSESSMENT PERFORMANCE")

df["skills_match_band"] = pd.cut(
    df["skills_match_percentage"],
    bins=[0, 60, 80, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

skills_summary = df.groupby(
    "skills_match_band",
    observed=False
)["combined_assessment_score"].agg(
    ["count", "mean"]
)

print(skills_summary.round(2))

skills_summary.to_csv(
    output_folder / "02_skills_analysis.csv"
)

skills_summary["mean"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Skills Match vs Combined Assessment Score")
plt.xlabel("Skills Match Level")
plt.ylabel("Average Assessment Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "02_skills_analysis.png")
plt.close()


# ==========================================================
# TASK 3: CERTIFICATION IMPACT ON PLACEMENT
# ==========================================================

print("\nTASK 3: CERTIFICATION IMPACT")

df["certification_group"] = df["certifications_count"].apply(
    lambda x: "4+" if x >= 4 else str(int(x))
)

certification_summary = df.groupby(
    "certification_group"
)["placed_binary"].agg(
    ["count", "mean"]
)

certification_summary["placement_rate_percentage"] = (
    certification_summary["mean"] * 100
)

print(certification_summary.round(2))

certification_summary.to_csv(
    output_folder / "03_certification_analysis.csv"
)

certification_summary["placement_rate_percentage"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Certification Impact on Placement")
plt.xlabel("Number of Certifications")
plt.ylabel("Placement Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "03_certification_analysis.png")
plt.close()


# ==========================================================
# TASK 4: COMPANY TIER VS PLACEMENT RATE
# ==========================================================

print("\nTASK 4: COMPANY TIER VS PLACEMENT RATE")

company_summary = df.groupby(
    "company_tier"
)["placed_binary"].agg(
    ["count", "mean"]
)

company_summary["placement_rate_percentage"] = (
    company_summary["mean"] * 100
)

print(company_summary.round(2))

company_summary.to_csv(
    output_folder / "04_company_tier_analysis.csv"
)

company_summary["placement_rate_percentage"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Placement Rate by Company Tier")
plt.xlabel("Company Tier")
plt.ylabel("Placement Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "04_company_tier_analysis.png")
plt.close()


# ==========================================================
# TASK 5: EXPERIENCE VS PLACEMENT SUCCESS
# ==========================================================

print("\nTASK 5: EXPERIENCE VS PLACEMENT")

df["experience_group"] = pd.cut(
    df["years_of_experience"],
    bins=[-0.001, 0, 2, float("inf")],
    labels=["Fresher", "Junior", "Senior"],
    include_lowest=True
)

experience_summary = df.groupby(
    "experience_group",
    observed=False
)["placed_binary"].agg(
    ["count", "mean"]
)

experience_summary["placement_rate_percentage"] = (
    experience_summary["mean"] * 100
)

print(experience_summary.round(2))

experience_summary.to_csv(
    output_folder / "05_experience_analysis.csv"
)

experience_summary["placement_rate_percentage"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Experience vs Placement Success")
plt.xlabel("Experience Category")
plt.ylabel("Placement Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "05_experience_analysis.png")
plt.close()


# ==========================================================
# TASK 6: INTERVIEW SCORE VS PLACEMENT PROBABILITY
# ==========================================================

print("\nTASK 6: ASSESSMENT SCORE VS PLACEMENT PROBABILITY")

correlation = df[
    ["combined_assessment_score", "placement_probability_score"]
].corr().iloc[0, 1]

print("Correlation:", round(correlation, 4))

df["assessment_band"] = pd.cut(
    df["combined_assessment_score"],
    bins=[0, 60, 75, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

probability_summary = df.groupby(
    "assessment_band",
    observed=False
)["placement_probability_score"].agg(
    ["count", "mean"]
)

print(probability_summary.round(2))

probability_summary.to_csv(
    output_folder / "06_probability_analysis.csv"
)

plot_data = df.sample(
    n=min(3000, len(df)),
    random_state=42
)

plt.figure(figsize=(8, 5))

plt.scatter(
    plot_data["combined_assessment_score"],
    plot_data["placement_probability_score"],
    alpha=0.3
)

plt.title("Assessment Score vs Placement Probability")
plt.xlabel("Combined Assessment Score")
plt.ylabel("Placement Probability (%)")
plt.tight_layout()
plt.savefig(output_folder / "06_probability_analysis.png")
plt.close()


# ==========================================================
# TASK 7: EMPLOYABILITY TEST SCORE ANALYSIS
# ==========================================================

print("\nTASK 7: EMPLOYABILITY SCORE ANALYSIS")

employability_columns = [
    "technical_score",
    "aptitude_score",
    "communication_score"
]

employability_summary = df.groupby(
    "status"
)[employability_columns].mean()

print(employability_summary.round(2))

employability_summary.to_csv(
    output_folder / "07_employability_analysis.csv"
)

employability_summary.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Employability Score Components vs Placement")
plt.xlabel("Placement Status")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "07_employability_analysis.png")
plt.close()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("ALL ANALYST TASKS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Total Candidates:", len(df))
print("Placed Candidates:", (df["status"] == "Placed").sum())
print("Not Placed Candidates:", (df["status"] == "Not Placed").sum())

print("\nAll analysis tables and charts saved in:")
print(output_folder.resolve())

print("\nIMPORTANT:")
print("Placement status is used, not independently verified offer acceptance.")
print("Combined assessment score is a proxy, not an official interview score.")
print("Technical, aptitude and communication scores are available")
print("assessment components, not a separately recorded employability test.")