
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ============================================================
# STEP 1: LOAD DATA AND TRAINED MODEL
# ============================================================

df = pd.read_csv("feature_engineered.csv")
best_model = joblib.load("best_placement_model.pkl")

print("=" * 60)
print("PLACEMENT PROBABILITY SCORE GENERATION")
print("=" * 60)

# Separate input features and target
X = df.drop(columns=["status"]).copy()
y = df["status"].copy()

# Use the same feature exclusions as the training script
redundant_columns = [
    "academic_average",
    "combined_assessment_score",
    "experience_category",
    "academic_performance_band",
    "skills_match_level",
    "interview_performance_category"
]

X = X.drop(
    columns=[c for c in redundant_columns if c in X.columns]
)

# ============================================================
# STEP 2: GENERATE OUT-OF-FOLD PROBABILITIES
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating out-of-fold predictions for all candidates.")
print("This may take a few minutes...")

probabilities = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)

# Identify the probability column for the Placed class
placed_index = list(best_model.classes_).index("Placed")

df["placement_probability_score"] = (
    probabilities[:, placed_index] * 100
).round(2)

# ============================================================
# STEP 3: VALIDATE SCORES
# ============================================================

print("\nProbability score validation:")
print("Missing scores:", df["placement_probability_score"].isna().sum())
print("Minimum score:", df["placement_probability_score"].min())
print("Maximum score:", df["placement_probability_score"].max())
print("Average score:", round(
    df["placement_probability_score"].mean(), 2
))

# ============================================================
# STEP 4: DISPLAY SAMPLE RESULTS
# ============================================================

print("\nSample candidate scores:")

print(
    df[
        [
            "status",
            "placement_probability_score"
        ]
    ].head(10).to_string(index=False)
)

# ============================================================
# STEP 5: SAVE OUTPUT
# ============================================================

df.to_csv("placement_probability_scores.csv", index=False)

print("\nSaved as placement_probability_scores.csv")
print("Placement probability score generation completed.")

