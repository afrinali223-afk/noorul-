
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("feature_engineered.csv")

print("=" * 65)
print("JOB PLACEMENT PREDICTION - FIVE MODEL COMPARISON")
print("=" * 65)

print("\nDataset shape:", df.shape)

# Target column
target = "status"

# Remove target from input features
X = df.drop(columns=[target]).copy()
y = df[target].copy()

print("\nUnique target values:", y.unique())

if y.isnull().any():
    raise ValueError("Target column contains missing values.")

if set(y.unique()) != {"Placed", "Not Placed"}:
    raise ValueError("Unexpected values found in status column.")

# Exclude redundant derived columns whose source columns are already
# included. This keeps the feature representation more consistent.
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

# IMPORTANT:
# Do not include identifiers or information that directly reveals
# the outcome, if such columns are added to the dataset in future.

print("Target:", target)
print("Input features:", X.shape[1])

print("\nFeatures used for model training:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# ============================================================
# 3. PREPROCESSING
# Fit preprocessing separately within each training fold
# using a Pipeline, avoiding leakage from test data.
# ============================================================

numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# ============================================================
# 4. DEFINE FIVE CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        n_jobs=-1
    )
}

# ============================================================
# 5. CROSS-VALIDATION COMPARISON
# Select using macro F1, which considers both target classes.
# The test set is not used for model selection.
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

results = []
pipelines = {}

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipelines[name] = pipeline

    print(f"\nCross-validating: {name}")

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    mean_score = scores.mean()

    results.append({
        "Model": name,
        "CV F1 Macro Mean": mean_score,
        "CV F1 Macro Std": scores.std()
    })

    print("Fold scores:", np.round(scores, 4))
    print("Mean F1 macro:", round(mean_score, 4))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="CV F1 Macro Mean",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 65)
print("FIVE MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

# Save the comparison table
results_df.to_csv("model_comparison.csv", index=False)

# Select the best model by cross-validation score
best_name = results_df.loc[0, "Model"]
best_model = pipelines[best_name]

print("\nSelected model:", best_name)
print("Selection basis: highest mean cross-validation macro F1")

# ============================================================
# 6. TRAIN SELECTED MODEL ON FULL TRAINING SET
# ============================================================

best_model.fit(X_train, y_train)

# ============================================================
# 7. EVALUATE ON HELD-OUT TEST SET
# ============================================================

y_pred = best_model.predict(X_test)

placed_index = list(best_model.classes_).index("Placed")
y_prob = best_model.predict_proba(X_test)[:, placed_index]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, pos_label="Placed", zero_division=0
)
recall = recall_score(
    y_test, y_pred, pos_label="Placed", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, pos_label="Placed", zero_division=0
)
macro_f1 = f1_score(y_test, y_pred, average="macro")
roc_auc = roc_auc_score(
    (y_test == "Placed").astype(int),
    y_prob
)

print("\n" + "=" * 65)
print("BEST MODEL - TEST SET RESULTS")
print("=" * 65)

print("Model:", best_name)
print(f"Accuracy:             {accuracy:.4f}")
print(f"Precision (Placed):   {precision:.4f}")
print(f"Recall (Placed):      {recall:.4f}")
print(f"F1-score (Placed):    {f1:.4f}")
print(f"Macro F1-score:       {macro_f1:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["Not Placed", "Placed"],
    zero_division=0
))

print("Confusion matrix (rows = actual, columns = predicted)")
print("Order: Not Placed, Placed")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Not Placed", "Placed"]
))

# ============================================================
# 8. SAVE BEST MODEL AND TEST METRICS
# ============================================================

joblib.dump(best_model, "best_placement_model.pkl")

test_metrics = pd.DataFrame([{
    "Selected Model": best_name,
    "Accuracy": accuracy,
    "Precision Placed": precision,
    "Recall Placed": recall,
    "F1 Placed": f1,
    "Macro F1": macro_f1,
    "ROC AUC": roc_auc
}])

test_metrics.to_csv("best_model_test_metrics.csv", index=False)

print("\nSaved files:")
print("1. model_comparison.csv")
print("2. best_model_test_metrics.csv")
print("3. best_placement_model.pkl")
print("\nModel training and evaluation completed.")
