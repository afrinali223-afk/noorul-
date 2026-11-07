
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Job Placement Prediction System",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "master_job_placement_dataset.csv"
MODEL_FILE = BASE_DIR / "best_placement_model.pkl"

# ==========================================================
# 2. LIGHTWEIGHT CACHED DATA LOADING
# ==========================================================

@st.cache_data(show_spinner="Loading candidate dataset...")
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    data["status"] = data["status"].astype(str).str.strip().str.title()
    return data


@st.cache_resource(show_spinner="Loading trained ML model...")
def load_model(model_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(model_path)

    return model, [str(w.message) for w in caught]


# ==========================================================
# 3. FILE VALIDATION
# ==========================================================

if not DATA_FILE.exists():
    st.error("Master dataset CSV not found in the project folder.")
    st.stop()

try:
    df = load_data(str(DATA_FILE))
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


required_columns = [
    "status",
    "ssc_percentage",
    "hsc_percentage",
    "degree_percentage",
    "skills_match_percentage",
    "certifications_count",
    "company_tier",
    "years_of_experience",
    "combined_assessment_score",
    "placement_probability_score",
    "technical_score",
    "aptitude_score",
    "communication_score"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


# ==========================================================
# 4. PREPARE DATA
# ==========================================================

df["placed_binary"] = (
    df["status"].str.lower() == "placed"
).astype(int)

df["skills_match_band"] = pd.cut(
    df["skills_match_percentage"],
    bins=[0, 60, 80, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

df["certification_group"] = df["certifications_count"].apply(
    lambda x: "4+" if x >= 4 else str(int(x))
)

df["experience_group"] = pd.cut(
    df["years_of_experience"],
    bins=[-0.001, 0, 2, float("inf")],
    labels=["Fresher", "Junior", "Senior"],
    include_lowest=True
)

df["assessment_band"] = pd.cut(
    df["combined_assessment_score"],
    bins=[0, 60, 75, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# ==========================================================
# 5. SIDEBAR
# ==========================================================

st.sidebar.title("📊 Dashboard Navigation")

st.sidebar.success("Dataset loaded successfully")

st.sidebar.write("Total records:", f"{len(df):,}")

risk_threshold = st.sidebar.slider(
    "High-risk probability threshold (%)",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)

st.sidebar.caption(
    "Candidates below this model-generated placement "
    "probability are classified as high risk."
)

st.sidebar.divider()

st.sidebar.caption(
    "Job Placement Prediction System"
)


# ==========================================================
# 6. DASHBOARD HEADER
# ==========================================================

st.title("📊 JOB PLACEMENT PREDICTION SYSTEM")

st.markdown(
    "#### Candidate Analysis | Placement Insights | "
    "Machine Learning Prediction"
)

st.caption(
    "An interactive data analytics dashboard using "
    "Python, Machine Learning and Streamlit."
)

st.divider()


# ==========================================================
# 7. CALCULATE KEY KPIs
# ==========================================================

total_candidates = len(df)

placed_candidates = int(df["placed_binary"].sum())

not_placed_candidates = (
    total_candidates - placed_candidates
)

placement_rate = (
    placed_candidates / total_candidates * 100
    if total_candidates > 0 else 0
)

average_skills_match = df[
    "skills_match_percentage"
].mean()

# Combined assessment score is used as a proxy.
average_interview_score_proxy = df[
    "combined_assessment_score"
].mean()

high_risk_candidates = (
    df["placement_probability_score"] < risk_threshold
).sum()

high_risk_percentage = (
    high_risk_candidates / total_candidates * 100
    if total_candidates > 0 else 0
)


# ==========================================================
# 8. KEY KPI CARDS
# ==========================================================

st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Candidates",
    f"{total_candidates:,}"
)

k2.metric(
    "Placement Rate (%)",
    f"{placement_rate:.2f}%"
)

k3.metric(
    "Job Acceptance Rate (%)",
    "N/A"
)

k4.metric(
    "Average Interview Score",
    f"{average_interview_score_proxy:.2f}"
)

k5, k6, k7 = st.columns(3)

k5.metric(
    "Average Skills Match %",
    f"{average_skills_match:.2f}%"
)

k6.metric(
    "Offer Dropout Rate",
    "N/A"
)

k7.metric(
    "High-Risk Candidate %",
    f"{high_risk_percentage:.2f}%"
)

st.caption(
    "Interview score is represented by the combined assessment "
    "score proxy. Job acceptance and offer dropout are unavailable "
    "because the dataset does not independently record these events. "
    f"High-risk threshold: placement probability below "
    f"{risk_threshold}%."
)

st.divider()


# ==========================================================
# 9. DASHBOARD TABS
# ==========================================================

tabs = st.tabs([
    "Overview",
    "Academic Analysis",
    "Skills Analysis",
    "Certifications",
    "Company Tier",
    "Experience",
    "Probability Analysis",
    "Assessment Analysis",
    "ML Prediction"
])


# ==========================================================
# TAB 1: OVERVIEW
# ==========================================================

with tabs[0]:

    st.subheader("Candidate Placement Overview")

    c1, c2 = st.columns(2)

    with c1:

        st.write("### Placement Distribution")

        distribution = pd.DataFrame({
            "Candidates": [
                placed_candidates,
                not_placed_candidates
            ]
        }, index=["Placed", "Not Placed"])

        st.bar_chart(distribution)

    with c2:

        st.write("### Placement Summary")

        summary = pd.DataFrame({
            "Category": [
                "Total Candidates",
                "Placed",
                "Not Placed"
            ],
            "Count": [
                total_candidates,
                placed_candidates,
                not_placed_candidates
            ]
        })

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

    st.write("### Dataset Information")

    st.write("Dataset shape:", df.shape)

    st.write(
        "Missing values:",
        int(df.isnull().sum().sum())
    )


# ==========================================================
# TAB 2: TASK 1 - ACADEMIC SCORES
# ==========================================================

with tabs[1]:

    st.subheader("1. Academic Scores vs Placement")

    academic_columns = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage"
    ]

    if "academic_average" in df.columns:
        academic_columns.append("academic_average")

    academic_analysis = (
        df.groupby("status")[academic_columns]
        .mean()
        .reindex(["Not Placed", "Placed"])
        .round(2)
    )

    st.write("Average Academic Scores")

    st.dataframe(
        academic_analysis,
        use_container_width=True
    )

    st.write("### Academic Score Comparison")

    st.bar_chart(academic_analysis)

    st.info(
        "Academic averages are almost identical between "
        "placed and not-placed candidates."
    )


# ==========================================================
# TAB 3: TASK 2 - SKILLS MATCH
# ==========================================================

with tabs[2]:

    st.subheader("2. Skills Match vs Assessment Performance")

    skills_analysis = (
        df.groupby("skills_match_band", observed=False)
        ["combined_assessment_score"]
        .agg(["count", "mean"])
        .reindex(["Low", "Medium", "High"])
        .round(2)
    )

    st.dataframe(
        skills_analysis,
        use_container_width=True
    )

    st.write("### Skills Match vs Average Assessment Score")

    st.bar_chart(skills_analysis[["mean"]])

    st.caption(
        "Combined assessment score is used for comparison; "
        "it is not an official interview score."
    )


# ==========================================================
# TAB 4: TASK 3 - CERTIFICATION IMPACT
# ==========================================================

with tabs[3]:

    st.subheader("3. Certification Impact on Placement")

    certification_analysis = (
        df.groupby("certification_group", observed=False)
        .agg(
            count=("placed_binary", "size"),
            placement_rate=("placed_binary", "mean")
        )
    )

    certification_analysis["placement_rate"] *= 100

    certification_analysis = (
        certification_analysis
        .reindex(["0", "1", "2", "3", "4+"])
        .round(2)
    )

    st.dataframe(
        certification_analysis,
        use_container_width=True
    )

    st.write("### Certification Count vs Placement Rate")

    st.bar_chart(
        certification_analysis[["placement_rate"]]
    )

    st.info(
        "Certification count does not show a clear increasing "
        "relationship with placement rate."
    )


# ==========================================================
# TAB 5: TASK 4 - COMPANY TIER
# ==========================================================

with tabs[4]:

    st.subheader("4. Company Tier vs Placement Rate")

    company_analysis = (
        df.groupby("company_tier", observed=False)
        .agg(
            count=("placed_binary", "size"),
            placement_rate=("placed_binary", "mean")
        )
    )

    company_analysis["placement_rate"] *= 100

    company_analysis = (
        company_analysis
        .reindex(["Tier 1", "Tier 2", "Tier 3"])
        .dropna(how="all")
        .round(2)
    )

    st.dataframe(
        company_analysis,
        use_container_width=True
    )

    st.write("### Placement Rate by Company Tier")

    st.bar_chart(
        company_analysis[["placement_rate"]]
    )


# ==========================================================
# TAB 6: TASK 5 - EXPERIENCE
# ==========================================================

with tabs[5]:

    st.subheader("5. Experience vs Placement Success")

    experience_analysis = (
        df.groupby("experience_group", observed=False)
        .agg(
            count=("placed_binary", "size"),
            placement_rate=("placed_binary", "mean")
        )
    )

    experience_analysis["placement_rate"] *= 100

    experience_analysis = (
        experience_analysis
        .reindex(["Fresher", "Junior", "Senior"])
        .round(2)
    )

    st.dataframe(
        experience_analysis,
        use_container_width=True
    )

    st.write("### Experience Group vs Placement Rate")

    st.bar_chart(
        experience_analysis[["placement_rate"]]
    )

    st.success(
        "Senior candidates show a higher placement rate "
        "than junior candidates and freshers."
    )


# ==========================================================
# TAB 7: TASK 6 - PLACEMENT PROBABILITY
# ==========================================================

with tabs[6]:

    st.subheader(
        "6. Assessment Score vs Placement Probability"
    )

    correlation = df[
        [
            "combined_assessment_score",
            "placement_probability_score"
        ]
    ].corr().iloc[0, 1]

    st.metric(
        "Assessment / Probability Correlation",
        f"{correlation:.4f}"
    )

    probability_analysis = (
        df.groupby("assessment_band", observed=False)
        ["placement_probability_score"]
        .agg(["count", "mean"])
        .reindex(["Low", "Medium", "High"])
        .round(2)
    )

    st.write("### Average Placement Probability")

    st.dataframe(
        probability_analysis,
        use_container_width=True
    )

    st.bar_chart(
        probability_analysis[["mean"]]
    )

    st.caption(
        "Probability values are generated by the trained "
        "machine-learning model. They are not verified "
        "job offer acceptance probabilities."
    )


# ==========================================================
# TAB 8: TASK 7 - ASSESSMENT ANALYSIS
# ==========================================================

with tabs[7]:

    st.subheader("7. Employability Assessment Analysis")

    score_columns = [
        "technical_score",
        "aptitude_score",
        "communication_score"
    ]

    score_analysis = (
        df.groupby("status")[score_columns]
        .mean()
        .reindex(["Not Placed", "Placed"])
        .round(2)
    )

    st.write("### Average Assessment Component Scores")

    st.dataframe(
        score_analysis,
        use_container_width=True
    )

    st.bar_chart(score_analysis)

    st.info(
        "Technical, aptitude and communication scores are "
        "the available assessment components. The dataset "
        "does not contain a separate employability test score."
    )


# ==========================================================
# TAB 9: MACHINE LEARNING PREDICTION
# ==========================================================

with tabs[8]:

    st.subheader("🎯 Candidate Placement Prediction")

    st.write(
        "Enter candidate details and click Predict to obtain "
        "a placement prediction from the trained Random Forest model."
    )

    st.warning(
        "The prediction is a model estimate, not a guarantee "
        "of actual placement or job offer acceptance."
    )

    if not MODEL_FILE.exists():

        st.error(
            "best_placement_model.pkl was not found. "
            "Keep the model file in the same folder as app.py."
        )

    else:

        try:

            model, model_warnings = load_model(str(MODEL_FILE))

            import sklearn

            st.caption(
                f"Current scikit-learn version: {sklearn.__version__}"
            )

            for warning_message in model_warnings:
                if "InconsistentVersionWarning" in warning_message:
                    st.warning(
                        "Model was saved using a different "
                        "scikit-learn version. Check compatibility "
                        "before relying on predictions."
                    )

            # Obtain the exact feature names used during training.
            feature_names = getattr(
                model,
                "feature_names_in_",
                None
            )

            if feature_names is None:
                st.error(
                    "Cannot identify the model's original input "
                    "features. Please check the saved model and "
                    "training script before using predictions."
                )

            else:

                feature_names = list(feature_names)

                unavailable_features = [
                    col for col in feature_names
                    if col not in df.columns
                ]

                if unavailable_features:

                    st.error(
                        "These model input features are missing "
                        f"from the master dataset: {unavailable_features}"
                    )

                else:

                    st.write("### Enter Candidate Information")

                    st.caption(
                        "Default values are taken from the dataset. "
                        "You can change them before predicting."
                    )

                    input_values = {}

                    with st.form("candidate_prediction_form"):

                        left, right = st.columns(2)

                        for index, col in enumerate(feature_names):

                            source = df[col]
                            target_column = (
                                left if index % 2 == 0 else right
                            )

                            with target_column:

                                if pd.api.types.is_numeric_dtype(source):

                                    numeric_source = pd.to_numeric(
                                        source,
                                        errors="coerce"
                                    ).dropna()

                                    if numeric_source.empty:
                                        default_value = 0
                                        minimum = 0
                                        maximum = 100

                                    else:
                                        minimum = float(
                                            numeric_source.min()
                                        )
                                        maximum = float(
                                            numeric_source.max()
                                        )
                                        default_value = float(
                                            numeric_source.median()
                                        )

                                    if minimum == maximum:
                                        minimum -= 1
                                        maximum += 1
                                        default_value = minimum + 1

                                    if pd.api.types.is_integer_dtype(source):

                                        input_values[col] = st.number_input(
                                            col.replace("_", " ").title(),
                                            min_value=int(np.floor(minimum)),
                                            max_value=int(np.ceil(maximum)),
                                            value=int(round(default_value)),
                                            step=1,
                                            key=f"prediction_{col}"
                                        )

                                    else:

                                        input_values[col] = st.number_input(
                                            col.replace("_", " ").title(),
                                            min_value=float(minimum),
                                            max_value=float(maximum),
                                            value=float(default_value),
                                            step=0.1,
                                            key=f"prediction_{col}"
                                        )

                                else:

                                    options = (
                                        source.dropna()
                                        .astype(str)
                                        .unique()
                                        .tolist()
                                    )

                                    options = sorted(options)

                                    if not options:
                                        options = ["Unknown"]

                                    default_value = (
                                        source.mode().iloc[0]
                                        if not source.mode().empty
                                        else options[0]
                                    )

                                    default_value = str(default_value)

                                    if default_value not in options:
                                        default_value = options[0]

                                    input_values[col] = st.selectbox(
                                        col.replace("_", " ").title(),
                                        options,
                                        index=options.index(default_value),
                                        key=f"prediction_{col}"
                                    )

                        predict_button = st.form_submit_button(
                            "🔮 Predict Placement",
                            use_container_width=True
                        )

                    if predict_button:

                        try:

                            input_df = pd.DataFrame(
                                [input_values],
                                columns=feature_names
                            )

                            prediction = model.predict(input_df)[0]

                            st.divider()

                            st.subheader("Prediction Result")

                            if str(prediction).strip().lower() == "placed":
                                st.success(
                                    "Prediction: PLACED"
                                )
                            elif str(prediction).strip().lower() == "not placed":
                                st.error(
                                    "Prediction: NOT PLACED"
                                )
                            else:
                                st.info(
                                    f"Predicted class: {prediction}"
                                )

                            if hasattr(model, "predict_proba"):

                                probabilities = model.predict_proba(
                                    input_df
                                )[0]

                                classes = list(model.classes_)

                                st.write(
                                    "### Model Prediction Probabilities"
                                )

                                probability_data = pd.DataFrame({
                                    "Outcome": [
                                        str(c) for c in classes
                                    ],
                                    "Probability (%)": [
                                        round(float(p) * 100, 2)
                                        for p in probabilities
                                    ]
                                })

                                st.dataframe(
                                    probability_data,
                                    hide_index=True,
                                    use_container_width=True
                                )

                                probability_chart = probability_data.set_index(
                                    "Outcome"
                                )

                                st.bar_chart(probability_chart)

                                st.caption(
                                    "These are the model's class probabilities "
                                    "for this candidate."
                                )

                        except Exception as prediction_error:

                            st.error(
                                f"Prediction failed: {prediction_error}"
                            )

        except Exception as model_error:

            st.error(
                f"Model loading failed: {model_error}"
            )

            st.warning(
                "Check the saved model's scikit-learn compatibility "
                "and the Python environment used for training."
            )


# ==========================================================
# 10. PROJECT FINDINGS AND LIMITATIONS
# ==========================================================

st.divider()

with st.expander("Project Findings and Limitations"):

    st.markdown("""
    ### Important Findings

    1. Senior candidates show a higher placement rate than freshers.
    2. Academic score averages are similar for placed and not-placed candidates.
    3. Certification count does not show a clear increasing placement trend.
    4. Company-tier placement rates are relatively close.
    5. Combined assessment score has a positive relationship with
       model-generated placement probability.
    6. Placed candidates have a higher average technical score.

    ### Project Limitations

    - Placement status is available; independently verified job
      acceptance is not available.
    - Combined assessment score is a proxy, not an official interview score.
    - Offer dropout information is not separately recorded.
    - Technical, aptitude and communication scores are available
      assessment components.
    - Correlation does not establish causation.
    """)


# ==========================================================
# 11. LIMITED DATASET PREVIEW
# ==========================================================

with st.expander("View First 10 Dataset Records"):

    st.dataframe(
        df.drop(columns=["placed_binary"], errors="ignore").head(10),
        use_container_width=True
    )


# ==========================================================
# 12. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Job Placement Prediction System | Python | Pandas | "
    "Scikit-learn | Random Forest | Streamlit"
)
