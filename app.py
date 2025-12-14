import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Student Grade Predictor",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("models/student_grade_model.pkl")
preprocess = model.named_steps["preprocess"]
regressor = model.named_steps["model"]

explainer = shap.TreeExplainer(regressor)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🎓 Student Final Grade Predictor</h1>",
    unsafe_allow_html=True
)
st.write(
    "Enter basic student information to predict the **final grade (G3)**."
)

# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------
st.markdown("### 📝 Student Information")

col1, col2 = st.columns(2)

with col1:
    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)

with col2:
    studytime = st.slider("Study Time (1–4)", 1, 4, 2)
    failures = st.slider("Past Failures", 0, 4, 0)
    absences = st.slider("Absences", 0, 50, 2)

# -------------------------------------------------
# REQUIRED COLUMNS + DEFAULTS
# -------------------------------------------------
required_cols = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
    'traveltime', 'studytime', 'failures', 'schoolsup',
    'famsup', 'paid', 'activities', 'nursery', 'higher',
    'internet', 'romantic', 'famrel', 'freetime', 'goout',
    'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3'
]

defaults = {
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 2,
    "Fedu": 2,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "schoolsup": "no",
    "famsup": "no",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 3,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "G1": 10,
    "G2": 10,
    "G3": 10
}

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
center = st.columns([1, 2, 1])
with center[1]:
    predict_btn = st.button("🎯 Predict Grade", use_container_width=True)

# -------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------
if predict_btn:

    input_df = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences
    }])

    for col in required_cols:
        if col not in input_df:
            input_df[col] = defaults[col]

    input_df = input_df[required_cols]

    X_trans = preprocess.transform(input_df)
    prediction = float(regressor.predict(X_trans)[0])
    prediction = max(0, min(20, round(prediction, 2)))

    st.session_state["input_df"] = input_df
    st.session_state["prediction"] = prediction

    st.success(f"### 🎉 Predicted Final Grade: **{prediction} / 20**")

# -------------------------------------------------
# RESULTS TABS
# -------------------------------------------------
if "prediction" in st.session_state:

    tab1, tab2 = st.tabs([
        "📈 Prediction Explanation",
        "🔥 Feature Importance"
    ])

    # ---------------- TAB 1: SHAP -----------------
    with tab1:
        st.subheader("📈 Why did the model predict this grade?")

        X_trans = preprocess.transform(st.session_state["input_df"])
        shap_values = explainer.shap_values(X_trans)
        feature_names = preprocess.get_feature_names_out()

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0]
        })

        shap_df["Impact"] = shap_df["SHAP Value"].abs()
        top_shap = shap_df.sort_values(
            "Impact", ascending=False
        ).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            top_shap["Feature"],
            top_shap["SHAP Value"],
            color="#6BB4FF",
            edgecolor="black"
        )

        ax.set_title(
            "Top 10 Factors Influencing This Prediction",
            fontsize=14,
            weight="bold"
        )
        ax.set_xlabel("Impact on Grade")
        ax.axvline(0, color="gray", linewidth=1)
        ax.invert_yaxis()
        plt.tight_layout()

        st.pyplot(fig)

        st.caption(
            "🔹 Positive values increase the grade\n"
            "🔹 Negative values decrease the grade"
        )

    # ---------------- TAB 2: FEATURE IMPORTANCE ----
    with tab2:
        st.subheader("🔥 Top 10 Most Important Features")

        fi_df = pd.DataFrame({
            "Feature": preprocess.get_feature_names_out(),
            "Importance": regressor.feature_importances_
        }).sort_values("Importance", ascending=False)

        top10 = fi_df.head(10)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(
            top10["Feature"],
            top10["Importance"],
            color="#4C78A8"
        )

        ax.set_title(
            "Top 10 Feature Importances",
            fontsize=15,
            weight="bold"
        )
        ax.set_xlabel("Importance Score")
        ax.invert_yaxis()

        for bar in bars:
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}",
                va="center",
                fontsize=10
            )

        plt.tight_layout()
        st.pyplot(fig)
