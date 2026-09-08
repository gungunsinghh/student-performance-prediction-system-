import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND ENCODERS
# ============================================================

MODEL_PATH = "models/tuned_logistic_regression.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter the student's information below to predict "
    "the expected performance category."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Student Information")


gender = st.selectbox(
    "Gender",
    ["male", "female"]
)


caste = st.selectbox(
    "Caste",
    ["General", "OBC", "SC", "ST"]
)


coaching = st.selectbox(
    "Coaching",
    ["NO", "WA", "OA"]
)


time = st.selectbox(
    "Time",
    ["ONE", "TWO", "THREE"]
)


class_ten_education = st.selectbox(
    "Class X Education",
    ["SEBA", "AHSEC", "CBSE", "OTHERS"]
)


twelve_education = st.selectbox(
    "Class XII Education",
    ["AHSEC", "CBSE", "OTHERS"]
)


medium = st.selectbox(
    "Medium",
    ["ENGLISH", "OTHERS"]
)


class_x_percentage = st.selectbox(
    "Class X Percentage",
    ["Average", "Good", "Vg", "Excellent"]
)


class_xii_percentage = st.selectbox(
    "Class XII Percentage",
    ["Average", "Good", "Vg", "Excellent"]
)


father_occupation = st.selectbox(
    "Father's Occupation",
    [
        "DOCTOR",
        "BUSINESS",
        "SCHOOL_TEACHER",
        "COLLEGE_TEACHER",
        "CULTIVATOR",
        "HOUSE_WIFE",
        "OTHERS"
    ]
)


mother_occupation = st.selectbox(
    "Mother's Occupation",
    [
        "DOCTOR",
        "BUSINESS",
        "SCHOOL_TEACHER",
        "COLLEGE_TEACHER",
        "CULTIVATOR",
        "HOUSE_WIFE",
        "OTHERS"
    ]
)


# ============================================================
# CREATE RAW INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "Gender": [gender],
    "Caste": [caste],
    "Coaching": [coaching],
    "Time": [time],
    "Class_Ten_Education": [class_ten_education],
    "Twelve_Education": [twelve_education],
    "Medium": [medium],
    "Class_X_Percentage": [class_x_percentage],
    "Class_XII_Percentage": [class_xii_percentage],
    "Father_Occupation": [father_occupation],
    "Mother_Occupation": [mother_occupation]
})


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔮 Predict Performance"):

    # --------------------------------------------------------
    # Apply SAME one-hot encoding used during model training
    # --------------------------------------------------------

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=False
    )


    # --------------------------------------------------------
    # Make columns exactly match training data
    # --------------------------------------------------------

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_encoded)

    predicted_class = label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]


    # --------------------------------------------------------
    # Display Prediction
    # --------------------------------------------------------

    st.success(
        f"Predicted Student Performance: **{predicted_class}**"
    )


    # --------------------------------------------------------
    # Prediction Probabilities
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_encoded
        )[0]

        probability_df = pd.DataFrame({
            "Performance": label_encoder.classes_,
            "Probability": probabilities * 100
        })

        probability_df["Probability"] = (
            probability_df["Probability"].round(2)
        )

        st.subheader("Prediction Probabilities")

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        st.bar_chart(
            probability_df.set_index("Performance")
        )