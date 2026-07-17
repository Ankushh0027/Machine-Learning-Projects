import streamlit as st
import pandas as pd
import joblib
import os

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ------------- Load Model -----------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "models", "student_performance_model.pkl")
preprocessor_path = os.path.join(current_dir, "models", "preprocessor.pkl")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


# -------------------- Title --------------------
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details to predict the Exam Score.")

st.markdown("---")

col1, col2 = st.columns(2)

# -------------------- Column 1 --------------------
with col1:

    hours_studied = st.number_input("Hours Studied", 0, 50, 10)

    attendance = st.number_input("Attendance (%)", 0, 100, 80)

    previous_scores = st.number_input("Previous Scores", 0, 100, 70)

    sleep_hours = st.number_input("Sleep Hours", 0, 12, 7)

    tutoring_sessions = st.number_input("Tutoring Sessions", 0, 20, 2)

    physical_activity = st.number_input("Physical Activity", 0, 20, 3)

    parental_involvement = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"]
    )

    access_to_resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"]
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    motivation = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"]
    )

# -------------------- Column 2 --------------------
with col2:

    internet = st.selectbox(
        "Internet Access",
        ["Yes", "No"]
    )

    family_income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )

    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"]
    )

    school_type = st.selectbox(
        "School Type",
        ["Public", "Private"]
    )

    peer = st.selectbox(
        "Peer Influence",
        ["Positive", "Neutral", "Negative"]
    )

    learning = st.selectbox(
        "Learning Disabilities",
        ["Yes", "No"]
    )

    parent_education = st.selectbox(
        "Parental Education Level",
        ["High School", "College", "Postgraduate"]
    )

    distance = st.selectbox(
        "Distance from Home",
        ["Near", "Moderate", "Far"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

st.markdown("---")

# -------------------- Prediction --------------------
if st.button("🎯 Predict Exam Score"):

    input_df = pd.DataFrame({

        "Hours_Studied":[hours_studied],
        "Attendance":[attendance],
        "Parental_Involvement":[parental_involvement],
        "Access_to_Resources":[access_to_resources],
        "Extracurricular_Activities":[extracurricular],
        "Sleep_Hours":[sleep_hours],
        "Previous_Scores":[previous_scores],
        "Motivation_Level":[motivation],
        "Internet_Access":[internet],
        "Tutoring_Sessions":[tutoring_sessions],
        "Family_Income":[family_income],
        "Teacher_Quality":[teacher_quality],
        "School_Type":[school_type],
        "Peer_Influence":[peer],
        "Physical_Activity":[physical_activity],
        "Learning_Disabilities":[learning],
        "Parental_Education_Level":[parent_education],
        "Distance_from_Home":[distance],
        "Gender":[gender]

    })

    processed_data = preprocessor.transform(input_df)

    prediction = model.predict(processed_data)

    score = prediction[0]

    st.success(f"### 🎯 Predicted Exam Score : {score:.2f}")

    if score >= 90:
        st.balloons()
        st.success("🌟 Performance : Excellent")

    elif score >= 75:
        st.info("✅ Performance : Good")

    elif score >= 60:
        st.warning("🙂 Performance : Average")

    else:
        st.error("⚠ Needs Improvement")

st.markdown("---")
st.caption("Developed by Ankush | Machine Learning Project | Streamlit")