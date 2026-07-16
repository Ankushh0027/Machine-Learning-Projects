import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict the exam score.")

# Model aur preprocessor load karo
model = joblib.load("models/student_performance_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Academic Details')
    Hours_Studied = st.number_input("Hours Studied", min_value=0, max_value=50, value=20)
    Attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
    Previous_Scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
    Tutoring_Sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=1)
    Sleep_Hours = st.number_input("Sleep Hours", min_value=0, max_value=12, value=7)
    Physical_Activity = st.number_input("Physical Activity Hours/Week", min_value=0, max_value=15, value=3)

with col2:
    st.subheader('Support & Resources')
    Parental_Involvement = st.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
    Access_to_Resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
    Internet_Access = st.selectbox("Internet Access", ['Yes', 'No'])
    Family_Income = st.selectbox("Family Income", ['Low', 'Medium', 'High'])
    Teacher_Quality = st.selectbox("Teacher Quality", ['Low', 'Medium', 'High'])
    School_Type = st.selectbox("School Type", ['Public', 'Private'])
    Parental_Education_Level = st.selectbox("Parental Education Level", ['High School', 'College', 'Postgraduate'])

with col3:
    st.subheader('Personal Details')
    Extracurricular_Activities = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
    Motivation_Level = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
    Peer_Influence = st.selectbox("Peer Influence", ['Negative', 'Neutral', 'Positive'])
    Learning_Disabilities = st.selectbox("Learning Disabilities", ['Yes', 'No'])
    Distance_from_Home = st.selectbox("Distance from Home", ['Near', 'Moderate', 'Far'])
    Gender = st.selectbox("Gender", ['Male', 'Female'])

if st.button('Predict Exam Score', type="primary"):
    # DataFrame banao - order training wala hi rakhna
    input_df = pd.DataFrame({
        'Hours_Studied': [Hours_Studied],
        'Attendance': [Attendance],
        'Parental_Involvement': [Parental_Involvement],
        'Access_to_Resources': [Access_to_Resources],
        'Extracurricular_Activities': [Extracurricular_Activities],
        'Sleep_Hours': [Sleep_Hours],
        'Previous_Scores': [Previous_Scores],
        'Motivation_Level': [Motivation_Level],
        'Internet_Access': [Internet_Access],
        'Tutoring_Sessions': [Tutoring_Sessions],
        'Family_Income': [Family_Income],
        'Teacher_Quality': [Teacher_Quality],
        'School_Type': [School_Type],
        'Peer_Influence': [Peer_Influence],
        'Physical_Activity': [Physical_Activity],
        'Learning_Disabilities': [Learning_Disabilities],
        'Parental_Education_Level': [Parental_Education_Level],
        'Distance_from_Home': [Distance_from_Home],
        'Gender': [Gender]
    })
    
    # Preprocessing apply karo jo training me kiya tha
    input_processed = preprocessor.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_processed)
    st.success(f'Predicted Exam Score: {prediction[0]:.2f} / 100')
    
    if prediction[0] >= 80:
        st.balloons()
        st.info('Excellent! Student likely to perform very well.')
    elif prediction[0] >= 60:
        st.info('Good! Student should pass with decent marks.')
    else:
        st.warning('Needs improvement. Consider more study hours or tutoring.')