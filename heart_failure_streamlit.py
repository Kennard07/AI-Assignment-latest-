import streamlit as st
import pandas as pd
from joblib import load

# Set page configuration
st.set_page_config(page_title='Heart Failure Prediction', page_icon='❤️', layout='wide')

# Load the models and preprocessing steps
heart_failure_model = load('heart_failure_model.joblib')
knn = heart_failure_model['knn']
scaler = heart_failure_model['scaler']
label_encoder = heart_failure_model['label_encoder']
poly = heart_failure_model['poly']
feature_selector = heart_failure_model['feature_selector']

# Apply custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Background color for the page */
    .css-1n76uvr {
        background-color: #f4f5f7;
        font-family: 'Arial', sans-serif;
    }

    /* Header styling */
    h1, h2, h3 {
        text-align: center;
        color: #003366;
        font-weight: 700;
    }

    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 12px;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }

    /* Input box styling */
    input[type=number], select {
        border: 2px solid #003366;
        border-radius: 6px;
        padding: 10px;
        font-size: 16px;
    }

    /* Container for the sidebar and main area */
    .stSidebar {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
    }

    /* Success and error box styling */
    .stSuccess, .stError {
        border-radius: 8px;
        padding: 15px;
        font-weight: bold;
    }

    /* Footer styling */
    footer {
        text-align: center;
        color: gray;
        margin-top: 20px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header section
st.title('Heart Failure Prediction 💓')
st.markdown("""
    Welcome to the **Heart Failure Prediction System**. This application uses a machine learning model to predict the likelihood of heart failure based on the information you provide.
    Please enter your data in the form below to receive a prediction.
""")

# Sidebar for explanation or additional information
with st.sidebar:
    st.header('Instructions')
    st.write("""
        - Please provide all the required details.
        - Click on 'Predict Heart Failure' for the results.
        - The app uses a **KNN Model** to make predictions.
    """)
    st.image('https://www.heart.org/-/media/images/heart/logos/aha-heart-failure.png', caption='Heart Failure Info')

# Main user input section
st.subheader('Patient Information')
col1, col2 = st.columns(2)  # Two-column layout

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, step=1, help="Enter the patient's age.")
    resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, step=1, help="Enter the resting blood pressure.")
    cholesterol = st.number_input('Cholesterol Level (mg/dl)', min_value=0, step=1, help="Enter cholesterol level.")

with col2:
    max_hr = st.number_input('Maximum Heart Rate (bpm)', min_value=0, step=1, help="Enter the maximum heart rate.")
    resting_ecg = st.selectbox('Resting ECG:', ['Normal', 'ST', 'LVH'], help="Select the ECG type.")

# Preprocess the data and handle the prediction
input_data = pd.DataFrame({
    'Age': [age],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'MaxHR': [max_hr],
    'RestingECG': [resting_ecg]
})

if resting_ecg not in label_encoder.classes_:
    st.error(f"Unknown value '{resting_ecg}' for RestingECG. Please select from {label_encoder.classes_}.")
else:
    input_data['RestingECG'] = label_encoder.transform(input_data['RestingECG'])
    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)
    input_selected = feature_selector.transform(input_poly)

    # Prediction button
    if st.button('Predict Heart Failure'):
        y_pred = knn.predict(input_selected)
        y_prob = knn.predict_proba(input_selected)[:, 1]
        probability = y_prob[0] * 100  # Convert to percentage
        result = "Yes" if y_pred[0] == 1 else "No"

        # Display results with interactive feedback
        st.subheader("Prediction Results")
        st.write(f"**Likelihood of Heart Failure:** {result}")
        st.write(f"**Probability:** {probability:.2f}%")
        
        if y_pred[0] == 1:
            st.error("⚠️ High risk of heart failure. Please consult a doctor.")
        else:
            st.success("✅ Low risk of heart failure. Continue maintaining a healthy lifestyle!")

# Footer
st.markdown("<footer>Powered by Streamlit | Heart Failure Prediction Model</footer>", unsafe_allow_html=True)
