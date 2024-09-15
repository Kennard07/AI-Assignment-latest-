import streamlit as st
import pandas as pd
from joblib import load

# Load the models and preprocessing steps
heart_failure_model = load('heart_failure_model.joblib')

# Extract models and preprocessing steps
knn = heart_failure_model['knn']
scaler = heart_failure_model['scaler']
label_encoder = heart_failure_model['label_encoder']
poly = heart_failure_model['poly']
feature_selector = heart_failure_model['feature_selector']

# Set the Streamlit page configuration
st.set_page_config(page_title='Heart Failure Prediction', page_icon='❤️', layout='wide')

# Simplified custom styles using markdown for background color, header color, and input field padding
st.markdown("""
    <style>
    /* General page styling */
    body {
        background-color: #f0f8ff;
    }

    h1, h2, h3 {
        color: #003366;
        text-align: center;
        font-weight: 600;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #e6f2ff;
    }

    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }

    /* Input fields styling */
    .stNumberInput input, .stSelectbox select {
        padding: 8px;
        font-size: 14px;
    }

    </style>
    """, unsafe_allow_html=True)

# Sidebar for app description or instructions
st.sidebar.header("About")
st.sidebar.write("""
    This app predicts the likelihood of heart failure based on user inputs such as age, resting blood pressure, cholesterol, maximum heart rate, and ECG results. 
    The prediction is based on a K-Nearest Neighbors (KNN) model.
""")

# Main container for content
with st.container():
    # Main title and description
    st.title('Heart Failure Prediction System 💓')
    st.write("""
        Enter the patient information to get a prediction of heart failure risk using machine learning.
    """)

    # Divider for visual separation
    st.divider()

    # Input section layout
    st.subheader('Enter Patient Information')
    col1, col2 = st.columns(2)  # Two-column layout

    with col1:
        age = st.number_input('Age:', min_value=0, max_value=120, step=1, help="Enter the age of the patient.")
        resting_bp = st.number_input('Resting Blood Pressure (mm Hg):', min_value=0, step=1, help="Enter the resting blood pressure.")
        cholesterol = st.number_input('Cholesterol Level (mg/dl):', min_value=0, step=1, help="Enter the cholesterol level.")

    with col2:
        max_hr = st.number_input('Maximum Heart Rate (bpm):', min_value=0, step=1, help="Enter the maximum heart rate.")
        resting_ecg = st.selectbox('Resting ECG:', ['Normal', 'ST', 'LVH'], help="Select the type of resting ECG.")

    # Divider for visual separation
    st.divider()

    # Data processing and prediction logic
    input_df = pd.DataFrame({
        'Age': [age],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'MaxHR': [max_hr],
        'RestingECG': [resting_ecg]
    })

    if resting_ecg not in label_encoder.classes_:
        st.error(f"Unknown value '{resting_ecg}' for RestingECG. Please select from {label_encoder.classes_}.")
    else:
        # Preprocessing
        input_df['RestingECG'] = label_encoder.transform(input_df['RestingECG'])
        input_df_scaled = scaler.transform(input_df)
        input_df_poly = poly.transform(input_df_scaled)
        input_df_selected = feature_selector.transform(input_df_poly)

        # Predict button
        if st.button('Predict Heart Failure'):
            y_pred = knn.predict(input_df_selected)
            y_prob = knn.predict_proba(input_df_selected)[:, 1]  # Get probability of heart failure
            probability = y_prob[0] * 100  # Convert to percentage
            heart_failure = "Yes" if y_pred[0] == 1 else "No"

            # Display entered details
            st.subheader("Entered Details")
            st.write(f"- **Age:** {age}")
            st.write(f"- **Resting Blood Pressure:** {resting_bp} mm Hg")
            st.write(f"- **Cholesterol Level:** {cholesterol} mg/dl")
            st.write(f"- **Maximum Heart Rate:** {max_hr} bpm")
            st.write(f"- **Resting ECG:** {resting_ecg}")

            # Display prediction results with color-coded message
            st.subheader("Prediction Results")
            st.write(f"- **Heart Failure Likelihood:** {heart_failure}")
            st.write(f"- **Predicted Probability of Heart Failure:** {probability:.2f}%")

            if y_pred[0] == 1:
                st.error("⚠️ High risk of heart failure. Please consult a healthcare professional.")
            else:
                st.success("✅ Low risk of heart failure. Maintain a healthy lifestyle!")

# Footer with additional resources or credits
st.sidebar.markdown("---")
st.sidebar.write("### Useful Resources")
st.sidebar.write("- [Heart Failure Information](https://www.heart.org/en/health-topics/heart-failure)")
st.sidebar.write("- [KNN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)")

# Credits (No explicit footer as Streamlit doesn't allow custom footers)
st.markdown("Powered by Streamlit | Heart Failure Prediction System © 2024", unsafe_allow_html=True)
