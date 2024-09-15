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

# Custom CSS for styling the app with black background
st.markdown("""
    <style>
    /* Set black background color */
    .main {
        background-color: #000000; /* Black */
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Title and subtitle styles */
    h1, h2, h3 {
        color: #FFFFFF; /* White text */
        text-align: center;
        font-family: 'Arial', sans-serif;
    }

    /* Text color styling for content */
    body, p, div {
        color: #FFFFFF; /* White text */
    }
    
    /* Button styling */
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #333333; /* Dark grey for sidebar */
    }

    /* Input field styling */
    .stNumberInput input, .stSelectbox select {
        border-radius: 8px;
        padding: 12px;
        font-size: 14px;
        border: 1px solid #d1d1d1;
        background-color: #222222; /* Dark background for input fields */
        color: #FFFFFF; /* White text for inputs */
    }

    /* Section styling */
    .section-title {
        color: #FFFFFF; /* White text */
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    
    /* Subtle hover effect for button */
    .stButton button:hover {
        background-color: #4CAF50;
        color: white;
        transition: background-color 0.3s ease;
    }

    </style>
    """, unsafe_allow_html=True)

# Sidebar for app description or instructions
st.sidebar.header("About the App")
st.sidebar.write("""
    This tool predicts the likelihood of heart failure based on factors such as age, cholesterol, heart rate, and resting ECG.
    It's powered by a K-Nearest Neighbors (KNN) model, trained on health data.
""")

st.sidebar.image("https://image-url-here.com/heart.png", use_column_width=True)

# Main content layout
with st.container():
    # Title and description
    st.title('Heart Failure Prediction System 💓')
    st.write("""
        Provide the required information below, and the system will estimate the likelihood of heart failure using machine learning.
    """)

    # Divider for clarity
    st.markdown("---")

    # Input section in two columns
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age:', min_value=0, max_value=120, step=1, help="Enter the patient's age.")
        resting_bp = st.number_input('Resting Blood Pressure (mm Hg):', min_value=0, step=1, help="Enter the resting blood pressure.")
        cholesterol = st.number_input('Cholesterol Level (mg/dl):', min_value=0, step=1, help="Enter the cholesterol level.")

    with col2:
        max_hr = st.number_input('Maximum Heart Rate (bpm):', min_value=0, step=1, help="Enter the maximum heart rate.")
        resting_ecg = st.selectbox('Resting ECG:', ['Normal', 'ST', 'LVH'], help="Select the type of resting ECG.")

    st.markdown("---")

    # DataFrame for input data
    input_df = pd.DataFrame({
        'Age': [age],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'MaxHR': [max_hr],
        'RestingECG': [resting_ecg]
    })

    # Check for unknown label
    if resting_ecg not in label_encoder.classes_:
        st.error(f"Unknown value '{resting_ecg}' for RestingECG. Please select from {label_encoder.classes_}.")
    else:
        # Preprocessing and prediction
        input_df['RestingECG'] = label_encoder.transform(input_df['RestingECG'])
        input_df_scaled = scaler.transform(input_df)
        input_df_poly = poly.transform(input_df_scaled)
        input_df_selected = feature_selector.transform(input_df_poly)

        # Predict button with progress bar
        if st.button('Predict Heart Failure'):
            with st.spinner('Predicting...'):
                y_pred = knn.predict(input_df_selected)
                y_prob = knn.predict_proba(input_df_selected)[:, 1]
                probability = y_prob[0] * 100
                heart_failure = "Yes" if y_pred[0] == 1 else "No"

            # Display prediction results
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
            st.write(f"- **Heart Failure Likelihood:** {heart_failure}")
            st.write(f"- **Predicted Probability of Heart Failure:** {probability:.2f}%")

            # Visual representation of the probability
            st.progress(int(probability))

            # Display the message with custom color coding
            if y_pred[0] == 1:
                st.error("⚠️ High risk of heart failure. Please consult a healthcare provider.")
            else:
                st.success("✅ Low risk of heart failure. Maintain a healthy lifestyle!")

# Footer section with resources
st.sidebar.markdown("---")
st.sidebar.write("### Useful Resources")
st.sidebar.write("- [Heart Failure Information](https://www.heart.org/en/health-topics/heart-failure)")
st.sidebar.write("- [KNN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)")
st.markdown("<center><p style='color:grey;'>Powered by Streamlit | Heart Failure Prediction System © 2024</p></center>", unsafe_allow_html=True)
