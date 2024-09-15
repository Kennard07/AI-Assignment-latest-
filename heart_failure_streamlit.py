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

# Apply custom CSS styles to improve design
st.markdown("""
    <style>
    /* General page styling */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f8ff;
        color: #003366;
    }

    h1, h2, h3 {
        color: #003366;
        text-align: center;
        font-weight: 600;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 10px;
    }

    /* Custom input fields styling */
    input[type=number], select {
        background-color: #fff;
        border: 2px solid #003366;
        border-radius: 5px;
        padding: 8px;
        font-size: 14px;
        width: 100%;
    }

    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: 0.3s;
    }

    /* Button hover effect */
    .stButton button:hover {
        background-color: #45a049;
    }

    /* Results section styling */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        padding: 15px;
    }

    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 8px;
        padding: 15px;
    }

    /* Footer styling */
    footer {
        text-align: center;
        color: gray;
        font-size: 12px;
        margin-top: 20px;
    }

    /* Add a subtle hover effect to input fields */
    input[type=number]:hover, select:hover {
        border-color: #66ccff;
        box-shadow: 0 0 5px rgba(0, 123, 255, 0.3);
    }

    /* Page container styling */
    .main-container {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for app description or instructions
st.sidebar.header("About")
st.sidebar.write("""
    This app predicts the likelihood of heart failure based on user inputs such as age, resting blood pressure, cholesterol, maximum heart rate, and ECG results. 
    The prediction is based on a K-Nearest Neighbors (KNN) model trained to evaluate these factors.
""")

# Main container for content
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    # Main title and description
    st.title('Heart Failure Prediction System 💓')
    st.write("""
        This tool uses machine learning to predict the likelihood of heart failure. Please enter the following information, and the system will provide a prediction based on the input factors.
    """)

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
                st.error("⚠️ High risk of heart failure. It is recommended to consult with a healthcare professional.")
            else:
                st.success("✅ Low risk of heart failure. Keep maintaining a healthy lifestyle!")

    # Close the main container
    st.markdown("</div>", unsafe_allow_html=True)

# Footer with additional resources or credits
st.sidebar.markdown("---")
st.sidebar.write("### Useful Resources")
st.sidebar.write("- [Heart Failure Information](https://www.heart.org/en/health-topics/heart-failure)")
st.sidebar.write("- [KNN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)")

# Footer for app credits
st.markdown("<footer>Powered by Streamlit | Heart Failure Prediction System © 2024</footer>", unsafe_allow_html=True)
