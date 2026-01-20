import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
    }
    .prediction-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .prediction-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚢 Titanic Survival Prediction System")
st.markdown("---")

# Load the trained model and scaler
@st.cache_resource
def load_models():
    """Load the trained model and scaler from disk"""
    try:
        model_path = Path(__file__).parent / "model" / "titanic_survival_model.pkl"
        scaler_path = Path(__file__).parent / "model" / "scaler.pkl"
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model has been trained and saved.")
        st.stop()

model, scaler = load_models()

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About This Application")
    st.markdown("""
    ### Titanic Survival Prediction
    
    This application uses a **Logistic Regression** machine learning model to predict 
    whether a Titanic passenger would have survived based on their characteristics.
    
    **Model Details:**
    - Algorithm: Logistic Regression
    - Persistence Method: Joblib
    - Features Used: Pclass, Sex, Age, SibSp, Fare
    - Training Accuracy: ~82%
    
    **Features:**
    - **Pclass**: Passenger ticket class (1=1st, 2=2nd, 3=3rd)
    - **Sex**: Passenger gender (Male/Female)
    - **Age**: Passenger age in years
    - **SibSp**: Number of siblings/spouses aboard
    - **Fare**: Ticket fare paid in pounds
    """)

# Main content
st.header("Enter Passenger Information")
st.markdown("Fill in the details below to predict survival:")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        help="1 = 1st Class (Upper), 2 = 2nd Class (Middle), 3 = 3rd Class (Lower)"
    )
    
    sex = st.radio(
        "Gender",
        options=["Female", "Male"],
        help="Select the passenger's gender"
    )
    # Convert to numeric: Female = 0, Male = 1
    sex_numeric = 1 if sex == "Male" else 0
    
    age = st.number_input(
        "Age (years)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="Passenger's age in years"
    )

with col2:
    sibsp = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of siblings or spouses traveling with the passenger"
    )
    
    fare = st.number_input(
        "Ticket Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=100.0,
        step=1.0,
        help="Amount paid for the ticket in British pounds"
    )

# Prediction button
st.markdown("---")
if st.button("🔮 Predict Survival", use_container_width=True, type="primary"):
    # Prepare the input data for the model
    passenger_data = np.array([[pclass, sex_numeric, age, sibsp, fare]])
    
    # Scale the data using the loaded scaler
    passenger_scaled = scaler.transform(passenger_data)
    
    # Make prediction
    prediction = model.predict(passenger_scaled)[0]
    probability = model.predict_proba(passenger_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.header("📊 Prediction Result")
    
    if prediction == 1:
        st.markdown(
            f"""
            <div class="prediction-success">
                <h3>✅ Prediction: SURVIVED</h3>
                <p><strong>Confidence: {probability[1]*100:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="prediction-danger">
                <h3>❌ Prediction: DID NOT SURVIVE</h3>
                <p><strong>Confidence: {probability[0]*100:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display passenger summary
    st.subheader("Passenger Summary")
    summary_data = {
        "Passenger Class": f"{pclass} ({['', '1st Class', '2nd Class', '3rd Class'][pclass]})",
        "Gender": sex,
        "Age": f"{age} years",
        "Siblings/Spouses": sibsp,
        "Ticket Fare": f"£{fare:.2f}",
        "Survival Probability": f"{probability[1]*100:.2f}%"
    }
    
    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Attribute", "Value"])
    st.table(summary_df)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p><small>Built with Streamlit | ML Model: Logistic Regression | Data: Titanic Dataset</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
