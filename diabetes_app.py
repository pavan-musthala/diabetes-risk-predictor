import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import plotly.graph_objects as go
import plotly.express as px

# Set page config with dark theme
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'Diabetes Risk Prediction System'
    }
)

# Custom CSS styling
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --background-color: #0E1117;
        --secondary-bg: #262730;
        --text-color: #FFFFFF;
        --accent-color: #FF4B4B;
    }

    /* Global styles */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 0rem 2rem;
    }

    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Button styles */
    .stButton>button {
        width: 100%;
        background-color: var(--accent-color);
        color: var(--text-color);
        height: 3rem;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: var(--text-color);
    }

    /* Card styles */
    .metric-card {
        background-color: var(--secondary-bg);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        color: var(--text-color);
    }

    /* Typography */
    h1 {
        color: var(--accent-color);
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        color: var(--accent-color);
        margin-bottom: 1rem;
    }
    p {
        color: var(--text-color);
    }

    /* Alert styles */
    .stAlert {
        background-color: var(--secondary-bg);
        color: var(--text-color);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }

    /* Expander and other elements */
    .streamlit-expanderHeader {
        background-color: var(--secondary-bg);
        color: var(--text-color);
    }
    .streamlit-expanderContent {
        background-color: var(--secondary-bg);
        color: var(--text-color);
    }

    /* Slider styles */
    .stSlider {
        background-color: var(--secondary-bg);
    }

    /* Custom classes */
    .intro-box {
        background-color: var(--secondary-bg);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .dark-card {
        background-color: var(--secondary-bg);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1>🏥 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class='intro-box'>
        <p style='font-size: 1.2rem; text-align: center;'>
            Welcome to the Diabetes Risk Prediction System. This tool uses machine learning to assess your diabetes risk 
            based on various health metrics. Enter your information below to get a personalized risk assessment.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')
    return data

data = load_data()

# Train model
def train_model():
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='linear', random_state=42, probability=True)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# Create two columns for the form
col1, col2 = st.columns(2)

# Input form with improved styling
with col1:
    st.markdown("<h2>📋 Personal Information</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        age = st.slider("Age", 0, 120, 33, help="Age in years")
        pregnancies = st.slider("Number of Pregnancies", 0, 20, 0)
        bmi = st.slider("BMI", 10.0, 70.0, 25.0, help="Body Mass Index")
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.47, help="A function that scores likelihood of diabetes based on family history")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("<h2>🔬 Clinical Measurements</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 300, 120, help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 200, 70, help="Diastolic blood pressure")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 79, help="2-Hour serum insulin")
        st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("📊 Analyze Diabetes Risk"):
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.decision_function(input_scaled)
    
    # Create columns for results
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("<h2>🎯 Prediction Result</h2>", unsafe_allow_html=True)
        if prediction[0] == 1:
            st.error("⚠️ **High Risk of Diabetes Detected**\n\nIt's recommended to consult a healthcare professional for a thorough evaluation.")
        else:
            st.success("✅ **Low Risk of Diabetes Detected**\n\nKeep maintaining a healthy lifestyle!")
        
        # Confidence gauge
        confidence = abs(prediction_proba[0])
        st.markdown("""
            <div class='dark-card'>
                <h4 style='color: var(--accent-color); margin-bottom: 10px;'>Understanding the Confidence Score</h4>
                <p style='margin-bottom: 15px;'>The confidence score indicates how certain the model is about its prediction:</p>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 8px;'>🎯 <strong>0-2:</strong> Low confidence - The prediction is less certain</li>
                    <li style='margin-bottom: 8px;'>🎯 <strong>2-3.5:</strong> Moderate confidence - The prediction has reasonable certainty</li>
                    <li style='margin-bottom: 8px;'>🎯 <strong>3.5-5:</strong> High confidence - The prediction is highly certain</li>
                </ul>
                <p style='font-style: italic; margin-top: 15px; font-size: 0.9rem;'>
                    Note: Higher confidence scores suggest that your health metrics align more strongly with patterns found in our training data.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence,
            title = {'text': "Confidence Score", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 5], 'tickfont': {'color': 'white'}},
                'bar': {'color': "#FF4B4B"},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 2,
                'bordercolor': 'gray',
                'steps': [
                    {'range': [0, 2], 'color': "#262730"},
                    {'range': [2, 3.5], 'color': "#363845"},
                    {'range': [3.5, 5], 'color': "#464B5F"}
                ]
            }
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with result_col2:
        st.markdown("<h2>📊 Your Metrics vs Average</h2>", unsafe_allow_html=True)
        # Prepare comparison data
        feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                        'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
        avg_values = data.mean()
        current_values = pd.Series([pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, dpf, age], index=feature_names)
        
        # Create comparison plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Your Values',
            x=feature_names,
            y=current_values,
            marker_color='#FF4B4B'
        ))
        fig.add_trace(go.Bar(
            name='Average Values',
            x=feature_names,
            y=avg_values[:-1],
            marker_color='#666666'
        ))
        fig.update_layout(
            barmode='group',
            height=400,
            title="Your Values vs Population Average"
        )
        st.plotly_chart(fig, use_container_width=True)

# Information section
with st.expander("ℹ️ About the Metrics"):
    st.markdown("""
    <div class='dark-card'>
        <h3 style='color: var(--accent-color);'>Understanding the Health Metrics</h3>
        <ul style='list-style-type: none; padding-left: 0;'>
            <li style='margin-bottom: 10px;'>🔹 <strong>Pregnancies:</strong> Number of times pregnant</li>
            <li style='margin-bottom: 10px;'>🔹 <strong>Glucose:</strong> Plasma glucose concentration after 2 hours in an oral glucose tolerance test</li>
            <li style='margin-bottom: 10px;'>🔹 <strong>Blood Pressure:</strong> Diastolic blood pressure (mm Hg)</li>
            <li style='margin-bottom: 10px;'>🔹 <strong>Skin Thickness:</strong> Triceps skin fold thickness (mm)</li>
            <li style='margin-bottom: 10px;'>🔹 <strong>Insulin:</strong> 2-Hour serum insulin (mu U/ml)</li>
            <li style='margin-bottom: 10px;'>🔹 <strong>BMI:</strong> Body mass index (weight in kg/(height in m)²)</li>
            <li style='margin-bottom: 10px;'>🔹 <strong>Diabetes Pedigree Function:</strong> A function scoring likelihood of diabetes based on family history</li>
            <li style='margin-bottom: 10px;'>🔹 <strong>Age:</strong> Age in years</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: var(--text-color);'>
        <p>Built with ❤️ using Streamlit • Data source: Pima Indians Diabetes Dataset</p>
        <p style='font-size: 0.8rem;'>Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)
