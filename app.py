import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration for a premium look without animations
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered", # Changed to centered for a focused premium look
    initial_sidebar_state="collapsed",
)

# Custom CSS for a high-end, centered, gradient-focused look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #ffffff;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0.1rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.8em;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        border: none;
        color: white;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.4);
        border: none;
        color: white;
    }
    
    .prediction-card {
        padding: 3rem;
        border-radius: 24px;
        background: #ffffff;
        box-shadow: 0 20px 40px rgba(0,0,0,0.06);
        text-align: center;
        margin-top: 3rem;
        border: 1px solid #f1f5f9;
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .price-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #0f172a;
        margin: 1rem 0;
        letter-spacing: -2px;
    }
    
    .price-label {
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* Remove streamlit footer and menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data_and_model():
    # Paths to files
    model_path = 'random_forest_model.joblib'
    le_map_path = 'label_encoder_map.joblib'
    csv_path = 'data/laptops.csv'

    # Load model and encoder map
    rf = joblib.load(model_path)
    le_map = joblib.load(le_map_path)
    
    # Load dataset for dropdown choices
    df_raw = pd.read_csv(csv_path)
    
    # Preprocessing (same as training)
    df_raw["rating"] = df_raw["rating"].fillna(df_raw["rating"].median())
    df_raw["no_of_ratings"] = df_raw["no_of_ratings"].fillna(df_raw["no_of_ratings"].median())
    df_raw["no_of_reviews"] = df_raw["no_of_reviews"].fillna(df_raw["no_of_reviews"].median())
    
    return rf, le_map, df_raw

def main():
    # Header Section
    st.markdown("<h1 class='gradient-text'>Laptop Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Professional Grade Machine Learning Estimation</p>", unsafe_allow_html=True)

    try:
        rf, le_map, df_raw = load_data_and_model()
    except Exception as e:
        st.error(f"Error loading model or data: {e}. Ensure all files are in the root directory.")
        return

    # Get unique values for dropdowns
    name_choices = sorted(df_raw['name'].unique().tolist())
    processor_choices = sorted(df_raw['processor'].unique().tolist())
    ram_choices = sorted(df_raw['ram'].unique().tolist())
    os_choices = sorted(df_raw['os'].unique().tolist())
    storage_choices = sorted(df_raw['storage'].unique().tolist())

    # Form Container
    with st.container():
        col1, col2 = st.columns(2, gap="large")

        with col1:
            name_val = st.selectbox("Brand & Model Name", name_choices)
            processor_val = st.selectbox("Processor Architecture", processor_choices)
            ram_val = st.selectbox("Memory (RAM)", ram_choices)
            os_val = st.selectbox("Operating System", os_choices)
            storage_val = st.selectbox("Storage Capacity", storage_choices)

        with col2:
            display_size_val = st.number_input("Display Size (Inches)", min_value=10.0, max_value=40.0, value=15.6, step=0.1)
            rating_val = st.slider("User Satisfaction Rating", min_value=1.0, max_value=5.0, value=4.3, step=0.1)
            no_of_ratings_val = st.number_input("Total Ratings Count", min_value=0, value=100)
            no_of_reviews_val = st.number_input("Total Reviews Count", min_value=0, value=15)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    if st.button("Calculate Market Value"):
        # Encode categorical inputs
        try:
            encoded_name = le_map['name'].transform([name_val])[0]
            encoded_processor = le_map['processor'].transform([processor_val])[0]
            encoded_ram = le_map['ram'].transform([ram_val])[0]
            encoded_os = le_map['os'].transform([os_val])[0]
            encoded_storage = le_map['storage'].transform([storage_val])[0]

            # Prepare feature vector
            feature_columns = ['name', 'processor', 'ram', 'os', 'storage', 'display(in inch)', 'rating', 'no_of_ratings', 'no_of_reviews']
            input_data = [
                encoded_name,
                encoded_processor,
                encoded_ram,
                encoded_os,
                encoded_storage,
                display_size_val,
                rating_val,
                no_of_ratings_val,
                no_of_reviews_val
            ]
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            # Predict
            prediction = rf.predict(input_df)[0]
            
            # Display result in a premium card
            st.markdown(f"""
            <div class='prediction-card'>
                <div class='price-label'>Estimated Market Value</div>
                <div class='price-value'>₹ {prediction:,.0f}</div>
                <div style='color: #64748b; font-size: 0.9rem;'>Based on current market trends and specifications</div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == '__main__':
    main()