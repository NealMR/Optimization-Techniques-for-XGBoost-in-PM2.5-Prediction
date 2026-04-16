import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from src.live_data_service import LiveDataService
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="PM2.5 Sentinel | Live Air Quality Monitor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PREMIUM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    .stSidebar { background-color: rgba(30, 41, 59, 0.7) !important; backdrop-filter: blur(10px); }
    .metric-card { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 15px; border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; }
    .metric-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }
    .metric-value { font-size: 2rem; font-weight: 600; margin: 5px 0; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stSidebar [data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }
    .stSidebar label { color: #f1f5f9 !important; }
    .stSidebar .stExpander { border: 1px solid rgba(255, 255, 255, 0.1); background: rgba(255, 255, 255, 0.02); }
    /* Dark theme for expanders to remove white background */
    .stSidebar [data-testid="stExpander"] { background-color: transparent !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }
    .stSidebar [data-testid="stExpander"] summary { background-color: rgba(255, 255, 255, 0.05) !important; color: #f1f5f9 !important; }
    .stSidebar [data-testid="stExpander"] summary:hover { background-color: rgba(255, 255, 255, 0.1) !important; }

    /* Hide the white top bar (Streamlit header) */
    header[data-testid="stHeader"] { background: transparent !important; }
    header[data-testid="stHeader"] { display: none !important; }

    /* Dark theme for inputs - ensuring visibility */
    .stSidebar input, .stSidebar [data-baseweb="input"] { 
        background-color: #1e293b !important; 
        color: #f8fafc !important; 
        border: 1px solid rgba(255, 255, 255, 0.2) !important; 
    }
    .stSidebar [data-baseweb="base-input"] { background-color: transparent !important; }
    .stSidebar input::placeholder { color: #94a3b8 !important; }
    
    button[kind="secondary"] { background-color: rgba(255, 255, 255, 0.05) !important; color: #f8fafc !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }
    .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = xgb.XGBRegressor()
    model.load_model("results/xgb_model.json")
    encoder = joblib.load("results/city_encoder.joblib")
    with open("results/feature_medians.json", "r") as f:
        medians = json.load(f)
    return model, encoder, medians

model, encoder, medians = load_assets()

# --- DEFAULT STATE INITIALIZATION ---
if "prediction" not in st.session_state:
    default_city = medians["Cities"][0]
    flist = ["PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
    default_data = {f: medians.get(f, 0.0) for f in flist}
    
    # Calculate initial prediction
    input_row = [default_data[f] for f in flist]
    input_row.append(encoder.transform([default_city])[0])
    
    st.session_state.prediction = model.predict([input_row])[0]
    st.session_state.live_data = default_data
    st.session_state.last_city = f"{default_city} (Baseline)"

# --- SIDEBAR: CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/wind.png", width=60)
st.sidebar.title("Sentinel Controls")

# API Key is now handled via .env for security
owm_api_key = os.getenv("OWM_API_KEY", "").strip()


# 1. Targeted Analytics
st.sidebar.markdown("### 🎯 Single City Detailed Scan")
selected_city = st.sidebar.selectbox("Target City", options=medians["Cities"])
if st.sidebar.button("Fetch Detail & Predict", use_container_width=True):
    if not owm_api_key:
        st.sidebar.error("OWM_API_KEY not found in environment.")
    else:
        service = LiveDataService(owm_api_key)
        data, err = service.fetch_live_data(selected_city)
        if data:
            f_list = ["PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
            input_row = [data[f] for f in f_list]
            input_row.append(encoder.transform([selected_city])[0])
            st.session_state.prediction = model.predict([input_row])[0]
            st.session_state.live_data = data
            st.session_state.last_city = selected_city
        else:
            st.sidebar.error(err)

# 2. Manual Override (For testing without API)
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Manual Input Override"):
    st.write("Simulate pollutant levels immediately.")
    flist = ["PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
    m_inputs = {}
    
    # Show inputs in two columns
    m_col1, m_col2 = st.columns(2)
    for i, feature in enumerate(flist):
        col = m_col1 if i % 2 == 0 else m_col2
        with col:
            # Determine appropriate step and max value based on feature name
            step = 0.1 if feature in ["CO", "Benzene", "Toluene", "Xylene"] else 1.0
            max_val = 50.0 if feature in ["CO", "Benzene", "Toluene", "Xylene"] else 1000.0
            m_inputs[feature] = st.number_input(feature, 0.0, max_val, float(medians.get(feature, 0.0)), step=step)
    
    if st.button("Run Manual Prediction", use_container_width=True):
        row = [m_inputs[f] for f in flist]
        row.append(encoder.transform([selected_city])[0])
        st.session_state.prediction = model.predict([row])[0]
        st.session_state.live_data = m_inputs
        st.session_state.last_city = f"{selected_city} (Manual)"
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("PM2.5 Sentinel Live Dashboard")

if "prediction" in st.session_state:
    st.subheader(f"Deep Analysis: {st.session_state.last_city}")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><p class="metric-label">Predicted PM2.5</p><p class="metric-value">{st.session_state.prediction:.2f}</p></div>', unsafe_allow_html=True)
    with m2:
        val = st.session_state.prediction
        cat = "Good" if val < 50 else ("Moderate" if val < 100 else "Hazardous")
        color = "#22c55e" if val < 50 else ("#eab308" if val < 100 else "#ef4444")
        st.markdown(f'<div class="metric-card"><p class="metric-label">AQI Category</p><p class="metric-value" style="background:none; -webkit-text-fill-color:{color};">{cat}</p></div>', unsafe_allow_html=True)
    with m3:
         st.markdown(f'<div class="metric-card"><p class="metric-label">Target Location</p><p class="metric-value">{st.session_state.last_city}</p></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    ld = st.session_state.live_data
    p_list = ["PM10", "CO", "NO2", "SO2", "O3", "NH3"]
    p_vals = [ld.get(p, 0) for p in p_list]
    fig_bar = px.bar(x=p_list, y=p_vals, color=p_vals, color_continuous_scale='Viridis', labels={'x':'Pollutant', 'y':'Conc'})
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=350)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.markdown("""
        <div style='display: flex; flex-direction: column; align-items:center; justify-content:center; height:500px; border:2px dashed rgba(255,255,255,0.1); border-radius:20px; background: rgba(255,255,255,0.02);'>
            <h3 style='color: #94a3b8; font-weight:300;'>Sentinel Analytics Standby</h3>
            <p style='color: #64748b; margin-top:10px;'>Select a city in the sidebar and click 'Fetch Detail' to begin analysis.</p>
        </div>
    """, unsafe_allow_html=True)
st.markdown("<br><p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Built with Python & XGBoost | Environment Secure</p>", unsafe_allow_html=True)
