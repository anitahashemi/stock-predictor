# Home.py
"""
Home — Stock Forecaster

A friendly landing page for the app. Explains what the project does in plain
language and points people to the key pages: Stock Recommendations and
Stock Analysis. Designed for quick orientation — pick a page from the sidebar
and get started.
"""

import streamlit as st

st.set_page_config(
    page_title="Stock Forecaster",
    layout="wide"
)

# Dark theme styling
st.markdown("""
<style>
.stApp{background:#0E1117;color:#E5E7EB}
[data-testid="stSidebar"]{background:#0B1220}
div[data-testid="stMetric"]{background:#111827;border:1px solid #1F2937;border-radius:12px;padding:8px 10px}
.big-title{font-size:3rem;font-weight:bold;text-align:center;margin:2rem 0}
.subtitle{font-size:1.2rem;text-align:center;color:#9CA3AF;margin-bottom:3rem}
.feature-box{background:#111827;padding:1.5rem;border-radius:12px;border:1px solid #1F2937;margin:1rem 0}
.feature-title{font-size:1.3rem;font-weight:bold;margin-bottom:0.5rem}
.step-box{background:#1F2937;padding:1rem;border-radius:8px;margin:0.5rem 0;border-left:4px solid #3B82F6}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="big-title">Stock Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML-Powered Stock Analysis & Personalized Recommendations</div>', unsafe_allow_html=True)

st.markdown("---")

# Main sections
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🎯 Stock Recommendations</div>
        <p>Get personalized stock picks tailored to your investment profile:</p>
        <ul>
            <li>Risk tolerance matching</li>
            <li>Sector preferences</li>
            <li>Investment timeline alignment</li>
            <li>Top 3 AI-ranked suggestions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">📊 Stock Analysis</div>
        <p>Deep technical analysis with machine learning:</p>
        <ul>
            <li>Interactive charts with zoom & hover</li>
            <li>15-60 day price forecasts</li>
            <li>Confidence intervals & metrics</li>
            <li>Historical backtesting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# How it works
st.markdown("### How It Works")

st.markdown("""
<div class="step-box">
    <strong>Step 1:</strong> Go to <strong>Stock Recommendations</strong> and answer questions about your investment preferences
</div>
<div class="step-box">
    <strong>Step 2:</strong> Receive your top 3 personalized stock recommendations based on AI analysis
</div>
<div class="step-box">
    <strong>Step 3:</strong> Navigate to <strong>Stock Analysis</strong> to explore detailed forecasts and metrics
</div>
<div class="step-box">
    <strong>Step 4:</strong> Make informed decisions using interactive charts and performance data
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Technical overview
st.markdown("### Technical Features")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    **ML Models**
    - Gradient Boosting
    - Random Forest
    - Ridge Regression
    - Auto model selection
    """)

with tech_col2:
    st.markdown("""
    **Visualization**
    - Interactive Plotly charts
    - Multiple confidence bands
    - Volume indicators
    - Custom time ranges
    """)

with tech_col3:
    st.markdown("""
    **Validation**
    - Walk-forward testing
    - MAPE, MAE, R² metrics
    - Direction accuracy
    - Performance tracking
    """)

st.markdown("---")

# Disclaimer
st.warning("""
**Disclaimer:** This tool is for educational purposes only. Stock investments carry risk. 
Always conduct your own research and consult licensed financial advisors before making investment decisions.
""")

st.markdown("---")

# CTA
st.info("Select a page from the sidebar to get started!")

# Footer
st.markdown("""
<div style="text-align:center;margin-top:3rem;color:#6B7280;font-size:0.9rem">
    <p>Data-driven insights • AI-powered predictions • User-friendly interface</p>
</div>
""", unsafe_allow_html=True)