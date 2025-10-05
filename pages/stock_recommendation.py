"""
Simple, friendly page that asks a few questions about your risk tolerance,
time horizon, and sector preferences, then suggests 3 stocks that match your
profile. Recommendations are a lightweight starting point — go to Stock
Analysis for deeper charts and forecasts. Educational only.
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Recommendations", layout="wide", page_icon="🎯")



st.title("🎯 Personalized Stock Recommendations")
st.caption("Answer a few questions to get stocks matched to your investment profile")

# Initialize session state
if 'recommended_stocks' not in st.session_state:
    st.session_state.recommended_stocks = []

# Load data
@st.cache_data
def load_stock_data():
    try:
        data = pd.read_csv("cleaned_data.csv")
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data.dropna(subset=['Date', 'Company', 'Close'])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_stock_data()

if data is None:
    st.error("Unable to load stock data. Ensure cleaned_data.csv exists.")
    st.stop()

# Stock categorization
STOCK_CATEGORIES = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'IBM'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'BK'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'BMY', 'LLY'],
    'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'TJX'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS', 'HON', 'LMT', 'RTX', 'DE', 'UNP'],
    'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'DLR', 'O', 'WELL', 'AVB'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM']
}

available_stocks = data['Company'].unique().tolist()
filtered_categories = {cat: [s for s in stocks if s in available_stocks] 
                      for cat, stocks in STOCK_CATEGORIES.items()}
filtered_categories = {k: v for k, v in filtered_categories.items() if v}

st.markdown("---")

# Questionnaire
col1, col2 = st.columns(2)

with col1:
    st.subheader("Investment Profile")
    
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
        value="Moderate"
    )
    
    investment_horizon = st.selectbox(
        "Investment Time Horizon",
        ["Short-term (< 1 year)", "Medium-term (1-3 years)", "Long-term (3-5 years)", "Very Long-term (5+ years)"]
    )
    
    investment_goal = st.multiselect(
        "Primary Investment Goals",
        ["Capital Growth", "Dividend Income", "Portfolio Diversification", "Hedge Against Inflation"],
        default=["Capital Growth"]
    )

with col2:
    st.subheader("🏭 Sector Preferences")
    
    preferred_sectors = st.multiselect(
        "Which sectors interest you?",
        list(filtered_categories.keys()),
        default=list(filtered_categories.keys())[:3] if len(filtered_categories) >= 3 else list(filtered_categories.keys())
    )
    
    market_cap_pref = st.selectbox(
        "Market Cap Preference",
        ["Any", "Large Cap (Stable)", "Mid Cap (Balanced)", "Small Cap (High Growth)"]
    )
    
    avoid_sectors = st.multiselect(
        "Sectors to Avoid (optional)",
        list(filtered_categories.keys())
    )

# Get Recommendations Button
if st.button("🔍 Generate Recommendations", type="primary", use_container_width=True):
    with st.spinner("Analyzing stocks..."):
        
        candidate_stocks = []
        for sector in preferred_sectors:
            if sector in filtered_categories:
                candidate_stocks.extend(filtered_categories[sector])
        
        for sector in avoid_sectors:
            if sector in filtered_categories:
                for stock in filtered_categories[sector]:
                    if stock in candidate_stocks:
                        candidate_stocks.remove(stock)
        
        candidate_stocks = list(set(candidate_stocks))
        
        if not candidate_stocks:
            st.error("No stocks match your criteria. Adjust your preferences.")
            st.stop()
        
        stock_scores = []
        for stock in candidate_stocks:
            stock_data = data[data['Company'] == stock].copy()
            if len(stock_data) < 30:
                continue
            
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            recent_prices = stock_data['Close'].tail(30).values
            returns = np.diff(recent_prices) / recent_prices[:-1]
            
            avg_return = np.mean(returns) * 100
            volatility = np.std(returns) * 100
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            risk_score = 0
            if risk_tolerance == "Very Conservative":
                risk_score = max(0, 10 - volatility * 2)
            elif risk_tolerance == "Conservative":
                risk_score = max(0, 8 - volatility * 1.5)
            elif risk_tolerance == "Moderate":
                risk_score = 5 if volatility < 3 else 3
            elif risk_tolerance == "Aggressive":
                risk_score = volatility * 1.5
            else:
                risk_score = volatility * 2
            
            trend_score = 10 if trend > 5 else (7 if trend > 0 else (4 if trend > -5 else 1))
            return_score = max(0, min(10, avg_return * 100))
            total_score = risk_score * 0.3 + trend_score * 0.4 + return_score * 0.3
            
            stock_scores.append({
                'Stock': stock,
                'Score': total_score,
                'Trend_30d': trend,
                'Volatility': volatility,
                'Current_Price': recent_prices[-1]
            })
        
        stock_scores = sorted(stock_scores, key=lambda x: x['Score'], reverse=True)
        top_recommendations = stock_scores[:3]
        st.session_state.recommended_stocks = [s['Stock'] for s in top_recommendations]
        
        st.markdown("---")
        st.subheader("✨ Your Top 3 Recommendations")
        
        for i, rec in enumerate(top_recommendations, 1):
            col_rank, col_info, col_metrics = st.columns([0.5, 2, 2])
            
            with col_rank:
                st.markdown(f"### #{i}")
            
            with col_info:
                st.markdown(f"### {rec['Stock']}")
                stock_sector = next((s for s, stocks in filtered_categories.items() if rec['Stock'] in stocks), "Unknown")
                st.caption(f"Sector: **{stock_sector}**")
            
            with col_metrics:
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"${rec['Current_Price']:.2f}")
                m2.metric("30d Trend", f"{rec['Trend_30d']:+.2f}%")
                m3.metric("Score", f"{rec['Score']:.1f}/10")
            
            st.markdown("---")
        
        st.info("💡 Go to **Stock Analysis** page to analyze these stocks in detail.")

# Sidebar display
if st.session_state.recommended_stocks:
    st.sidebar.markdown("### 🎯 Your Recommendations")
    for stock in st.session_state.recommended_stocks:
        st.sidebar.markdown(f"- **{stock}**")
    if st.sidebar.button("Clear"):
        st.session_state.recommended_stocks = []
        st.rerun()

