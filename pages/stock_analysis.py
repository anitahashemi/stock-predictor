""" 
Stock Analysis and Forecasting App using Streamlit
"""

import warnings
warnings.filterwarnings("ignore")

import os, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

import streamlit as st


# Config
# ===============================
NOTES_PATH = "user_notes.csv"  # where notes are saved

# ===============================
# Helpers
# ===============================
def to_numeric_safe(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("$", "", regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def create_features(df):
    df = df.copy()
    df['MA_5']  = df['Close'].rolling(window=5,  min_periods=1).mean()
    df['MA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Momentum_5']  = df['Close'].pct_change(5).rolling(window=3, min_periods=1).mean()
    df['Momentum_10'] = df['Close'].pct_change(10).rolling(window=3, min_periods=1).mean()
    df['Volatility']  = df['Close'].rolling(window=20, min_periods=1).std()
    df['Trend'] = (df['MA_5'] - df['MA_20']) / df['MA_20']
    df['Close_1'] = df['Close'].shift(1)
    df['Close_2'] = df['Close'].shift(2)
    df['Close_5'] = df['Close'].shift(5)
    return df.fillna(method='bfill').fillna(method='ffill')

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def forecast_realistic(model, last_data, feature_cols, n_days=15):
    """Iterative forecast with ±3% daily cap + damped recent trend."""
    out = []
    features = last_data[feature_cols].iloc[-1].values.copy()
    current_price = float(last_data['Close'].iloc[-1])
    lookback = min(30, len(last_data)-1) if len(last_data) > 1 else 1
    recent_trend = (last_data['Close'].iloc[-1] - last_data['Close'].iloc[-lookback]) / max(1, lookback)

    for day in range(n_days):
        pred_price = float(model.predict(features.reshape(1, -1))[0])
        max_change = current_price * 0.03
        change = np.clip(pred_price - current_price, -max_change, max_change)
        pred_price = current_price + change + recent_trend * (0.95 ** day)
        out.append(pred_price)

        # roll features: ['MA_5','MA_10','MA_20','Momentum_5','Momentum_10','Volatility','Trend','Close_1','Close_2','Close_5']
        features[9] = features[8]           # Close_5 <- Close_2
        features[8] = features[7]           # Close_2 <- Close_1
        features[7] = current_price         # Close_1 <- current
        features[0] = (features[0] * 4 + pred_price) / 5.0  # MA_5
        features[1] = (features[1] * 9 + pred_price) / 10.0 # MA_10
        features[2] = (features[2] * 19 + pred_price) / 20.0# MA_20
        if features[2] != 0:
            features[6] = (features[0] - features[2]) / features[2]  # Trend
        current_price = pred_price
    return np.array(out, dtype=float)

def load_notes():
    if os.path.exists(NOTES_PATH):
        try:
            return pd.read_csv(NOTES_PATH, parse_dates=["timestamp"])
        except Exception:
            return pd.DataFrame(columns=["timestamp","company","text"])
    return pd.DataFrame(columns=["timestamp","company","text"])

def add_note(company, text):
    df = load_notes()
    new_row = {"timestamp": pd.Timestamp.now(), "company": company, "text": text}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(NOTES_PATH, index=False)
    return df

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Your Stock Forecaster", layout="wide", page_icon="📊")
st.title("Your Stock Forecaster")
st.caption("Preloaded: cleaned_data.csv · Educational only.")

# ---------- Quick Guide (deep explanations) ----------
with st.expander("ℹ️ Quick Guide — what every term means", expanded=False):
    st.markdown("""
**Data prep**
- **Close/Open/High/Low/Volume**: standard OHLCV columns.
- **MA 5/10/20**: moving averages over 5/10/20 days (smoothed prices).
- **Momentum 5/10**: average percent change over last 5/10 days (short-term speed).
- **Volatility**: standard deviation of recent closes (higher = more jittery).
- **Trend**: (MA5 − MA20) / MA20 — positive ⇒ short-term > long-term (bullish tilt).

**Train/Test split**
- We **keep time order**: the earliest *Train %* of rows are used to train; the rest are used to test.

**Models**
- **Gradient Boosting**: strong tree-based model for non-linear patterns.
- **Ridge Regression**: simple linear baseline with L2 regularization.

**Metrics**
- **MAPE**: mean absolute percentage error (average % miss). ↓ is better.
- **MAE**: mean absolute error in dollars. ↓ is better.
- **R²**: how much of the price variance the model explains (0–1).
- **Direction Accuracy**: how often our predicted *up/down* matches actual *up/down* moves.

**Forecast**
- We forecast iteratively, capping daily moves at ±3% and adding a **damped recent trend**.
- The shaded area is a **confidence range** built from recent volatility (wider further out).

**Charts**
- **Validation**: actual vs predicted on the test window.
- **Continuous**: last 60/30 days connected to forecast; **Today** is the dotted red line.

**Notes + Chat**
- **Notes** you type are saved to `user_notes.csv` and can be searched.
- The **mini chat** answers questions about your selected stock, metrics, forecast, and your notes.

**Remember**: this is for **education**, not trading advice. Markets are wild and poetic.
""")

# ---------- Controls ----------
"""
Stock Analysis (Streamlit)

Interactive page for exploring a single stock: shows validation plots,
continuous forecasts, simple metrics, and lets you save quick notes. Meant to
be approachable — use the sliders to change the forecast horizon and training
split, then explore charts. Educational only.
"""
horizon   = st.slider("Forecast days", 15, 30, 15, 5, help="How many future days to predict.")
train_pct = st.slider("Train size (%)", 60, 90, 80, 5, help="Earliest % of rows become training; the recent rows become test.")
show_vol  = st.toggle("Show Volume chart (separate)", value=False, help="Toggle a dedicated volume chart below.")
chart_style = st.selectbox("Chart style (history view)", ["Line", "Candlestick"], index=0)  # <-- added
notes_help = "Write any observation or hypothesis — it's saved to user_notes.csv and searchable in chat."

# ---------- Load CSV ----------
csv_path = "cleaned_data.csv"
if not os.path.exists(csv_path):
    st.error(f"Missing file: **{csv_path}** in the current folder.")
    st.stop()

data = pd.read_csv(csv_path)

# Basic parsing
if "Date" not in data.columns or "Company" not in data.columns:
    st.error("CSV must contain at least 'Date' and 'Company' columns.")
    st.stop()

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = to_numeric_safe(data, ["Open","High","Low","Close","Volume"])
data = data.dropna(subset=["Date","Company","Close"]).sort_values("Date").reset_index(drop=True)

companies = sorted(data["Company"].astype(str).unique().tolist())
default_company = "AAPL" if "AAPL" in companies else companies[0]
company = st.selectbox("Pick a company", companies, index=companies.index(default_company))

stock_data = data[data["Company"] == company].copy().sort_values("Date").reset_index(drop=True)

if len(stock_data) < 20:
    st.warning("Not enough rows for this symbol.")
    st.stop()

st.write(f"**Analyzing {company}:** {len(stock_data)} observations · **Range:** {stock_data['Date'].min().date()} → {stock_data['Date'].max().date()}")

# ---------- Features ----------
stock_data = create_features(stock_data)
feature_cols = ['MA_5','MA_10','MA_20','Momentum_5','Momentum_10','Volatility','Trend','Close_1','Close_2','Close_5']

# Split
train_size = int(len(stock_data) * (train_pct/100.0))
train_df = stock_data.iloc[:train_size]
test_df  = stock_data.iloc[train_size:]

X_train = train_df[feature_cols].values
y_train = train_df['Close'].values.astype(float)
X_test  = test_df[feature_cols].values
y_test  = test_df['Close'].values.astype(float)

# ---------- Train (pick best of 2) ----------
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred  = gb_model.predict(X_test)
mape_gb  = mape(y_test, gb_pred)

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)
ridge_pred  = ridge_model.predict(X_test)
mape_ridge  = mape(y_test, ridge_pred)

if mape_gb < mape_ridge:
    model       = gb_model
    model_name  = "Gradient Boosting"
    predictions = gb_pred
    mape_val    = mape_gb
else:
    model       = ridge_model
    model_name  = "Ridge Regression"
    predictions = ridge_pred
    mape_val    = mape_ridge

mae = mean_absolute_error(y_test, predictions)
r2  = r2_score(y_test, predictions)

# Direction accuracy
y_test_changes = np.diff(y_test)
pred_changes   = np.diff(predictions)
direction_accuracy = float(np.mean((y_test_changes > 0) == (pred_changes > 0)) * 100.0) if len(y_test_changes) else 0.0

# Reliability score
score = 0
score += 3 if mape_val < 5 else (2 if mape_val < 10 else 1)
score += 3 if r2 > 0.8 else (2 if r2 > 0.6 else 1)
score += 2 if direction_accuracy > 55 else (1 if direction_accuracy > 50 else 0)
if score >= 7:
    rating ="🟢RELIABLE"; rec = "Safe for practice with risk management."
elif score >= 5:
    rating ="🟢RELIABLE"; rec = "Good for short-term; combine with other signals."
elif score >= 3:
    rating ="🟡MODERATELY RELIABLE"; rec = "Use as ONE indicator, not the only one."
else:
    rating ="🔴UNRELIABLE"; rec = "Model needs improvement."

# ---------- Metrics ----------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Best model", model_name)
m2.metric("Accuracy (≈100 − MAPE)", f"{max(0.0,100-mape_val):.2f}%")
m3.metric("MAPE (↓)", f"{mape_val:.2f}%")
m4.metric("MAE ($)", f"${mae:,.2f}")
m5.metric("R²", f"{r2:.4f}")
st.caption(f"Direction Accuracy: **{direction_accuracy:.2f}%** · Reliability: **{rating}**\n{rec}")

# ---------- Forecast ----------
forecast_vals = forecast_realistic(model, stock_data, feature_cols, n_days=horizon)
last_date     = stock_data['Date'].max()
future_dates  = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

volatility = stock_data['Close'].tail(60).std()
decay      = np.linspace(1, 1.3, horizon)
upper      = forecast_vals + (volatility * decay)
lower      = forecast_vals - (volatility * decay)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Validation (No Future)", "Last 60d + Future (Continuous)", "Last 30d + Future (Continuous)"])

with tab1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=test_df['Date'].values, y=y_test, mode='lines',
                              name='Actual Price', line=dict(color='#2E86AB', width=2.5),
                              hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> $%{y:,.2f}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=test_df['Date'].values, y=predictions, mode='lines',
                              name='Predicted Price', line=dict(color='#F77F00', width=2.5, dash='dash'),
                              hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted:</b> $%{y:,.2f}<extra></extra>'))
    fig1.update_layout(title=dict(text=f'{company} - Validation · {model_name}', font=dict(size=16)),
                       xaxis_title='Date', yaxis_title='Price ($)', hovermode='x unified',
                       template='plotly_white', height=500, showlegend=True,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

def _cont_plot(lookback_days, title_suffix, style="Line"):
    recent_idx = max(0, len(stock_data) - lookback_days)
    recent = stock_data.iloc[recent_idx:].copy()

    # build combined arrays for forecast continuation
    combined_dates  = np.concatenate([recent['Date'].values, future_dates.values])
    combined_prices = np.concatenate([recent['Close'].values.astype(float), forecast_vals])

    fig = go.Figure()

    if style == "Candlestick":
        # historical candlesticks
        fig.add_trace(go.Candlestick(
            x=recent["Date"], open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"],
            name=f'Historical (Last {lookback_days}d)'
        ))
        # forecast as a smooth line from the last close
        fig.add_trace(go.Scatter(
            x=combined_dates[len(recent)-1:],
            y=combined_prices[len(recent)-1:],
            mode='lines', name=f'Predicted (Next {horizon}d)', line=dict(width=3)
        ))
    else:
        # simple line history
        fig.add_trace(go.Scatter(
            x=recent["Date"], y=recent["Close"], mode='lines',
            name=f'Historical (Last {lookback_days}d)', line=dict(width=3)
        ))
        # forecast continuation
        fig.add_trace(go.Scatter(
            x=combined_dates[len(recent)-1:],
            y=combined_prices[len(recent)-1:],
            mode='lines', name=f'Predicted (Next {horizon}d)', line=dict(width=3)
        ))

    # confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself', fillcolor='rgba(6, 214, 160, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), name='Confidence Range',
        hoverinfo='skip', showlegend=True
    ))

    # "today" marker
    fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper",
                  line=dict(color="red", width=2, dash="dot"))
    fig.add_annotation(x=last_date, y=1, yref="paper", text="Today", showarrow=False, yshift=10,
                       font=dict(color="red"))

    fig.update_layout(title=dict(text=f'{company} - {title_suffix}', font=dict(size=16)),
                      xaxis_title='Date', yaxis_title='Price ($)', hovermode='x unified',
                      template='plotly_white', height=500, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

with tab2:
    st.plotly_chart(_cont_plot(60, "60d History + Future (Continuous)", chart_style), use_container_width=True)

with tab3:
    st.plotly_chart(_cont_plot(30, "30d History + Future (Continuous)", chart_style), use_container_width=True)

# ---------- Optional Volume (separate chart) ----------
if show_vol and "Volume" in stock_data.columns:
    vfig = go.Figure()
    vfig.add_trace(go.Bar(x=stock_data["Date"], y=stock_data["Volume"], name="Volume",
                          marker_color='rgba(100,150,250,0.6)'))
    vfig.update_layout(template='plotly_white', height=280, title=f"{company} · Volume (full history)",
                       xaxis_title="Date", yaxis_title="Shares", showlegend=False)
    st.plotly_chart(vfig, use_container_width=True)

# ---------- Summary table + download ----------
current = float(stock_data['Close'].iloc[-1])
trend = "BULLISH 📈" if forecast_vals[-1] > current else "BEARISH 📉"
total_change = (forecast_vals[-1] / current - 1.0) * 100.0

st.subheader("Forecast Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Current", f"${current:,.2f}")
c2.metric(f"Day {horizon}", f"${forecast_vals[-1]:,.2f}", f"{total_change:+.2f}%")
c3.metric("Trend", trend)

summary = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": forecast_vals,
    "Pct_Change_From_Current": (forecast_vals / current - 1.0) * 100.0,
    "Lower": lower, "Upper": upper
})
st.dataframe(summary, use_container_width=True)
st.download_button("Download forecast CSV",
                   data=summary.to_csv(index=False).encode("utf-8"),
                   file_name=f"{company}_forecast_{horizon}d.csv",
                   mime="text/csv", use_container_width=True)

# ---------- Notes (persistent) ----------
st.subheader("Notes (saved to user_notes.csv)")
notes_df = load_notes()
with st.form("add_note_form", clear_on_submit=True):
    note_text = st.text_area("Write your note:", height=100, help=notes_help)
    submitted = st.form_submit_button("Save note")
    if submitted and note_text.strip():
        notes_df = add_note(company, note_text.strip())
        st.success("Saved!")

colN1, colN2 = st.columns([1,2])
with colN1:
    only_this = st.toggle("Show only this company", True)
with colN2:
    search_q = st.text_input("Search notes (keyword)", "")
view_notes = notes_df.copy()
if only_this:
    view_notes = view_notes[view_notes["company"] == company]
if search_q.strip():
    view_notes = view_notes[view_notes["text"].str.contains(search_q, case=False, na=False)]
st.dataframe(view_notes.sort_values("timestamp", ascending=False), use_container_width=True)
st.download_button("Download all notes CSV",
                   notes_df.to_csv(index=False).encode("utf-8"),
                   file_name="user_notes.csv", use_container_width=True)

# ---------- Mini Chat (data-aware) ----------
st.subheader("Ask the data (mini chat)")
if "chat" not in st.session_state: st.session_state.chat = []
for role, msg in st.session_state.chat:
    with st.chat_message(role): st.markdown(msg)

GLOSSARY = {
    "mape": "Mean Absolute Percentage Error — average percent miss. Lower is better.",
    "mae": "Mean Absolute Error — average absolute dollar miss.",
    "r2": "R² — how much variance the model explains (0 to 1).",
    "direction": "Share of days where predicted up/down matches actual up/down.",
    "confidence": "Confidence range shows uncertainty, widening into the future.",
    "volatility": "How much price wiggles; we estimate from a rolling std of closes.",
    "gradient boosting": "Ensemble of trees that learns step-by-step (non-linear power).",
    "ridge": "Simple linear model with L2 regularization (fast baseline).",
    "trend feature": "(MA5 − MA20) / MA20, a short-vs-long tilt.",
}

future_dates_list = [d.date() for d in future_dates]

def answer_query(q: str) -> str:
    ql = q.lower().strip()

    # Definitions
    m = re.search(r"(what is|define)\s+(.+)", ql)
    if m:
        term = m.group(2).strip().replace("?", "")
        for k,v in GLOSSARY.items():
            if k in term:
                return f"**{k.title()}** — {v}"
        return ("Ask me about MAPE, MAE, R², direction, confidence, volatility, gradient boosting, ridge, trend feature.")

    # Split / model
    if "how did you split" in ql or "train test" in ql:
        return f"We used a chronological split with Train {train_pct}% and Test {100-train_pct}%."
    if "which model" in ql or "what model" in ql:
        return f"Best model: **{model_name}** (lowest MAPE). MAPE **{mape_val:.2f}%**, MAE **${mae:.2f}**, R² **{r2:.3f}**, Direction **{direction_accuracy:.1f}%**."

    # Point lookup: "close on 2023-11-02"
    m = re.search(r"(open|high|low|close|volume)\s+on\s+(\d{4}-\d{2}-\d{2})", ql)
    if m:
        col, dstr = m.group(1).capitalize(), m.group(2)
        row = stock_data[stock_data["Date"].dt.date == pd.to_datetime(dstr).date()]
        if row.empty: return f"I couldn't find data on {dstr}."
        val = float(row.iloc[0][col])
        return f"{company} {col} on {dstr}: **${val:,.2f}**" if col!="Volume" else f"{company} Volume on {dstr}: **{int(val):,}**"

    # Averages: "average close last 30 days"
    m = re.search(r"(avg|average)\s+(open|high|low|close|volume)\s+last\s+(\d+)\s+days", ql)
    if m:
        col, nd = m.group(2).capitalize(), int(m.group(3))
        val = stock_data[col].tail(nd).mean()
        return f"Average {col} over last {nd} days: **{val:,.2f}**" if col!="Volume" else f"Average Volume last {nd} days: **{int(val):,}**"

    # Max/min between dates
    m = re.search(r"(max|min)\s+(open|high|low|close)\s+between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", ql)
    if m:
        kind, col, d1, d2 = m.groups()
        mask = (stock_data["Date"]>=pd.to_datetime(d1)) & (stock_data["Date"]<=pd.to_datetime(d2))
        sub = stock_data.loc[mask, col.capitalize()]
        if sub.empty: return "No rows in that window."
        val = sub.max() if kind=="max" else sub.min()
        return f"{kind.upper()} {col} in that window: **${val:,.2f}**"

    # Forecast by date or step
    m = re.search(r"forecast\s+on\s+(\d{4}-\d{2}-\d{2})", ql)
    if m:
        d = pd.to_datetime(m.group(1)).date()
        if d in future_dates_list:
            i = future_dates_list.index(d)
            return f"Forecast for {d}: **${forecast_vals[i]:,.2f}**"
        return "That date is outside the current forecast window."
    m = re.search(r"forecast\s+in\s+(\d+)\s+days?", ql)
    if m:
        i = int(m.group(1)) - 1
        if 0 <= i < len(forecast_vals): return f"Forecast in {i+1} days: **${forecast_vals[i]:,.2f}**"
        return "That step is outside the current forecast window."

    # Quick metrics
    if "mape" in ql: return f"MAPE (avg % miss): **{mape_val:.2f}%**"
    if "mae" in ql:  return f"MAE (avg $ miss): **${mae:.2f}**"
    if "r2" in ql or "r²" in ql: return f"R² (fit quality): **{r2:.4f}**"
    if "direction" in ql or "dir" in ql: return f"Direction accuracy: **{direction_accuracy:.1f}%**"
    if "trend" in ql:
        trend_now = "BULLISH 📈" if forecast_vals[-1] > float(stock_data['Close'].iloc[-1]) else "BEARISH 📉"
        return f"Trend: **{trend_now}**"
    if "current" in ql or "price now" in ql:
        return f"Current close: **${float(stock_data['Close'].iloc[-1]):,.2f}**"

    # Show last N rows
    m = re.search(r"last\s+(\d+)\s+rows", ql)
    if m:
        n = max(1, min(int(m.group(1)), 100))
        st.dataframe(stock_data[["Date","Open","High","Low","Close","Volume"]].tail(n), use_container_width=True)
        return f"Displayed last **{n}** rows above."

    # Notes helpers
    if "notes" in ql or "my notes" in ql:
        df = load_notes()
        df = df if df.empty else df.sort_values("timestamp", ascending=False).head(10)
        if df.empty: return "You have no saved notes yet."
        st.dataframe(df, use_container_width=True)
        return "Shown your latest notes above."
    m = re.search(r"find note about\s+(.+)", ql)
    if m:
        term = m.group(1).strip()
        df = load_notes()
        hit = df[df["text"].str.contains(term, case=False, na=False)]
        if hit.empty: return f"No notes mentioning **{term}**."
        st.dataframe(hit.sort_values("timestamp", ascending=False), use_container_width=True)
        return f"Found {len(hit)} note(s) mentioning **{term}** (shown above)."

    return ("I can help with:\n"
            "- `close on 2023-11-02`, `average close last 30 days`\n"
            "- `max close between 2023-01-01 and 2023-03-01`\n"
            "- `forecast on 2024-12-15` or `forecast in 7 days`\n"
            "- `what model`, `how did you split`, `what is MAPE / MAE / R2 / confidence`\n"
            "- `notes` or `find note about earnings`")

prompt = st.chat_input("Ask about this dataset, forecast, stocks, or your notes…")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    reply = answer_query(prompt)
    st.session_state.chat.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

st.caption("⚠️ Educational purposes only. Markets are volatile.")
