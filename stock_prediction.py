"""
stock_prediction.py

Quick command-line tool for training simple models and printing a short
forecast summary. It loads `cleaned_data.csv`, fits a couple of models, then
prints metrics and a 15-day forecast that is easy to read. For learning and
experimentation — not production trading advice.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv("cleaned_data.csv")
data['Date'] = pd.to_datetime(data['Date'])

# User input
print("Available companies:", ', '.join(data['Company'].unique()))
company_name = input("\nEnter stock symbol: ").strip().upper()

stock_data = data[data['Company'] == company_name].copy()
if stock_data.empty:
    print(f"No data found for {company_name}")
    exit()

stock_data = stock_data.sort_values('Date').reset_index(drop=True)
print(f"\nAnalyzing {company_name}: {len(stock_data)} observations")
print(f"Date range: {stock_data['Date'].min().date()} to {stock_data['Date'].max().date()}")

# Create features
def create_features(df):
    df = df.copy()
    df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Momentum_5'] = df['Close'].pct_change(5).rolling(window=3).mean()
    df['Momentum_10'] = df['Close'].pct_change(10).rolling(window=3).mean()
    df['Volatility'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['Trend'] = (df['MA_5'] - df['MA_20']) / df['MA_20']
    df['Close_1'] = df['Close'].shift(1)
    df['Close_2'] = df['Close'].shift(2)
    df['Close_5'] = df['Close'].shift(5)
    return df.fillna(method='bfill').fillna(method='ffill')

stock_data = create_features(stock_data)

# Split data
train_size = int(len(stock_data) * 0.8)
train_df = stock_data[:train_size]
test_df = stock_data[train_size:]

feature_cols = ['MA_5', 'MA_10', 'MA_20', 'Momentum_5', 'Momentum_10', 
                'Volatility', 'Trend', 'Close_1', 'Close_2', 'Close_5']

X_train = train_df[feature_cols].values
y_train = train_df['Close'].values
X_test = test_df[feature_cols].values
y_test = test_df['Close'].values

# Train models
print("\nTraining models...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, 
                                     learning_rate=0.05, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
mape_gb = np.mean(np.abs((y_test - gb_pred) / y_test)) * 100

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
mape_ridge = np.mean(np.abs((y_test - ridge_pred) / y_test)) * 100

# Select best model
if mape_gb < mape_ridge:
    model = gb_model
    model_name = "Gradient Boosting"
    predictions = gb_pred
    mape = mape_gb
else:
    model = ridge_model
    model_name = "Ridge Regression"
    predictions = ridge_pred
    mape = mape_ridge

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n{'='*60}")
print(f"BEST MODEL: {model_name}")
print(f"{'='*60}")
print(f"Accuracy: {100 - mape:.2f}%")
print(f"MAPE: {mape:.2f}%")
print(f"MAE: ${mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Reliability analysis
print(f"\n{'='*60}")
print("PREDICTION RELIABILITY ANALYSIS")
print(f"{'='*60}")

# Direction accuracy
y_test_changes = np.diff(y_test)
pred_changes = np.diff(predictions)
direction_accuracy = np.mean((y_test_changes > 0) == (pred_changes > 0)) * 100

print(f"\n1️⃣ Direction Accuracy: {direction_accuracy:.2f}%")
if direction_accuracy > 55:
    print(f"   ✅ GOOD: Better than random (50%)")
elif direction_accuracy > 50:
    print(f"   ⚠️ FAIR: Slightly better than random")
else:
    print(f"   ❌ POOR: Not reliable")

print(f"\n2️⃣ MAPE: {mape:.2f}%")
if mape < 3:
    print(f"   ✅ EXCELLENT: Very accurate")
elif mape < 5:
    print(f"   ✅ GOOD: Reliable for forecasting")
elif mape < 10:
    print(f"   ⚠️ FAIR: Use with caution")
else:
    print(f"   ❌ POOR: Not reliable")

print(f"\n3️⃣ R² Score: {r2:.4f}")
if r2 > 0.9:
    print(f"   ✅ EXCELLENT: Explains {r2*100:.1f}% of price movements")
elif r2 > 0.7:
    print(f"   ✅ GOOD: Explains {r2*100:.1f}% of movements")
elif r2 > 0.5:
    print(f"   ⚠️ FAIR: Explains {r2*100:.1f}%")
else:
    print(f"   ❌ POOR: Only explains {r2*100:.1f}%")

# Overall rating
score = 0
if mape < 5: score += 3
elif mape < 10: score += 2
else: score += 1

if r2 > 0.8: score += 3
elif r2 > 0.6: score += 2
else: score += 1

if direction_accuracy > 55: score += 2
elif direction_accuracy > 50: score += 1

print(f"\n{'='*60}")
print("OVERALL RELIABILITY RATING")
print(f"{'='*60}")

if score >= 7:
    rating = "⭐⭐⭐⭐⭐ HIGHLY RELIABLE"
    recommendation = "Safe for investment decisions with risk management"
elif score >= 5:
    rating = "⭐⭐⭐⭐ RELIABLE"
    recommendation = "Good for short-term, combine with other analysis"
elif score >= 3:
    rating = "⭐⭐⭐ MODERATELY RELIABLE"
    recommendation = "Use as ONE indicator, not sole decision maker"
else:
    rating = "⭐⭐ LOW RELIABILITY"
    recommendation = "NOT recommended - model needs improvement"

print(f"Rating: {rating}")
print(f"Score: {score}/8\n")
print(f"💡 {recommendation}")

# Generate forecast
print(f"\n{'='*60}")
print("GENERATING 15-DAY FORECAST")
print(f"{'='*60}")

def forecast_realistic(model, last_data, n_days=15):
    forecast = []
    features = last_data[feature_cols].iloc[-1].values.copy()
    current_price = last_data['Close'].iloc[-1]
    recent_trend = (last_data['Close'].iloc[-1] - last_data['Close'].iloc[-30]) / 30
    
    for day in range(n_days):
        pred_price = model.predict(features.reshape(1, -1))[0]
        
        # Limit daily change to 3%
        max_change = current_price * 0.03
        change = np.clip(pred_price - current_price, -max_change, max_change)
        
        # Add decaying trend
        trend = recent_trend * (0.95 ** day)
        pred_price = current_price + change + trend
        
        forecast.append(pred_price)
        
        # Update features
        features[7] = features[8]
        features[8] = current_price
        features[0] = (features[0] * 4 + pred_price) / 5
        features[1] = (features[1] * 9 + pred_price) / 10
        features[2] = (features[2] * 19 + pred_price) / 20
        
        current_price = pred_price
    
    return np.array(forecast)

forecast_15 = forecast_realistic(model, stock_data, n_days=15)

last_date = stock_data['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=15)
test_dates = stock_data['Date'][train_size:].values

volatility = stock_data['Close'].tail(60).std()

# Create three plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16))

# PLOT 1: Full Validation (No Future)
ax1.plot(test_dates, y_test, label='Actual Price', color='#2E86AB', linewidth=1.5)
ax1.plot(test_dates, predictions, label='Predicted Price', color='#F77F00', 
         linewidth=1.5, linestyle='--', alpha=0.8)
ax1.set_title(f'{company_name} - Model Validation Only (No Future Prediction)\n' + 
              f'{model_name} | Accuracy: {100-mape:.2f}% | MAPE: {mape:.2f}%', 
              fontsize=15, fontweight='bold', pad=15)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3)

# PLOT 2: Last 2 Months + 15 Days Future (No Gap)
lookback_days = 60
recent_data_idx = max(0, len(stock_data) - lookback_days)
recent_dates = stock_data['Date'][recent_data_idx:].values
recent_prices = stock_data['Close'][recent_data_idx:].values

# Combine historical and future for continuous line
combined_dates = np.concatenate([recent_dates, future_dates])
combined_prices = np.concatenate([recent_prices, forecast_15])

# Plot as one continuous line
ax2.plot(combined_dates[:len(recent_dates)], combined_prices[:len(recent_dates)], 
         label='Historical (Last 2 Months)', color='#2E86AB', linewidth=1.5, alpha=0.9)
ax2.plot(combined_dates[len(recent_dates)-1:], combined_prices[len(recent_dates)-1:], 
         label='Predicted (Next 15 Days)', color='#06D6A0', linewidth=1.5, alpha=0.95)

# Confidence interval - only for future
decay = np.linspace(1, 1.3, 15)
upper = forecast_15 + (volatility * decay)
lower = forecast_15 - (volatility * decay)
ax2.fill_between(future_dates, lower, upper, alpha=0.15, color='#06D6A0', 
                 label='Confidence Range')

# Today line
ax2.axvline(x=last_date, color='red', linestyle=':', linewidth=2, alpha=0.7, 
            label='Today', zorder=5)

ax2.set_title(f'{company_name} - 2 Months Historical + 15 Days Prediction (Continuous)', 
              fontsize=15, fontweight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11, framealpha=0.95, loc='best')
ax2.grid(True, alpha=0.3)

# PLOT 3: Last 1 Month + 15 Days Future (Zoomed, No Gap)
lookback_days_short = 30
recent_data_idx_short = max(0, len(stock_data) - lookback_days_short)
recent_dates_short = stock_data['Date'][recent_data_idx_short:].values
recent_prices_short = stock_data['Close'][recent_data_idx_short:].values

# Combine for continuous line
combined_dates_short = np.concatenate([recent_dates_short, future_dates])
combined_prices_short = np.concatenate([recent_prices_short, forecast_15])

# Plot as one continuous line
ax3.plot(combined_dates_short[:len(recent_dates_short)], combined_prices_short[:len(recent_dates_short)], 
         label='Historical (Last Month)', color='#2E86AB', linewidth=1.5, alpha=0.9)
ax3.plot(combined_dates_short[len(recent_dates_short)-1:], combined_prices_short[len(recent_dates_short)-1:], 
         label='Predicted (Next 15 Days)', color='#06D6A0', linewidth=1.5, alpha=0.95)

# Confidence interval
ax3.fill_between(future_dates, lower, upper, alpha=0.15, color='#06D6A0', 
                 label='Confidence Range')

# Today line
ax3.axvline(x=last_date, color='red', linestyle=':', linewidth=2, alpha=0.7, 
            label='Today', zorder=5)

ax3.set_title(f'{company_name} - 1 Month Historical + 15 Days Prediction (Continuous)', 
              fontsize=15, fontweight='bold', pad=15)
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=11, framealpha=0.95, loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Forecast summary
current = stock_data['Close'].iloc[-1]
print(f"\nCurrent Price: ${current:.2f}")

print(f"\n{'='*60}")
print("15-DAY PREDICTION (Daily Breakdown)")
print(f"{'='*60}")

# Show every day for 15 days
for day in range(1, 16):
    price = forecast_15[day-1]
    change = ((price / current) - 1) * 100
    date = (last_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d')
    
    print(f"  Day {day:2} ({date}): ${price:.2f} ({change:+.2f}%)")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

trend = "BULLISH 📈" if forecast_15[-1] > current else "BEARISH 📉"
total_change = ((forecast_15[-1] / current) - 1) * 100

print(f"Starting Price (Today): ${current:.2f}")
print(f"Ending Price (Day 15): ${forecast_15[-1]:.2f}")
print(f"Total Change: {total_change:+.2f}%")
print(f"15-Day Trend: {trend}")

print(f"\nExpected Price Range:")
print(f"  Lowest:  ${forecast_15.min():.2f}")
print(f"  Highest: ${forecast_15.max():.2f}")
print(f"  Average: ${forecast_15.mean():.2f}")

print(f"\n{'='*60}")