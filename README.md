Your Stock Forecaster

A tiny, friendly Streamlit app that helps you explore stock prices, learn basic forecasting, save notes, and ask a mini chat about your data.

Built with care by Anita · Sahibjeet · Bhuvesh · Gautham.

Educational only. Markets can be wild — learn gently, risk carefully.

✨ What it does (in simple words)

Pick a company from your CSV

Choose a Train % (earlier rows train; recent rows test)

Pick Forecast days (e.g., 15)

See charts: Validation, Last 60d + Future, Last 30d + Future

Toggle a separate Volume chart

Read metrics (MAPE / MAE / R² / Direction)

See uncertainty bands that widen into the future

Save Notes (kept in user_notes.csv)

Ask the Mini Chat questions about prices, metrics, forecasts, and your notes

🚀 Quickstart
# 1) Create a fresh env (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install
pip install streamlit pandas numpy scikit-learn plotly

# 3) Put your CSV next to app.py
#    name it exactly: cleaned_data(in).csv

# 4) Run
streamlit run app.py


Example CSV row

Date,Company,Open,High,Low,Close,Volume
2023-01-03,AAPL,130.28,130.90,124.17,125.07,112117500


The app auto-cleans values like "1,234" or "$123".

🧭 How to use (step by step)

Pick a company from the dropdown.

Set Train size (%) and Forecast days.

(Optional) Show Volume to see a separate volume chart.

Read the metrics:

MAPE = average % miss (lower is better)

MAE = average $ miss (lower is better)

R² = fit quality (0–1, higher is better)

Direction = how often we got up/down right

Explore the charts and bands.

Add Notes (saved to user_notes.csv).

Use the Mini Chat:

close on 2023-11-02

average close last 30 days

max close between 2023-01-01 and 2023-03-01

forecast in 7 days / forecast on 2024-12-15

what is mape / what model / how did you split

notes or find note about earnings

🧠 How prediction works (super simple)

We build simple features from prices: MA(5/10/20), momentum, volatility, a short-vs-long trend, and lagged closes.

We keep time order: early data → train, recent data → test.

We try Ridge Regression (linear) and Gradient Boosting (trees) and pick the lower MAPE model.

We forecast day-by-day: predict tomorrow → cap the jump to ±3% → add a tiny fading trend → update features → repeat.

The shaded band uses recent volatility, so it widens further out.

Your example

Today = $100 → model says $104
Cap to +3% ⇒ $103, add tiny trend +$0.20 ⇒ $103.20 (tomorrow).
Update averages → predict day 2… and so on.


🔗 Downloading CSV from the app
# inside app.py where you want a download:
csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button("Download forecast CSV", data=csv_bytes,
                   file_name=f"{company}_forecast_{horizon}d.csv", mime="text/csv")

☁️ Shipping large CSVs to GitHub (pick one)

≤100 MB: commit normally.

>100 MB: use Git LFS

git lfs install
git lfs track "*.csv"
git add .gitattributes data/my_big.csv
git commit -m "Add big CSV via LFS"
git push origin main


Very large / stable link: upload as a GitHub Release asset and download in the app on first run.

❓ Troubleshooting

“Missing file: cleaned_data(in).csv” → put your CSV next to app.py with that exact name.

“CSV must contain 'Date' and 'Company'” → check headers and casing.

Too few rows → you’ll want ~60+ rows for a nice demo.

Module not found → pip install streamlit pandas numpy scikit-learn plotly.

📜 License

MIT — free to use & learn. Please keep attribution.

🙏 Credits

Anita · Sahibjeet · Bhuvesh · Gautham
we built it simple on purpose — so anyone can learn, one day at a time. 💫
