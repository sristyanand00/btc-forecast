import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timezone

st.set_page_config(page_title="BTC Next-Hour Forecast", page_icon="₿", layout="wide")

# ================================================================
# PART C — PERSISTENCE (Local File)
# ================================================================
HISTORY_FILE = "prediction_history.jsonl"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_prediction(record):
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def update_actuals(history, prices):
    price_dict = {str(t): float(p) for t, p in zip(prices.index, prices.values)}
    updated = []
    for r in history:
        if r["actual"] is None:
            actual_price = price_dict.get(r["predicted_for"])
            if actual_price:
                r["actual"] = round(actual_price, 2)
                r["hit"]    = int(r["lower_95"] <= actual_price <= r["upper_95"])
        updated.append(r)
    return updated

# ================================================================
# DATA FETCH
# ================================================================
@st.cache_data(ttl=300)
def fetch_btc(limit=500):
    url = "https://data-api.binance.vision/api/v3/klines"
    r   = requests.get(
        url,
        params={"symbol": "BTCUSDT", "interval": "1h", "limit": limit},
        timeout=15
    )
    df  = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["close"]     = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.set_index("open_time")

# ================================================================
# PREDICTION MODEL — matches backtest exactly
# ================================================================
def predict_next_hour(prices):
    log_ret    = np.log(prices / prices.shift(1)).dropna()
    recent_100 = log_ret.iloc[-100:]
    recent_vol = log_ret.iloc[-20:].std()
    full_vol   = recent_100.std()
    try:
        df_t, loc_t, scale_t = stats.t.fit(recent_100)
        scale_adj = scale_t * (recent_vol / full_vol) * 1.4 if full_vol > 0 else scale_t * 1.4
        df_t = max(df_t, 4.0)
    except:
        df_t, loc_t, scale_adj = 5.0, 0.0, recent_vol
    S0         = float(prices.iloc[-1])
    sim_ret    = stats.t.rvs(df_t, loc=loc_t, scale=scale_adj, size=50_000)
    sim_prices = S0 * np.exp(sim_ret)
    lower = float(np.percentile(sim_prices, 2.5))
    upper = float(np.percentile(sim_prices, 97.5))
    return lower, upper

# ================================================================
# PART A BACKTEST METRICS — hardcoded from Colab run
# ================================================================
METRICS = {
    "coverage_95"     : 0.9500,
    "avg_width"       : 1450.70,
    "mean_winkler_95" : 1884.17
}

# ================================================================
# DASHBOARD
# ================================================================
st.title("₿ BTC/USDT — Next Hour Forecast")
st.caption("Student-t GBM + Volatility Clustering | AlphaI × Polaris Challenge")

with st.spinner("Fetching live BTC data from Binance..."):
    df = fetch_btc(500)

prices        = df["close"]
current_price = float(prices.iloc[-1])
lower, upper  = predict_next_hour(prices)
next_bar_time = prices.index[-1] + pd.Timedelta(hours=1)
now_utc       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ── Section 1: Live Prediction ───────────────────────────────────
st.subheader("🔴 Live Prediction")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current BTC",    f"${current_price:,.2f}")
c2.metric("Predicted Low",  f"${lower:,.2f}")
c3.metric("Predicted High", f"${upper:,.2f}")
c4.metric("Range Width",    f"${upper - lower:,.2f}")

st.divider()

# ── Section 2: Backtest Metrics ──────────────────────────────────
st.subheader("📊 30-Day Backtest Results (Part A — 720 predictions)")
m1, m2, m3 = st.columns(3)
m1.metric("Coverage 95%",    f"{METRICS['coverage_95']*100:.1f}%",   delta="Target: 95.0%")
m2.metric("Avg Range Width", f"${METRICS['avg_width']:,.0f}")
m3.metric("Mean Winkler",    f"{METRICS['mean_winkler_95']:,.2f}",   delta="Lower = Better")

st.divider()

# ── Section 3: Chart ─────────────────────────────────────────────
st.subheader("📈 Last 50 Hours + Next Hour Forecast Range")
last50 = prices.iloc[-50:]
fig    = go.Figure()

fig.add_trace(go.Scatter(
    x=last50.index, y=last50.values,
    name="BTC Close",
    line=dict(color="#F7931A", width=2)
))

fig.add_shape(
    type="rect",
    x0=str(last50.index[-1]), x1=str(next_bar_time),
    y0=lower, y1=upper,
    fillcolor="rgba(0,200,100,0.2)",
    line=dict(width=0)
)

fig.add_hline(
    y=lower, line_dash="dot", line_color="#00c864",
    annotation_text=f"Low ${lower:,.0f}",
    annotation_position="bottom right"
)
fig.add_hline(
    y=upper, line_dash="dot", line_color="#00c864",
    annotation_text=f"High ${upper:,.0f}",
    annotation_position="top right"
)

fig.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Time (UTC)",
    yaxis_title="Price (USDT)",
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Last closed bar: {prices.index[-1]} UTC | Predicting for: {next_bar_time} UTC | Page loaded: {now_utc} | Refreshes every 5 min")

st.divider()

# ================================================================
# PART C — Prediction History
# ================================================================
st.subheader("🕐 Prediction History (Part C)")
st.caption("Every dashboard visit saves a prediction. Actuals fill in automatically when the bar closes.")

history = load_history()
history = update_actuals(history, prices)


# Save current prediction if not already saved for this bar
predicted_for = str(next_bar_time)
already_saved = any(r["predicted_for"] == predicted_for for r in history)

if not already_saved:
    new_record = {
        "saved_at"     : now_utc,
        "predicted_for": predicted_for,
        "lower_95"     : round(lower, 2),
        "upper_95"     : round(upper, 2),
        "actual"       : None,
        "hit"          : None
    }
    save_prediction(new_record)
    history.append(new_record)

# Rewrite file with updated actuals
with open(HISTORY_FILE, "w") as f:
    for r in history:
        f.write(json.dumps(r) + "\n")

if history:
    hist_df = pd.DataFrame(history[::-1])  # newest first
    hist_df["width"] = (hist_df["upper_95"] - hist_df["lower_95"]).round(2)
    hist_df["hit"]   = hist_df["hit"].apply(
        lambda x: "✅ Hit" if x == 1 else ("❌ Miss" if x == 0 else "⏳ Pending")
    )
    hist_df = hist_df.rename(columns={
        "saved_at"     : "Saved At (UTC)",
        "predicted_for": "Predicting For",
        "lower_95"     : "Lower $",
        "upper_95"     : "Upper $",
        "actual"       : "Actual $",
        "width"        : "Width $",
        "hit"          : "Result"
    })
    st.dataframe(hist_df, use_container_width=True, height=300)

    # Summary stats from history
    hits    = history[-50:]  # last 50 predictions with actuals
    settled = [r for r in hits if r["hit"] is not None]
    if settled:
        live_coverage = sum(r["hit"] for r in settled) / len(settled)
        st.info(f"Live coverage from {len(settled)} settled predictions: **{live_coverage*100:.1f}%** (target: 95.0%)")
else:
    st.info("No history yet. Visit again after the next bar closes to see predictions fill in.")
