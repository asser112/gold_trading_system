import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml
import sqlite3
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Gold Trading System Dashboard")

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to DB
conn = sqlite3.connect(config['data']['db_path'])

# Sidebar
st.sidebar.header("Parameters")
interval = st.sidebar.selectbox("Interval", ["m1", "m5", "m15", "h1"], index=1)
lookback_days = st.sidebar.slider("Lookback days", 1, 90, 30)

# Load OHLC data
def load_ohlc(interval, days):
    query = f"SELECT * FROM ohlc_{interval} WHERE timestamp >= datetime('now', '-{days} days') ORDER BY timestamp"
    df = pd.read_sql(query, conn, index_col='timestamp', parse_dates=['timestamp'])
    return df

df = load_ohlc(interval, lookback_days)

# Equity curve (if backtest results exist)
st.header("Equity Curve")
try:
    # Try to load from HTML (backtesting.py creates an HTML file)
    import os
    if os.path.exists('backtest_reports/equity_curve.html'):
        with open('backtest_reports/equity_curve.html', 'r') as f:
            html = f.read()
        st.components.v1.html(html, height=600)
    else:
        st.warning("No backtest results found. Run backtester first.")
except Exception as e:
    st.warning(f"Could not load equity curve: {e}")

# Latest signal
st.header("Latest Signal")
signal_file = config['trading']['signal_file']
try:
    with open(signal_file, 'r') as f:
        content = f.read()
    if content:
        import json
        signal = json.loads(content)
        st.write(f"**Timestamp:** {signal['timestamp']}")
        st.write(f"**Signal:** {signal['signal']}")
        st.write(f"**Confidence:** {signal['confidence']:.2f}")
        st.write(f"**SL:** {signal['sl']:.2f}")
        st.write(f"**TP:** {signal['tp']:.2f}")
    else:
        st.info("No current signal.")
except:
    st.error("Could not read signal file.")

# News sentiment chart
st.header("News Sentiment")
try:
    news_df = pd.read_sql("SELECT timestamp, sentiment_score FROM news_processed ORDER BY timestamp DESC LIMIT 500", conn, parse_dates=['timestamp'])
    if not news_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=news_df['timestamp'], y=news_df['sentiment_score'], mode='markers', name='Sentiment'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No news data available.")
except Exception as e:
    st.info("No news data available.")

# Recent trades (if any)
st.header("Recent Trades")
try:
    trades = pd.read_sql("SELECT * FROM trades ORDER BY close_time DESC LIMIT 20", conn)
    st.dataframe(trades)
except:
    st.info("No trade history.")

# Live price
st.header("Live Price")
if not df.empty:
    last_price = df['close'].iloc[-1]
    st.metric("Last Price", f"{last_price:.2f}")
    # Add simple chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close'])])
    st.plotly_chart(fig, use_container_width=True)