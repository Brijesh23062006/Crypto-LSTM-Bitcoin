
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import plotly.graph_objects as go

from streamlit_autorefresh import st_autorefresh


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Crypto LSTM Prediction",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin LSTM Prediction Dashboard")

st.caption(
    "Cryptocurrency price monitoring and machine learning forecasting"
)


# ---------------------------------
# LIVE BITCOIN PRICE GRAPH
# ---------------------------------

import requests
from datetime import datetime

st.subheader("🔴 Live Bitcoin Price Movement")

# Refresh dashboard every 5 seconds
st_autorefresh(
    interval=5000,
    key="live_btc_refresh"
)

# Store live prices
if "live_prices" not in st.session_state:
    st.session_state.live_prices = []


# Get latest Bitcoin price
def get_live_price():

    url = "https://api.binance.com/api/v3/ticker/price"

    response = requests.get(
        url,
        params={"symbol": "BTCUSDT"},
        timeout=10
    )

    response.raise_for_status()

    return float(response.json()["price"])


# Fetch and store new price
try:

    live_price = get_live_price()

    current_time = datetime.now()

    st.session_state.live_prices.append({
        "Time": current_time,
        "Price": live_price
    })

    # Keep recent 300 samples
    st.session_state.live_prices = (
        st.session_state.live_prices[-300:]
    )

    st.success("Live Bitcoin data connected")

except Exception as error:

    st.warning(
        "Live price unavailable. "
        "Check your internet connection."
    )


# Display live graph
if st.session_state.live_prices:

    live_data = pd.DataFrame(
        st.session_state.live_prices
    )

    current_price = live_data["Price"].iloc[-1]

    col1, col2 = st.columns(2)

    col1.metric(
        "Live BTC Price",
        f"${current_price:,.2f}"
    )

    col2.metric(
        "Last Update",
        datetime.now().strftime("%H:%M:%S")
    )

    # Create graph
    live_fig = go.Figure()

    live_fig.add_trace(
        go.Scatter(
            x=live_data["Time"],
            y=live_data["Price"],
            mode="lines",
            name="Live Bitcoin Price"
        )
    )

    live_fig.update_layout(
        title="Bitcoin Live Market Movement",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        live_fig,
        use_container_width=True
    )

    st.caption(
        "Live market data is periodically fetched. "
        "The graph changes when new price samples arrive."
    )


# ---------------------------------
# AUTO REFRESH
# ---------------------------------

st_autorefresh(
    interval=60000,
    key="bitcoin_refresh"
)


# ---------------------------------
# LOAD MODEL AND SCALERS
# ---------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "crypto_model/lstm_model.keras"
    )

    scaler = joblib.load(
        "crypto_model/feature_scaler.pkl"
    )

    close_scaler = joblib.load(
        "crypto_model/close_scaler.pkl"
    )

    with open("crypto_model/features.json", "r") as f:
        features = json.load(f)

    return model, scaler, close_scaler, features


model, scaler, close_scaler, features = load_model()


# ---------------------------------
# DOWNLOAD HISTORICAL DATA
# ---------------------------------

@st.cache_data(ttl=60)
def get_data():

    data = yf.download(
        "BTC-USD",
        period="2y",
        interval="1d",
        auto_adjust=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[
        ["Open", "High", "Low", "Close", "Volume"]
    ]

    data = data.dropna()

    return data


data = get_data()


# ---------------------------------
# FEATURE ENGINEERING
# ---------------------------------

def create_features(data):

    data = data.copy()

    data["MA_7"] = data["Close"].rolling(7).mean()

    data["MA_30"] = data["Close"].rolling(30).mean()

    data["Daily_Return"] = data["Close"].pct_change()

    data["Log_Return"] = np.log(
        data["Close"] / data["Close"].shift(1)
    )

    data["Volatility"] = (
        data["Daily_Return"].rolling(7).std()
    )

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["Volume_Change"] = data["Volume"].pct_change()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    return data


data = create_features(data)


# ---------------------------------
# LIVE PRICE
# ---------------------------------

latest_price = float(data["Close"].iloc[-1])

previous_price = float(data["Close"].iloc[-2])

change = latest_price - previous_price

change_percent = (
    change / previous_price
) * 100


# ---------------------------------
# PRICE DISPLAY
# ---------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Bitcoin Price",
    f"${latest_price:,.2f}",
    f"{change_percent:.2f}%"
)

col2.metric(
    "Previous Close",
    f"${previous_price:,.2f}"
)

col3.metric(
    "Latest Data Date",
    str(data.index[-1].date())
)


# ---------------------------------
# HISTORICAL PRICE GRAPH
# ---------------------------------

st.subheader("Bitcoin Historical Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        name="Bitcoin Price",
        mode="lines"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA_7"],
        name="7 Day Moving Average"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA_30"],
        name="30 Day Moving Average"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------
# CANDLESTICK GRAPH
# ---------------------------------

st.subheader("Bitcoin Candlestick Chart")

candle = go.Figure(
    data=[
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="BTC"
        )
    ]
)

candle.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=500
)

st.plotly_chart(
    candle,
    use_container_width=True
)


# ---------------------------------
# LSTM PREDICTION
# ---------------------------------

st.subheader("LSTM Next Day Prediction")

sequence_length = 60

if len(data) >= sequence_length:

    latest_data = data[features].iloc[-sequence_length:]

    scaled_data = scaler.transform(
        latest_data
    )

    input_data = scaled_data.reshape(
        1,
        sequence_length,
        len(features)
    )

    prediction_scaled = model.predict(
        input_data,
        verbose=0
    )

    prediction = close_scaler.inverse_transform(
        prediction_scaled
    )[0][0]

    prediction_change = (
        prediction - latest_price
    )

    prediction_percent = (
        prediction_change / latest_price
    ) * 100

    st.metric(
        "Predicted Next Day Closing Price",
        f"${prediction:,.2f}",
        f"{prediction_percent:.2f}%"
    )

    st.caption(
        "This is a model forecast, not a guaranteed future price."
    )


    # ---------------------------------
    # PREDICTION GRAPH
    # ---------------------------------

    historical = data["Close"].iloc[-60:]

    dates = list(historical.index)

    prices = list(historical.values)

    next_date = dates[-1] + pd.Timedelta(days=1)

    dates.append(next_date)

    prices.append(prediction)

    prediction_fig = go.Figure()

    prediction_fig.add_trace(
        go.Scatter(
            x=dates[:-1],
            y=prices[:-1],
            name="Historical Price",
            mode="lines"
        )
    )

    prediction_fig.add_trace(
        go.Scatter(
            x=dates[-2:],
            y=prices[-2:],
            name="Predicted Movement",
            mode="lines+markers",
            line=dict(dash="dash")
        )
    )

    prediction_fig.update_layout(
        template="plotly_dark",
        title="Historical Bitcoin and LSTM Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=450
    )

    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )


# ---------------------------------
# RSI GRAPH
# ---------------------------------

st.subheader("Bitcoin RSI")

rsi_fig = go.Figure()

rsi_fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["RSI"],
        name="RSI",
        mode="lines"
    )
)

rsi_fig.add_hline(y=70)

rsi_fig.add_hline(y=30)

rsi_fig.update_layout(
    template="plotly_dark",
    yaxis_title="RSI",
    height=350
)

st.plotly_chart(
    rsi_fig,
    use_container_width=True
)


# ---------------------------------
# MACD GRAPH
# ---------------------------------

st.subheader("Bitcoin MACD")

macd_fig = go.Figure()

macd_fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MACD"],
        name="MACD",
        mode="lines"
    )
)

macd_fig.update_layout(
    template="plotly_dark",
    yaxis_title="MACD",
    height=350
)

st.plotly_chart(
    macd_fig,
    use_container_width=True
)


# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.caption(
    "Bitcoin LSTM Forecasting Project | "
    "Historical data and model-based predictions"
)
