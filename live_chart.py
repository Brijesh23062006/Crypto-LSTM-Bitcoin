
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

from datetime import datetime
from streamlit_autorefresh import st_autorefresh


# --------------------------------
# PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Bitcoin Live Moving Graph",
    layout="wide"
)

st.title("Bitcoin Live Moving Graph")

st.write("Live Bitcoin market price")


# --------------------------------
# AUTOMATIC REFRESH
# --------------------------------

st_autorefresh(
    interval=2000,
    key="bitcoin_live_refresh"
)


# --------------------------------
# SESSION STORAGE
# --------------------------------

if "live_prices" not in st.session_state:

    st.session_state.live_prices = []


# --------------------------------
# FETCH LIVE PRICE
# --------------------------------

def get_live_price():

    url = "https://api.binance.com/api/v3/ticker/price"

    params = {
        "symbol": "BTCUSDT"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    result = response.json()

    return float(result["price"])


# --------------------------------
# GET NEW PRICE
# --------------------------------

try:

    price = get_live_price()

    current_time = datetime.now()

    # Add latest price
    st.session_state.live_prices.append(
        {
            "Time": current_time,
            "Price": price
        }
    )

    # Keep last 300 price samples
    st.session_state.live_prices = (
        st.session_state.live_prices[-300:]
    )

    st.success("Live market data connected")

except Exception as e:

    st.error(
        "Unable to fetch live price. "
        "Check your internet connection or API access."
    )

    st.caption(str(e))


# --------------------------------
# DISPLAY CURRENT PRICE
# --------------------------------

if st.session_state.live_prices:

    prices = pd.DataFrame(
        st.session_state.live_prices
    )

    latest_price = prices["Price"].iloc[-1]

    if len(prices) > 1:

        previous_price = prices["Price"].iloc[-2]

        difference = latest_price - previous_price

    else:

        difference = 0


    col1, col2 = st.columns(2)

    col1.metric(
        "Bitcoin Live Price",
        f"${latest_price:,.2f}",
        f"${difference:,.2f}"
    )

    col2.metric(
        "Latest Update",
        datetime.now().strftime("%H:%M:%S")
    )


    # --------------------------------
    # LIVE MOVING GRAPH
    # --------------------------------

    st.subheader("Live Bitcoin Price Movement")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prices["Time"],
            y=prices["Price"],
            mode="lines",
            name="BTC Live Price",
            line=dict(width=2)
        )
    )

    fig.update_layout(
        title="Bitcoin Live Price (BTC/USDT)",

        xaxis_title="Time",

        yaxis_title="Price (USDT)",

        template="plotly_dark",

        height=600,

        xaxis=dict(
            type="date",
            tickformat="%H:%M:%S"
        ),

        yaxis=dict(
            autorange=True
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------
    # LIVE DATA TABLE
    # --------------------------------

    st.subheader("Latest Price Updates")

    st.dataframe(
        prices.tail(20).sort_values(
            "Time",
            ascending=False
        ),

        use_container_width=True
    )


else:

    st.info("Waiting for live Bitcoin price...")


st.caption(
    "Updates every 2 seconds using Binance public market data. "
    "Price updates depend on market activity and API availability."
)
