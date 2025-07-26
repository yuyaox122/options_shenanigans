import streamlit as st
import yfinance as yf
import numpy as np
import datetime
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(S, K, T, r, market_price, option_type='call'):
    try:
        result = minimize_scalar(
            lambda sigma: (black_scholes_price(S, K, T, r, sigma, option_type) - market_price) ** 2,
            bounds=(0.01, 3.0),
            method='bounded'
        )
        return result.x if result.success else None
    except:
        return None

def plot_iv_smile(strikes, ivs, ticker_symbol, expiry):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=ivs, mode='lines+markers', name='IV'))
    fig.update_layout(
        title=f"Implied Volatility Smile for {ticker_symbol.upper()} ({expiry})",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        template="plotly_white"
    )
    return fig

st.title("🧮 Implied Volatility Estimator")
ticker_symbol = st.text_input("Enter stock ticker", value="AAPL")

try:
    ticker = yf.Ticker(ticker_symbol)
except:
    st.error("Invalid ticker symbol. Please try again.")
    st.stop()