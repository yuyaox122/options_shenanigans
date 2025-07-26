import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import plotly.graph_objects as go
import yfinance as yf
import options_greeks as og

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

def black_scholes(S, K, T, r, sigma, type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == 'call':
        return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

C = black_scholes(r, S, K, T, sigma, type='call')
print(f"Call Option Price: {C:.4f}")

def implied_voltatility(C, S, K, T, r, type='call', initial_guess=0.2):
    def objective_function(sigma):
        return (black_scholes(S, K, T, r, sigma, type) - C) ** 2

    implied_vol = newton(objective_function, initial_guess)
    return implied_vol

implied_vol = implied_voltatility(C, S, K, T, r, type='call')
price_check = black_scholes(S, K, T, r, implied_vol, type='call')
print(f"Implied Volatility: {implied_vol:.1%}")
print(f"Call Option Price with Implied Volatility: {price_check:.4f}")
print(f"Put Option Price: {black_scholes(S, K, T, r, implied_vol, type='put'):.4f}")

def plot_black_scholes_surface(prices, volatilities, maturities):
    volatility_surface_fig = go.Figure(data=go.Surface(z=prices, x=volatilities, y=maturities))
    volatility_surface_fig.update_layout(title='Black-Scholes Call Option Price Surface',
                    scene=dict(xaxis_title='Volatility',
                                yaxis_title='Maturity (Years)',
                                zaxis_title='Call Option Price'),
                    width=800, height=600)
    # volatility_surface_fig.show()

maturities = np.linspace(0.01, 1, 100)
volatilities = np.linspace(0.01, 1, 100)
prices = np.array([[black_scholes(S, K, T, r, sigma) for sigma in volatilities] for T in maturities])
plot_black_scholes_surface(prices, volatilities, maturities)

def fetch_options_data(symbol):
    ticker = yf.Ticker(symbol)
    options = ticker.options
    options_data = ticker.option_chain(options[0])
    return options_data.calls, options_data.puts

def plot_options_data(symbol):
    calls, puts = fetch_options_data(symbol)
    plot_plt = go.Figure()
    plot_plt.add_trace(go.Scatter(x=calls['strike'], y=calls['lastPrice'], mode='markers', name='JPM Calls'))
    plot_plt.add_trace(go.Scatter(x=puts['strike'], y=puts['lastPrice'], mode='markers', name='JPM Puts'))
    plot_plt.update_layout(title=f'{symbol} Options Prices',
                      xaxis_title='Strike Price',
                        yaxis_title='Last Price',
                        width=800, height=600)
    # plot_plt.show()

plot_options_data('AAPL')




