import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from itertools import product
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

st.set_page_config(page_title="Model Comparison", page_icon="📈")

def black_scholes_price(S, K, r, T, sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

@st.cache_data
def generate_data(noise_scale, random_state=42):
    np.random.seed(random_state)
    S = np.arange(40, 61)
    K = np.arange(30, 71, 2)
    r = np.arange(0, 0.051, 0.01)
    T = np.arange(0.1, 2.01, 0.1)
    sigma = np.arange(0.1, 0.61, 0.1)
    option_prices = pd.DataFrame(product(S, K, r, T, sigma), columns=["S", "K", "r", "T", "sigma"])
    option_prices["black_scholes"] = black_scholes_price(
        option_prices["S"].values, 
        option_prices["K"].values, 
        option_prices["r"].values, 
        option_prices["T"].values, 
        option_prices["sigma"].values,
    )
    option_prices["observed_price"] = option_prices["black_scholes"] + np.random.normal(scale=noise_scale, size=len(option_prices))
    return option_prices

st.title("Option Pricing Models")

# Sidebar controls
st.sidebar.header("Model Settings")

noise_scale = st.sidebar.slider("Noise scale in observed price", 0.0, 1.0, 0.15, 0.05)
test_size = st.sidebar.slider("Test set size (fraction)", 0.005, 0.1, 0.01, 0.005)

# Model selection
models_to_run = st.sidebar.multiselect(
    "Select models to train & show",
    options=["Random Forest", "Neural Network", "Deep Neural Network", "Polynomial + Lasso"],
    default=["Random Forest"]
)

# RF hyperparameters
if "Random Forest" in models_to_run:
    n_estimators = st.sidebar.slider("RF: n_estimators", 10, 100, 20, 10)
    min_samples_leaf = st.sidebar.slider("RF: min_samples_leaf", 1000, 10000, 5000, 1000)

# NN hyperparameters
if "Neural Network" in models_to_run:
    hidden_layer_size = st.sidebar.slider("NN: hidden_layer_sizes", 1, 50, 5, 1)

# Deep NN hyperparameters
if "Deep Neural Network" in models_to_run:
    deep_hidden_layer_size = st.sidebar.slider("Deep NN: hidden_layer_sizes", 1, 100, 10, 1)

# Polynomial + Lasso hyperparameters
if "Polynomial + Lasso" in models_to_run:
    poly_degree = st.sidebar.slider("Poly degree", 1, 5, 2, 1)
    lasso_alpha = st.sidebar.slider("Lasso alpha", 0.0, 1.0, 0.01, 0.01)

# Generate and split data
option_prices = generate_data(noise_scale)
train_data, test_data = train_test_split(option_prices, test_size=test_size, random_state=42)

preprocessor = ColumnTransformer([
    ("normalize", StandardScaler(), ["S", "K", "r", "T", "sigma"])
])

results = pd.DataFrame()

# Train Random Forest
if "Random Forest" in models_to_run:
    start_forest = time.time()
    with st.spinner("Training Random Forest..."):
        rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)
        rf_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", rf_model)])
        rf_pipeline.fit(train_data.drop(columns=["observed_price"]), train_data["observed_price"])
        rf_pred = rf_pipeline.predict(test_data[["S", "K", "r", "T", "sigma"]])
        results["Random Forest"] = rf_pred
        end_forest = time.time()
    st.success("Random Forest trained in " + str(end_forest - start_forest) + " seconds")

# Train Neural Network
if "Neural Network" in models_to_run:
    start_nnet = time.time()
    with st.spinner("Training Neural Network..."):
        nnet_model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, max_iter=500, random_state=42)
        nnet_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", nnet_model)])
        nnet_pipeline.fit(train_data.drop(columns=["observed_price"]), train_data["observed_price"])
        nnet_pred = nnet_pipeline.predict(test_data[["S", "K", "r", "T", "sigma"]])
        results["Neural Network"] = nnet_pred
        end_nnet = time.time()
    st.success("Neural Network trained in " + str(end_nnet - start_nnet) + " seconds")


# Train Deep Neural Network
if "Deep Neural Network" in models_to_run:
    start_deepnnet = time.time()
    with st.spinner("Training Deep Neural Network..."):
        deepnnet_model = MLPRegressor(hidden_layer_sizes=(deep_hidden_layer_size, deep_hidden_layer_size), max_iter=500, random_state=42)
        deepnnet_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", deepnnet_model)])
        deepnnet_pipeline.fit(train_data.drop(columns=["observed_price"]), train_data["observed_price"])
        deepnnet_pred = deepnnet_pipeline.predict(test_data[["S", "K", "r", "T", "sigma"]])
        results["Deep Neural Network"] = deepnnet_pred
        end_deepnnet = time.time()
    st.success("Deep Neural Network trained in " + str(end_deepnnet - start_deepnnet) + " seconds")

# Train Polynomial + Lasso
if "Polynomial + Lasso" in models_to_run:
    start_lasso = time.time()
    with st.spinner("Training Polynomial + Lasso..."):
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        lasso_model = Lasso(alpha=lasso_alpha, random_state=42)
        lasso_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("poly_features", poly_features),
            ("regressor", lasso_model)
        ])
        lasso_pipeline.fit(train_data.drop(columns=["observed_price"]), train_data["observed_price"])
        lasso_pred = lasso_pipeline.predict(test_data[["S", "K", "r", "T", "sigma"]])
        results["Polynomial + Lasso"] = lasso_pred
        end_lasso = time.time()
    st.success("Polynomial + Lasso trained in " + str(end_lasso - start_lasso) + " seconds")


if results.empty:
    st.warning("Please select at least one model to display results.")
    st.stop()

# Combine with test data
plot_df = pd.concat([test_data.reset_index(drop=True), results], axis=1)
plot_df = plot_df.melt(
    id_vars=["S", "K", "r", "T", "sigma", "black_scholes", "observed_price"],
    var_name="Model",
    value_name="Predicted"
)
plot_df["moneyness"] = plot_df["S"] - plot_df["K"]
plot_df["pricing_error"] = np.abs(plot_df["Predicted"] - plot_df["black_scholes"])

# Plot
fig = px.scatter(
    plot_df,
    x="moneyness",
    y="pricing_error",
    color="Model",
    facet_col="Model",
    title="Pricing Error vs Moneyness by Model",
    trendline="ols",
    trendline_color_override="black"
)
fig.update_layout(
    xaxis_title="Moneyness (S - K)",
    yaxis_title="Pricing Error",
    legend_title="Model"
)

st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show raw test data"):
    st.write(test_data.head())
