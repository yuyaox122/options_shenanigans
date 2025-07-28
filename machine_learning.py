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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import streamlit as st

def black_scholes_price(S, K, r, T, sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    price = S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    
    return price

def generate_data(random_state=42):
    np.random.seed(random_state)

    S = np.arange(40, 61)
    K = np.arange(20, 91)
    r = np.arange(0, 0.051, 0.01)
    T = np.arange(3/12, 2.01, 1/12)
    sigma = np.arange(0.1, 0.81, 0.1)

    option_prices = pd.DataFrame(
    product(S, K, r, T, sigma),
    columns=["S", "K", "r", "T", "sigma"]
    )

    option_prices["black_scholes"] = black_scholes_price(
    option_prices["S"].values, 
    option_prices["K"].values, 
    option_prices["r"].values, 
    option_prices["T"].values, 
    option_prices["sigma"].values,
    )

    # Generate observed prices with some noise
    option_prices = (option_prices
    .assign(
        observed_price=lambda x: (
        x["black_scholes"] + np.random.normal(scale=0.15)
        )
    )
    )
    return option_prices

random_state = 42
option_prices = generate_data(random_state)
train_data, test_data = train_test_split(option_prices, test_size=0.01, random_state=random_state)

preprocessor = ColumnTransformer(
  transformers=[(
    "normalize_predictors", 
     StandardScaler(),
     ["S", "K", "r", "T", "sigma"]
  )],
  remainder="drop"
)

max_iter = 1000

nnet_model = MLPRegressor(
  hidden_layer_sizes=10, 
  max_iter=max_iter, 
  random_state=random_state
)

nnet_pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("regressor", nnet_model)
])

nnet_fit = nnet_pipeline.fit(
  train_data.drop(columns=["observed_price"]), 
  train_data.get("observed_price")
)

print("Neural network model fitted.")

rf_model = RandomForestRegressor(
  n_estimators=20, 
  min_samples_leaf=2000, 
  random_state=random_state
)

rf_pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("regressor", rf_model)
])

rf_fit = rf_pipeline.fit(
  train_data.drop(columns=["observed_price"]), 
  train_data.get("observed_price")
)

print("Random forest model fitted.")

# deepnnet_model = MLPRegressor(
#   hidden_layer_sizes=(10, 10, 10),
#   activation="logistic", 
#   solver="lbfgs",
#   max_iter=max_iter, 
#   random_state=random_state
# )
                              
# deepnnet_pipeline = Pipeline([
#   ("preprocessor", preprocessor),
#   ("regressor", deepnnet_model)
# ])

# deepnnet_fit = deepnnet_pipeline.fit(
#   train_data.drop(columns=["observed_price"]),
#   train_data.get("observed_price")
# )

# # Polynomial regression with Lasso regularisation
# lm_pipeline = Pipeline([
#   ("polynomial", PolynomialFeatures(degree=5, 
#                                     interaction_only=False, 
#                                     include_bias=True)),
#   ("scaler", StandardScaler()),
#   ("regressor", Lasso(alpha=0.01))
# ])

# lm_fit = lm_pipeline.fit(
#   train_data.get(["S", "K", "r", "T", "sigma"]),
#   train_data.get("observed_price")
# )

test_X = test_data.get(["S", "K", "r", "T", "sigma"])
test_y = test_data.get("observed_price")

predictive_performance = (pd.concat(
    [test_data.reset_index(drop=True), 
     pd.DataFrame({"Random forest": rf_fit.predict(test_X),
                   "Single layer": nnet_fit.predict(test_X),
                  #  "Deep NN": deepnnet_fit.predict(test_X),
                  #  "Lasso": lm_fit.predict(test_X)
                  })
    ], axis=1)
  .melt(
    id_vars=["S", "K", "r", "T", "sigma",
             "black_scholes", "observed_price"],
    var_name="Model",
    value_name="Predicted"
  )
  .assign(
    moneyness=lambda x: x["S"]-x["K"],
    pricing_error=lambda x: np.abs(x["Predicted"]-x["black_scholes"])
  )
)

fig = px.scatter(
    predictive_performance,
    x="moneyness",
    y="pricing_error",
    title="Pricing Error vs Moneyness by Model"
)
fig.update_layout(
    xaxis_title="Moneyness (S - K)",
    yaxis_title="Pricing Error",
    legend_title="Model"
)

st.title("Machine Learning Models for Option Pricing")
st.sidebar.header("Model Settings")
st.plotly_chart(fig, use_container_width=True)
