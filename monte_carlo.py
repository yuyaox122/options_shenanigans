import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
# The expected value is the probability weighted average of all possible outcomes.
# The analytical expectation is the long-term average of the payoffs.

def bernoulli_distribution(p, n):
    # The formula for the Bernoulli distribution is:
    # P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
    # p is the probability of success, n is the number of trials.
    return np.random.binomial(1, p, n)

def exponential_distribution(lam, size):
    # The formula for the exponential distribution is:
    # f(x; λ) = λ * exp(-λx) for x >= 0
    # lam is the rate parameter, which is the inverse of the mean.
    # size is the number of samples to generate
    return np.random.exponential(1/lam, size)

# Plot the distributions 

def plot_distributions(bernoulli_data, exponential_data):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bernoulli_data, name='Bernoulli Distribution', opacity=0.75))
    fig.add_trace(go.Scatter(x=exponential_data, name='Exponential Distribution', mode='lines'))
    
    fig.update_layout(title='Comparison of Bernoulli and Exponential Distributions',
                      xaxis_title='Value',
                      yaxis_title='Frequency',
                      barmode='overlay')
    fig.write_html("bernoulli_vs_exponential.html")

plot_distributions(
    bernoulli_distribution(0.7, 1000),
    exponential_distribution(2, 1000)
)

# Intractable distributions can be approximated using Monte Carlo methods.
# The Monte Carlo simulation approximates the expected value by simulating a large number of random paths.
