import numpy as np
import plotly.graph_objects as go
import streamlit as st
from svi import svi, calibrate_svi
from scipy.optimize import least_squares
from option_pricing import call_price
from implied_vol import implied_volatility
from scipy.stats import norm
st.set_page_config(
    page_title="Monte Carlo Option Pricing",
    layout="wide"
)
st.title("Monte Carlo Option Pricing")

# Parameters
S0 = st.sidebar.number_input(
    "Stock Price ($)", min_value=1.0, value=100.0
)

K = st.sidebar.number_input(
    "Strike Price ($)", min_value=1.0, value=100.0
)

T = 1
r = 0.05

sigma = st.sidebar.slider(
    "Volatility", 0.05, 0.80, 0.20
)

n_paths = 10000
n_steps = 252

# Generate random numbers
rng = np.random.default_rng(42)
dt = T / n_steps

Z = rng.standard_normal((n_steps, n_paths))

# Simulate stock prices
returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

paths = S0 * np.vstack([
    np.ones(n_paths),
    np.exp(np.cumsum(returns, axis=0))
])

# Calculate option price
payoff = np.maximum(paths[-1] - K, 0)

price = np.exp(-r * T) * np.mean(payoff)

st.write("Monte Carlo Call Price:", round(price, 4))

# Create interactive chart
fig = go.Figure()

time = np.linspace(0, T, n_steps + 1)

for i in range(40):
    fig.add_trace(
        go.Scatter(
            x=time,
            y=paths[:, i],
            mode="lines",
            showlegend=False
        )
    )

    fig.update_layout(
    template="plotly_dark",
    title="Monte Carlo Stock Price Simulation",
    xaxis_title="Time (Years)",
    yaxis_title="Stock Price ($)",
    height=700,
    autosize=True
)

st.plotly_chart(fig, width="stretch")
# OPTION PRICING HEATMAP

st.subheader("Black-Scholes Option Pricing Heatmap")

# Generate stock prices and volatilities
stock_prices = np.linspace(70, 130, 12)
volatilities = np.linspace(0.05, 0.80, 10)

# Create a grid
S_grid, V_grid = np.meshgrid(stock_prices, volatilities)

# Black-Scholes calculations
d1 = (
    np.log(S_grid / K)
    + (r + 0.5 * V_grid**2) * T
) / (V_grid * np.sqrt(T))

d2 = d1 - V_grid * np.sqrt(T)

call_prices = (
    S_grid * norm.cdf(d1)
    - K * np.exp(-r * T) * norm.cdf(d2)
)

# Create heatmap
fig_heatmap = go.Figure(
    data=go.Heatmap(
        x=stock_prices,
        y=volatilities,
        z=call_prices,
        colorscale="Viridis",
        text=np.round(call_prices, 2),
        texttemplate="%{text:.1f}",
        colorbar=dict(title="Call Price ($)")
    )
)

# Chart design
fig_heatmap.update_layout(
    template="plotly_dark",
    title="Option Price Sensitivity",
    xaxis_title="Stock Price ($)",
    yaxis_title="Volatility",
    yaxis_tickformat=".0%",
    height=650
)

st.plotly_chart(fig_heatmap, width="stretch")
# 3D VOLATILITY SURFACE

st.subheader("3D Volatility Surface")

# Generate strike prices and maturities
strikes = np.linspace(S0 * 0.7, S0 * 1.3, 60)
maturities = np.linspace(0.1, 2.0, 45)

# Create 2D grids
K_grid, T_grid = np.meshgrid(strikes, maturities)

# Illustrative volatility model
iv_surface = (
    sigma
    + 0.12 * (K_grid / S0 - 1)**2
    - 0.07 * (K_grid / S0 - 1)
    + 0.02 * np.sqrt(T_grid)
)

# More detailed illustrative volatility surface

m = np.log(K_grid / S0)

iv_surface = (
    sigma
    - 0.10 * m
    + 0.25 * m**2
    + 0.015 * np.log1p(T_grid)
)

iv_surface = np.clip(iv_surface, 0.01, 1.0)
# Create interactive 3D surface
fig_3d = go.Figure(
    data=go.Surface(
        x=K_grid,
        y=T_grid,
        z=iv_surface * 100,
        colorscale="Viridis",
        colorbar=dict(title="Volatility (%)")
    )
)

fig_3d.update_layout(
    template="plotly_dark",

    title="Synthetic Implied Volatility Surface",

    scene=dict(
        xaxis_title="Strike Price ($)",
        yaxis_title="Maturity (Years)",
        zaxis_title="Implied Volatility (%)",

        aspectmode="manual",

        aspectratio=dict(
            x=1.5,
            y=1.2,
            z=0.6
        ),

        camera=dict(
            eye=dict(
                x=1.6,
                y=-1.8,
                z=0.9
            )
        )
    ),

    height=700,

    margin=dict(
        l=0,
        r=0,
        t=60,
        b=0
    )
)

fig_3d.update_layout(
    scene_aspectmode="manual",

    scene_aspectratio=dict(
        x=1.6,
        y=1.2,
        z=0.9
    ),

    scene_camera=dict(
        eye=dict(
            x=1.6,
            y=-1.8,
            z=1.3
        )
    ),

    height=800
)
st.plotly_chart(fig_3d, width="stretch")
# VOLATILITY SMILE

st.subheader("Implied Volatility Smile")

strikes_smile = np.linspace(S0 * 0.7, S0 * 1.3, 25)

recovered_ivs = []

for strike in strikes_smile:

    m = np.log(strike / S0)

    # Synthetic volatility smile
    synthetic_iv = (
        sigma
        - 0.10 * m
        + 0.25 * m**2
    )

    # Generate synthetic option price
    market_price = call_price(
        S0, strike, T, r, synthetic_iv
    )

    # Recover implied volatility
    iv = implied_volatility(
        market_price, S0, strike, T, r
    )

    recovered_ivs.append(iv * 100)


fig_smile = go.Figure()

fig_smile.add_trace(
    go.Scatter(
        x=strikes_smile,
        y=recovered_ivs,
        mode="lines+markers",
        name="Recovered IV"
    )
)

fig_smile.update_layout(
    template="plotly_dark",
    title="Synthetic Implied Volatility Smile",
    xaxis_title="Strike Price ($)",
    yaxis_title="Implied Volatility (%)",
    height=550
)

st.plotly_chart(fig_smile, width="stretch")
# SVI CALIBRATION

st.subheader("SVI Calibration")

observed_iv = np.array(recovered_ivs) / 100

forward = S0 * np.exp(r * T)

params = calibrate_svi(
    strikes_smile,
    observed_iv,
    T,
    forward
)

k = np.log(strikes_smile / forward)

fitted_variance = svi(k, *params)

fitted_iv = np.sqrt(
    np.maximum(fitted_variance, 0) / T
) * 100

fig_svi = go.Figure()

fig_svi.add_trace(go.Scatter(
    x=strikes_smile,
    y=recovered_ivs,
    mode="markers",
    name="Original IV"
))

fig_svi.add_trace(go.Scatter(
    x=strikes_smile,
    y=fitted_iv,
    mode="lines",
    name="SVI Fitted IV"
))

fig_svi.update_layout(
    template="plotly_dark",
    title="SVI Model Calibration",
    xaxis_title="Strike Price ($)",
    yaxis_title="Implied Volatility (%)",
    height=600
)

st.plotly_chart(fig_svi, width="stretch")

rmse = np.sqrt(
    np.mean((fitted_iv - recovered_ivs)**2)
)

st.metric("Calibration RMSE", f"{rmse:.4f} percentage points")