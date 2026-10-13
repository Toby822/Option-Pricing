"""Interactive option pricing lab with manual and market-data workflows.

Place this file beside svi.py in the Option-Pricing repository and run it with
``streamlit run option_pricing_app.py``. Market data is for research, not trading.
"""

from datetime import datetime, timedelta, timezone
from io import BytesIO
import re
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import streamlit as st
import yfinance as yf

from svi import calibrate_svi, svi


MIN_QUOTES = 8
NY_TZ = ZoneInfo("America/New_York")
TREASURY_COLUMNS = ("DGS1MO", "DGS3MO", "DGS6MO", "DGS1")
TREASURY_TENORS = np.array([1 / 12, 3 / 12, 6 / 12, 1.0])


def black_scholes_call(spot, strike, maturity, rate, dividend_yield, volatility):
    """European call with a continuous dividend-yield assumption."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    if (np.any(spot <= 0) or np.any(strike <= 0) or maturity <= 0
            or np.any(volatility <= 0)):
        raise ValueError("Spot, strike, maturity and volatility must be positive")
    root_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike)
          + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    return (spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1)
            - strike * np.exp(-rate * maturity) * norm.cdf(d2))


def simulate_call(spot, strike, maturity, rate, dividend_yield, volatility,
                  n_paths=10000, n_steps=64, seed=42):
    """Risk-neutral GBM paths and a European call Monte Carlo estimate."""
    if min(spot, strike, maturity, volatility) <= 0 or n_paths < 2 or n_steps < 1:
        raise ValueError("Invalid simulation inputs")
    dt = maturity / n_steps
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((n_steps, n_paths))
    increments = ((rate - dividend_yield - 0.5 * volatility**2) * dt
                  + volatility * np.sqrt(dt) * draws)
    paths = spot * np.vstack((np.ones(n_paths), np.exp(np.cumsum(increments, axis=0))))
    discounted = np.exp(-rate * maturity) * np.maximum(paths[-1] - strike, 0)
    return paths, float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def synthetic_iv(spot, strikes, maturities, atm_iv, skew, curvature, term_slope):
    log_moneyness = np.log(np.asarray(strikes) / spot)
    maturity = np.asarray(maturities)
    return np.clip(atm_iv + skew * log_moneyness + curvature * log_moneyness**2
                   + term_slope * np.log1p(maturity), 0.01, 2.0)


def fit_svi(strikes, observed_iv, maturity, forward):
    params = calibrate_svi(strikes, observed_iv, maturity, forward)
    total_variance = np.asarray(svi(np.log(strikes / forward), *params), dtype=float)
    if not np.all(np.isfinite(total_variance)) or np.any(total_variance <= 0):
        raise ValueError("SVI produced invalid total variance")
    fitted_iv = np.sqrt(total_variance / maturity)
    rmse_pp = float(np.sqrt(np.mean((fitted_iv - observed_iv) ** 2)) * 100)
    return fitted_iv, rmse_pp, params


def pricing_heatmap(spot, strike, maturity, rate, dividend_yield, volatility):
    spots = np.linspace(spot * 0.7, spot * 1.3, 17)
    volatilities = np.linspace(max(0.01, volatility * 0.4), min(2.0, volatility * 1.8), 15)
    spot_grid, vol_grid = np.meshgrid(spots, volatilities)
    prices = black_scholes_call(spot_grid, strike, maturity, rate, dividend_yield, vol_grid)
    fig = go.Figure(go.Heatmap(
        x=spots, y=volatilities * 100, z=prices, colorscale="Viridis",
        colorbar=dict(title="Call ($)"),
        hovertemplate="Spot $%{x:.2f}<br>Volatility %{y:.1f}%<br>Call $%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="European call price sensitivity", xaxis_title="Scenario spot ($)",
                      yaxis_title="Scenario volatility (%)", height=500)
    return fig


def sampled_paths_figure(paths, maturity):
    time = np.linspace(0, maturity, len(paths))
    fig = go.Figure()
    for index in range(min(40, paths.shape[1])):
        fig.add_trace(go.Scatter(x=time, y=paths[:, index], mode="lines",
                                 line=dict(width=1), opacity=0.55, showlegend=False))
    fig.update_layout(title="Monte Carlo sample paths | model scenarios",
                      xaxis_title="Years from valuation", yaxis_title="Simulated price ($)", height=520)
    return fig


def scenario_fan_figure(paths, maturity):
    time = np.linspace(0, maturity, len(paths))
    q05, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=q95, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=q05, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(68, 141, 180, 0.18)", name="5th-95th percentile"))
    fig.add_trace(go.Scatter(x=time, y=q75, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=q25, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(68, 141, 180, 0.35)", name="25th-75th percentile"))
    fig.add_trace(go.Scatter(x=time, y=q50, mode="lines",
                             line=dict(color="#efbd56", width=2), name="Median simulation"))
    fig.update_layout(title="GBM simulation range | not a forecast",
                      xaxis_title="Years from valuation", yaxis_title="Simulated price ($)", height=480)
    return fig


def terminal_distribution_figure(paths, strike, spot):
    fig = go.Figure(go.Histogram(x=paths[-1], nbinsx=60,
                                 marker_color="#4b9db5", name="Terminal price"))
    fig.add_vline(x=strike, line_dash="dash", line_color="#efbd56")
    fig.add_vline(x=spot, line_dash="dot", line_color="#e7e7e7")
    fig.update_layout(title="Simulated price at expiry | model distribution",
                      xaxis_title="Simulated price ($)", yaxis_title="Paths", height=400)
    return fig


def smile_figure(strikes, observed_iv, fitted_iv, label, maturity):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.72, 0.28])
    fig.add_trace(go.Scatter(x=strikes, y=observed_iv * 100, mode="markers",
                             marker=dict(size=7), name=label), row=1, col=1)
    fig.add_trace(go.Scatter(x=strikes, y=fitted_iv * 100, mode="lines",
                             name="SVI fit"), row=1, col=1)
    fig.add_trace(go.Bar(x=strikes, y=(fitted_iv - observed_iv) * 100,
                         marker_color="#8ea6b5", name="Fit minus input"), row=2, col=1)
    fig.update_layout(title=f"Volatility smile | {maturity:.2f} years", height=570)
    fig.update_yaxes(title_text="IV (%)", row=1, col=1)
    fig.update_yaxes(title_text="Error (pp)", row=2, col=1)
    fig.update_xaxes(title_text="Strike ($)", row=2, col=1)
    return fig


def synthetic_surface_figure(spot, atm_iv, skew, curvature, term_slope):
    strikes = np.linspace(spot * 0.7, spot * 1.3, 60)
    maturities = np.linspace(0.1, 2.0, 45)
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    values = synthetic_iv(spot, strike_grid, maturity_grid, atm_iv, skew, curvature, term_slope)
    fig = go.Figure(go.Surface(x=strike_grid, y=maturity_grid,
                               z=values * 100, colorscale="Viridis",
                               colorbar=dict(title="IV (%)")))
    fig.update_layout(title="Synthetic volatility surface | assumed shape", height=650,
                      scene=dict(xaxis_title="Strike ($)", yaxis_title="Years to expiry",
                                 zaxis_title="Volatility (%)",
                                 aspectmode="manual", aspectratio=dict(x=1.4, y=1, z=0.65)))
    return fig


def synthetic_term_figure(spot, atm_iv, skew, curvature, term_slope):
    maturities = np.linspace(0.1, 2.0, 60)
    fig = go.Figure()
    for factor, label in ((0.9, "90% of spot"), (1.0, "At spot"), (1.1, "110% of spot")):
        values = synthetic_iv(spot, spot * factor, maturities, atm_iv, skew, curvature, term_slope)
        fig.add_trace(go.Scatter(x=maturities, y=values * 100, mode="lines", name=label))
    fig.update_layout(title="Synthetic volatility term structure",
                      xaxis_title="Years to expiry", yaxis_title="Volatility (%)", height=380)
    return fig


def render_manual():
    with st.sidebar:
        st.header("Model inputs")
        spot = st.number_input("Stock price ($)", min_value=1.0, value=100.0, step=1.0)
        strike = st.number_input("Strike ($)", min_value=1.0, value=100.0, step=1.0)
        maturity = st.number_input("Years to expiry", min_value=0.02, max_value=10.0,
                                   value=1.0, step=0.05)
        rate_pct = st.number_input("Annual risk-free rate (%)", value=5.0, step=0.25)
        dividend_pct = st.number_input("Annual dividend yield (%)", min_value=0.0,
                                       max_value=50.0, value=0.0, step=0.25)
        volatility_pct = st.slider("Volatility (%)", 1, 150, 20)
        n_paths = st.slider("Monte Carlo paths", 1000, 20000, 10000, step=1000)
        n_steps = st.slider("Time steps", 12, 252, 64, step=4)
        seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42)
        with st.expander("Synthetic surface shape"):
            skew = st.slider("Skew", -0.5, 0.5, -0.1, step=0.01)
            curvature = st.slider("Smile curvature", 0.0, 1.0, 0.25, step=0.01)
            term_slope = st.slider("Term slope", -0.2, 0.2, 0.015, step=0.005)

    rate, dividend_yield, volatility = rate_pct / 100, dividend_pct / 100, volatility_pct / 100
    paths, mc_price, mc_se = simulate_call(spot, strike, maturity, rate, dividend_yield,
                                            volatility, n_paths, n_steps, int(seed))
    bs_price = float(black_scholes_call(spot, strike, maturity, rate, dividend_yield, volatility))

    st.title("Option Pricing Lab")
    st.caption("Manual scenario | all inputs are assumptions, not observed market data.")
    price_tab, volatility_tab, simulation_tab = st.tabs(["Pricing", "Volatility", "Simulation"])
    with price_tab:
        first, second, third = st.columns(3)
        first.metric("European BS call", f"${bs_price:,.2f}")
        second.metric("Monte Carlo call", f"${mc_price:,.2f}")
        third.metric("MC standard error", f"${mc_se:.3f}")
        st.caption(f"95% Monte Carlo sampling interval: \\${mc_price - 1.96 * mc_se:.2f}"
                   f"-\\${mc_price + 1.96 * mc_se:.2f}. This excludes model uncertainty.")
        st.plotly_chart(pricing_heatmap(spot, strike, maturity, rate, dividend_yield,
                                         volatility), width="stretch")
    with volatility_tab:
        st.plotly_chart(synthetic_term_figure(spot, volatility, skew, curvature,
                                               term_slope), width="stretch")
        st.plotly_chart(synthetic_surface_figure(spot, volatility, skew, curvature,
                                                  term_slope), width="stretch")
        smile_strikes = np.linspace(spot * 0.7, spot * 1.3, 25)
        assumed_iv = synthetic_iv(spot, smile_strikes, maturity, volatility,
                                  skew, curvature, term_slope)
        forward = spot * np.exp((rate - dividend_yield) * maturity)
        try:
            fitted_iv, rmse_pp, _ = fit_svi(smile_strikes, assumed_iv, maturity, forward)
            st.metric("SVI in-sample RMSE", f"{rmse_pp:.3f} pp",
                      help="Percentage points of implied volatility, measured on the fitting data.")
            st.plotly_chart(smile_figure(smile_strikes, assumed_iv, fitted_iv,
                                          "Assumed IV", maturity), width="stretch")
            st.caption("This SVI fit uses generated values. It is not market calibration or out-of-sample accuracy.")
        except Exception as exc:
            st.warning(f"SVI fit unavailable for these inputs: {exc}")
    with simulation_tab:
        st.plotly_chart(sampled_paths_figure(paths, maturity), width="stretch")
        st.plotly_chart(scenario_fan_figure(paths, maturity), width="stretch")
        st.plotly_chart(terminal_distribution_figure(paths, strike, spot), width="stretch")
        st.caption("Yellow dashed line: strike; white dotted line: starting spot. "
                   "GBM paths are model scenarios, not predictions of future prices.")


@st.cache_data(ttl=900, show_spinner=False)
def get_expirations(symbol):
    return tuple(yf.Ticker(symbol).options)


@st.cache_data(ttl=900, show_spinner=False)
def get_chain(symbol, expiration):
    chain = yf.Ticker(symbol).option_chain(expiration)
    return chain.calls, chain.underlying, datetime.now(timezone.utc)


@st.cache_data(ttl=3600, show_spinner=False)
def get_history(symbol):
    return yf.Ticker(symbol).history(period="1y", auto_adjust=True, actions=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_treasury_curve(today):
    start = (today - timedelta(days=30)).isoformat()
    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?"
           f"id={','.join(TREASURY_COLUMNS)}&cosd={start}")
    request = Request(url, headers={"User-Agent": "Option-Pricing research dashboard"})
    with urlopen(request, timeout=8) as response:
        frame = pd.read_csv(BytesIO(response.read()))
    required = {"observation_date", *TREASURY_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError("FRED Treasury curve has missing columns")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    for column in TREASURY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame.loc[frame["observation_date"].dt.date <= today]
    if frame.empty:
        raise ValueError("FRED has no complete recent Treasury curve")
    latest = frame.sort_values("observation_date").iloc[-1]
    observed_on = latest["observation_date"].date()
    if (today - observed_on).days > 10:
        raise ValueError(f"Latest complete Treasury curve is stale ({observed_on})")
    return observed_on, latest[list(TREASURY_COLUMNS)].to_numpy(dtype=float)


def treasury_rate_for_maturity(curve_rates_pct, maturity):
    if not np.all(np.isfinite(curve_rates_pct)) or np.any(curve_rates_pct < 0):
        raise ValueError("Treasury curve contains invalid yields")
    return float(np.interp(maturity, TREASURY_TENORS, curve_rates_pct))


def trailing_dividend_yield(history, spot, today):
    if history is None or history.empty or "Dividends" not in history:
        raise ValueError("Dividend history is unavailable")
    dividends = pd.to_numeric(history["Dividends"], errors="coerce").fillna(0)
    dates = pd.to_datetime(history.index).date
    valid = ((dates >= today - timedelta(days=365)) & (dates <= today)
             & (dividends.to_numpy() > 0))
    count = int(np.count_nonzero(valid))
    if count < 3:
        raise ValueError(f"Only {count} dividends found in the past year")
    cash = float(dividends.to_numpy()[valid].sum())
    if not np.isfinite(cash) or cash <= 0 or spot <= 0:
        raise ValueError("Dividend yield cannot be calculated")
    return 100 * cash / spot, cash, count


def filter_calls(calls, spot, strike_bounds=(0.8, 1.2), max_spread=0.5):
    required = ("strike", "bid", "ask", "impliedVolatility")
    if calls is None or any(column not in calls for column in required):
        raise ValueError("Option chain is missing required columns")
    quotes = calls.copy()
    for column in required:
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes = quotes.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
    quotes["spread_pct"] = (quotes["ask"] - quotes["bid"]) / quotes["mid"]
    valid = (quotes["strike"].between(strike_bounds[0] * spot, strike_bounds[1] * spot)
             & (quotes["bid"] > 0) & (quotes["ask"] >= quotes["bid"])
             & quotes["impliedVolatility"].between(0.05, 2.0)
             & (quotes["spread_pct"] <= max_spread))
    return quotes.loc[valid].sort_values("strike").drop_duplicates("strike")


def choose_surface_dates(expirations, today, selected):
    chosen = [selected]
    for start, end in ((14, 45), (45, 90), (90, 150), (150, 240)):
        match = next((day for day in expirations
                      if start <= (pd.Timestamp(day).date() - today).days < end), None)
        if match and match not in chosen:
            chosen.append(match)
    return sorted(chosen)


def market_mid_heatmap(all_quotes, symbol):
    fields = ("mid", "bid", "ask", "impliedVolatility", "spread_pct")
    grids = {field: all_quotes.pivot_table(index="expiration", columns="strike",
                                           values=field, aggfunc="mean").sort_index()
             for field in fields}
    mids = grids["mid"]
    aligned = [grids[field].reindex_like(mids).to_numpy(dtype=float) for field in fields]
    customdata = np.stack(aligned[1:], axis=-1)
    fig = go.Figure(go.Heatmap(
        x=mids.columns.to_numpy(dtype=float), y=mids.index.tolist(),
        z=mids.to_numpy(dtype=float), customdata=customdata, colorscale="Viridis",
        colorbar=dict(title="Mid ($)"), hoverongaps=False,
        hovertemplate=("Expiry %{y}<br>Strike $%{x:.2f}<br>Mid $%{z:.2f}<br>"
                       "Bid $%{customdata[0]:.2f} / Ask $%{customdata[1]:.2f}<br>"
                       "IV %{customdata[2]:.1%}<br>Spread %{customdata[3]:.1%}<extra></extra>"),
    ))
    fig.update_layout(title=f"{symbol} call bid/ask mid-price heatmap",
                      xaxis_title="Strike ($)", yaxis_title="Expiration", height=500)
    return fig


def historical_figure(history, symbol):
    required = {"Open", "High", "Low", "Close", "Volume"}
    if history is None or history.empty or not required.issubset(history.columns):
        raise ValueError("Adjusted OHLCV history is unavailable")
    log_returns = np.log(history["Close"]).diff()
    rv_21 = log_returns.rolling(21).std() * np.sqrt(252) * 100
    rv_63 = log_returns.rolling(63).std() * np.sqrt(252) * 100
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        row_heights=[0.55, 0.17, 0.28])
    fig.add_trace(go.Candlestick(x=history.index, open=history["Open"], high=history["High"],
                                  low=history["Low"], close=history["Close"],
                                  name="Adjusted OHLC"), row=1, col=1)
    fig.add_trace(go.Bar(x=history.index, y=history["Volume"], name="Volume",
                         marker_color="#638aa3"), row=2, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=rv_21, name="21d realized vol"), row=3, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=rv_63, name="63d realized vol"), row=3, col=1)
    fig.update_layout(title=f"{symbol} adjusted OHLC, volume and realized volatility | past year",
                      height=760, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.03))
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Shares", row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


def term_structure_figure(all_quotes, spot, symbol):
    indexes = all_quotes.groupby("expiration")["strike"].apply(
        lambda strikes: (strikes - spot).abs().idxmin())
    atm = all_quotes.loc[indexes].sort_values("days_to_expiry")
    fig = go.Figure(go.Scatter(
        x=atm["days_to_expiry"], y=atm["impliedVolatility"] * 100,
        mode="lines+markers", name="Nearest-to-spot call IV",
        customdata=atm[["expiration", "strike", "spread_pct"]].to_numpy(),
        hovertemplate=("Expiry %{customdata[0]}<br>Days %{x}<br>"
                       "Strike $%{customdata[1]:.2f}<br>IV %{y:.2f}%<br>"
                       "Spread %{customdata[2]:.1%}<extra></extra>"),
    ))
    fig.update_layout(title=f"{symbol} near-the-money IV term structure",
                      xaxis_title="Days to expiry", yaxis_title="Implied volatility (%)", height=380)
    return fig


def market_svi_surface(all_quotes, spot, rate, dividend_yield, symbol):
    slices, rejected = [], []
    for expiration, quotes in all_quotes.groupby("expiration", sort=True):
        if len(quotes) < MIN_QUOTES:
            rejected.append(f"{expiration}: insufficient quotes")
            continue
        try:
            maturity = float(quotes["days_to_expiry"].iloc[0]) / 365
            forward = spot * np.exp((rate - dividend_yield) * maturity)
            strikes = quotes["strike"].to_numpy(dtype=float)
            observed_iv = quotes["impliedVolatility"].to_numpy(dtype=float)
            _, rmse_pp, params = fit_svi(strikes, observed_iv, maturity, forward)
            log_min, log_max = np.log(strikes.min() / forward), np.log(strikes.max() / forward)
            slices.append((expiration, quotes, maturity, forward, params, rmse_pp,
                           log_min, log_max))
        except Exception as exc:
            rejected.append(f"{expiration}: {exc}")
    if len(slices) < 3:
        return None, pd.DataFrame(), rejected, None
    log_min, log_max = max(row[6] for row in slices), min(row[7] for row in slices)
    if log_max <= log_min:
        rejected.append("No common observed forward-moneyness range")
        return None, pd.DataFrame(), rejected, None
    log_grid = np.linspace(log_min, log_max, 70)
    maturities = np.array([row[2] for row in slices])
    total_variances = np.stack([np.asarray(svi(log_grid, *row[4]), dtype=float)
                                for row in slices])
    if not np.all(np.isfinite(total_variances)) or np.any(total_variances <= 0):
        rejected.append("Invalid fitted SVI total variance")
        return None, pd.DataFrame(), rejected, None
    decreasing = int(np.count_nonzero(np.diff(total_variances, axis=0) < -1e-8))
    dense_maturities = np.linspace(maturities.min(), maturities.max(), 45)
    dense_variance = np.column_stack([
        np.interp(dense_maturities, maturities, total_variances[:, column])
        for column in range(len(log_grid))
    ])
    iv_mesh = np.sqrt(dense_variance / dense_maturities[:, None]) * 100
    forward_mesh = spot * np.exp((rate - dividend_yield) * dense_maturities[:, None])
    strike_mesh = forward_mesh * np.exp(log_grid[None, :])
    day_mesh = np.broadcast_to(dense_maturities[:, None] * 365, iv_mesh.shape)
    observed = pd.concat([row[1] for row in slices], ignore_index=True)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=strike_mesh, y=day_mesh, z=iv_mesh, colorscale="Viridis",
                              opacity=0.86, colorbar=dict(title="IV (%)"),
                              name="SVI fitted surface"))
    fig.add_trace(go.Scatter3d(x=observed["strike"], y=observed["days_to_expiry"],
                                z=observed["impliedVolatility"] * 100, mode="markers",
                                marker=dict(size=2.5, color="#f5f5f5", opacity=0.65),
                                name="Reported IV"))
    fig.update_layout(title=f"{symbol} IV surface | SVI slices and market quotes", height=670,
                      scene=dict(xaxis_title="Strike ($)", yaxis_title="Days to expiry",
                                 zaxis_title="IV (%)", aspectmode="manual",
                                 aspectratio=dict(x=1.4, y=1, z=0.65)))
    diagnostics = pd.DataFrame({
        "Expiration": [row[0] for row in slices],
        "Quotes": [len(row[1]) for row in slices],
        "SVI RMSE (pp)": [round(row[5], 3) for row in slices],
    })
    return fig, diagnostics, rejected, decreasing


def render_market():
    st.title("Option Pricing Lab")
    st.caption("Market quotes and history via Yahoo Finance/yfinance; delayed research data, not live trading quotes.")
    with st.sidebar:
        symbol = st.text_input("Ticker", value="AAPL", max_chars=15).strip().upper()
    if not re.fullmatch(r"[A-Z0-9^.=\-]{1,15}", symbol):
        st.error("Enter a valid ticker symbol, for example AAPL, MSFT or SPY.")
        return
    today = datetime.now(NY_TZ).date()
    try:
        expirations = tuple(day for day in get_expirations(symbol)
                            if 14 <= (pd.Timestamp(day).date() - today).days <= 365)
    except Exception as exc:
        st.error(f"Could not load {symbol} option expirations: {exc}")
        return
    if not expirations:
        st.warning(f"No eligible call expirations found for {symbol}. Try another ticker or Manual scenario.")
        return
    default_index = next((i for i, day in enumerate(expirations)
                          if (pd.Timestamp(day).date() - today).days >= 30), 0)
    with st.sidebar:
        expiration = st.selectbox("Pricing expiration", expirations, index=default_index)
        with st.expander("Quote filters"):
            strike_range = st.slider("Strike range (% of spot)", 50, 150, (80, 120), step=5)
            max_spread_pct = st.slider("Maximum bid/ask spread (%)", 5, 100, 50, step=5)
    strike_bounds = tuple(bound / 100 for bound in strike_range)
    max_spread = max_spread_pct / 100
    try:
        history = get_history(symbol)
        history_error = None
    except Exception as exc:
        history, history_error = None, str(exc)
    try:
        calls, underlying, fetched_at = get_chain(symbol, expiration)
        underlying = underlying or {}
        spot = float(underlying.get("regularMarketPrice", np.nan))
        if (not np.isfinite(spot) or spot <= 0) and history is not None and not history.empty:
            spot = float(history["Close"].iloc[-1])
            spot_source = "latest available adjusted close"
        else:
            spot_source = "chain underlying quote"
        if not np.isfinite(spot) or spot <= 0:
            raise ValueError("Reference stock price is unavailable")
        selected_quotes = filter_calls(calls, spot, strike_bounds, max_spread)
        if len(selected_quotes) < MIN_QUOTES:
            raise ValueError(f"Only {len(selected_quotes)} usable calls; at least {MIN_QUOTES} needed")
    except Exception as exc:
        st.error(f"Could not prepare {symbol} option chain: {exc}")
        return

    selected_quotes = selected_quotes.assign(
        expiration=expiration,
        days_to_expiry=(pd.Timestamp(expiration).date() - today).days,
        fetched_at_utc=fetched_at.strftime("%Y-%m-%d %H:%M UTC"),
    )
    maturity = float(selected_quotes["days_to_expiry"].iloc[0]) / 365
    try:
        treasury_asof, treasury_curve = get_treasury_curve(today)
        rate_default = treasury_rate_for_maturity(treasury_curve, maturity)
        rate_error = None
    except Exception as exc:
        treasury_asof, rate_default, rate_error = None, None, str(exc)
    try:
        dividend_default, trailing_cash, dividend_count = trailing_dividend_yield(
            history, spot, today)
        dividend_error = None
    except Exception as exc:
        dividend_default, trailing_cash, dividend_count = None, None, None
        dividend_error = str(exc)

    with st.sidebar:
        strike = st.selectbox("Pricing strike ($)", selected_quotes["strike"].tolist(),
                              index=int(np.argmin(abs(selected_quotes["strike"].to_numpy() - spot))))
        rate_pct = st.number_input("Annual model risk-free rate (%)",
                                   value=round(rate_default, 3) if rate_default is not None else None,
                                   step=0.05, format="%.3f", placeholder="Enter annual rate",
                                   key=f"rate_{symbol}_{expiration}_{treasury_asof}")
        if rate_default is None:
            st.warning(f"Treasury rate unavailable: {rate_error}. Enter an assumption to enable model charts.")
        else:
            st.caption(f"Default: FRED U.S. Treasury par yield proxy, as of {treasury_asof}.")
        dividend_pct = st.number_input("Annual model dividend yield (%)", min_value=0.0,
                                       value=round(dividend_default, 3) if dividend_default is not None else None,
                                       step=0.05, format="%.3f", placeholder="Enter annual yield",
                                       key=f"dividend_{symbol}_{expiration}_{today}")
        if dividend_default is None:
            st.warning(f"Dividend proxy unavailable: {dividend_error}. Enter an assumption (0% only if intended).")
        else:
            st.caption(f"Default: \\${trailing_cash:.2f} across {dividend_count} cash payments "
                       "in the past year / reference spot.")
        n_paths = st.slider("Monte Carlo paths", 1000, 20000, 10000, step=1000)
        n_steps = st.slider("Time steps", 12, 252, 64, step=4)
        seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42)
    priced_quote = selected_quotes.loc[selected_quotes["strike"] == strike].iloc[0]

    quote_sets, skipped = [selected_quotes], []
    for day in choose_surface_dates(expirations, today, expiration):
        if day == expiration:
            continue
        try:
            day_calls, _, chain_fetched_at = get_chain(symbol, day)
            quotes = filter_calls(day_calls, spot, strike_bounds, max_spread)
            if len(quotes) < MIN_QUOTES:
                skipped.append(f"{day}: too few usable quotes")
                continue
            quote_sets.append(quotes.assign(
                expiration=day,
                days_to_expiry=(pd.Timestamp(day).date() - today).days,
                fetched_at_utc=chain_fetched_at.strftime("%Y-%m-%d %H:%M UTC"),
            ))
        except Exception as exc:
            skipped.append(f"{day}: {exc}")
    all_quotes = pd.concat(quote_sets, ignore_index=True)

    first, second, third = st.columns(3)
    first.metric(f"{symbol} reference price", f"${spot:,.2f}")
    second.metric("Selected call market mid", f"${priced_quote['mid']:,.2f}")
    third.metric("Valid quotes / expirations",
                 f"{len(all_quotes)} / {all_quotes['expiration'].nunique()}")
    st.caption(f"Selected chain fetched {fetched_at:%Y-%m-%d %H:%M UTC}; "
               f"reference spot source: {spot_source}. Option bid/ask timestamps unavailable. "
               "Different expirations may have different fetch times.")
    quote_time = underlying.get("regularMarketTime")
    try:
        quoted_at = datetime.fromtimestamp(int(quote_time), timezone.utc)
        st.caption(f"Underlying quote timestamp: {quoted_at:%Y-%m-%d %H:%M UTC}.")
        if (datetime.now(timezone.utc) - quoted_at).total_seconds() > 86_400:
            st.warning("The underlying reference quote is more than 24 hours old. Check market hours and data freshness.")
    except (TypeError, ValueError, OverflowError):
        st.caption("Underlying quote timestamp was not supplied by the data provider.")
    if skipped:
        st.warning("Some expirations were excluded: " + "; ".join(skipped))
    st.caption(f"Selected contract {priced_quote.get('contractSymbol', '')} | "
               f"bid \\${priced_quote['bid']:.2f} / ask \\${priced_quote['ask']:.2f} | "
               f"spread {priced_quote['spread_pct'] * 100:.1f}% | "
               f"reported IV {priced_quote['impliedVolatility'] * 100:.2f}%.")

    market_tab, volatility_tab, model_tab = st.tabs(["Market data", "Volatility", "Pricing model"])
    with market_tab:
        if history_error:
            st.warning(f"Historical data source unavailable: {history_error}")
        try:
            st.plotly_chart(historical_figure(history, symbol), width="stretch")
            st.caption("Adjusted historical OHLC; rolling log-return volatility annualized with 252 trading days.")
        except Exception as exc:
            st.warning(f"Historical chart unavailable: {exc}")
        st.plotly_chart(market_mid_heatmap(all_quotes, symbol), width="stretch")
        st.caption("Cells are observed bid/ask midpoints. Missing quotes are not filled.")
        quality = all_quotes.groupby("expiration", sort=True).agg(
            Quotes=("strike", "count"),
            MedianSpreadPct=("spread_pct", lambda values: round(float(values.median() * 100), 1)),
            MinStrike=("strike", "min"), MaxStrike=("strike", "max"),
            FetchedAtUTC=("fetched_at_utc", "first"),
        )
        with st.expander("Quote coverage and quality"):
            st.dataframe(quality, width="stretch")
            st.caption(f"Screened: strike {strike_range[0]}%-{strike_range[1]}% of spot, "
                       f"positive bid, ask >= bid, IV 5%-200%, spread <= {max_spread_pct}%. "
                       "This does not guarantee executable quotes.")

    with volatility_tab:
        st.plotly_chart(term_structure_figure(all_quotes, spot, symbol), width="stretch")
        st.caption("Nearest-to-spot listed call at each expiry; observed IV, not a forecast.")
        if rate_pct is None or dividend_pct is None:
            st.info("Enter both annual model rates in the sidebar to fit SVI using forward moneyness.")
        else:
            rate, dividend_yield = rate_pct / 100, dividend_pct / 100
            forward = spot * np.exp((rate - dividend_yield) * maturity)
            strikes = selected_quotes["strike"].to_numpy(dtype=float)
            observed_iv = selected_quotes["impliedVolatility"].to_numpy(dtype=float)
            try:
                fitted_iv, rmse_pp, _ = fit_svi(strikes, observed_iv, maturity, forward)
                st.metric("SVI in-sample RMSE", f"{rmse_pp:.3f} pp",
                          help="Percentage points of implied volatility, measured on the fitting quotes.")
                st.plotly_chart(smile_figure(strikes, observed_iv, fitted_iv,
                                              "Reported IV", maturity), width="stretch")
                st.caption("SVI error is measured on fitting quotes, not held-out data.")
            except Exception as exc:
                st.warning(f"SVI smile unavailable: {exc}")
            surface, diagnostics, rejected, decreasing = market_svi_surface(
                all_quotes, spot, rate, dividend_yield, symbol)
            if surface is None:
                st.warning("3D surface needs three fitted expirations with common forward-moneyness support.")
            else:
                st.plotly_chart(surface, width="stretch")
                st.caption("White dots: reported IVs. The sheet interpolates SVI-fitted total "
                           "variance across expirations within common observed support.")
                st.metric("Negative adjacent total-variance moves", decreasing,
                          help="Calendar-consistency screen, not a complete arbitrage test.")
                if decreasing:
                    st.warning("Some fitted slices fail a calendar-consistency screen; "
                               "do not use this as an arbitrage-free pricing surface.")
                with st.expander("SVI diagnostics by expiration"):
                    st.dataframe(diagnostics, width="stretch")
            if rejected:
                st.caption("Excluded from surface: " + "; ".join(rejected))

    with model_tab:
        if rate_pct is None or dividend_pct is None:
            st.info("Enter both annual model rates in the sidebar to show pricing and simulation charts. "
                    "Missing inputs are not treated as 0%.")
        else:
            rate, dividend_yield = rate_pct / 100, dividend_pct / 100
            volatility = float(priced_quote["impliedVolatility"])
            paths, mc_price, mc_se = simulate_call(spot, strike, maturity, rate,
                                                    dividend_yield, volatility,
                                                    n_paths, n_steps, int(seed))
            bs_price = float(black_scholes_call(spot, strike, maturity, rate,
                                                dividend_yield, volatility))
            one, two, three = st.columns(3)
            one.metric("Observed market mid", f"${priced_quote['mid']:,.2f}")
            two.metric("Monte Carlo benchmark", f"${mc_price:,.2f}")
            three.metric("European BS benchmark", f"${bs_price:,.2f}")
            st.caption(f"MC 95% sampling interval: \\${mc_price - 1.96 * mc_se:.2f}"
                       f"-\\${mc_price + 1.96 * mc_se:.2f}. Rates {rate_pct:.3f}% and "
                       f"dividend yield {dividend_pct:.3f}% are model inputs. Treasury par "
                       "yields and trailing dividends are proxies, not synchronized forward inputs.")
            st.caption("The simulation uses the reported IV as an input; matching the "
                       "option premium is not independent evidence of forecasting accuracy.")
            st.plotly_chart(pricing_heatmap(spot, strike, maturity, rate, dividend_yield,
                                             volatility), width="stretch")
            st.plotly_chart(sampled_paths_figure(paths, maturity), width="stretch")
            st.plotly_chart(scenario_fan_figure(paths, maturity), width="stretch")
            st.plotly_chart(terminal_distribution_figure(paths, strike, spot), width="stretch")
            st.caption("Yellow dashed line: strike; white dotted line: reference spot. "
                       "Exercise style is not verified; listed U.S. equity options are often "
                       "American-style. These European GBM/BS results are educational benchmarks, not tradable "
                       "prices or stock-price forecasts.")

    with st.expander("Option quotes used"):
        columns = ("expiration", "fetched_at_utc", "contractSymbol", "strike", "bid",
                   "ask", "mid", "spread_pct", "impliedVolatility", "openInterest",
                   "lastTradeDate")
        st.dataframe(all_quotes[[column for column in columns if column in all_quotes]],
                     width="stretch")


def main():
    st.set_page_config(page_title="Option Pricing Lab", layout="wide")
    with st.sidebar:
        source = st.radio("Data source", ("Manual scenario", "Market quotes"))
    if source == "Manual scenario":
        render_manual()
    else:
        render_market()


if __name__ == "__main__":
    main()
