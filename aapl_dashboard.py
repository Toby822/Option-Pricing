"""AAPL market-data dashboard for the Option-Pricing repository.

Run this file as the Streamlit entrypoint from the repository root. It keeps
the existing synthetic monte_carlo.py untouched.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.stats import norm
import streamlit as st
import yfinance as yf

from svi import calibrate_svi, svi


SYMBOL = "AAPL"
MIN_QUOTES = 8
NY_TZ = ZoneInfo("America/New_York")


@st.cache_data(ttl=900, show_spinner=False)
def get_expirations():
    return tuple(yf.Ticker(SYMBOL).options)


@st.cache_data(ttl=900, show_spinner=False)
def get_chain(expiration):
    chain = yf.Ticker(SYMBOL).option_chain(expiration)
    return chain.calls, chain.underlying, datetime.now(timezone.utc)


@st.cache_data(ttl=3600, show_spinner=False)
def get_history():
    return yf.Ticker(SYMBOL).history(period="1y", auto_adjust=False)


def filter_calls(calls, spot):
    required = ("strike", "bid", "ask", "impliedVolatility")
    if calls is None or any(column not in calls for column in required):
        raise ValueError("The option chain is missing required columns")

    quotes = calls.copy()
    for column in required:
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes = quotes.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
    quotes["spread_pct"] = (quotes["ask"] - quotes["bid"]) / quotes["mid"]
    valid = (
        quotes["strike"].between(0.8 * spot, 1.2 * spot)
        & (quotes["bid"] > 0)
        & (quotes["ask"] >= quotes["bid"])
        & quotes["impliedVolatility"].between(0.05, 2.0)
        & (quotes["spread_pct"] <= 0.5)
    )
    return quotes.loc[valid].sort_values("strike").drop_duplicates("strike")


def choose_surface_dates(expirations, today, selected):
    windows = ((14, 45), (45, 90), (90, 150), (150, 240))
    chosen = [selected]
    for start, end in windows:
        match = next(
            (
                day for day in expirations
                 if start <= (pd.Timestamp(day).date() - today).days < end),
            None,
        )
        if match and match not in chosen:
            chosen.append(match)
    return sorted(chosen)


def load_market(today, expiration):
    calls, underlying, fetched_at = get_chain(expiration)
    spot = float((underlying or {}).get("regularMarketPrice", np.nan))
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("The AAPL reference price is unavailable")
    selected_quotes = filter_calls(calls, spot)
    if len(selected_quotes) < MIN_QUOTES:
        raise ValueError(
            f"Only {len(selected_quotes)} usable quotes for {expiration}; "
            f"at least {MIN_QUOTES} are needed"
        )
    selected_quotes = selected_quotes.assign(
        expiration=expiration,
        days_to_expiry=(pd.Timestamp(expiration).date() - today).days,
    )
    return selected_quotes, spot, underlying, fetched_at


def fit_smile(quotes, spot, rate, dividend_yield):
    maturity = float(quotes["days_to_expiry"].iloc[0]) / 365.0
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    strikes = quotes["strike"].to_numpy(dtype=float)
    observed_iv = quotes["impliedVolatility"].to_numpy(dtype=float)
    params = calibrate_svi(strikes, observed_iv, maturity, forward)
    fitted_variance = np.asarray(svi(np.log(strikes / forward), *params), dtype=float)
    if not np.all(np.isfinite(fitted_variance)) or np.any(fitted_variance <= 0):
        raise ValueError("SVI produced invalid total variance")
    fitted_iv = np.sqrt(fitted_variance / maturity)
    rmse_pp = float(np.sqrt(np.mean((fitted_iv - observed_iv) ** 2)) * 100)
    return strikes, observed_iv, fitted_iv, rmse_pp


def market_mid_heatmap(all_quotes):
    mids = all_quotes.pivot_table(
        index="expiration", columns="strike", values="mid", aggfunc="mean"
    ).sort_index()
    fig = go.Figure(
        go.Heatmap(
            x=mids.columns.to_numpy(dtype=float),
            y=mids.index.tolist(),
            z=mids.to_numpy(dtype=float),
            colorscale="Viridis",
            colorbar=dict(title="Mid ($)"),
            hovertemplate="Expiry %{y}<br>Strike $%{x:.2f}<br>Mid $%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="AAPL call bid/ask mid-price heatmap",
        xaxis_title="Strike ($)",
        yaxis_title="Expiration",
        height=510,
    )
    return fig


def volatility_surface(all_quotes):
    points = all_quotes[["strike", "days_to_expiry"]].to_numpy(dtype=float)
    iv = all_quotes["impliedVolatility"].to_numpy(dtype=float)
    if all_quotes["expiration"].nunique() < 3 or len(points) < 24:
        return None

    strike_grid = np.linspace(points[:, 0].min(), points[:, 0].max(), 55)
    day_grid = np.linspace(points[:, 1].min(), points[:, 1].max(), 40)
    strike_mesh, day_mesh = np.meshgrid(strike_grid, day_grid)
    # No nearest-neighbor fill: areas outside observed quote support stay blank.
    iv_mesh = griddata(points, iv, (strike_mesh, day_mesh), method="linear")

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=strike_mesh,
            y=day_mesh,
            z=iv_mesh * 100,
            colorscale="Viridis",
            opacity=0.82,
            colorbar=dict(title="IV (%)"),
            name="Interpolated IV",
            showscale=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=iv * 100,
            mode="markers",
            marker=dict(size=3, color="white", opacity=0.75),
            name="Observed quotes",
        )
    )
    fig.update_layout(
        title="AAPL market IV surface | linear interpolation",
        height=680,
        scene=dict(
            xaxis_title="Strike ($)",
            yaxis_title="Days to expiry",
            zaxis_title="Implied volatility (%)",
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1, z=0.65),
        ),
    )
    return fig


def european_call_benchmark(spot, strike, maturity, rate, dividend_yield, volatility):
    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    return (
        spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1)
        - strike * np.exp(-rate * maturity) * norm.cdf(d2)
    )


def market_calibrated_simulation(spot, strike, maturity, rate, dividend_yield, volatility):
    n_paths = 10000
    n_steps = 64
    dt = maturity / n_steps
    rng = np.random.default_rng(42)
    z = rng.standard_normal((n_steps, n_paths))
    increments = (
        (rate - dividend_yield - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * z
    )
    paths = spot * np.vstack((np.ones(n_paths), np.exp(np.cumsum(increments, axis=0))))
    discounted = np.exp(-rate * maturity) * np.maximum(paths[-1] - strike, 0)
    estimate = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return paths[:, :30], estimate, standard_error


def main():
    st.set_page_config(page_title="AAPL Option-Pricing", layout="wide")
    st.title("AAPL Option-Pricing")
    st.caption("Market quotes and history: Yahoo Finance via yfinance. Not real-time trading data.")

    today = datetime.now(NY_TZ).date()
    try:
        expirations = tuple(
            day for day in get_expirations()
            if 14 <= (pd.Timestamp(day).date() - today).days <= 365
        )
    except Exception as exc:
        st.error(f"Could not load AAPL expirations: {exc}")
        st.stop()
    if not expirations:
        st.error("No usable AAPL expirations are available.")
        st.stop()

    default_index = next(
        (i for i, day in enumerate(expirations)
         if (pd.Timestamp(day).date() - today).days >= 30),
        0,
    )
    with st.sidebar:
        expiration = st.selectbox("Pricing expiration", expirations, index=default_index)
        rate_pct = st.number_input("Assumed risk-free rate (%)", value=0.0, step=0.1)
        dividend_pct = st.number_input("Assumed dividend yield (%)", value=0.0, step=0.1)

    try:
        selected_quotes, spot, underlying, fetched_at = load_market(today, expiration)
    except Exception as exc:
        st.error(f"Could not load the AAPL option chain: {exc}")
        st.stop()

    with st.sidebar:
        strike = st.selectbox(
            "Pricing strike ($)",
            selected_quotes["strike"].tolist(),
            index=int(np.argmin(np.abs(selected_quotes["strike"].to_numpy() - spot))),
        )
    rate = rate_pct / 100.0
    dividend_yield = dividend_pct / 100.0
    priced_quote = selected_quotes.loc[selected_quotes["strike"] == strike].iloc[0]

    surface_dates = choose_surface_dates(expirations, today, expiration)
    quote_sets = [selected_quotes]
    skipped = []
    for day in surface_dates:
        if day == expiration:
            continue
        try:
            calls, _, _ = get_chain(day)
            quotes = filter_calls(calls, spot)
            if len(quotes) < MIN_QUOTES:
                skipped.append(f"{day}: too few usable quotes")
                continue
            quote_sets.append(
                quotes.assign(
                    expiration=day,
                    days_to_expiry=(pd.Timestamp(day).date() - today).days,
                )
            )
        except Exception as exc:
            skipped.append(f"{day}: {exc}")
    all_quotes = pd.concat(quote_sets, ignore_index=True)

    first, second, third = st.columns(3)
    first.metric("AAPL reference price", f"${spot:,.2f}")
    second.metric("Market call mid", f"${priced_quote['mid']:,.2f}")
    third.metric("Usable option quotes", len(all_quotes))

    quote_time = (underlying or {}).get("regularMarketTime")
    try:
        quoted_at = datetime.fromtimestamp(int(quote_time), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OverflowError):
        quoted_at = "not supplied"
    st.caption(
        f"Selected chain fetched {fetched_at:%Y-%m-%d %H:%M UTC}; "
        f"underlying quote time: {quoted_at}. Option bid/ask timestamps are unavailable. "
        "Other expirations may have different fetch times."
    )
    if skipped:
        st.warning("Some expirations were excluded: " + "; ".join(skipped))

    market_tab, volatility_tab, model_tab = st.tabs(
        ["Market data", "Volatility", "Pricing model"]
    )

    with market_tab:
        try:
            history = get_history()
            if history is None or history.empty or "Close" not in history:
                raise ValueError("AAPL historical close data is unavailable")
            history_fig = go.Figure(
                go.Scatter(
                    x=history.index, y=history["Close"], mode="lines", name="AAPL close"
                )
            )
            history_fig.update_layout(
                title="AAPL observed daily close | past year",
                xaxis_title="Date", yaxis_title="Close ($)", height=430,
            )
            st.plotly_chart(history_fig, width="stretch")
        except Exception as exc:
            st.warning(f"Historical price chart unavailable: {exc}")
        st.plotly_chart(market_mid_heatmap(all_quotes), width="stretch")
        st.caption("Heatmap cells are observed bid/ask mid-prices. Missing cells are not invented.")

    with volatility_tab:
        try:
            strikes, observed_iv, fitted_iv, rmse_pp = fit_smile(
                selected_quotes, spot, rate, dividend_yield
            )
            st.metric("SVI in-sample RMSE", f"{rmse_pp:.3f} percentage points")
            smile_fig = go.Figure()
            smile_fig.add_trace(
                go.Scatter(
                    x=strikes, y=observed_iv * 100,
                    mode="markers", name="Yahoo-reported IV",
                )
            )
            smile_fig.add_trace(
                go.Scatter(x=strikes, y=fitted_iv * 100, mode="lines", name="SVI fit")
            )
            smile_fig.update_layout(
                title=f"AAPL call volatility smile | {expiration}",
                xaxis_title="Strike ($)", yaxis_title="Implied volatility (%)",
                height=480,
            )
            st.plotly_chart(smile_fig, width="stretch")
        except Exception as exc:
            st.warning(f"SVI smile unavailable: {exc}")

        try:
            surface = volatility_surface(all_quotes)
            if surface is None:
                st.warning("At least three expirations with enough quotes are needed for a 3D IV surface.")
            else:
                st.plotly_chart(surface, width="stretch")
                st.caption(
                    "White markers are reported IVs. The colored sheet is linearly interpolated "
                    "between them; blank areas lie outside observed quote support. "
                    "No cross-expiry arbitrage validation has been applied."
                )
        except Exception as exc:
            st.warning(f"3D IV surface unavailable: {exc}")

    with model_tab:
        maturity = float(priced_quote["days_to_expiry"]) / 365.0
        volatility = float(priced_quote["impliedVolatility"])
        paths, mc_price, mc_se = market_calibrated_simulation(
            spot, strike, maturity, rate, dividend_yield, volatility
        )
        bs_price = european_call_benchmark(
            spot, strike, maturity, rate, dividend_yield, volatility
        )
        one, two, three = st.columns(3)
        one.metric("Observed market mid", f"${priced_quote['mid']:,.2f}")
        two.metric("Monte Carlo estimate", f"${mc_price:,.2f}")
        three.metric("European BS benchmark", f"${bs_price:,.2f}")
        st.caption(
            f"Market bid/ask: ${priced_quote['bid']:,.2f} / ${priced_quote['ask']:,.2f}; "
            f"Yahoo-reported IV: {volatility * 100:.2f}%; "
            f"Monte Carlo standard error: ${mc_se:.3f}. "
            "The two model prices use the sidebar's assumed rate and dividend yield."
        )
        path_fig = go.Figure()
        elapsed_days = np.linspace(0, int(priced_quote["days_to_expiry"]), len(paths))
        for path in paths.T:
            path_fig.add_trace(
                go.Scatter(x=elapsed_days, y=path, mode="lines", showlegend=False)
            )
        path_fig.update_layout(
            title="AAPL market-IV-calibrated simulated paths | not observed future prices",
            xaxis_title="Days from reference quote",
            yaxis_title="Simulated AAPL price ($)",
            height=480,
        )
        st.plotly_chart(path_fig, width="stretch")
        st.caption(
            "AAPL equity options are American-style. These European-style GBM/BS "
            "calculations are educational benchmarks, not tradable prices or forecasts."
        )

    with st.expander("Option quotes used"):
        columns = (
            "expiration", "contractSymbol", "strike", "bid", "ask", "mid",
            "spread_pct", "impliedVolatility", "openInterest",
        )
        st.dataframe(
            all_quotes[[column for column in columns if column in all_quotes]],
            width="stretch",
        )


if __name__ == "__main__":
    main()

