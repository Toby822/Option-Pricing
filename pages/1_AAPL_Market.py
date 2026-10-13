"""AAPL market-data analysis for the Option-Pricing Streamlit app.

Use this as the Streamlit entrypoint or as a page in the pages directory.
It does not modify the synthetic pricing examples.
"""

from datetime import datetime, timedelta, timezone
from io import BytesIO
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


SYMBOL = "AAPL"
MIN_QUOTES = 8
NY_TZ = ZoneInfo("America/New_York")
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO,DGS3MO,DGS6MO,DGS1"
TREASURY_TENORS_YEARS = np.array([1 / 12, 3 / 12, 6 / 12, 1.0])
TREASURY_COLUMNS = ("DGS1MO", "DGS3MO", "DGS6MO", "DGS1")


@st.cache_data(ttl=900, show_spinner=False)
def get_expirations():
    return tuple(yf.Ticker(SYMBOL).options)


@st.cache_data(ttl=900, show_spinner=False)
def get_chain(expiration):
    chain = yf.Ticker(SYMBOL).option_chain(expiration)
    return chain.calls, chain.underlying, datetime.now(timezone.utc)


@st.cache_data(ttl=3600, show_spinner=False)
def get_history():
    return yf.Ticker(SYMBOL).history(period="1y", auto_adjust=True, actions=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_treasury_curve(today):
    request = Request(FRED_CSV_URL, headers={"User-Agent": "Option-Pricing research dashboard"})
    with urlopen(request, timeout=12) as response:
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
        raise ValueError("FRED has no complete Treasury curve for this date")
    latest = frame.sort_values("observation_date").iloc[-1]
    observed_on = latest["observation_date"].date()
    if (today - observed_on).days > 10:
        raise ValueError(f"The latest FRED Treasury curve is stale ({observed_on})")
    return observed_on, latest[list(TREASURY_COLUMNS)].to_numpy(dtype=float)


def treasury_rate_for_maturity(curve_rates_pct, maturity):
    if not np.all(np.isfinite(curve_rates_pct)) or np.any(curve_rates_pct < 0):
        raise ValueError("Treasury curve contains invalid yields")
    return float(np.interp(maturity, TREASURY_TENORS_YEARS, curve_rates_pct))


def trailing_dividend_yield(history, spot, today):
    if history is None or history.empty or "Dividends" not in history:
        raise ValueError("AAPL dividend history is unavailable")
    dividends = pd.to_numeric(history["Dividends"], errors="coerce").fillna(0)
    dates = pd.to_datetime(history.index).date
    valid = (
        (dates >= today - timedelta(days=365))
        & (dates <= today)
        & (dividends.to_numpy() > 0)
    )
    count = int(np.count_nonzero(valid))
    if count < 3:
        raise ValueError(f"Only {count} cash dividends found in the past year")
    annual_cash = float(dividends.to_numpy()[valid].sum())
    if not np.isfinite(annual_cash) or annual_cash <= 0 or spot <= 0:
        raise ValueError("AAPL dividend yield cannot be calculated")
    return 100 * annual_cash / spot, annual_cash, count


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
        fetched_at_utc=fetched_at.strftime("%Y-%m-%d %H:%M UTC"),
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
    return strikes, observed_iv, fitted_iv, rmse_pp, params


def market_mid_heatmap(all_quotes):
    fields = ("mid", "bid", "ask", "impliedVolatility", "spread_pct")
    grids = {
        field: all_quotes.pivot_table(
            index="expiration", columns="strike", values=field, aggfunc="mean"
        ).sort_index()
        for field in fields
    }
    mids = grids["mid"]
    aligned = [grids[field].reindex_like(mids).to_numpy(dtype=float) for field in fields]
    customdata = np.stack(aligned[1:], axis=-1)
    fig = go.Figure(
        go.Heatmap(
            x=mids.columns.to_numpy(dtype=float),
            y=mids.index.tolist(),
            z=mids.to_numpy(dtype=float),
            customdata=customdata,
            colorscale="Viridis",
            colorbar=dict(title="Mid ($)"),
            hovertemplate=(
                "Expiry %{y}<br>Strike $%{x:.2f}<br>Mid $%{z:.2f}<br>"
                "Bid $%{customdata[0]:.2f} / Ask $%{customdata[1]:.2f}<br>"
                "IV %{customdata[2]:.1%}<br>Spread %{customdata[3]:.1%}"
                "<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="AAPL call bid/ask mid-price heatmap",
        xaxis_title="Strike ($)",
        yaxis_title="Expiration",
        height=510,
    )
    return fig


def svi_surface(all_quotes, spot, rate, dividend_yield):
    slices = []
    rejected = []
    for expiration, quotes in all_quotes.groupby("expiration", sort=True):
        if len(quotes) < MIN_QUOTES:
            rejected.append(f"{expiration}: insufficient quotes")
            continue
        try:
            _, _, _, rmse_pp, params = fit_smile(quotes, spot, rate, dividend_yield)
            maturity = float(quotes["days_to_expiry"].iloc[0]) / 365.0
            forward = spot * np.exp((rate - dividend_yield) * maturity)
            log_min = np.log(float(quotes["strike"].min()) / forward)
            log_max = np.log(float(quotes["strike"].max()) / forward)
            slices.append((expiration, quotes, maturity, forward, params, rmse_pp, log_min, log_max))
        except Exception as exc:
            rejected.append(f"{expiration}: {exc}")
    if len(slices) < 3:
        return None, pd.DataFrame(), rejected, None

    log_min = max(item[6] for item in slices)
    log_max = min(item[7] for item in slices)
    if log_max <= log_min:
        rejected.append("No common observed moneyness range across expirations")
        return None, pd.DataFrame(), rejected, None

    log_grid = np.linspace(log_min, log_max, 70)
    maturities = np.array([item[2] for item in slices])
    total_variances = np.stack(
        [np.asarray(svi(log_grid, *item[4]), dtype=float) for item in slices]
    )
    if not np.all(np.isfinite(total_variances)) or np.any(total_variances <= 0):
        rejected.append("A fitted SVI slice has invalid total variance")
        return None, pd.DataFrame(), rejected, None

    # A negative adjacent difference is a useful screen, not a full arbitrage test.
    decreasing_variance = int(np.count_nonzero(np.diff(total_variances, axis=0) < -1e-8))
    dense_maturities = np.linspace(maturities.min(), maturities.max(), 45)
    dense_variance = np.column_stack(
        [np.interp(dense_maturities, maturities, total_variances[:, j])
         for j in range(len(log_grid))]
    )
    iv_mesh = np.sqrt(dense_variance / dense_maturities[:, None]) * 100
    log_mesh = np.broadcast_to(log_grid, iv_mesh.shape)
    day_mesh = np.broadcast_to(dense_maturities[:, None] * 365, iv_mesh.shape)
    forward_mesh = spot * np.exp((rate - dividend_yield) * dense_maturities[:, None])
    strike_mesh = forward_mesh * np.exp(log_mesh)

    observed = pd.concat([item[1] for item in slices], ignore_index=True)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=strike_mesh,
            y=day_mesh,
            z=iv_mesh,
            colorscale="Viridis",
            opacity=0.86,
            colorbar=dict(title="IV (%)"),
            name="SVI fitted surface",
            showscale=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=observed["strike"],
            y=observed["days_to_expiry"],
            z=observed["impliedVolatility"] * 100,
            mode="markers",
            marker=dict(size=2.5, color="#f5f5f5", opacity=0.65),
            name="Yahoo-reported IV",
        )
    )
    fig.update_layout(
        title="AAPL IV surface | SVI slices fitted to market quotes",
        height=680,
        scene=dict(
            xaxis_title="Strike ($)",
            yaxis_title="Days to expiry",
            zaxis_title="Implied volatility (%)",
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1, z=0.65),
        ),
    )
    diagnostics = pd.DataFrame(
        {
            "Expiration": [item[0] for item in slices],
            "Quotes": [len(item[1]) for item in slices],
            "SVI RMSE (pp)": [round(item[5], 3) for item in slices],
        }
    )
    return fig, diagnostics, rejected, decreasing_variance


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
    return paths, estimate, standard_error


def historical_figure(history):
    required = {"Open", "High", "Low", "Close", "Volume"}
    if history is None or history.empty or not required.issubset(history.columns):
        raise ValueError("Adjusted AAPL OHLCV history is unavailable")

    log_returns = np.log(history["Close"]).diff()
    rv_21 = log_returns.rolling(21).std() * np.sqrt(252) * 100
    rv_63 = log_returns.rolling(63).std() * np.sqrt(252) * 100
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        row_heights=[0.55, 0.17, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=history.index, open=history["Open"], high=history["High"],
            low=history["Low"], close=history["Close"], name="Adjusted OHLC",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=history.index, y=history["Volume"], name="Volume", marker_color="#638aa3"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=history.index, y=rv_21, mode="lines", name="21d realized vol"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=history.index, y=rv_63, mode="lines", name="63d realized vol"),
        row=3, col=1,
    )
    fig.update_layout(
        title="AAPL observed adjusted OHLC, volume and realized volatility | past year",
        height=760, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.03),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Shares", row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


def term_structure_figure(all_quotes, spot):
    atm = all_quotes.loc[
        all_quotes.groupby("expiration")["strike"].apply(
            lambda strikes: (strikes - spot).abs().idxmin()
        )
    ].sort_values("days_to_expiry")
    fig = go.Figure(
        go.Scatter(
            x=atm["days_to_expiry"],
            y=atm["impliedVolatility"] * 100,
            mode="lines+markers", name="Nearest-to-spot call IV",
            customdata=atm[["expiration", "strike", "spread_pct"]].to_numpy(),
            hovertemplate=(
                "Expiry %{customdata[0]}<br>Days %{x}<br>"
                "Strike $%{customdata[1]:.2f}<br>IV %{y:.2f}%<br>"
                "Spread %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="AAPL near-the-money IV term structure",
        xaxis_title="Days to expiry", yaxis_title="Implied volatility (%)",
        height=390,
    )
    return fig


def scenario_fan_figure(paths, days_to_expiry):
    days = np.linspace(0, days_to_expiry, len(paths))
    q05, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=q95, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=days, y=q05, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(68, 141, 180, 0.18)",
            name="5th-95th percentile",
        )
    )
    fig.add_trace(go.Scatter(x=days, y=q75, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=days, y=q25, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(68, 141, 180, 0.35)",
            name="25th-75th percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days, y=q50, mode="lines", line=dict(color="#efbd56", width=2),
            name="Median simulation",
        )
    )
    fig.update_layout(
        title="AAPL market-IV-calibrated simulation range | not a forecast",
        xaxis_title="Days from reference quote", yaxis_title="Simulated price ($)",
        height=500,
    )
    return fig


def terminal_distribution_figure(paths, strike, spot):
    fig = go.Figure(
        go.Histogram(x=paths[-1], nbinsx=60, marker_color="#4b9db5", name="Simulated final price")
    )
    fig.add_vline(x=strike, line_dash="dash", line_color="#efbd56")
    fig.add_vline(x=spot, line_dash="dot", line_color="#e7e7e7")
    fig.update_layout(
        title="Simulated AAPL price at expiry | model distribution",
        xaxis_title="Simulated price ($)", yaxis_title="Paths", height=390,
    )
    return fig


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

    try:
        selected_quotes, spot, underlying, fetched_at = load_market(today, expiration)
    except Exception as exc:
        st.error(f"Could not load the AAPL option chain: {exc}")
        st.stop()

    maturity = float(selected_quotes["days_to_expiry"].iloc[0]) / 365.0
    try:
        treasury_asof, treasury_rates = get_treasury_curve(today)
        rate_default = treasury_rate_for_maturity(treasury_rates, maturity)
        rate_error = None
    except Exception as exc:
        treasury_asof, rate_default, rate_error = None, None, str(exc)
    try:
        history = get_history()
        dividend_default, trailing_cash, dividend_count = trailing_dividend_yield(
            history, spot, today
        )
        dividend_error = None
    except Exception as exc:
        history = None
        dividend_default, trailing_cash, dividend_count = None, None, None
        dividend_error = str(exc)

    with st.sidebar:
        strike = st.selectbox(
            "Pricing strike ($)",
            selected_quotes["strike"].tolist(),
            index=int(np.argmin(np.abs(selected_quotes["strike"].to_numpy() - spot))),
        )
        rate_pct = st.number_input(
            "Model risk-free rate (%)",
            value=round(rate_default, 3) if rate_default is not None else None,
            step=0.05, format="%.3f", placeholder="Enter a rate",
            key=f"risk_free_{expiration}_{treasury_asof}",
        )
        if rate_default is None:
            st.warning(f"Treasury rate unavailable: {rate_error}. Enter a rate above.")
        else:
            st.caption(
                f"Default: FRED U.S. Treasury 1M/3M/6M/1Y curve, "
                f"interpolated to this expiry; as of {treasury_asof}."
            )
        dividend_pct = st.number_input(
            "Model dividend yield (%)",
            value=round(dividend_default, 3) if dividend_default is not None else None,
            step=0.05, format="%.3f", placeholder="Enter a yield",
            key=f"dividend_{expiration}_{today}",
        )
        if dividend_default is None:
            st.warning(f"Dividend history unavailable: {dividend_error}. Enter a yield above.")
        else:
            st.caption(
                f"Default: ${trailing_cash:.2f} cash dividends across {dividend_count} "
                f"payments in the past year / current AAPL reference price."
            )
    if rate_pct is None or dividend_pct is None:
        st.error("Enter both model rates in the sidebar; missing data is not treated as 0%.")
        st.stop()
    rate_source = (
        f"FRED Treasury curve, {treasury_asof}"
        if rate_default is not None and abs(rate_pct - round(rate_default, 3)) < 0.0005
        else "manual input"
    )
    dividend_source = (
        "Yahoo trailing cash dividends / AAPL reference price"
        if dividend_default is not None
        and abs(dividend_pct - round(dividend_default, 3)) < 0.0005
        else "manual input"
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
            calls, _, chain_fetched_at = get_chain(day)
            quotes = filter_calls(calls, spot)
            if len(quotes) < MIN_QUOTES:
                skipped.append(f"{day}: too few usable quotes")
                continue
            quote_sets.append(
                quotes.assign(
                    expiration=day,
                    days_to_expiry=(pd.Timestamp(day).date() - today).days,
                    fetched_at_utc=chain_fetched_at.strftime("%Y-%m-%d %H:%M UTC"),
                )
            )
        except Exception as exc:
            skipped.append(f"{day}: {exc}")
    all_quotes = pd.concat(quote_sets, ignore_index=True)

    first, second, third = st.columns(3)
    first.metric("AAPL reference price", f"${spot:,.2f}")
    second.metric("Selected call market mid", f"${priced_quote['mid']:,.2f}")
    third.metric("Valid quotes / expirations", f"{len(all_quotes)} / {all_quotes['expiration'].nunique()}")

    quote_time = (underlying or {}).get("regularMarketTime")
    try:
        quoted_at_dt = datetime.fromtimestamp(int(quote_time), timezone.utc)
        quoted_at = quoted_at_dt.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OverflowError):
        quoted_at_dt = None
        quoted_at = "not supplied"
    st.caption(
        f"Selected chain fetched {fetched_at:%Y-%m-%d %H:%M UTC}; "
        f"underlying quote time: {quoted_at}. Option bid/ask timestamps are unavailable. "
        "Other expirations may have different fetch times."
    )
    if quoted_at_dt is not None and (datetime.now(timezone.utc) - quoted_at_dt).total_seconds() > 86400:
        st.warning("The underlying reference quote is more than 24 hours old. Check market hours and data freshness.")
    st.caption(
        f"Model inputs: risk-free {rate_pct:.3f}% ({rate_source}); "
        f"dividend yield {dividend_pct:.3f}% ({dividend_source}). "
        "Treasury par yields and trailing dividends are proxies, not synchronized "
        "zero-coupon rates or forward dividends."
    )
    if rate_pct == 0 or dividend_pct == 0:
        st.warning("A 0% model input is active. Check the sidebar before interpreting model prices.")
    st.caption(
        f"Selected contract {priced_quote.get('contractSymbol', '')} | "
        f"bid ${priced_quote['bid']:.2f} / ask ${priced_quote['ask']:.2f} | "
        f"relative spread {priced_quote['spread_pct'] * 100:.1f}% | "
        f"Yahoo-reported IV {priced_quote['impliedVolatility'] * 100:.2f}%."
    )
    if skipped:
        st.warning("Some expirations were excluded: " + "; ".join(skipped))

    market_tab, volatility_tab, model_tab = st.tabs(
        ["Market data", "Volatility", "Pricing model"]
    )

    with market_tab:
        try:
            st.plotly_chart(historical_figure(history), width="stretch")
            st.caption(
                "Historical OHLC is adjusted for corporate actions by the data provider. "
                "Realized volatility uses rolling log returns, annualized with 252 trading days."
            )
        except Exception as exc:
            st.warning(f"Historical price chart unavailable: {exc}")
        st.plotly_chart(market_mid_heatmap(all_quotes), width="stretch")
        st.caption(
            "Each colored cell is the midpoint of a reported bid and ask; hover to inspect "
            "the underlying quote and spread. Missing cells are not filled."
        )

        quality = all_quotes.groupby("expiration", sort=True).agg(
            Quotes=("strike", "count"),
            MedianSpreadPct=("spread_pct", lambda s: round(float(s.median() * 100), 1)),
            MinStrike=("strike", "min"),
            MaxStrike=("strike", "max"),
            FetchedAtUTC=("fetched_at_utc", "first"),
        )
        with st.expander("Quote coverage and quality"):
            st.dataframe(quality, width="stretch")
            st.caption(
                "Included: strikes within 80%-120% of reference spot, positive bid, "
                "ask >= bid, IV 5%-200%, and relative spread <= 50%. "
                "These screens do not guarantee synchronized or executable quotes."
            )

    with volatility_tab:
        st.plotly_chart(term_structure_figure(all_quotes, spot), width="stretch")
        st.caption(
            "Each point uses the listed call strike nearest the reference spot at that expiry. "
            "It is an observed IV term structure, not a forward-volatility forecast."
        )
        try:
            strikes, observed_iv, fitted_iv, rmse_pp, _ = fit_smile(
                selected_quotes, spot, rate, dividend_yield
            )
            st.metric("SVI in-sample RMSE", f"{rmse_pp:.3f} percentage points")
            smile_fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                row_heights=[0.72, 0.28],
            )
            smile_fig.add_trace(
                go.Scatter(
                    x=strikes, y=observed_iv * 100,
                    mode="markers", name="Yahoo-reported IV",
                    marker=dict(size=8),
                    customdata=selected_quotes[["bid", "ask", "spread_pct"]].to_numpy(),
                    hovertemplate=(
                        "Strike $%{x:.2f}<br>IV %{y:.2f}%<br>"
                        "Bid $%{customdata[0]:.2f} / Ask $%{customdata[1]:.2f}<br>"
                        "Spread %{customdata[2]:.1%}<extra></extra>"
                    ),
                ), row=1, col=1,
            )
            smile_fig.add_trace(
                go.Scatter(x=strikes, y=fitted_iv * 100, mode="lines", name="SVI fit"),
                row=1, col=1,
            )
            smile_fig.add_trace(
                go.Bar(
                    x=strikes, y=(fitted_iv - observed_iv) * 100,
                    marker_color="#8ea6b5", name="Fit minus quote",
                ), row=2, col=1,
            )
            smile_fig.update_layout(
                title=f"AAPL call volatility smile | {expiration}",
                height=620,
            )
            smile_fig.update_yaxes(title_text="IV (%)", row=1, col=1)
            smile_fig.update_yaxes(title_text="Error (pp)", row=2, col=1)
            smile_fig.update_xaxes(title_text="Strike ($)", row=2, col=1)
            st.plotly_chart(smile_fig, width="stretch")
            st.caption("SVI error is measured on the same quotes used to fit it; it is not out-of-sample accuracy.")
        except Exception as exc:
            st.warning(f"SVI smile unavailable: {exc}")

        try:
            surface, diagnostics, rejected, decreasing_variance = svi_surface(
                all_quotes, spot, rate, dividend_yield
            )
            if surface is None:
                st.warning("At least three valid SVI-fitted expirations with overlapping strikes are needed.")
            else:
                st.plotly_chart(surface, width="stretch")
                st.caption(
                    "Dots are Yahoo-reported IVs. The sheet fits SVI separately to each expiry, "
                    "then linearly interpolates fitted total variance between expiries. "
                    "It stays within the shared observed forward-moneyness range."
                )
                st.metric(
                    "Negative adjacent total-variance moves",
                    decreasing_variance,
                    help="Screened across 70 common forward-moneyness points; not a full no-arbitrage test.",
                )
                if decreasing_variance:
                    st.warning(
                        "The fitted slices fail a calendar-consistency screen at some sampled points. "
                        "Do not use this surface as an arbitrage-free pricing input."
                    )
                with st.expander("SVI fit diagnostics by expiration"):
                    st.dataframe(diagnostics, width="stretch")
            if rejected:
                st.caption("Excluded from surface: " + "; ".join(rejected))
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
        lower_95, upper_95 = mc_price - 1.96 * mc_se, mc_price + 1.96 * mc_se
        st.caption(
            f"Market bid/ask: ${priced_quote['bid']:,.2f} / ${priced_quote['ask']:,.2f}; "
            f"Yahoo-reported IV: {volatility * 100:.2f}%; "
            f"Monte Carlo 95% sampling interval: ${lower_95:.2f}-${upper_95:.2f}. "
            "The model prices use the sidebar's assumed rate and dividend yield. "
            "Sampling uncertainty is not model or market-data uncertainty."
        )
        st.caption(
            "The simulation uses the option's reported IV as an input, so agreement with "
            "the market premium is not independent evidence of forecasting accuracy."
        )
        st.plotly_chart(
            scenario_fan_figure(paths, int(priced_quote["days_to_expiry"])),
            width="stretch",
        )
        st.plotly_chart(terminal_distribution_figure(paths, strike, spot), width="stretch")
        st.caption(
            "Yellow dashed line: selected strike; white dotted line: AAPL reference price. "
            "These markers may nearly coincide when the strike is near the reference price."
        )
        st.caption(
            "AAPL equity options are American-style. These European-style GBM/BS "
            "calculations are educational benchmarks, not tradable prices or forecasts."
        )

    with st.expander("Option quotes used"):
        columns = (
            "expiration", "fetched_at_utc", "contractSymbol", "strike", "bid", "ask", "mid",
            "spread_pct", "impliedVolatility", "openInterest", "lastTradeDate",
        )
        st.dataframe(
            all_quotes[[column for column in columns if column in all_quotes]],
            width="stretch",
        )


main()