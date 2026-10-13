# Option Pricing Lab

I built this Streamlit app to explore call-option pricing and implied volatility. It has two modes: a manual scenario for changing model assumptions, and a market-quotes view for examining option chains and price history.

## What the app does

**Manual scenario:** Change the stock price, strike, time to expiry, interest rate, dividend yield, volatility and number of Monte Carlo paths. The app compares Black-Scholes and Monte Carlo call prices, reports the Monte Carlo standard error and a 95% sampling interval, and plots how the call price changes with stock price and volatility. It also shows simulated stock-price paths and an illustrative implied-volatility surface. These inputs and surfaces are assumed, not observed market data.

**Market quotes:** Enter a stock ticker to load option chains and adjusted price history. The app filters invalid or wide bid and ask quotes, maps option mid prices by strike and expiry, and calculates realized volatility over rolling windows of 21 and 63 trading days. It fits an SVI implied-volatility curve for each eligible expiry and displays the fitted surface alongside the quoted values. In-sample RMSE and a calendar-consistency check help inspect the fit.

## Monte Carlo convergence check

I ran a separate test to see how the number of simulated paths affects pricing error. The fixed scenario uses a $100 stock price and strike, one year to expiry, a 5% interest rate, no dividends and 20% volatility. For each path count, `research_mc.py` runs 20 random seeds and compares the Monte Carlo estimate with the Black-Scholes price of $10.4506.

| Paths | Mean absolute error | Mean Monte Carlo standard error |
| ---: | ---: | ---: |
| 1,000 | $0.3896 | $0.4651 |
| 10,000 | $0.0950 | $0.1476 |
| 100,000 | $0.0431 | $0.0466 |

The mean absolute error fell as the path count increased **in this fixed test**. This checks numerical convergence against the analytic price; it does not show that the model predicts market prices. The 100,000-path runs use the separate research script, not the app's interactive slider. Run-level data and the summary are in `research_results/`.

## Run locally

```bash
python -m pip install numpy pandas scipy plotly streamlit yfinance
python -m streamlit run option_pricing_app.py
```

To reproduce the convergence check, run `python research_mc.py` from the repository root.

## How to read the results

Market quotes can be delayed, missing or too wide to be useful. The SVI error is measured on the same quotes used for fitting, and the calendar check is not a complete arbitrage test. The pricing comparison uses a European call model, while many listed equity calls can be exercised early. Monte Carlo intervals describe simulation sampling error only; simulated paths are scenarios, not forecasts.
