Option Pricing Lab

I built this Streamlit app to explore call-option pricing and implied volatility. It has two modes: a manual scenario for changing model assumptions, and a market-quotes view for examining option chains and price history.

What the app does

Manual scenario: Change the stock price, strike, time to expiry, interest rate, dividend yield, volatility and number of Monte Carlo paths. The app compares Black-Scholes and Monte Carlo call prices, reports the Monte Carlo standard error and a 95% sampling interval, and plots how the call price changes with stock price and volatility. It also shows simulated stock-price paths and an illustrative implied-volatility surface. These inputs and surfaces are assumed, not observed market data.

Market quotes: Enter a stock ticker to load option chains and adjusted price history. The app filters invalid or wide bid and ask quotes, maps option mid prices by strike and expiry, and calculates realized volatility over rolling windows of 21 and 63 trading days. It fits an SVI implied-volatility curve for each eligible expiry and displays the fitted surface alongside the quoted values. In-sample RMSE and a calendar-consistency check help inspect the fit.
