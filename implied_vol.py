from scipy.optimize import brentq
from option_pricing import call_price

def implied_volatility(market_price, S, K, T, r):

    def difference(sigma):
        return call_price(S, K, T, r, sigma) - market_price

    iv = brentq(difference, 0.0001, 3.0)

    return iv


if __name__ == "__main__":

    market_price = call_price(100, 100, 1, 0.05, 0.20)

    iv = implied_volatility(
        market_price, 100, 100, 1, 0.05
    )

    print("Implied Volatility:", round(iv * 100, 2), "%")