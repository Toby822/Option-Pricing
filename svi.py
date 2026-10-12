import numpy as np
from scipy.optimize import least_squares

# SVI mathematical model
def svi(k, a, b, rho, m, eta):
    return a + b * (
        rho * (k - m)
        + np.sqrt((k - m)**2 + eta**2)
    )

# Fit SVI to implied volatility data
def calibrate_svi(strikes, iv, T, forward):

    k = np.log(strikes / forward)
    market_variance = iv**2 * T

    def residuals(params):
        return svi(k, *params) - market_variance

    result = least_squares(
        residuals,
        x0=[0.02, 0.1, -0.3, 0.0, 0.1],
        bounds=(
            [0, 0, -0.999, -2, 0.001],
            [1, 5, 0.999, 2, 2]
        )
    )

    return result.x