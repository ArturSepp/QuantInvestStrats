"""
Demonstrate ``qis.fit_multivariate_ols``: multivariate OLS that returns
predictions, fitted parameters, and a formatted equation string.

Run with:
    python -m qis.examples.models.multivariate_ols
"""
import numpy as np
import pandas as pd

from qis import fit_multivariate_ols


def run() -> None:
    rng = np.random.default_rng(42)
    n = 200

    # Synthetic data: house price as a linear function of size, bedrooms, age,
    # plus noise. Coefficients chosen to be recoverable.
    x = pd.DataFrame({
        'house_size': rng.normal(2000, 500, n),  # square feet
        'bedrooms': rng.integers(1, 6, n),
        'age': rng.integers(1, 50, n),
    })
    y = pd.Series(
        150 * x['house_size']
        + 10_000 * x['bedrooms']
        - 500 * x['age']
        + 300_000
        + rng.normal(0, 20_000, n),
        name='house_price',
    )

    # ── With intercept ─────────────────────────────────────────────────────
    prediction, params, label = fit_multivariate_ols(
        x=x, y=y,
        fit_intercept=True,
        verbose=True,        # prints the full statsmodels OLS summary
    )

    print(f"\nFitted equation: {label}")
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,.2f}")

    # Sanity check: in-sample R^2 from the returned predictions
    ss_res = np.sum((y - prediction) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"In-sample R^2: {r2:.3f}")
    print(f"In-sample RMSE: {np.sqrt(ss_res / n):,.0f}")

    # ── No intercept, custom formatting ────────────────────────────────────
    _, _, label_noint = fit_multivariate_ols(
        x=x, y=y,
        fit_intercept=False,
        verbose=False,
        beta_format='{0:+0.1f}',
        alpha_format='{0:+0.1f}',
    )
    print(f"\nFitted equation (no intercept): {label_noint}")


if __name__ == '__main__':
    run()
