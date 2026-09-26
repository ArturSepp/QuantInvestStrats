"""The displayed benchmark betas and the beta attribution use one default estimator.

``PortfolioData.compute_portfolio_benchmark_betas`` and
``PortfolioData.compute_portfolio_benchmark_attribution`` describe the same beta; with default
arguments they must estimate it on the same grid ('B') with the same quarterly span (63 business
days at AN=252), so the betas a report plots are the betas its attribution applies.
"""

# packages
import inspect

import numpy as np
import pandas as pd

# qis
import qis


def _portfolio_and_benchmark() -> tuple:
    """A backtest with a benchmark that drives two of its three assets.

    Returns:
        The portfolio and a one-column benchmark price frame.
    """
    dates = pd.bdate_range('2021-01-01', periods=400)
    rng = np.random.default_rng(5)
    market = rng.normal(0.0002, 0.01, size=len(dates))
    returns = pd.DataFrame({'Asset A': 1.2 * market + rng.normal(0.0, 0.005, size=len(dates)),
                            'Asset B': 0.4 * market + rng.normal(0.0, 0.004, size=len(dates)),
                            'Asset C': rng.normal(0.0001, 0.006, size=len(dates))},
                           index=dates)
    returns.iloc[0, :] = 0.0
    prices = 100.0 * (1.0 + returns).cumprod()
    market[0] = 0.0
    benchmark = pd.DataFrame({'Market': 100.0 * np.cumprod(1.0 + market)}, index=dates)
    portfolio = qis.backtest_model_portfolio(prices=prices, weights=np.array([0.5, 0.3, 0.2]),
                                             rebalancing_freq='ME', ticker='beta defaults')
    return portfolio, benchmark


def test_default_beta_arguments_agree() -> None:
    """Both methods default to business-day returns and a 63-period span."""
    betas = inspect.signature(qis.PortfolioData.compute_portfolio_benchmark_betas).parameters
    attribution = inspect.signature(
        qis.PortfolioData.compute_portfolio_benchmark_attribution).parameters
    for name in ('freq_beta', 'factor_beta_span'):
        assert betas[name].default == attribution[name].default, name
    assert betas['freq_beta'].default == 'B'
    assert betas['factor_beta_span'].default == 63


def test_default_betas_are_the_betas_the_attribution_applies() -> None:
    """The default attribution equals the one rebuilt from the default displayed betas."""
    portfolio, benchmark = _portfolio_and_benchmark()
    betas = portfolio.compute_portfolio_benchmark_betas(benchmark_prices=benchmark)
    explicit = portfolio.compute_portfolio_benchmark_betas(benchmark_prices=benchmark,
                                                           freq_beta='B', factor_beta_span=63)
    pd.testing.assert_frame_equal(betas, explicit)

    attribution = portfolio.compute_portfolio_benchmark_attribution(benchmark_prices=benchmark)
    rebuilt = qis.compute_benchmarks_beta_attribution_from_prices(
        portfolio_nav=portfolio.get_portfolio_nav(), benchmark_prices=benchmark,
        portfolio_benchmark_betas=betas)
    pd.testing.assert_frame_equal(attribution, rebuilt)
