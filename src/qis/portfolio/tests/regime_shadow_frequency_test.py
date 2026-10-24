"""Report sampling must not change the benchmark returns used for regime shading."""
import numpy as np
import pandas as pd
import pytest
import qis


@pytest.mark.parametrize("display_freq", ["B", "W-FRI", "ME", "QE"])
def test_regime_shadows_keep_benchmark_sampling(monkeypatch, display_freq):
    """A quarterly exposure chart still classifies actual monthly benchmark returns."""
    dates = pd.bdate_range("2018-01-01", periods=1300)
    rng = np.random.default_rng(29)
    prices = pd.Series(np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))),
                       index=dates, name="SPY")
    portfolio = qis.backtest_model_portfolio(prices.to_frame(), np.array([1.0]))
    multi = qis.MultiPortfolioData([portfolio], benchmark_prices=prices.to_frame())
    display = prices.loc["2019-01-01":"2022-06-30"].resample(display_freq).last().index
    expected = qis.to_returns(prices.loc[display.min():display.max()], freq="ME")
    observed = []

    def inspect_benchmark(*, ax, pivot_prices, regime_classifier):
        """Check the return input independently of the chart rendering implementation."""
        actual = qis.to_returns(pivot_prices, freq="ME")
        pd.testing.assert_series_equal(actual, expected)
        observed.append(actual)

    monkeypatch.setattr(qis, "add_bnb_regime_shadows", inspect_benchmark)
    multi.add_regime_shadows(ax=None, regime_benchmark="SPY", index=display)
    assert len(observed) == 1
