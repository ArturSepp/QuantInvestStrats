"""EwmLinearModel estimates are point in time, and portfolio betas keep genuine zeros.

The loadings dated t must not change when later returns are appended or altered, the fit must
leave the supplied return panels untouched, and a portfolio that holds nothing has beta zero
rather than the beta of the last invested date.
"""

import numpy as np
import pandas as pd
import pytest

import qis

FACTORS = ['F1', 'F2']
ASSETS = ['A1', 'A2']


def _panels(n: int = 120, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly factor and asset returns with non-zero means, so a mean seed matters."""
    rng = np.random.default_rng(seed)
    index = pd.date_range('2020-01-01', periods=n, freq='W-WED')
    x = pd.DataFrame(0.004 + rng.normal(0.0, [0.02, 0.01], size=(n, 2)), index=index,
                     columns=FACTORS)
    y = pd.DataFrame(0.002 + x.to_numpy() @ np.array([[1.0, 0.3], [0.2, 0.8]])
                     + rng.normal(0.0, 0.01, size=(n, 2)), index=index, columns=ASSETS)
    return x, y


@pytest.mark.parametrize('mean_adj_type', [qis.MeanAdjType.EWMA, qis.MeanAdjType.EXPANDING,
                                           qis.MeanAdjType.NONE])
def test_default_fit_does_not_use_later_returns(mean_adj_type: qis.MeanAdjType) -> None:
    """Loadings up to a cut-off are identical whatever happens after it."""
    x, y = _panels()
    cut = 80
    shocked_x, shocked_y = x.copy(), y.copy()
    shocked_x.iloc[cut:] += 0.05  # a large change in the later means
    shocked_y.iloc[cut:] -= 0.05

    full = qis.EwmLinearModel(x=x, y=y)
    full.fit(span=31, mean_adj_type=mean_adj_type)
    shocked = qis.EwmLinearModel(x=shocked_x, y=shocked_y)
    shocked.fit(span=31, mean_adj_type=mean_adj_type)

    for factor in FACTORS:
        pd.testing.assert_frame_equal(full.loadings[factor].iloc[:cut],
                                      shocked.loadings[factor].iloc[:cut])


def test_mean_seed_remains_available_and_is_forward_looking() -> None:
    """``InitType.MEAN`` is still accepted, and it is what used later data."""
    x, y = _panels()
    cut = 80
    shocked_x = x.copy()
    shocked_x.iloc[cut:] += 0.05

    full = qis.EwmLinearModel(x=x, y=y)
    full.fit(span=31, mean_adj_type=qis.MeanAdjType.EWMA, init_type=qis.InitType.MEAN)
    shocked = qis.EwmLinearModel(x=shocked_x, y=y)
    shocked.fit(span=31, mean_adj_type=qis.MeanAdjType.EWMA, init_type=qis.InitType.MEAN)

    difference = (full.loadings['F1'] - shocked.loadings['F1']).iloc[:cut].abs().max().max()
    assert difference > 1.0e-3


def test_fit_leaves_the_supplied_return_panels_unchanged() -> None:
    """Mean adjustment is for estimation only; ``x`` and ``y`` keep the supplied returns."""
    x, y = _panels()
    model = qis.EwmLinearModel(x=x, y=y)

    model.fit(span=31, mean_adj_type=qis.MeanAdjType.EWMA)

    pd.testing.assert_frame_equal(model.x, x)
    pd.testing.assert_frame_equal(model.y, y)
    alpha, explained = model.get_factor_alpha(lag=1)
    assert alpha.iloc[:22].isna().all().all()  # 21 warm-up rows plus the one-period lag
    pd.testing.assert_frame_equal((alpha + explained).iloc[22:], y.iloc[22:],
                                  check_exact=False, atol=1.0e-15)
    # The residual keeps the intercept: its mean is the asset drift not explained by the betas.
    assert (alpha.iloc[22:].mean() > 0.0).all()


def test_warm_up_masks_warmup_period_plus_one_rows() -> None:
    """``warmup_period=20`` leaves positions 0 to 20, which is 21 rows, missing."""
    x, y = _panels()
    model = qis.EwmLinearModel(x=x, y=y)

    model.fit(span=31, warmup_period=20)

    loadings = model.loadings['F1']
    assert loadings.iloc[:21].isna().all().all()
    assert loadings.iloc[21:].notna().all().all()


def test_portfolio_benchmark_beta_is_zero_when_the_portfolio_is_in_cash() -> None:
    """A genuine zero beta is kept instead of being replaced by the previous beta."""
    rng = np.random.default_rng(3)
    index = pd.bdate_range('2021-01-01', periods=400)
    bench_logs = rng.normal(0.0, 0.01, size=400)
    benchmark_prices = pd.DataFrame({'Bench': 100.0 * np.exp(np.cumsum(bench_logs))},
                                    index=index)
    instrument_logs = 1.2 * bench_logs[:, None] + rng.normal(0.0, 0.005, size=(400, 2))
    instrument_prices = pd.DataFrame(100.0 * np.exp(np.cumsum(instrument_logs, axis=0)),
                                     index=index, columns=ASSETS)
    weights = pd.DataFrame(0.5, index=index, columns=ASSETS)
    weights.iloc[300:] = 0.0  # fully in cash from day 300

    betas = qis.compute_portfolio_ewm_benchmark_betas(
        instrument_prices=instrument_prices, weights=weights, benchmark_prices=benchmark_prices,
        factor_beta_span=63)

    assert betas['Bench'].iloc[:21].isna().all()  # positions 0 to 20 are the warm-up
    assert betas['Bench'].iloc[21:].notna().all()
    assert (betas['Bench'].iloc[300:] == 0.0).all()
    assert betas['Bench'].iloc[100:300].between(1.0, 1.4).all()


def test_portfolio_benchmark_betas_use_weights_as_of_calendar_month_ends() -> None:
    """Monthly betas on calendar month-ends read the last business-day weights."""
    rng = np.random.default_rng(5)
    index = pd.bdate_range('2018-01-01', '2023-12-29')
    n = len(index)
    bench_logs = rng.normal(0.0, 0.01, size=n)
    benchmark_prices = pd.DataFrame({'Bench': 100.0 * np.exp(np.cumsum(bench_logs))},
                                    index=index)
    instrument_logs = np.column_stack([1.5 * bench_logs, 0.5 * bench_logs]) + rng.normal(
        0.0, 0.004, size=(n, 2))
    instrument_prices = pd.DataFrame(100.0 * np.exp(np.cumsum(instrument_logs, axis=0)),
                                     index=index, columns=ASSETS)
    in_odd_month = (index.month % 2 == 1).astype(float)
    weights = pd.DataFrame({'A1': in_odd_month, 'A2': 1.0 - in_odd_month}, index=index)

    betas = qis.compute_portfolio_ewm_benchmark_betas(
        instrument_prices=instrument_prices, weights=weights, benchmark_prices=benchmark_prices,
        freq_beta='ME', factor_beta_span=12)

    model = qis.EwmLinearModel(
        x=qis.to_returns(benchmark_prices, freq='ME', is_log_returns=True),
        y=qis.to_returns(instrument_prices, freq='ME', is_log_returns=True))
    model.fit(span=12, mean_adj_type=qis.MeanAdjType.EWMA, init_type=qis.InitType.X0)
    month_end_weights = weights.reindex(index=betas.index, method='ffill')
    expected = (model.loadings['Bench'] * month_end_weights).sum(axis=1, min_count=2)
    assert (~betas.index.isin(index)).any()  # some month-ends fall on a weekend
    pd.testing.assert_series_equal(betas['Bench'], expected, check_names=False)
