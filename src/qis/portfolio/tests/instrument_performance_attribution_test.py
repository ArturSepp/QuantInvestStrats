"""Instrument attribution spends each period's gain on the portfolio's actual capital.

Independent currency accounting distinguishes portfolio contributions from standalone
instrument NAVs: a second 10% gain on capital 110 earns 11, not 10, on initial capital 100.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.perfstats.returns import to_total_returns


def _rebalanced_portfolio(second_price: float) -> qis.PortfolioData:
    """Return a fee-free book whose two gains or gain/loss belong to different holdings."""
    dates = pd.date_range("2025-01-31", periods=3, freq="ME", name="date")
    prices = pd.DataFrame(
        {"A": [100.0, 120.0, 120.0], "B": [100.0, 100.0, second_price]}, index=dates
    )
    targets = pd.DataFrame(0.5, index=dates[:2], columns=prices.columns)
    return qis.backtest_model_portfolio(
        prices,
        targets,
        initial_nav=100.0,
        management_fee=0.0,
        rebalancing_costs=0.0,
        ticker="Portfolio",
    )


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float64"])
@pytest.mark.parametrize("second_price,second_earning", [(120.0, 11.0), (80.0, -11.0)])
def test_get_instruments_performance_attribution_reconciles_currency_earnings(
    dtype: str,
    second_price: float,
    second_earning: float,
) -> None:
    """The full vector, not just its total, assigns the second holding's currency P&L."""
    portfolio = _rebalanced_portfolio(second_price)
    portfolio.instrument_pnl = portfolio.instrument_pnl.astype(dtype)
    expected = pd.Series([0.1, second_earning / 100.0], index=["A", "B"], name="Portfolio")

    actual = portfolio.get_instruments_performance_attribution()
    pd.testing.assert_series_equal(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(portfolio.nav, [100.0, 110.0, 110.0 + second_earning])
    assert actual.sum() == pytest.approx(portfolio.nav.iloc[-1] / 100.0 - 1.0)
    pd.testing.assert_series_equal(
        portfolio.get_performance_attribution_data(qis.AttributionMetric.PNL), actual
    )

    # Standalone instrument paths intentionally do not include another holding's gains.
    standalone = portfolio.get_instruments_navs()
    np.testing.assert_allclose(standalone.iloc[-1], [1.1, 1.0 + second_earning / 110.0], rtol=1e-6)
    pd.testing.assert_frame_equal(
        portfolio.get_performance_attribution_data(qis.AttributionMetric.INST_PNL), standalone
    )


@pytest.mark.parametrize("columns", [["A"], ["A", "Inactive"]])
def test_get_instruments_performance_attribution_preserves_same_holding_growth(columns) -> None:
    """One holding earning both returns still contributes 21% of initial capital."""
    dates = pd.date_range("2025-01-31", periods=3, freq="ME")
    pnl = pd.DataFrame(0.0, index=dates, columns=columns)
    pnl["A"] = [0.0, 0.1, 0.1]
    portfolio = qis.PortfolioData(
        nav=pd.Series([100.0, 110.0, 121.0], index=dates, name="Portfolio"), instrument_pnl=pnl
    )
    expected = pd.Series([0.21] + [0.0] * (len(columns) - 1), index=columns, name="Portfolio")
    pd.testing.assert_series_equal(portfolio.get_instruments_performance_attribution(), expected)


@pytest.mark.parametrize("arithmetic", [False, True])
@pytest.mark.parametrize("second_price,window_return", [(120.0, 0.1), (80.0, -0.1)])
def test_get_instruments_performance_attribution_resets_selected_baseline(
    arithmetic: bool,
    second_price: float,
    window_return: float,
) -> None:
    """An off-grid start uses the first selected NAV, not its preceding holding-period gain."""
    portfolio = _rebalanced_portfolio(second_price)
    period = qis.TimePeriod("2025-02-01", "2025-03-31")
    expected = pd.Series([0.0, window_return], index=["A", "B"], name="Portfolio")
    pd.testing.assert_series_equal(
        portfolio.get_instruments_performance_attribution(period, arithmetic), expected
    )
    baseline_only = qis.TimePeriod("2025-03-31", "2025-03-31")
    expected[:] = np.nan
    pd.testing.assert_series_equal(
        portfolio.get_instruments_performance_attribution(baseline_only, arithmetic), expected
    )


def test_get_instruments_performance_attribution_preserves_arithmetic_normalization() -> None:
    """Legacy arithmetic NAVs still divide by their initial level when the first P&L is nonzero."""
    portfolio = _rebalanced_portfolio(120.0)
    portfolio.instrument_pnl.iloc[0] = [0.1, 0.0]
    expected = pd.Series([0.1 / 1.1, 0.1], index=["A", "B"], name="Portfolio")
    pd.testing.assert_series_equal(
        portfolio.get_instruments_performance_attribution(constant_trade_level=True), expected
    )
    # The compounded report also excludes the P&L attached to its baseline observation.
    expected[:] = [0.1, 0.11]
    pd.testing.assert_series_equal(portfolio.get_instruments_performance_attribution(), expected)


@pytest.mark.parametrize("multi_index", [False, True])
@pytest.mark.parametrize("dtype,missing", [("float64", np.nan), ("Float64", pd.NA)])
@pytest.mark.parametrize(
    "start,contributions",
    [
        ("2025-01-31", [0.1, -0.11526, 0.0226, 0.03, 0.0, 0.0]),
        ("2025-02-01", [0.0, -0.102, 0.02, 0.0, 0.0, 0.0]),
    ],
)
def test_get_instruments_performance_attribution_preserves_mixed_panel_and_inputs(
    dtype: str,
    missing: object,
    start: str,
    contributions: list[float],
    multi_index: bool,
) -> None:
    """Legacy zero-filled gaps stay local; aliases and result mutation cannot alter stored P&L."""
    dates = pd.date_range("2025-01-31", periods=4, freq="ME", name="date")
    pnl = pd.DataFrame(
        {
            "A": [0.0, 0.1, missing, 0.0],
            "B": [0.0, 0.0, missing, -0.1],
            "Ragged": [missing, missing, 0.02, 0.0],
            "Ended": [0.0, 0.03, missing, missing],
            "Unavailable": [missing] * 4,
            "Inactive": [0.0] * 4,
        },
        index=dates,
        dtype=dtype,
    )
    pnl.columns.name = "instrument"
    if multi_index:
        pnl.columns = pd.MultiIndex.from_product(
            [["Assets"], pnl.columns], names=["group", "instrument"]
        )
    portfolio = qis.PortfolioData(
        nav=pd.Series([100.0, 113.0, 115.26, 103.734], index=dates, name="Portfolio"),
        instrument_pnl=pnl,
        tickers_to_names_map={"B": "Second holding"},
    )
    before = portfolio.instrument_pnl.copy(deep=True)
    # Full-history capital is 113 then 115.26. A later report resets at February's NAV:
    # its 2% gain then 10% loss contribute .02 and -.102 of the new baseline.
    # The native summary's unnamed axis keeps reset_index() output columns unchanged.
    expected_index = pd.Index(["A", "Second holding", "Ragged", "Ended", "Unavailable", "Inactive"])
    if multi_index:
        expected_index = pd.MultiIndex.from_product([["Assets"], expected_index])
    expected = pd.Series(
        contributions,
        index=expected_index,
        name="Portfolio",
    )
    actual = portfolio.get_instruments_performance_attribution(qis.TimePeriod(start, dates[-1]))
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(portfolio.instrument_pnl, before)
    actual.iloc[0] = 999.0
    actual.index.names = ["changed result"] * actual.index.nlevels
    pd.testing.assert_frame_equal(portfolio.instrument_pnl, before)


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(pd.Index(["A", "B"], name="instrument"), id="strings"),
        pytest.param(pd.Index(["A", "B"], dtype=object, name="instrument"), id="object-strings"),
        pytest.param(
            pd.CategoricalIndex(
                ["A", "B"], categories=["B", "A", "unused"], ordered=True, name="instrument"
            ),
            id="ordered-categorical",
        ),
        pytest.param(
            pd.date_range("2024-01-31", periods=2, freq="ME", name="instrument"),
            id="dated",
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [("Equity", "A"), ("Equity", "B")], names=["asset_class", "instrument"]
            ),
            id="multi-index",
        ),
    ],
)
def test_get_instruments_performance_attribution_preserves_native_result_index(columns) -> None:
    """Native conversion, not the input axis, defines the summary's index representation."""
    dates = pd.date_range("2025-01-31", periods=3, freq="ME")
    pnl = pd.DataFrame([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], index=dates, columns=columns)
    portfolio = qis.PortfolioData(
        nav=pd.Series([100.0, 110.0, 121.0], index=dates, name="Portfolio"), instrument_pnl=pnl
    )
    # Standalone NAVs retain the accepted axis conversion but are not a financial-value oracle:
    # the second holding earns 11 on capital 110, rather than 10 on the initial capital 100.
    native_index = to_total_returns(portfolio.get_instruments_navs()).index
    expected = pd.Series([0.1, 0.11], index=native_index, name="Portfolio")
    actual = portfolio.get_instruments_performance_attribution()
    pd.testing.assert_series_equal(actual, expected, check_index_type=True, check_freq=True)


@pytest.mark.parametrize("missing_position", [0, 1, 2])
def test_get_instruments_performance_attribution_preserves_missing_date_warning(
    missing_position: int,
) -> None:
    """An undated contribution must not become a finite, silently dated performance claim."""
    dates = list(pd.date_range("2025-01-01", periods=3))
    dates[missing_position] = pd.NaT
    dates = pd.DatetimeIndex(dates)
    pnl = pd.DataFrame([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], index=dates, columns=["A", "B"])
    portfolio = qis.PortfolioData(
        nav=pd.Series([100.0, 110.0, 121.0], index=dates, name="Portfolio"), instrument_pnl=pnl
    )
    before = pnl.copy(deep=True)
    expected = pd.Series([np.nan, np.nan], index=["A", "B"], name="Portfolio")
    with pytest.warns(UserWarning, match="total return has inconsistent dates"):
        actual = portfolio.get_instruments_performance_attribution()
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(portfolio.instrument_pnl, before)


@pytest.mark.parametrize(
    "dates,contributions",
    [
        (["2025-01-03", "2025-01-02", "2025-01-01"], [-1.0 / 11.0, -1.0 / 11.0]),
        (["2025-01-01", "2025-01-03", "2025-01-02"], [0.1, 0.0]),
        (["2025-01-01", "2025-01-03", "2025-01-03"], [0.1, 0.1]),
    ],
    ids=["descending", "shuffled", "duplicate-terminal"],
)
def test_get_instruments_performance_attribution_preserves_native_irregular_dates(
    dates: list[str], contributions: list[float]
) -> None:
    """Retain native endpoint behavior without inventing a chronology for irregular histories."""
    dates = pd.DatetimeIndex(dates)
    pnl = pd.DataFrame([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], index=dates, columns=["A", "B"])
    portfolio = qis.PortfolioData(
        nav=pd.Series([100.0, 110.0, 121.0], index=dates, name="Portfolio"), instrument_pnl=pnl
    )
    # Native NAVs compound in physical order before selecting dated endpoints. For descending
    # rows this divides 1 by 1.1; sorting contributions first would select a different contract.
    expected = pd.Series(contributions, index=["A", "B"], name="Portfolio")
    native = to_total_returns(portfolio.get_instruments_navs()).rename("Portfolio")
    pd.testing.assert_series_equal(native, expected)
    pd.testing.assert_series_equal(portfolio.get_instruments_performance_attribution(), expected)


@pytest.mark.parametrize(
    "dates,error",
    [
        (pd.DatetimeIndex(["2025-01-01", "2025-01-01", "2025-01-03"]), TypeError),
        (pd.DatetimeIndex([pd.NaT] * 3), ValueError),
        (pd.RangeIndex(3), AttributeError),
    ],
    ids=["duplicate-initial", "all-undated", "non-date"],
)
def test_get_instruments_performance_attribution_preserves_native_date_rejection(
    dates: pd.Index, error: type[Exception]
) -> None:
    """The accounting correction must not silently admit histories the native path rejects."""
    pnl = pd.DataFrame([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], index=dates, columns=["A", "B"])
    portfolio = qis.PortfolioData(
        nav=pd.Series([100.0, 110.0, 121.0], index=dates, name="Portfolio"), instrument_pnl=pnl
    )
    with pytest.raises(error):
        to_total_returns(portfolio.get_instruments_navs())
    with pytest.raises(error):
        portfolio.get_instruments_performance_attribution()


@pytest.mark.parametrize(
    "dates",
    [
        pd.date_range("2025-01-01", periods=3, tz="UTC"),
        pd.date_range("2025-03-08", periods=3, tz="America/New_York"),
        pd.date_range("2025-01-01", periods=3, freq="h"),
    ],
    ids=["UTC", "daylight-saving", "intraday"],
)
def test_get_instruments_performance_attribution_compounds_valid_date_variants(dates) -> None:
    """A date-compatibility fallback must not disable compounding for valid nearby histories."""
    pnl = pd.DataFrame([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], index=dates, columns=["A", "B"])
    portfolio = qis.PortfolioData(
        nav=pd.Series([100.0, 110.0, 121.0], index=dates, name="Portfolio"), instrument_pnl=pnl
    )
    expected = pd.Series([0.1, 0.11], index=["A", "B"], name="Portfolio")
    pd.testing.assert_series_equal(portfolio.get_instruments_performance_attribution(), expected)


def test_get_instruments_performance_attribution_retains_supplied_gross_basis() -> None:
    """Unattributed costs or fees in NAV must not be silently charged to an instrument."""
    portfolio = _rebalanced_portfolio(120.0)
    # Drag before the second return distinguishes supplied-P&L wealth from net NAV wealth.
    portfolio.nav.iloc[1:] = [109.0, 119.0]
    portfolio.realized_costs.iloc[1] = [1.0, 0.0]
    portfolio.realized_costs.iloc[-1] = [1.0, 1.0]
    expected = pd.Series([0.1, 0.11], index=["A", "B"], name="Portfolio")
    pd.testing.assert_series_equal(portfolio.get_instruments_performance_attribution(), expected)
    assert expected.sum() != pytest.approx(portfolio.nav.iloc[-1] / portfolio.nav.iloc[0] - 1.0)


@pytest.mark.parametrize("second_price,second_contribution", [(120.0, 0.11), (80.0, -0.11)])
def test_plot_performance_attribution_renders_currency_contributions(
    second_price: float,
    second_contribution: float,
) -> None:
    """The factsheet's public PNL panel must display the independently derived vector."""
    portfolio = _rebalanced_portfolio(second_price)
    portfolio.tickers_to_names_map = {"A": "First holding", "B": "Second holding"}
    fig, ax = plt.subplots(figsize=(6, 3))
    try:
        portfolio.plot_performance_attribution(
            attribution_metric=qis.AttributionMetric.PNL,
            ax=ax,
            max_bars=0,
            shorten_instrument_names=False,
            legend_loc=None,
        )
        fig.canvas.draw()
        np.testing.assert_allclose(
            [bar.get_height() for bar in ax.patches], [0.1, second_contribution], atol=1e-14
        )
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "First holding",
            "Second holding",
        ]
    finally:
        plt.close(fig)
