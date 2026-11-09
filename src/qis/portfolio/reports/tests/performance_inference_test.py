"""Appendix sample rules, independent return/cash checks and summary-page preservation."""

import warnings
from io import BytesIO, StringIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.sharpe_inference import (
    SharpeInference,
    sharpe_ratio_test,
)
from qis.portfolio.reports._performance_inference import (
    compute_performance_inference,
    plot_performance_inference,
)


@pytest.fixture(scope="module")
def prices():
    return generate_synthetic_prices(end="2025-12-31", apply_quirks=False)[["SEQ_US", "SBD_TSY"]]


@pytest.fixture(autouse=True)
def close_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


def compute(prices, start="2005-01-01", rates=None, freq="ME"):
    return compute_performance_inference(
        prices.SEQ_US,
        prices[["SBD_TSY"]],
        qis.TimePeriod(start, prices.index[-1]),
        freq=freq,
        rates_data=rates,
        instrument_prices=prices,
        weights=pd.DataFrame({"SEQ_US": 1.0}, index=prices.index),
        factor_beta_span=12,
    )


def test_tables_match_independent_price_and_cash_reference(prices):
    rates = pd.Series(0.03, index=prices.index)
    data = compute(prices, rates=rates)
    # Direct endpoint arithmetic and lagged ACT/365 cash, not report sampling helpers.
    levels = prices.resample("ME").last()
    total = levels.divide(levels.shift(1)).subtract(1)
    accrual = 0.03 * levels.index.to_series().diff().dt.days / 365
    expected = total.subtract(accrual, axis=0).iloc[1:]
    np.testing.assert_allclose(data.returns.Strategy, expected.SEQ_US, rtol=1e-12, atol=1e-14)
    row = data.windows.loc["Full history"]
    theta = expected.SEQ_US.mean() / expected.SEQ_US.std(ddof=0)
    assert row.sharpe == pytest.approx(np.sqrt(12) * theta)
    iid = sharpe_ratio_test(expected.SEQ_US, method=SharpeInference.IID_NORMAL, af=12)
    np.testing.assert_allclose([row.iid_lower, row.iid_upper], [iid.ci_lower, iid.ci_upper])
    # Different-way HAC reference: explicitly sum Bartlett-weighted score cross-products.
    values = expected.SEQ_US.to_numpy()
    standardized = (values - values.mean()) / values.std(ddof=0)
    score = standardized - 0.5 * theta * (standardized**2 - 1)
    score -= score.mean()
    lag = int(row.hac_lags)
    cross_products = score @ score
    for offset in range(1, lag + 1):
        cross_products += 2 * (1 - offset / (lag + 1)) * (score[offset:] @ score[:-offset])
    reference_se = np.sqrt(12 * cross_products) / len(score)
    assert row.hac_se == pytest.approx(reference_se)
    np.testing.assert_allclose(
        [row.hac_lower, row.hac_upper],
        row.sharpe + np.array([-1, 1]) * 1.959963984540054 * reference_se,
    )
    assert data.metadata["return_basis"] == "excess of supplied cash"
    # Raw realised payoff identity; beta is the prior-date, holdings-weighted exposure.
    reference_alpha = expected.SEQ_US - data.alpha_diagnostics.lagged_beta * expected.SBD_TSY
    np.testing.assert_allclose(data.alpha_returns, reference_alpha, equal_nan=True)
    alpha = data.alpha_returns.dropna()
    row = data.alpha_windows.loc["Full history"]
    centered = alpha.to_numpy() - alpha.mean()
    lag = int(row.hac_lags)
    kernel = np.maximum(1 - np.abs(np.subtract.outer(np.arange(len(alpha)),
                                                  np.arange(len(alpha)))) / (lag + 1), 0)
    # Different-way dense Bartlett kernel and one-parameter T/(T-1) correction.
    se = np.sqrt(centered @ kernel @ centered / (len(alpha) * (len(alpha) - 1)))
    assert row.alpha == pytest.approx(12 * alpha.mean())
    assert row.hac_se == pytest.approx(12 * se)
    np.testing.assert_allclose([row.hac_lower, row.hac_upper],
                               12 * (alpha.mean() + norm.isf(0.025) * se * np.array([-1, 1])))
    assert row.pvalue == pytest.approx(2 * norm.sf(abs(alpha.mean() / se)))


@pytest.mark.parametrize(("years", "fallback"), [(4, 3), (2, 2), (1, None)])
def test_rolling_fallback_warns_and_insufficient_stays_unavailable(prices, years, fallback):
    # Include the opening month-end so exactly years*12 realised returns exist.
    monthly = prices.resample("ME").last()
    short = monthly.loc[monthly.index >= monthly.index[-1] - pd.DateOffset(years=years)]
    with pytest.warns(UserWarning, match="Rolling chart"):
        data = compute(short, start=short.index[0])
    assert data.metadata["rolling_years"] == fallback
    if years == 1:
        assert data.windows.hac_lower.isna().all()
        assert data.expanding.empty and data.rolling.empty
        figure = plot_performance_inference(data, "Short sample")
        figure.canvas.draw()
        assert any("Insufficient" in text.get_text() for ax in figure.axes for text in ax.texts)
    else:
        assert not data.rolling.empty
        first = data.rolling.date.iloc[0]
        cutoff = first - pd.DateOffset(years=fallback)
        expected = data.returns.Strategy.loc[
            (data.returns.index > cutoff) & (data.returns.index <= first)
        ]
        assert len(expected) == fallback * 12


def test_interior_gap_not_filled_or_condensed(prices):
    levels = prices.resample("ME").last()
    levels.loc["2015-06-30", "SEQ_US"] = np.nan
    with pytest.warns(UserWarning, match="interior missing"):
        data = compute(levels)
    assert data.returns.loc["2015-06-30":"2015-07-31", "Strategy"].isna().all()
    assert pd.isna(data.windows.loc["Full history", "sharpe"])
    assert np.isfinite(data.windows.loc["Recent 3 years", "sharpe"])
    # A completely absent month is also missing, not a manufactured zero return.
    missing = prices.resample("ME").last().drop(index=pd.Timestamp("2015-06-30"))
    with pytest.warns(UserWarning, match="interior missing"):
        absent = compute(missing)
    assert absent.returns.loc["2015-06-30":"2015-07-31", "Strategy"].isna().all()


def test_late_reference_does_not_shorten_strategy(prices):
    reference = prices[["SBD_TSY"]].copy()
    reference.loc[:"2018-12-31"] = np.nan
    data = compute_performance_inference(
        prices.SEQ_US, reference, qis.TimePeriod("2005-01-01", "2025-12-31"), "ME",
        instrument_prices=prices, weights=pd.DataFrame({"SEQ_US": 1.0}, index=prices.index),
        factor_beta_span=12,
    )
    assert data.windows.loc["Full history", "n_obs"] > data.alpha_windows.iloc[0].n_obs
    assert data.alpha_windows.iloc[0].start > pd.Timestamp("2020-01-31")


def test_undefined_constant_returns_warn_without_fabricated_interval(prices):
    constant = prices.copy()
    constant.SEQ_US = 100.0
    with pytest.warns(UserWarning, match="undefined Sharpe"):
        data = compute(constant)
    assert data.windows.sharpe.isna().all()
    assert data.alpha_windows.alpha.isna().all()


@pytest.mark.parametrize("prewhitened", [False, True])
def test_integration_default_order_and_identical_first_page(prices, prewhitened):
    import inspect
    from qis.perfstats.significance import InterceptInference

    assert (
        inspect.signature(qis.generate_strategy_factsheet)
        .parameters["add_performance_inference_sheet"]
        .default
        is True
    )
    portfolio = qis.backtest_model_portfolio(prices, weights=np.array([0.6, 0.4]), ticker="Test")
    before = portfolio.nav.copy()
    settings = qis.fetch_default_report_kwargs(
        qis.TimePeriod("2010-01-01", "2025-12-31"), add_rates_data=False
    )
    options = dict(
        alpha_inference=InterceptInference.PREWHITENED_AR1 if prewhitened else
        InterceptInference.BARTLETT_HAC,
        benchmark_prices=prices[["SBD_TSY"]],
        time_period=qis.TimePeriod("2010-01-01", "2025-12-31"),
        **settings,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = qis.generate_strategy_factsheet(
            portfolio, add_performance_inference_sheet=False, **options
        )
        new = qis.generate_strategy_factsheet(portfolio, **options)
    assert len(old) == 2 and len(new) == 3
    assert new[1].axes[0].get_title() == "Monthly Returns - Full History"
    assert "Performance confidence and stability" in new[2].get_suptitle()
    primary = "PW" if prewhitened else "B"
    assert any(f"Primary: {primary} |" in text.get_text() for text in new[2].texts)
    old[0].canvas.draw()
    new[0].canvas.draw()
    np.testing.assert_array_equal(
        np.asarray(old[0].canvas.buffer_rgba()), np.asarray(new[0].canvas.buffer_rgba())
    )
    pd.testing.assert_series_equal(portfolio.nav, before)
    new[2].canvas.draw()
    renderer = new[2].canvas.get_renderer()
    # Long benchmark sample labels must not be clipped outside the page's left margin.
    for label in new[2].axes[-1].get_yticklabels():
        bounds = label.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= new[2].bbox.width
    # The appendix also follows the summary directly when no monthly appendix is needed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recent = qis.generate_strategy_factsheet(
            portfolio,
            benchmark_prices=prices[["SBD_TSY"]],
            time_period=qis.TimePeriod("2021-01-01", "2025-12-31"),
            **settings,
        )
    assert len(recent) == 2 and "Performance confidence" in recent[1].get_suptitle()


def test_documentation_footnotes_are_linked_and_fit_page(prices):
    figure = plot_performance_inference(compute(prices), "Documentation links")
    chapter = "https://quantinveststrats.readthedocs.io/en/latest/"
    expected = {
        "qis docs: Sharpe CI (IID/HAC)": (
            chapter + "performance_analytics_and_sharpe.html#sampling-uncertainty"
        ),
        "qis docs: Holdings-weighted alpha": (
            chapter + "model_layer_attribution.html#"
            "holdings-exposure-versus-model-layer-nav-regression"),
        "qis docs: HAC limitations": chapter + "factor_and_backtest_significance.html",
    }
    links = {text.get_text(): text for text in figure.texts if text.get_url()}
    assert set(links) == set(expected)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for label, url in expected.items():
        assert links[label].get_url() == url
        bounds = links[label].get_window_extent(renderer)
        assert 0 <= bounds.x0 < bounds.x1 <= figure.bbox.width
        assert 0 <= bounds.y0 < bounds.y1 < 0.055 * figure.bbox.height
    assert not links[next(iter(expected))].get_window_extent(renderer).overlaps(
        links[list(expected)[1]].get_window_extent(renderer)
    )
    # Vector output retains the URLs; raster previews can only display their labels.
    svg = StringIO()
    figure.savefig(svg, format="svg")
    assert all(f'xlink:href="{url}"' in svg.getvalue() for url in expected.values())
    pdf = BytesIO()
    figure.savefig(pdf, format="pdf")
    assert all(f"/URI ({url})".encode() in pdf.getvalue() for url in expected.values())


def test_alpha_uses_only_first_benchmark_and_its_own_frequency(prices):
    options = dict(nav=prices.SEQ_US, instrument_prices=prices,
                   weights=pd.DataFrame({"SEQ_US": 1.0}, index=prices.index),
                   time_period=qis.TimePeriod("2010-01-01", "2025-12-31"),
                   freq="ME", freq_beta="QE", factor_beta_span=12)
    first = compute_performance_inference(benchmark_prices=prices[["SBD_TSY"]], **options)
    both = compute_performance_inference(benchmark_prices=prices[["SBD_TSY", "SEQ_US"]], **options)
    pd.testing.assert_series_equal(first.alpha_returns, both.alpha_returns)
    pd.testing.assert_frame_equal(first.alpha_windows, both.alpha_windows)
    assert both.metadata["alpha"]["benchmark"] == "SBD_TSY"
    assert both.metadata["af"] == 12 and both.metadata["alpha"]["af"] == 4
    assert both.alpha_returns.index[0] > pd.Timestamp("2010-01-01")
    assert both.alpha_windows.loc["Full history", "alpha"] == pytest.approx(
        4 * both.alpha_returns.mean())
    # Quarterly 3/5-year windows have <24 returns: no annualisation-based bypass.
    assert both.alpha_windows.loc["Recent 5 years", "n_obs"] == 20
    assert pd.isna(both.alpha_windows.loc["Recent 5 years", "alpha"])


def test_alpha_missing_holdings_and_interior_coverage_do_not_damage_sharpe(prices):
    period = qis.TimePeriod("2010-01-01", "2025-12-31")
    with pytest.warns(UserWarning, match="holdings unavailable"):
        missing = compute_performance_inference(prices.SEQ_US, prices[["SBD_TSY"]], period, "ME")
    assert missing.alpha_windows.alpha.isna().all()
    assert missing.windows.sharpe.notna().all()
    weights = pd.DataFrame({"SEQ_US": 1.0}, index=prices.index)
    weights.loc["2015-06", "SEQ_US"] = np.nan
    with pytest.warns(UserWarning, match="latest continuous"):
        gapped = compute_performance_inference(
            prices.SEQ_US, prices[["SBD_TSY"]], period, "ME", instrument_prices=prices,
            weights=weights, factor_beta_span=12)
    assert pd.notna(gapped.alpha_windows.loc["Full history", "alpha"])
    assert gapped.alpha_windows.iloc[0].start == pd.Timestamp("2015-08-31")
    assert pd.isna(gapped.alpha_returns.loc["2015-07-31"])
    assert pd.notna(gapped.alpha_windows.loc["Recent 3 years", "alpha"])
    pd.testing.assert_frame_equal(gapped.windows, missing.windows)


def test_two_columns_and_alpha_curve_ends_match_raw_samples(prices):
    data = compute(prices)
    assert data.alpha_returns.isna().any()  # warm-up is not zero-return evidence
    assert data.alpha_windows.iloc[0].start == data.alpha_returns.first_valid_index()
    assert data.alpha_expanding.iloc[-1].estimate == pytest.approx(data.alpha_windows.iloc[0].alpha)
    last = data.alpha_rolling.iloc[-1]
    sample = data.alpha_returns.loc[data.alpha_returns.index > last.date - pd.DateOffset(years=5)]
    assert last.estimate == pytest.approx(12 * sample.mean())
    figure = plot_performance_inference(data, "Two-column test")
    assert len(figure.axes) == 6
    assert any(text.get_text() == "Alpha diagnosis" for text in figure.texts)
    # Col-major axes: table/expanding/rolling, then the same three alpha panels.
    assert figure.axes[0].get_position().x1 < figure.axes[3].get_position().x0
    assert figure.axes[1].get_ylabel() == "Annualised Sharpe"
    assert figure.axes[4].get_ylabel() == "Annualised mean alpha"


def test_single_endpoint_shows_interval_not_only_point(prices):
    data = compute(prices)
    data.expanding = data.expanding.iloc[-1:]
    data.alpha_expanding = data.alpha_expanding.iloc[-1:]
    figure = plot_performance_inference(data, "One endpoint")
    for axis in (figure.axes[1], figure.axes[4]):
        intervals = [item for item in axis.containers if isinstance(item, ErrorbarContainer)]
        assert len(intervals) == 1 and intervals[0].has_yerr


def test_alpha_unavailable_grid_warns_without_breaking_sharpe(prices):
    monthly = prices.resample("ME").last()
    with pytest.warns(UserWarning, match="Alpha: sampling grid unavailable"):
        data = compute_performance_inference(
            monthly.SEQ_US, monthly[["SBD_TSY"]],
            qis.TimePeriod("2010-01-01", "2025-12-31"), "ME", freq_beta="W-WED",
            instrument_prices=monthly, weights=pd.DataFrame({"SEQ_US": 1.0}, index=monthly.index))
    assert data.windows.sharpe.notna().all()
    assert data.alpha_returns.empty and data.alpha_windows.alpha.isna().all()


@pytest.mark.parametrize("prewhitened", [False, True])
def test_latest_alpha_block_counts_exclusions_and_matches_dense_reference(prices, prewhitened):
    from qis.perfstats.significance import InterceptInference
    from qis.portfolio.reports._performance_inference import _inference_views
    from qis.portfolio.reports.tests.alpha_inference_options_test import dense_reference

    sample = prices.SEQ_US.resample("ME").last().pct_change().iloc[1:]
    sample.iloc[:3] = np.nan
    sample.iloc[70:80] = np.nan
    sample.iloc[140:150] = np.nan
    sample.iloc[-2:] = np.nan
    original = sample.copy()
    expected = sample.iloc[150:-2]
    issues = []
    method = (InterceptInference.PREWHITENED_AR1 if prewhitened else
              InterceptInference.BARTLETT_HAC)
    windows, expanding, rolling, meta = _inference_views(
        sample, "ME", sample.index[-1], issues, alpha=True, alpha_inference=method)
    row = windows.iloc[0]
    assert row.start == expected.index[0] and row.end == expected.index[-1]
    assert row.n_obs == row.n_periods == len(expected)
    assert meta["sample_fallback"] is True
    assert meta["report_n_periods"] == len(sample)
    assert meta["available_n_obs"] == int(sample.notna().sum())
    assert meta["unavailable_n_obs"] == 25
    assert meta["discarded_valid_n_obs"] == int(sample.iloc[:150].notna().sum())
    assert meta["excluded_ranges"] == [
        dict(start=sample.index[0], end=sample.index[149], n_periods=150),
        dict(start=sample.index[-2], end=sample.index[-1], n_periods=2),
    ]
    assert sum(item["n_periods"] for item in meta["unavailable_ranges"]) == 25
    assert any("latest continuous" in item and "excluded" in item for item in issues)
    se, _ = dense_reference(expected.to_numpy(), int(row.hac_lags), prewhitened)
    assert row.alpha == pytest.approx(expected.mean() * 12)
    assert row.hac_se == pytest.approx(se)
    assert row.comparison_estimate == pytest.approx(row.alpha)
    assert expanding.iloc[-1].estimate == pytest.approx(row.alpha)
    # No invented recent data when the report ends after the final eligible return.
    assert rolling.empty and meta["rolling_years"] is None
    pd.testing.assert_series_equal(sample, original)


def test_recent_gaps_keep_true_counts_and_short_latest_block_is_not_replaced(prices):
    from qis.portfolio.reports._performance_inference import _inference_views

    sample = prices.SEQ_US.resample("ME").last().pct_change().iloc[1:]
    sample.iloc[-13] = np.nan
    windows, expanding, rolling, meta = _inference_views(
        sample, "ME", sample.index[-1], [], alpha=True)
    assert windows.iloc[0].n_obs == 12  # Not the earlier, longer block or concatenated blocks.
    assert windows.alpha.isna().all()
    assert windows.loc["Recent 5 years", "n_obs"] == 59
    assert windows.loc["Recent 5 years", "n_periods"] == 60
    assert windows.loc["Recent 3 years", "n_obs"] == 35
    assert expanding.empty and rolling.empty
    assert meta["sample_fallback"] is True


def test_absent_alpha_calendar_date_is_a_gap_and_all_missing_counts_zero(prices):
    from qis.portfolio.reports._performance_inference import _inference_views

    sample = prices.SEQ_US.resample("ME").last().pct_change().iloc[1:]
    missing_date = sample.index[100]
    windows, _, _, meta = _inference_views(
        sample.drop(missing_date), "ME", sample.index[-1], [], alpha=True)
    assert windows.iloc[0].start == sample.index[101]
    assert meta["unavailable_ranges"] == [
        dict(start=missing_date, end=missing_date, n_periods=1)]
    windows, expanding, rolling, meta = _inference_views(
        sample * np.nan, "ME", sample.index[-1], [], alpha=True)
    assert (windows.n_obs == 0).all() and windows.alpha.isna().all()
    assert expanding.empty and rolling.empty
    assert meta["unavailable_n_obs"] == len(sample)


@pytest.mark.parametrize("trailing_gap", [False, True])
def test_fallback_page_labels_excluded_dates_and_valid_n_without_clipping(prices, trailing_gap):
    weights = pd.DataFrame({"SEQ_US": 1.0}, index=prices.index)
    weights.loc["2015-06", "SEQ_US"] = np.nan
    if trailing_gap:
        weights.loc["2025-11":, "SEQ_US"] = np.nan
    with pytest.warns(UserWarning, match="latest continuous"):
        data = compute_performance_inference(
            prices.SEQ_US, prices[["SBD_TSY"]], qis.TimePeriod("2010", "2025-12-31"),
            "ME", instrument_prices=prices, weights=weights, factor_beta_span=12)
    figure = plot_performance_inference(data, "Coverage fallback")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    table = figure.axes[3].tables[0]
    assert "Latest continuous" in table[1, 0].get_text().get_text()
    assert "N=" in table[1, 0].get_text().get_text()
    for cell in table.get_celld().values():
        box = cell.get_window_extent(renderer)
        text = cell.get_text().get_window_extent(renderer)
        assert box.x0 <= text.x0 < text.x1 <= box.x1
        assert box.y0 <= text.y0 < text.y1 <= box.y1
    note = next(t for t in figure.axes[3].texts if "Excluded" in t.get_text())
    assert "28Feb2010-31Jul2015" in note.get_text()
    assert "unavailable" in note.get_text() and "earlier valid" in note.get_text()
    box = note.get_window_extent(renderer)
    assert box.x0 >= 0 and box.x1 <= figure.bbox.width
    assert box.y0 > figure.axes[4].get_tightbbox(renderer).y1
    assert box.y1 < table.get_window_extent(renderer).y0


def test_benchmark_prehistory_warms_betas_without_extending_measurement(prices):
    options = dict(nav=prices.SEQ_US, time_period=qis.TimePeriod("2010", "2025-12-31"),
                   freq="ME", instrument_prices=prices,
                   weights=pd.DataFrame({"SEQ_US": 1.0}, index=prices.index),
                   factor_beta_span=12)
    full = compute_performance_inference(benchmark_prices=prices[["SBD_TSY"]], **options)
    cropped = compute_performance_inference(
        benchmark_prices=prices.loc["2010":, ["SBD_TSY"]], **options)
    assert full.alpha_diagnostics.complete_coverage.iloc[0]
    assert full.alpha_returns.first_valid_index() == pd.Timestamp("2010-02-28")
    assert cropped.alpha_returns.first_valid_index() > full.alpha_returns.first_valid_index()
    assert full.metadata["alpha"]["beta_warmup"] == cropped.metadata["alpha"]["beta_warmup"] == 12
    pd.testing.assert_frame_equal(full.windows, cropped.windows)


def test_latest_block_rolling_falls_back_but_fixed_five_year_test_stays_unavailable(prices):
    from qis.portfolio.reports._performance_inference import _inference_views

    sample = prices.SEQ_US.resample("ME").last().pct_change().iloc[1:]
    sample.iloc[-39] = np.nan
    issues = []
    windows, expanding, rolling, meta = _inference_views(
        sample, "ME", sample.index[-1], issues, alpha=True)
    assert windows.iloc[0].n_obs == 38 and np.isfinite(windows.iloc[0].alpha)
    assert pd.isna(windows.loc["Recent 5 years", "alpha"])
    assert windows.loc["Recent 5 years", "n_obs"] == 59
    assert np.isfinite(windows.loc["Recent 3 years", "alpha"])
    assert meta["rolling_years"] == 3
    assert any("3-year fallback" in item for item in issues)
    assert expanding.estimate.notna().all() and rolling.estimate.notna().all()
    assert rolling.iloc[-1].estimate == pytest.approx(sample.iloc[-36:].mean() * 12)


@pytest.mark.parametrize("freq", ["QE", "QE-JAN"])
@pytest.mark.parametrize(("alpha", "paired"), [(False, False), (False, True),
                                             (True, False), (True, True)])
def test_quarterly_rolling_uses_six_years_and_matches_direct_reference(prices, freq, alpha, paired):
    from qis.portfolio.reports._performance_inference import _inference_views

    levels = prices.resample(freq).last().loc[:prices.index[-1]]
    returns = levels.pct_change(fill_method=None).iloc[1:]
    sample = returns if paired and not alpha else returns.SEQ_US
    if alpha and paired:
        sample = returns.SEQ_US - returns.SBD_TSY
    issues = []
    windows, _, rolling, meta = _inference_views(
        sample, freq, sample.index[-1], issues, alpha=alpha, paired=paired)
    assert meta["rolling_years"] == 6
    assert meta["minimum_observations"] == 24
    assert not any("Rolling chart" in message for message in issues)
    # Fixed summary windows are not silently extended to six years.
    assert windows.loc["Recent 5 years", "n_obs"] == 20
    assert pd.isna(windows.loc["Recent 5 years", "hac_lower"])
    assert windows.loc["Recent 3 years", "n_obs"] == 12
    assert rolling.date.iloc[0] == sample.index[23]
    assert len(rolling) == len(sample) - 23
    for date in rolling.date:
        observed = sample.loc[(sample.index > date - pd.DateOffset(years=6)) &
                              (sample.index <= date)]
        assert len(observed) == 24
    # Different-way calculation: last 24 chronological observations and a dense HAC kernel.
    values = sample.iloc[-24:].to_numpy()
    n = len(values)
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    kernel = np.maximum(1 - distance / (lag + 1), 0)
    if alpha:
        centered = values - values.mean()
        expected = 4 * values.mean()
        se = 4 * np.sqrt(centered @ kernel @ centered / (n * (n - 1)))
    else:
        theta = values.mean(axis=0) / values.std(axis=0, ddof=0)
        standardized = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
        score = standardized - .5 * theta * (standardized ** 2 - 1)
        if paired:
            theta = theta[0] - theta[1]
            score = score[:, 0] - score[:, 1]
        score -= score.mean()
        expected = 2 * theta
        se = 2 * np.sqrt(score @ kernel @ score) / n
    last = rolling.iloc[-1]
    assert last.estimate == pytest.approx(expected)
    np.testing.assert_allclose([last.ci_lower, last.ci_upper],
                               expected + norm.isf(.025) * se * np.array([-1, 1]))


@pytest.mark.parametrize("length", [23, 24])
def test_quarterly_rolling_minimum_is_not_relaxed(prices, length):
    from qis.portfolio.reports._performance_inference import _inference_views

    sample = prices.SEQ_US.resample("QE").last().pct_change(fill_method=None).iloc[-length:]
    issues = []
    _, _, rolling, meta = _inference_views(sample, "QE", sample.index[-1], issues, alpha=True)
    if length == 23:
        assert rolling.empty and meta["rolling_years"] is None
        assert any("6-year history unavailable" in message for message in issues)
    else:
        assert len(rolling) == 1 and meta["rolling_years"] == 6
        assert np.isfinite(rolling.iloc[0].estimate)


def test_quarterly_rolling_page_labels_and_beta_warmup(prices):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = compute(prices, freq="QE")
    assert data.metadata["alpha"]["beta_warmup"] == 12
    assert data.metadata["rolling_years"] == data.metadata["alpha"]["rolling_years"] == 6
    figure = plot_performance_inference(data, "Quarterly sample")
    figure.canvas.draw()
    assert "6-year rolling" in figure.axes[2].get_title()
    assert "6-year rolling" in figure.axes[5].get_title()
    footer = next(text for text in figure.texts if "Expanding warm-up:" in text.get_text())
    assert "6 years at quarterly" in footer.get_text()
    assert footer.get_window_extent(figure.canvas.get_renderer()).x1 <= figure.bbox.width


def test_mixed_inference_grids_select_windows_independently(prices):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = compute_performance_inference(
            prices.SEQ_US, prices[["SBD_TSY"]], qis.TimePeriod("2010", "2025-12-31"),
            "ME", freq_beta="QE", instrument_prices=prices,
            weights=pd.DataFrame({"SEQ_US": 1.0}, index=prices.index), factor_beta_span=12)
    assert data.metadata["rolling_years"] == 5
    assert data.metadata["alpha"]["rolling_years"] == 6
    assert data.metadata["alpha"]["beta_warmup"] == 12
    assert data.rolling.estimate.notna().all() and data.alpha_rolling.estimate.notna().all()
