"""
Structural regression tests for the reporting-frequency convention (ROADMAP workstream 1) and the
FactsheetConfig <-> generator name-matching contract (workstream 4).

These lock the conventions established across the reporting-frequency work:
  * the per-frequency / per-horizon window + grid + regime presets,
  * long-vs-short horizon auto-selection from the reporting span,
  * the up-sampling guard (no reporting finer than the data),
  * frequency-invariant vs frequency-dependent statistics (incl. the skew-frequency fix),
  * the frequency labels carried by every rendered panel title (multi-asset and the three
    portfolio reports), with no anchored 'QE-DEC' leak,
  * the qis.factsheet facade dispatch + PDF output,
  * the FactsheetConfig field / generator-parameter contract.

Everything runs on a deterministic synthetic GBM panel - no network, reproducible.
Run:           pytest qis/tests/test_reporting_conventions.py
Fast subset:   pytest qis/tests/test_reporting_conventions.py -k "not label and not facade"
(the rendering tests - 'label' and 'facade' - dominate runtime; the rest are sub-second.)
"""
import inspect
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless rendering for the panel-title tests
import matplotlib.pyplot as plt
import pytest

import qis
from qis import ReportingFrequency, PerfParams
from qis.portfolio.reports.config import (FactsheetConfig, make_factsheet_config,
                                          fetch_default_report_kwargs, validate_reporting_frequency)
from qis.portfolio.reports.strategy_factsheet import generate_strategy_factsheet
from qis.portfolio.reports.strategy_benchmark_factsheet import generate_strategy_benchmark_factsheet_plt
from qis.portfolio.reports.multi_strategy_factsheet import generate_multi_portfolio_factsheet
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet


# --------------------------------------------------------------------------------------------------
# deterministic synthetic data (module-scoped, built once)
# --------------------------------------------------------------------------------------------------
def _make_synthetic_prices(n_assets: int = 6, n_years: int = 14, seed: int = 7) -> pd.DataFrame:
    """reproducible correlated-GBM price panel on a business-day grid (no network)."""
    rng = np.random.default_rng(seed)
    n = n_years * 260
    idx = pd.bdate_range(end=pd.Timestamp('2025-12-31'), periods=n)
    mu = rng.uniform(0.03, 0.12, n_assets) / 260.0
    sig = rng.uniform(0.10, 0.30, n_assets) / np.sqrt(260.0)
    common = rng.standard_normal((n, 1))
    idio = rng.standard_normal((n, n_assets))
    rets = mu + sig * (0.5 * common + 0.85 * idio)  # mild common factor + idiosyncratic
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f'A{i}' for i in range(n_assets)])


@pytest.fixture(scope='module')
def prices() -> pd.DataFrame:
    return _make_synthetic_prices()


@pytest.fixture(scope='module')
def benchmark(prices) -> str:
    return str(prices.columns[0])


# expected presets (period counts in the base grid), from docs/reporting_frequencies.md:
# (reporting_frequency, grid, sharpe/vol LONG, sharpe/vol SHORT, beta LONG, beta SHORT)
_PRESETS = [
    (ReportingFrequency.DAILY,     'B',     260, 260, 780, 260),
    (ReportingFrequency.WEEKLY,    'W-WED', 156,  52, 156,  52),
    (ReportingFrequency.MONTHLY,   'ME',     36,  12,  36,  12),
    (ReportingFrequency.QUARTERLY, 'QE',     12,   4,  12,   4),
]
_FREQS = [p[0] for p in _PRESETS]


# --------------------------------------------------------------------------------------------------
# 1. preset calibration (no rendering)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rf, grid, s_long, s_short, b_long, b_short", _PRESETS,
                         ids=[p[0].name for p in _PRESETS])
def test_preset_windows_and_grid(rf, grid, s_long, s_short, b_long, b_short):
    c_long = make_factsheet_config(rf, is_long_period=True)
    c_short = make_factsheet_config(rf, is_long_period=False)
    # base grid is the frequency's own pandas rule
    assert c_long.freq == grid == rf.value
    # vol and sharpe share the same rolling window in every preset
    assert c_long.sharpe_rolling_window == s_long == c_long.vol_rolling_window
    assert c_short.sharpe_rolling_window == s_short == c_short.vol_rolling_window
    # factor-beta span
    assert c_long.factor_beta_span == b_long
    assert c_short.factor_beta_span == b_short
    # every freq_* field uses the frequency's own grid
    for field in ('vol_freq', 'freq_sharpe', 'freq_beta', 'corr_freq', 'freq_reg', 'freq_drawdown'):
        assert getattr(c_long, field) == grid, f"{rf.name} long {field}"
        assert getattr(c_short, field) == grid, f"{rf.name} short {field}"


@pytest.mark.parametrize("rf", _FREQS, ids=[f.name for f in _FREQS])
def test_regime_frequency_by_horizon(rf):
    # regimes classify on QE (long) / ME (short) regardless of the reporting frequency
    assert make_factsheet_config(rf, is_long_period=True).freq_regime == 'QE'
    assert make_factsheet_config(rf, is_long_period=False).freq_regime == 'ME'


# --------------------------------------------------------------------------------------------------
# 2. long vs short horizon auto-selection from the reporting span
# --------------------------------------------------------------------------------------------------
def test_horizon_autoselect_from_span():
    long_tp = qis.TimePeriod(pd.Timestamp('2005-01-01'), pd.Timestamp('2025-12-31'))   # > 5y
    short_tp = qis.TimePeriod(pd.Timestamp('2023-06-01'), pd.Timestamp('2025-12-31'))  # < 5y
    kw_long = fetch_default_report_kwargs(time_period=long_tp, reporting_frequency=ReportingFrequency.MONTHLY)
    kw_short = fetch_default_report_kwargs(time_period=short_tp, reporting_frequency=ReportingFrequency.MONTHLY)
    # long horizon -> 36 (3y monthly) + QE regimes; short horizon -> 12 (1y monthly) + ME regimes
    assert kw_long['sharpe_rolling_window'] == 36 and kw_long['freq_regime'] == 'QE'
    assert kw_short['sharpe_rolling_window'] == 12 and kw_short['freq_regime'] == 'ME'


# --------------------------------------------------------------------------------------------------
# 3. up-sampling guard
# --------------------------------------------------------------------------------------------------
def test_guard_allows_equal_or_coarser(prices):
    # daily data: every reporting frequency is equal or coarser -> all allowed (no raise)
    for rf in ReportingFrequency:
        validate_reporting_frequency(prices, rf)


@pytest.mark.parametrize("rf", [ReportingFrequency.DAILY, ReportingFrequency.WEEKLY],
                         ids=['daily', 'weekly'])
def test_guard_blocks_upsampling(prices, rf):
    monthly = prices.resample('ME').last()
    with pytest.raises(ValueError):
        validate_reporting_frequency(monthly, rf)  # finer than monthly data -> raise


def test_guard_blocks_in_generator(prices, benchmark):
    # the generator must refuse weekly reporting on monthly data (guard fires before rendering)
    monthly = prices.resample('ME').last()
    tp = qis.get_time_period(df=monthly)
    kw = fetch_default_report_kwargs(time_period=tp, reporting_frequency=ReportingFrequency.WEEKLY)
    with pytest.raises(ValueError):
        generate_multi_asset_factsheet(prices=monthly, benchmark=benchmark, time_period=tp, **kw)


# --------------------------------------------------------------------------------------------------
# 4. frequency-invariant vs frequency-dependent statistics (perf engine + skew-frequency fix)
# --------------------------------------------------------------------------------------------------
def _ra_table(prices, grid):
    return qis.compute_ra_perf_table(prices, perf_params=PerfParams(freq=grid, freq_vol=grid, freq_skewness=grid))


def test_total_return_is_frequency_invariant(prices):
    native_total = prices.iloc[-1] / prices.iloc[0] - 1.0  # raw endpoints, frequency-independent
    base = None
    for grid in ('B', 'W-WED', 'ME', 'QE'):
        total = _ra_table(prices, grid)['Total']
        # engine total matches the raw endpoint total at every frequency ...
        pd.testing.assert_series_equal(total, native_total, check_names=False, rtol=1e-3, atol=1e-4)
        # ... and is identical across frequencies
        if base is None:
            base = total
        else:
            pd.testing.assert_series_equal(total, base, check_names=False, rtol=1e-6, atol=1e-6)


def test_vol_and_skew_are_frequency_dependent(prices):
    daily = _ra_table(prices, 'B')
    monthly = _ra_table(prices, 'ME')
    quarterly = _ra_table(prices, 'QE')
    # annualised vol shifts with sampling frequency; skew strongly so (the skew-frequency fix)
    assert (daily['Vol'] - monthly['Vol']).abs().max() > 1e-3
    assert (daily['Skewness'] - quarterly['Skewness']).abs().max() > 1e-2


# --------------------------------------------------------------------------------------------------
# 5. FactsheetConfig <-> generator parameter contract (workstream 4)
# --------------------------------------------------------------------------------------------------
_GENERATORS = [generate_strategy_factsheet, generate_strategy_benchmark_factsheet_plt,
               generate_multi_portfolio_factsheet, generate_multi_asset_factsheet]


def _looks_frequency_critical(name: str) -> bool:
    return (name.startswith('freq_') or name.endswith('_freq')
            or name.endswith('_rolling_window') or name.endswith('_rolling_period')
            or name.endswith('_span'))


@pytest.mark.parametrize("generator", _GENERATORS, ids=[g.__name__ for g in _GENERATORS])
def test_generators_accept_config_spread(generator):
    # the config is spread as **kwargs into every generator -> each must keep a **kwargs catch-all
    params = inspect.signature(generator).parameters.values()
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params), \
        f"{generator.__name__} lost **kwargs - the config spread would break"


@pytest.mark.parametrize("generator", _GENERATORS, ids=[g.__name__ for g in _GENERATORS])
def test_frequency_critical_params_are_fed_by_config(generator):
    # every frequency-critical *named* parameter must be supplied by the report kwargs, otherwise it
    # silently falls back to its own default and is not frequency-calibrated (the name-matching drift)
    tp = qis.TimePeriod(pd.Timestamp('2005-01-01'), pd.Timestamp('2025-12-31'))
    provided = set(fetch_default_report_kwargs(time_period=tp, reporting_frequency=ReportingFrequency.MONTHLY))
    freq_params = {p for p in inspect.signature(generator).parameters if _looks_frequency_critical(p)}
    missing = freq_params - provided
    assert not missing, f"{generator.__name__} freq-critical params not fed by config: {sorted(missing)}"


def test_config_contract_invariants():
    tp = qis.TimePeriod(pd.Timestamp('2005-01-01'), pd.Timestamp('2025-12-31'))
    provided = set(fetch_default_report_kwargs(time_period=tp, reporting_frequency=ReportingFrequency.MONTHLY))
    # 'freq' is popped before the spread (generators take the grid via perf_params.freq)
    assert 'freq' not in provided
    # perf_params and the regime classifier are injected by the config layer
    assert 'perf_params' in provided and 'regime_classifier' in provided
    # the public field contract stays stable
    assert {'sharpe_rolling_window', 'factor_beta_span', 'freq_beta', 'corr_freq',
            'freq_regime'}.issubset(set(FactsheetConfig._fields))


# --------------------------------------------------------------------------------------------------
# 6. rendered panel frequency labels (multi-asset report across the frequency grid)
# --------------------------------------------------------------------------------------------------
def _titles(fig):
    return [ax.get_title() for ax in fig.axes if ax.get_title()]


@pytest.mark.parametrize("rf, grid, s_long, s_short, b_long, b_short", _PRESETS,
                         ids=[p[0].name for p in _PRESETS])
def test_panel_frequency_labels(prices, benchmark, rf, grid, s_long, s_short, b_long, b_short):
    tp = qis.get_time_period(df=prices)  # full history (14y) -> long horizon
    kw = fetch_default_report_kwargs(time_period=tp, reporting_frequency=rf)
    fig = generate_multi_asset_factsheet(prices=prices, benchmark=benchmark, time_period=tp, **kw)
    blob = " || ".join(_titles(fig))
    try:
        # cumulative stats labelled at the reporting grid
        assert f"({grid}-freq stats)" in blob
        # drawdowns / under-water labelled at the native (business-day) grid, not the reporting grid
        assert "Running Drawdowns (B-freq)" in blob
        assert "Rolling time under water (B-freq)" in blob
        # rolling beta and correlation carry the reporting grid
        assert f"rolling Beta of {grid}-freq returns" in blob
        assert f"Correlation of {grid}-freq returns" in blob
        # the long-horizon rolling window for this frequency is the one actually used
        assert f"roll_period={s_long}" in blob
        # regimes stay on QE (long horizon) regardless of the reporting frequency
        assert "QE-freq regimes" in blob
        # no anchored quarter label must leak into any title
        assert "QE-DEC" not in blob
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------------------
# 7. qis.factsheet facade smoke (dispatch + PDF output)
# --------------------------------------------------------------------------------------------------
def test_facade_returns_figures(prices, benchmark):
    figs = qis.factsheet(prices, benchmark=benchmark, reporting_frequency='monthly')
    try:
        assert isinstance(figs, list) and len(figs) >= 1
        assert isinstance(figs[0], plt.Figure)
    finally:
        plt.close('all')


def test_facade_writes_pdf(prices, benchmark, tmp_path):
    out = qis.factsheet(prices, benchmark=benchmark, reporting_frequency='quarterly',
                        file_name='facade_smoke', local_path=str(tmp_path) + os.sep)
    plt.close('all')
    assert isinstance(out, str) and out.endswith('.pdf')
    assert os.path.exists(out)


# --------------------------------------------------------------------------------------------------
# 8. rendered panel frequency labels - portfolio reports (strategy / multi-strategy / benchmark)
#    these share the FactsheetConfig + the MultiPortfolioData plotting layer covered above; here we
#    confirm their panels carry the same frequency labels and never leak an anchored quarter label.
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def portfolios(prices):
    """two simple constant-weight portfolios (equal-weight + a tilt) and a MultiPortfolioData.

    Built with qis primitives only (no dependency on the examples package) so the suite stays
    self-contained; the portfolio content is irrelevant - the panels just need to render.
    """
    rebal = prices.resample('ME').last().index
    n = prices.shape[1]
    ew = pd.DataFrame(1.0 / n, index=rebal, columns=prices.columns)
    tilt = ew.copy()
    tilt.iloc[:, 0] = 0.5
    tilt.iloc[:, 1:] = 0.5 / (n - 1)
    p_ew = qis.backtest_model_portfolio(prices=prices, weights=ew, rebalancing_costs=0.0010, ticker='EW')
    p_tilt = qis.backtest_model_portfolio(prices=prices, weights=tilt, rebalancing_costs=0.0010, ticker='Tilt')
    mpd = qis.MultiPortfolioData(portfolio_datas=[p_ew, p_tilt], benchmark_prices=prices[['A0']])
    return p_ew, mpd


def _render_portfolio_report(report_kind, p_strategy, mpd, benchmark_prices, time_period, report_kwargs):
    if report_kind == 'strategy':
        return generate_strategy_factsheet(portfolio_data=p_strategy, benchmark_prices=benchmark_prices,
                                           time_period=time_period, **report_kwargs)
    if report_kind == 'multi_strategy':
        return generate_multi_portfolio_factsheet(multi_portfolio_data=mpd, time_period=time_period,
                                                  **report_kwargs)
    if report_kind == 'strategy_benchmark':
        # brinson attribution needs grouped data; off here since we test frequency labels, not brinson
        return generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=mpd, time_period=time_period,
                                                         add_brinson_attribution=False, **report_kwargs)
    raise ValueError(report_kind)


@pytest.mark.parametrize("report_kind", ['strategy', 'multi_strategy', 'strategy_benchmark'])
@pytest.mark.parametrize("rf, grid, s_long", [(ReportingFrequency.MONTHLY, 'ME', 36),
                                              (ReportingFrequency.QUARTERLY, 'QE', 12)],
                         ids=['monthly', 'quarterly'])
def test_portfolio_report_labels(portfolios, prices, report_kind, rf, grid, s_long):
    p_strategy, mpd = portfolios
    tp = qis.get_time_period(df=prices)  # full history -> long horizon
    kw = fetch_default_report_kwargs(time_period=tp, reporting_frequency=rf)
    figs = _render_portfolio_report(report_kind, p_strategy, mpd, prices[['A0']], tp, kw)
    figs = list(figs) if isinstance(figs, (list, tuple)) else [figs]
    blob = " || ".join(ax.get_title() for f in figs for ax in f.axes if ax.get_title())
    try:
        # cumulative stats labelled at the reporting grid (shared across all four report types)
        assert f"({grid}-freq stats)" in blob, report_kind
        # drawdowns and under-water both on the native business-day grid
        assert blob.count("(B-freq)") >= 2, report_kind
        # turnover / cost panels carry the reporting grid (the QE-DEC fix lives here)
        assert f"{grid}-freq Turnover" in blob, report_kind
        # the long-horizon rolling window for this frequency is the one actually used
        assert f"roll_period={s_long}" in blob, report_kind
        # a benchmark bull/bear/normal regime panel is present
        assert "Bear/Normal/Bull" in blob, report_kind
        # no anchored quarter label leaks anywhere
        assert "QE-DEC" not in blob, report_kind
    finally:
        for f in figs:
            plt.close(f)
