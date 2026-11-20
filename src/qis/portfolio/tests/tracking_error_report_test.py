"""Smoke tests for covariance RiskModel panels in the tracking-error report."""
import warnings

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_universe
from qis.portfolio.backtester import backtest_model_portfolio
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.portfolio.reports.strategy_benchmark_tre_factsheet import (
    _compute_ex_ante_benchmark_beta,
    _compute_ex_post_benchmark_series,
    weights_tracking_error_report_by_ac_subac,
)
from qis.portfolio.risk.risk_model import RiskModel
from qis.portfolio.reports._one_year_tail_risk import get_one_year_tail_risk_table
from qis.utils.df_to_weights import generate_static_weights_schedule


PRE_EXISTING_FIG_KEYS = {
    'navs',
    'ra_table',
    'strategy_benchmark_weights_stack',
    'strategy_benchmark_weights_box',
    'ex_anti_vols',
    'time_series_risk_contrib',
    'risk_contributions_boxplot',
    'brinson_table_ac',
    'brinson_total_time_series',
    'brinson_grouped_allocation_return',
    'brinson_grouped_selection_return',
    'brinson_table_subac',
    'tre_time_series',
    'tre_group_time_series',
    'joint_turnover',
    'group_turnover',
    'returns_pdfs',
}
PRE_EXISTING_DF_KEYS = {
    'nav',
    'ra_perf_table',
    'ex_anti_vols',
    'brinson_table_ac',
    'brinson_active_total',
    'brinson_table_subac',
    'ac_tracking_error',
    'ac_turnover',
}
NEW_COVAR_PANEL_KEYS = {
    'tre_ex_ante_vs_ex_post',
    'benchmark_beta_time_series',
    'ex_post_alpha_time_series',
}
NEW_NAV_ONLY_PANEL_KEYS = {
    'tre_ex_ante_vs_ex_post',
    'ex_post_alpha_time_series',
}


@pytest.fixture(scope='module')
def report_inputs():
    universe = generate_synthetic_universe(
        start='2021-01-04', end='2022-12-30', seed=20260725, apply_quirks=False)
    prices = universe.prices
    assets = prices.columns

    strategy_target = pd.Series(1.0 / len(assets), index=assets)
    benchmark_target = pd.Series(0.0, index=assets)
    benchmark_target.loc['SEQ_US'] = 0.6
    benchmark_target.loc['SBD_TSY'] = 0.4
    strategy_weights = generate_static_weights_schedule(
        prices=prices, weights=strategy_target, rebalancing_freq='QE')
    benchmark_weights = generate_static_weights_schedule(
        prices=prices, weights=benchmark_target, rebalancing_freq='QE')
    strategy = backtest_model_portfolio(
        prices=prices, weights=strategy_weights, ticker='Synthetic strategy')
    benchmark = backtest_model_portfolio(
        prices=prices, weights=benchmark_weights, ticker='Synthetic benchmark')

    covar = prices.pct_change(fill_method=None).dropna().cov() * 260.0
    covar_dates = prices.index[[60, 120, 180, 240, 300, 360, 420, 480]]
    covar_dict = {date: covar * scale for date, scale in zip(
        covar_dates, np.linspace(0.9, 1.1, len(covar_dates)))}

    factors = pd.Index(['Market', *universe.group_order])
    loadings = pd.DataFrame(0.0, index=assets, columns=factors)
    loadings.loc[:, 'Market'] = 0.5
    for asset, group in universe.group_data.items():
        loadings.loc[asset, group] = 0.5
    factor_covar = pd.DataFrame(
        np.diag(np.linspace(0.008, 0.012, len(factors))),
        index=factors,
        columns=factors)
    residual_vars = pd.Series(0.006, index=assets)
    factor_model = RiskModel(
        covar=covar_dict,
        factor_loadings={date: loadings.copy() for date in covar_dates},
        factor_covar={date: factor_covar.copy() for date in covar_dates},
        residual_vars={date: residual_vars.copy() for date in covar_dates})
    covar_model = RiskModel(covar=covar_dict)

    with_covar = MultiPortfolioData(
        portfolio_datas=[strategy, benchmark],
        benchmark_prices=universe.benchmark_prices.copy(),
        covar_dict=covar_dict)
    without_covar = MultiPortfolioData(
        portfolio_datas=[strategy, benchmark],
        benchmark_prices=universe.benchmark_prices.copy())
    return universe, with_covar, without_covar, covar_model, factor_model


def _run_report(multi_portfolio_data, universe, covar_risk_model=None, **kwargs):
    return weights_tracking_error_report_by_ac_subac(
        multi_portfolio_data=multi_portfolio_data,
        ac_group_data=universe.group_data,
        ac_group_order=universe.group_order,
        sub_ac_group_data=universe.group_data,
        sub_ac_group_order=universe.group_order,
        turnover_groups=universe.group_data,
        turnover_order=universe.group_order,
        covar_risk_model=covar_risk_model,
        add_benchmarks_to_navs=False,
        add_titles=False,
        figsize=(4.0, 3.0), **kwargs)


def _close_figures(figures) -> None:
    for figure in figures.values():
        plt.close(figure)


def test_ex_ante_beta_applies_report_period_before_history_calculation() -> None:
    assets = pd.Index(['A', 'B'])
    dates = pd.DatetimeIndex(['2024-01-02', '2024-01-04', '2024-01-08'])
    covar = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    risk_model = RiskModel(covar={date: covar for date in dates})
    benchmark_weights = pd.DataFrame(
        [[0.6, 0.4]], index=dates[[1]], columns=assets)
    portfolio_weights = pd.DataFrame(
        [[0.6, 0.4]], index=dates[[0]], columns=assets)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = _compute_ex_ante_benchmark_beta(
            risk_model=risk_model,
            benchmark_weights=benchmark_weights,
            portfolio_weights=portfolio_weights,
            time_period=qis.TimePeriod(start=dates[1], end=dates[2]),
        )

    expected = pd.Series(
        [1.0, 1.0], index=dates[1:], name='Benchmark beta')
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize('supply_history', [True, False])
def test_one_year_tail_appendix_matches_export(report_inputs, supply_history):
    """Read every displayed cell back from the opted-in public report at a past date."""
    universe, pair, _, _, _ = report_inputs
    prices = universe.prices
    end = prices.index[-30]
    monthly = prices.resample('ME').last().pct_change(fill_method=None) if supply_history else None
    figs, dfs = _run_report(pair, universe, add_one_year_tail_risk=True, tail_risk_currency='USD',
                            tail_risk_monthly_returns=monthly, time_period=qis.TimePeriod(end=end))
    try:
        frame = dfs['one_year_tail_risk']
        assert list(figs)[-1] == list(dfs)[-1] == 'one_year_tail_risk'
        assert frame.as_of.eq(end).all()
        vintage = max(d for d in pair.covar_dict if d <= end)
        assert frame.iloc[0].covariance_date == vintage
        target = pair.portfolio_datas[0].input_weights.loc[:end].iloc[-1]
        assert frame.iloc[0].strategy_target_date == target.name
        expected_var = 2.32634787404084 * np.sqrt(target @ pair.covar_dict[vintage] @ target)
        np.testing.assert_allclose(frame.iloc[0].strategy_var, expected_var)
        ax = figs['one_year_tail_risk'].axes[0]
        grid = ax.tables[0]
        for i, (method, row) in enumerate(frame.iterrows(), 1):
            assert grid[i, 0].get_text().get_text() == method
            columns = ['strategy_var', 'strategy_es', 'benchmark_var', 'benchmark_es']
            for j, col in enumerate(columns, 1):
                value = row[col]
                expected_text = '—' if pd.isna(value) else f'{value:.2%}'
                assert grid[i, j].get_text().get_text() == expected_text
        notes = ' '.join(t.get_text() for t in ax.texts)
        assert ('not supplied' in notes) == (not supply_history)
        assert pd.notna(frame.iloc[-1].benchmark_var) == supply_history
        if supply_history:
            assert frame.iloc[-1].sample_end <= end
    finally:
        _close_figures(figs)


def test_tail_snapshot_excludes_future_targets_and_covariance(report_inputs):
    """Use date-specific targets and accept a static input when the benchmark has drifted."""
    from dataclasses import replace

    _, pair, _, _, _ = report_inputs
    original, reference = pair.portfolio_datas
    dates = original.prices.index[[100, 101, 102]]
    assets = original.prices.columns
    targets = pd.DataFrame(0., index=dates[[0, 2]], columns=assets)
    targets.loc[dates[0], assets[0]] = 1.
    targets.loc[dates[2], assets[-1]] = 1.
    strategy = replace(original, input_weights=targets)
    benchmark = replace(reference, input_weights={assets[-1]: 1.})
    cov = pd.DataFrame(np.diag(np.linspace(.01, .09, len(assets))), index=assets, columns=assets)
    history = {dates[0]: cov, dates[2]: 4 * cov}
    table = get_one_year_tail_risk_table(strategy, benchmark, history, dates[1], None, 'CHF')
    analytic = table.iloc[0]
    np.testing.assert_allclose([analytic.strategy_var, analytic.benchmark_var],
                               [0.1 * 2.32634787404084, 0.3 * 2.32634787404084])
    assert analytic.strategy_target_date == analytic.covariance_date == dates[0]
    assert pd.isna(analytic.benchmark_target_date) and analytic.currency == 'CHF'
    assert table.iloc[-1].paths == 0 and pd.isna(table.iloc[-1].seed)
    with pytest.raises(ValueError, match='no target weights'):
        get_one_year_tail_risk_table(strategy, benchmark, history, dates[0] - pd.Timedelta(days=1),
                                     None, 'CHF')


def test_ra_performance_table_is_rendered_once(report_inputs) -> None:
    universe, with_covar, _, _, _ = report_inputs
    figs, _ = _run_report(with_covar, universe)
    try:
        assert len(figs['ra_table'].axes) == 1
        assert len(figs['ra_table'].axes[0].tables) == 1
    finally:
        _close_figures(figs)


def test_covariance_only_model_preserves_pre_existing_output_keys(report_inputs) -> None:
    universe, with_covar, _, covar_model, _ = report_inputs
    absent_figs, absent_dfs = _run_report(with_covar, universe)
    try:
        assert set(absent_figs) == PRE_EXISTING_FIG_KEYS | NEW_COVAR_PANEL_KEYS
        assert set(absent_dfs) == PRE_EXISTING_DF_KEYS | NEW_COVAR_PANEL_KEYS
        absent_fig_keys = set(absent_figs)
        absent_df_keys = set(absent_dfs)
        absent_pre_existing_dfs = {
            key: absent_dfs[key].copy() for key in PRE_EXISTING_DF_KEYS
        }
    finally:
        _close_figures(absent_figs)

    model_figs, model_dfs = _run_report(with_covar, universe, covar_model)
    try:
        assert set(model_figs) == absent_fig_keys
        assert set(model_dfs) == absent_df_keys
        for key, expected in absent_pre_existing_dfs.items():
            pd.testing.assert_frame_equal(model_dfs[key], expected, rtol=1e-12, atol=0.0)
    finally:
        _close_figures(model_figs)


def test_complete_factor_model_adds_panels_and_supplies_missing_covariance(
        report_inputs) -> None:
    universe, _, without_covar, _, factor_model = report_inputs
    # The complete factor-model report intentionally creates and returns 22 figures in one call.
    # Matplotlib warns when the 21st figure opens, before this test receives the figure dictionary
    # and can close it. Pin the default threshold to make the warning deterministic, assert it here,
    # and close every returned figure in the finally block below.
    with matplotlib.rc_context({'figure.max_open_warning': 20}):
        with pytest.warns(RuntimeWarning, match='More than 20 figures have been opened'):
            figs, dfs = _run_report(without_covar, universe, factor_model)
    try:
        assert {'tre_decomposition', 'factor_exposures'} <= set(figs)
        assert {'tre_decomposition', 'factor_exposures'} <= set(dfs)
        assert NEW_COVAR_PANEL_KEYS <= set(figs)
        assert NEW_COVAR_PANEL_KEYS <= set(dfs)
        assert dfs['tre_decomposition'].columns.tolist() == [
            'tracking_error', 'factor_te', 'residual_te']
        assert dfs['factor_exposures'].columns.tolist() == [
            'Market', *universe.group_order]
    finally:
        _close_figures(figs)


def test_nav_only_report_keeps_realised_tre_and_alpha_panels(report_inputs) -> None:
    universe, _, without_covar, _, _ = report_inputs

    figs, dfs = _run_report(without_covar, universe)
    try:
        assert NEW_NAV_ONLY_PANEL_KEYS <= set(figs)
        assert NEW_NAV_ONLY_PANEL_KEYS <= set(dfs)
        assert 'benchmark_beta_time_series' not in figs
        assert 'benchmark_beta_time_series' not in dfs
        assert dfs['tre_ex_ante_vs_ex_post'].columns.tolist() == [
            'Realised TRE (EWMA 36m)'
        ]
    finally:
        _close_figures(figs)


def test_new_panel_frames_match_independent_ewma_references(report_inputs) -> None:
    universe, with_covar, _, _, _ = report_inputs
    figs, dfs = _run_report(with_covar, universe)
    try:
        assert dfs['tre_ex_ante_vs_ex_post'].columns.tolist() == [
            'Ex-ante TRE', 'Realised TRE (EWMA 36m)'
        ]
        assert dfs['benchmark_beta_time_series'].columns.tolist() == [
            'Ex-ante beta', 'Ex-post beta (EWMA 36m)'
        ]
        assert dfs['ex_post_alpha_time_series'].columns.tolist() == [
            'Ex-post alpha (EWMA 36m, annualised)'
        ]
    finally:
        _close_figures(figs)

    long_universe = generate_synthetic_universe(
        start='2018-01-01',
        end='2022-12-30',
        seed=20260725,
        apply_quirks=False,
    )
    strategy_nav = long_universe.prices['SEQ_US']
    benchmark_nav = long_universe.benchmark_prices.iloc[:, 0]
    navs = pd.concat(
        [strategy_nav.rename('strategy'), benchmark_nav.rename('benchmark')],
        axis=1,
    ).ffill()
    monthly_returns = navs.asfreq('ME', method='ffill').pct_change(fill_method=None).dropna()
    monthly_returns.index.name = None
    direct_realised_tre = qis.compute_ewm_vol(
        data=monthly_returns['strategy'] - monthly_returns['benchmark'],
        span=36,
        annualize=True,
        warmup_period=36,
    ).rename('Realised TRE (EWMA 36m)')
    realised_tre, report_beta, report_alpha = _compute_ex_post_benchmark_series(
        strategy_nav=strategy_nav,
        benchmark_nav=benchmark_nav,
    )
    direct_realised_tre.index.name = realised_tre.index.name
    pd.testing.assert_series_equal(
        realised_tre.rename('Realised TRE (EWMA 36m)'),
        direct_realised_tre,
        rtol=1e-10,
        atol=0.0,
    )

    direct_beta, direct_alpha, _, _, _, _ = qis.compute_ewm_beta_alpha_forecast(
        x_data=monthly_returns['benchmark'],
        y_data=monthly_returns[['strategy']],
        span=36,
        init_type=qis.InitType.X0,
        beta_init_value=1.0,
    )
    expected_alpha = (
        direct_alpha.iloc[:, 0] * qis.get_annualization_factor('ME')
    ).rename('Ex-post alpha (EWMA 36m, annualised)')
    expected_alpha.index.name = strategy_nav.index.name
    pd.testing.assert_series_equal(report_alpha, expected_alpha, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(report_beta.iloc[0], 1.0, rtol=0.0, atol=0.0)
    expected_beta = direct_beta.iloc[:, 0].rename('Ex-post beta (EWMA 36m)')
    expected_beta.index.name = strategy_nav.index.name
    pd.testing.assert_series_equal(
        report_beta,
        expected_beta,
        rtol=1e-12,
        atol=0.0,
    )

    unseeded_beta = qis.compute_one_factor_ewm_betas(
        x=monthly_returns['benchmark'],
        y=monthly_returns[['strategy']],
        span=36,
    )
    seeded_terminal = report_beta.iloc[-1]
    unseeded_terminal = unseeded_beta.iloc[-1, 0]
    assert 1.0 < seeded_terminal < unseeded_terminal
    np.testing.assert_allclose(
        seeded_terminal,
        unseeded_terminal,
        rtol=0.0,
        atol=0.05,
    )
