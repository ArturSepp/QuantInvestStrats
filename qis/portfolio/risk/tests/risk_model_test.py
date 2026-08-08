"""Policy and validation tests for the point-in-time risk model."""

import numpy as np
import pandas as pd
import pytest

from qis.datasets.synthetic import generate_synthetic_universe
from qis.portfolio.backtester import backtest_model_portfolio
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.portfolio.risk.risk_model import RiskModel, UNASSIGNED_GROUP, WEIGHT_TOL


DATE_1 = pd.Timestamp('2024-01-02')
DATE_2 = pd.Timestamp('2024-01-04')
DATE_3 = pd.Timestamp('2024-01-08')
ASSETS = pd.Index(['A', 'B', 'C'])
FACTORS = pd.Index(['Market', 'Rates'])


def _covar(scale: float = 1.0) -> pd.DataFrame:
    values = scale * np.array([
        [0.040, 0.006, -0.002],
        [0.006, 0.025, 0.004],
        [-0.002, 0.004, 0.016],
    ])
    return pd.DataFrame(values, index=ASSETS, columns=ASSETS)


def _loadings() -> pd.DataFrame:
    return pd.DataFrame([
        [1.0, 0.1],
        [0.8, -0.2],
        [0.2, 1.1],
    ], index=ASSETS, columns=FACTORS)


def _factor_covar() -> pd.DataFrame:
    return pd.DataFrame([[0.03, 0.002], [0.002, 0.01]],
                        index=FACTORS, columns=FACTORS)


def _residual_vars() -> pd.Series:
    return pd.Series([0.01, 0.008, 0.006], index=ASSETS)


def _model() -> RiskModel:
    return RiskModel(covar={DATE_1: _covar(), DATE_2: _covar(1.1), DATE_3: _covar(0.9)})


def _factor_model() -> RiskModel:
    dates = (DATE_1, DATE_2, DATE_3)
    return RiskModel(
        covar={date: _covar() for date in dates},
        factor_loadings={date: _loadings() for date in dates},
        factor_covar={date: _factor_covar() for date in dates},
        residual_vars={date: _residual_vars() for date in dates})


def test_missing_and_nan_weights_are_zero_on_covar_universe() -> None:
    model = _model()
    actual = model._align_weights(
        weights=pd.Series({'A': 0.7, 'B': np.nan}),
        date=DATE_1,
        role='portfolio_weights')
    expected = pd.Series([0.7, 0.0, 0.0], index=ASSETS)
    pd.testing.assert_series_equal(actual, expected)


def test_extra_material_weight_raises_with_ticker_and_weight() -> None:
    model = _model()
    with pytest.raises(ValueError, match=r"portfolio_weights.*OUTSIDE.*0.25"):
        model._align_weights(
            weights=pd.Series({'A': 0.75, 'OUTSIDE': 0.25}),
            date=DATE_1,
            role='portfolio_weights')


@pytest.mark.parametrize('extra_weight', [0.0, WEIGHT_TOL, np.nan])
def test_extra_zero_tolerance_or_nan_weight_is_dropped(extra_weight: float) -> None:
    model = _model()
    actual = model._align_weights(
        weights=pd.Series({'A': 1.0, 'OUTSIDE': extra_weight}),
        date=DATE_1,
        role='portfolio_weights')
    expected = pd.Series([1.0, 0.0, 0.0], index=ASSETS)
    pd.testing.assert_series_equal(actual, expected)


def test_strict_false_silently_drops_extra_material_weight() -> None:
    model = _model()
    actual = model._align_weights(
        weights=pd.Series({'A': 0.75, 'OUTSIDE': 0.25}),
        date=DATE_1,
        role='portfolio_weights',
        strict=False)
    expected = pd.Series([0.75, 0.0, 0.0], index=ASSETS)
    pd.testing.assert_series_equal(actual, expected)


def test_nan_covar_raises_with_date_and_field() -> None:
    covar = _covar()
    covar.loc['A', 'B'] = np.nan
    with pytest.raises(ValueError, match=r"covar\[2024-01-02.*non-finite"):
        RiskModel(covar={DATE_1: covar})


def test_off_grid_date_names_nearest_earlier_grid_date() -> None:
    with pytest.raises(KeyError, match=r"2024-01-03.*2024-01-02"):
        _model()._covar_at_date(pd.Timestamp('2024-01-03'))


def test_off_grid_date_before_grid_says_no_earlier_date() -> None:
    with pytest.raises(KeyError, match=r"2023-12-31.*none"):
        _model()._covar_at_date(pd.Timestamp('2023-12-31'))


def test_history_uses_asof_ffill_and_leading_zero_without_lookahead() -> None:
    weights = pd.DataFrame(
        [[0.2, 0.8], [0.6, 0.4]],
        index=[pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-05')],
        columns=['A', 'B'])
    actual = _model()._aligned_weight_history(weights, role='portfolio_weights')
    expected = pd.DataFrame(
        [[0.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.6, 0.4, 0.0]],
        index=pd.DatetimeIndex([DATE_1, DATE_2, DATE_3]),
        columns=ASSETS)
    pd.testing.assert_frame_equal(actual, expected)


def test_group_alignment_exposes_missing_labels_as_unassigned() -> None:
    groups = pd.Series({'A': 'Equity', 'C': np.nan, 'OUTSIDE': 'Ignored'})
    actual = _model()._align_groups(group_data=groups, date=DATE_1)
    expected = pd.Series(['Equity', UNASSIGNED_GROUP, UNASSIGNED_GROUP], index=ASSETS)
    pd.testing.assert_series_equal(actual, expected)


def test_factor_loading_asset_mismatch_names_date_and_tickers() -> None:
    loadings = _loadings().drop(index='C')
    loadings.loc['OUTSIDE'] = [0.3, 0.4]
    with pytest.raises(ValueError, match=r"2024-01-02.*missing \['C'\].*OUTSIDE"):
        RiskModel(covar={DATE_1: _covar()}, factor_loadings={DATE_1: loadings})


def test_factor_covar_factor_mismatch_names_date_and_tickers() -> None:
    factor_covar = _factor_covar().rename(index={'Rates': 'Credit'},
                                          columns={'Rates': 'Credit'})
    with pytest.raises(ValueError, match=r"2024-01-02.*Rates.*Credit"):
        RiskModel(
            covar={DATE_1: _covar()},
            factor_loadings={DATE_1: _loadings()},
            factor_covar={DATE_1: factor_covar},
            residual_vars={DATE_1: _residual_vars()})


def test_residual_variance_asset_mismatch_names_date_and_tickers() -> None:
    residual_vars = _residual_vars().drop(index='C')
    residual_vars.loc['OUTSIDE'] = 0.01
    with pytest.raises(ValueError, match=r"2024-01-02.*missing \['C'\].*OUTSIDE"):
        RiskModel(
            covar={DATE_1: _covar()},
            factor_loadings={DATE_1: _loadings()},
            factor_covar={DATE_1: _factor_covar()},
            residual_vars={DATE_1: residual_vars})


def test_factor_date_grid_mismatch_names_missing_date() -> None:
    with pytest.raises(ValueError, match=r"factor_loadings.*2024-01-04"):
        RiskModel(
            covar={DATE_1: _covar(), DATE_2: _covar()},
            factor_loadings={DATE_1: _loadings()})


def test_factor_covar_without_loadings_names_missing_field() -> None:
    with pytest.raises(ValueError, match="factor_loadings"):
        RiskModel(
            covar={DATE_1: _covar()},
            factor_covar={DATE_1: _factor_covar()},
            residual_vars={DATE_1: _residual_vars()})


def test_partial_factor_block_names_residual_vars() -> None:
    with pytest.raises(ValueError, match="residual_vars"):
        RiskModel(
            covar={DATE_1: _covar()},
            factor_loadings={DATE_1: _loadings()},
            factor_covar={DATE_1: _factor_covar()})


def test_covariance_validation_rejects_asymmetry_at_stated_tolerance() -> None:
    covar = _covar()
    covar.loc['A', 'B'] += 1e-8
    with pytest.raises(ValueError, match=r"not symmetric within 1e-12"):
        RiskModel(covar={DATE_1: covar})


def test_covariance_only_model_names_factor_loadings_when_required() -> None:
    with pytest.raises(ValueError, match="factor_loadings"):
        _model()._require_factor_loadings()


def test_loadings_only_model_names_factor_covar_when_block_required() -> None:
    model = RiskModel(covar={DATE_1: _covar()},
                      factor_loadings={DATE_1: _loadings()})
    with pytest.raises(ValueError, match="factor_covar"):
        model._require_factor_block()


def test_complete_factor_block_is_normalised_to_covar_and_factor_order() -> None:
    model = _factor_model()
    assert model.factor_loadings is not None
    assert model.factor_covar is not None
    assert model.residual_vars is not None
    assert model.factor_loadings[DATE_1].index.equals(ASSETS)
    assert model.factor_covar[DATE_1].index.equals(FACTORS)
    assert model.residual_vars[DATE_1].index.equals(ASSETS)


def test_tre_matches_explicit_double_loop_reference_on_seeded_inputs() -> None:
    # Seed 20260808. The reference deliberately uses scalar loops, not the implementation's
    # labelled quadratic form or any alignment helper.
    rng = np.random.default_rng(20260808)
    random_matrix = rng.normal(size=(5, 5))
    covar_values = random_matrix @ random_matrix.T / 100.0
    assets = pd.Index([f'A{idx}' for idx in range(5)])
    covar = pd.DataFrame(covar_values, index=assets, columns=assets)
    benchmark = pd.Series(rng.normal(size=5), index=assets)
    portfolio = pd.Series(rng.normal(size=5), index=assets)
    active = portfolio - benchmark
    variance_reference = 0.0
    for i in range(len(assets)):
        for j in range(len(assets)):
            variance_reference += active.iloc[i] * covar.iloc[i, j] * active.iloc[j]
    expected = np.sqrt(variance_reference)

    actual = RiskModel(covar={DATE_1: covar}).compute_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_tre_group_mask_matches_covariance_block_reference() -> None:
    benchmark = pd.Series([0.4, 0.3, 0.3], index=ASSETS)
    portfolio = pd.Series([0.6, 0.1, 0.4], index=ASSETS)
    groups = pd.Series(['Group 1', 'Group 1', 'Group 2'], index=ASSETS)
    actual = _model().compute_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1,
        group_data=groups)
    assert isinstance(actual, pd.Series)

    tickers = ['A', 'B']
    active_block = (portfolio - benchmark).loc[tickers]
    covar_block = _covar().loc[tickers, tickers]
    expected_group = np.sqrt(active_block @ covar_block @ active_block)
    np.testing.assert_allclose(actual['Group 1'], expected_group, rtol=1e-12, atol=0.0)
    assert actual['Group 1'] + actual['Group 2'] != pytest.approx(actual['Total'])


def test_tre_zero_active_weights_is_exactly_zero() -> None:
    weights = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
    actual = _model().compute_tre_at_date(
        benchmark_weights=weights,
        portfolio_weights=weights,
        date=DATE_1)
    assert actual == 0.0


def test_tre_public_method_routes_through_strict_aligner() -> None:
    benchmark = pd.Series([1.0], index=['A'])
    portfolio = pd.Series([0.8, 0.2], index=['A', 'OUTSIDE'])
    with pytest.raises(ValueError, match=r"portfolio_weights.*OUTSIDE"):
        _model().compute_tre_at_date(
            benchmark_weights=benchmark,
            portfolio_weights=portfolio,
            date=DATE_1)


def test_tre_history_matches_legacy_mpd_total_and_groups_on_synthetic_panel() -> None:
    # Frozen generator seed 20260725; clean paths isolate TE arithmetic from reporting defects.
    universe = generate_synthetic_universe(
        start='2021-01-04', end='2021-06-30', seed=20260725, apply_quirks=False)
    prices = universe.prices
    assets = prices.columns
    covar_base = prices.pct_change(fill_method=None).dropna().cov() * 260.0
    covar_dates = prices.index[[20, 40, 60, 80]]
    covar_dict = {
        date: covar_base * scale
        for date, scale in zip(covar_dates, [0.9, 1.0, 1.1, 1.2])
    }

    strategy_rows = np.full((2, len(assets)), 1.0 / len(assets))
    strategy_rows[1, 0] += 0.08
    strategy_rows[1, 3] -= 0.08
    benchmark_rows = np.zeros((2, len(assets)))
    benchmark_rows[:, assets.get_loc('SEQ_US')] = 0.6
    benchmark_rows[:, assets.get_loc('SBD_TSY')] = 0.4
    weight_dates = covar_dates[[0, 2]]
    strategy_weights = pd.DataFrame(strategy_rows, index=weight_dates, columns=assets)
    benchmark_weights = pd.DataFrame(benchmark_rows, index=weight_dates, columns=assets)

    strategy = backtest_model_portfolio(
        prices=prices,
        weights=strategy_weights,
        ticker='Synthetic strategy')
    benchmark = backtest_model_portfolio(
        prices=prices,
        weights=benchmark_weights,
        ticker='Synthetic benchmark')
    legacy = MultiPortfolioData(
        portfolio_datas=[strategy, benchmark],
        covar_dict=covar_dict)
    model = RiskModel(covar=covar_dict)

    legacy_total = legacy.compute_tracking_error_implied_by_covar()
    actual_total = model.compute_tre_history(
        benchmark_weights=benchmark_weights,
        portfolio_weights=strategy_weights,
        strict=False)
    pd.testing.assert_series_equal(actual_total, legacy_total, rtol=1e-12, atol=0.0)

    legacy_groups = legacy.compute_tracking_error_implied_by_covar(
        is_grouped=True,
        group_data=universe.group_data,
        group_order=universe.group_order)
    actual_groups = model.compute_tre_history(
        benchmark_weights=benchmark_weights,
        portfolio_weights=strategy_weights,
        group_data=universe.group_data,
        strict=False)
    pd.testing.assert_frame_equal(actual_groups, legacy_groups, rtol=1e-12, atol=0.0)


def test_exposures_match_literal_assets_by_factors_hand_calculation() -> None:
    weights = pd.Series([0.5, -0.25, 0.75], index=ASSETS)
    actual = _factor_model().compute_exposures_at_date(
        portfolio_weights=weights,
        date=DATE_1)
    # Market = 0.5*1.0 - 0.25*0.8 + 0.75*0.2 = 0.45
    # Rates = 0.5*0.1 - 0.25*(-0.2) + 0.75*1.1 = 0.925
    expected = pd.Series([0.45, 0.925], index=FACTORS)
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=0.0)


def test_exposures_satisfy_seeded_linearity_identity() -> None:
    # Seed 20260808. Linearity independently catches a transpose or accidental normalisation.
    rng = np.random.default_rng(20260808)
    weights_1 = pd.Series(rng.normal(size=3), index=ASSETS)
    weights_2 = pd.Series(rng.normal(size=3), index=ASSETS)
    scalar_1, scalar_2 = 1.7, -0.4
    model = _factor_model()
    combined = model.compute_exposures_at_date(
        portfolio_weights=scalar_1 * weights_1 + scalar_2 * weights_2,
        date=DATE_1)
    separate = (
        scalar_1 * model.compute_exposures_at_date(
            portfolio_weights=weights_1, date=DATE_1)
        + scalar_2 * model.compute_exposures_at_date(
            portfolio_weights=weights_2, date=DATE_1)
    )
    pd.testing.assert_series_equal(combined, separate, rtol=1e-12, atol=0.0)


def test_exposures_history_uses_asof_weights_and_leading_zero() -> None:
    weights = pd.DataFrame(
        [[0.5, 0.25, 0.25]],
        index=[pd.Timestamp('2024-01-03')],
        columns=ASSETS)
    actual = _factor_model().compute_exposures_history(portfolio_weights=weights)
    expected_nonzero = _loadings().T @ weights.iloc[0]
    expected = pd.DataFrame(
        [[0.0, 0.0], expected_nonzero.to_numpy(), expected_nonzero.to_numpy()],
        index=pd.DatetimeIndex([DATE_1, DATE_2, DATE_3]),
        columns=FACTORS)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=0.0)


def test_exposures_covariance_only_model_names_factor_loadings() -> None:
    with pytest.raises(ValueError, match="factor_loadings"):
        _model().compute_exposures_at_date(
            portfolio_weights=pd.Series([1.0], index=['A']),
            date=DATE_1)


def test_exposures_public_method_routes_through_strict_aligner() -> None:
    with pytest.raises(ValueError, match=r"portfolio_weights.*OUTSIDE"):
        _factor_model().compute_exposures_at_date(
            portfolio_weights=pd.Series([0.8, 0.2], index=['A', 'OUTSIDE']),
            date=DATE_1)


def test_benchmark_beta_loadings_match_seeded_sliced_constituent_reference() -> None:
    # Seed 20260808. The reference slices strict-subset constituents before multiplication.
    rng = np.random.default_rng(20260808)
    assets = pd.Index([f'A{idx}' for idx in range(6)])
    random_matrix = rng.normal(size=(6, 6))
    covar = pd.DataFrame(random_matrix @ random_matrix.T,
                         index=assets, columns=assets)
    constituents = pd.Index(['A0', 'A2', 'A5'])
    benchmark_constituent_weights = pd.Series([0.5, 0.3, 0.2], index=constituents)
    benchmark = benchmark_constituent_weights.reindex(assets).fillna(0.0)
    denominator = float(
        benchmark_constituent_weights
        @ covar.loc[constituents, constituents]
        @ benchmark_constituent_weights)
    expected = (
        covar.loc[assets, constituents] @ benchmark_constituent_weights / denominator)

    actual = RiskModel(covar={DATE_1: covar}).compute_benchmark_beta_loadings_at_date(
        benchmark_weights=benchmark,
        date=DATE_1)
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=0.0)


def test_benchmark_beta_identities_and_loading_linearity() -> None:
    benchmark = pd.Series([0.55, 0.35, 0.10], index=ASSETS)
    portfolio = pd.Series([0.40, 0.45, 0.15], index=ASSETS)
    model = _model()
    beta = model.compute_benchmark_beta_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1)
    benchmark_beta = model.compute_benchmark_beta_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=benchmark,
        date=DATE_1)
    loadings = model.compute_benchmark_beta_loadings_at_date(
        benchmark_weights=benchmark,
        date=DATE_1)
    covar = _covar()
    benchmark_variance = float(benchmark @ covar @ benchmark)
    active_beta = float((portfolio - benchmark) @ covar @ benchmark) / benchmark_variance

    assert benchmark_beta == 1.0
    np.testing.assert_allclose(beta - 1.0, active_beta, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(loadings @ portfolio, beta, rtol=1e-12, atol=0.0)


def test_benchmark_beta_history_uses_asof_portfolio_weights() -> None:
    benchmark = pd.Series([0.6, 0.4, 0.0], index=ASSETS)
    portfolio_history = pd.DataFrame(
        [[0.5, 0.4, 0.1]],
        index=[pd.Timestamp('2024-01-03')],
        columns=ASSETS)
    model = _model()
    actual = model.compute_benchmark_beta_history(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio_history)
    expected_nonzero = model.compute_benchmark_beta_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio_history.iloc[0],
        date=DATE_2)
    expected = pd.Series(
        [0.0, expected_nonzero, expected_nonzero],
        index=pd.DatetimeIndex([DATE_1, DATE_2, DATE_3]),
        name='Benchmark beta')
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=0.0)


def test_benchmark_beta_rejects_degenerate_benchmark_with_value() -> None:
    zero_benchmark = pd.Series(0.0, index=ASSETS)
    with pytest.raises(ValueError, match=r"benchmark variance.*0.0"):
        _model().compute_benchmark_beta_at_date(
            benchmark_weights=zero_benchmark,
            portfolio_weights=pd.Series([1.0, 0.0, 0.0], index=ASSETS),
            date=DATE_1)


def test_benchmark_beta_r8_error_names_joint_universe_remedy() -> None:
    benchmark = pd.Series({'A': 0.8, 'OUTSIDE': 0.2})
    portfolio = pd.Series({'A': 1.0})
    with pytest.raises(ValueError, match=r"OUTSIDE.*joint universe.*benchmark constituents"):
        _model().compute_benchmark_beta_at_date(
            benchmark_weights=benchmark,
            portfolio_weights=portfolio,
            date=DATE_1)
