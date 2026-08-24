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


def _consistent_factor_model() -> RiskModel:
    # Seed 20260808. Covariance is assembled independently as B F B' + D.
    rng = np.random.default_rng(20260808)
    assets = pd.Index([f'X{idx}' for idx in range(5)])
    factors = pd.Index([f'F{idx}' for idx in range(3)])
    loadings = pd.DataFrame(rng.normal(size=(5, 3)), index=assets, columns=factors)
    factor_root = rng.normal(size=(3, 3))
    factor_covar = pd.DataFrame(
        factor_root @ factor_root.T / 100.0,
        index=factors,
        columns=factors)
    residual_vars = pd.Series(rng.uniform(0.003, 0.012, size=5), index=assets)
    scales = {DATE_1: 1.0, DATE_2: 1.1, DATE_3: 0.9}
    factor_covars = {date: factor_covar * scale for date, scale in scales.items()}
    residuals = {date: residual_vars * scale for date, scale in scales.items()}
    covars = {
        date: loadings @ factor_covars[date] @ loadings.T + np.diag(residuals[date])
        for date in scales
    }
    return RiskModel(
        covar=covars,
        factor_loadings={date: loadings.copy() for date in scales},
        factor_covar=factor_covars,
        residual_vars=residuals)


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


def test_tre_group_loadings_match_independent_fractional_overlapping_reference() -> None:
    # Seed 20260824. The oracle evaluates all quadratic forms independently with einsum.
    rng = np.random.default_rng(20260824)
    assets = pd.Index([f'L{idx}' for idx in range(6)])
    covar_root = rng.normal(size=(len(assets), len(assets)))
    covar = pd.DataFrame(covar_root @ covar_root.T, index=assets, columns=assets)
    benchmark = pd.Series(rng.normal(size=len(assets)), index=assets)
    portfolio = pd.Series(rng.normal(size=len(assets)), index=assets)
    group_loadings = pd.DataFrame(
        rng.uniform(-0.5, 1.0, size=(len(assets), 3)),
        index=assets,
        columns=['Rates', 'Credit', 'Equity'],
    )
    group_loadings['All'] = 1.0

    actual = RiskModel(covar={DATE_1: covar}).compute_tre_by_group_loadings_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1,
        group_loadings=group_loadings,
    )

    active = (portfolio - benchmark).to_numpy(dtype=float)
    loadings = group_loadings.to_numpy(dtype=float)
    masked_active = active[:, None] * loadings
    covar_values = covar.to_numpy(dtype=float)
    expected = np.sqrt(np.concatenate((
        [np.einsum('i,ij,j->', active, covar_values, active)],
        np.einsum('ig,ij,jg->g', masked_active, covar_values, masked_active),
    )))
    assert actual.index.tolist() == ['Total', 'Rates', 'Credit', 'Equity', 'All']
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-12, atol=0.0)
    assert actual['All'] == actual['Total']
    assert actual[['Rates', 'Credit', 'Equity']].sum() != pytest.approx(actual['Total'])


def test_tre_group_loadings_one_hot_match_categorical_groups() -> None:
    benchmark = pd.Series([0.4, 0.3, 0.3], index=ASSETS)
    portfolio = pd.Series([0.6, 0.1, 0.4], index=ASSETS)
    groups = pd.Series(['Group 1', 'Group 1', 'Group 2'], index=ASSETS)
    loadings = pd.DataFrame({
        'Group 1': [1.0, 1.0, 0.0],
        'Group 2': [0.0, 0.0, 1.0],
    }, index=ASSETS)

    categorical = _model().compute_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1,
        group_data=groups)
    loading_based = _model().compute_tre_by_group_loadings_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1,
        group_loadings=loadings)

    pd.testing.assert_series_equal(loading_based, categorical)


def test_tre_group_loadings_align_missing_nan_and_extra_assets() -> None:
    benchmark = pd.Series(0.0, index=ASSETS)
    portfolio = pd.Series([0.2, -0.1, 0.3], index=ASSETS)
    loadings = pd.DataFrame(
        {'Partial': [0.5, np.nan, 9.0], 'Zero': [0.0, np.nan, 2.0]},
        index=['A', 'C', 'OUTSIDE'],
    )

    actual = _model().compute_tre_by_group_loadings_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1,
        group_loadings=loadings)

    expected_partial_weights = pd.Series([0.1, 0.0, 0.0], index=ASSETS)
    expected_partial = np.sqrt(expected_partial_weights @ _covar() @ expected_partial_weights)
    assert actual['Partial'] == pytest.approx(expected_partial)
    assert actual['Zero'] == 0.0


@pytest.mark.parametrize(
    ('group_loadings', 'message'),
    [
        (pd.Series([1.0], index=['A']), 'pd.DataFrame'),
        (pd.DataFrame([[1.0], [1.0]], index=['A', 'A'], columns=['Group']),
         'duplicate assets'),
        (pd.DataFrame([[1.0, 0.0]], index=['A'], columns=['Group', 'Group']),
         'duplicate groups'),
        (pd.DataFrame({'Group': ['bad']}, index=['A']), 'numeric values'),
        (pd.DataFrame({'Group': [np.inf]}, index=['A']), 'infinite values'),
    ],
)
def test_tre_group_loadings_reject_invalid_matrices(group_loadings, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _model().compute_tre_by_group_loadings_at_date(
            benchmark_weights=pd.Series(0.0, index=ASSETS),
            portfolio_weights=pd.Series(0.0, index=ASSETS),
            date=DATE_1,
            group_loadings=group_loadings)


def test_tre_group_loadings_reject_total_label_collision() -> None:
    with pytest.raises(ValueError, match='total_column.*duplicates'):
        _model().compute_tre_by_group_loadings_at_date(
            benchmark_weights=pd.Series(0.0, index=ASSETS),
            portfolio_weights=pd.Series(0.0, index=ASSETS),
            date=DATE_1,
            group_loadings=pd.DataFrame({'Total': 1.0}, index=ASSETS))


def test_tre_group_loadings_zero_active_weights_are_exact_zeros() -> None:
    weights = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
    actual = _model().compute_tre_by_group_loadings_at_date(
        benchmark_weights=weights,
        portfolio_weights=weights,
        date=DATE_1,
        group_loadings=pd.DataFrame({'All': 1.0, 'None': 0.0}, index=ASSETS))
    expected = pd.Series({'Total': 0.0, 'All': 0.0, 'None': 0.0})
    pd.testing.assert_series_equal(actual, expected)


def test_tre_group_loadings_route_through_strict_weight_aligner() -> None:
    with pytest.raises(ValueError, match=r"portfolio_weights.*OUTSIDE"):
        _model().compute_tre_by_group_loadings_at_date(
            benchmark_weights=pd.Series([1.0], index=['A']),
            portfolio_weights=pd.Series([0.8, 0.2], index=['A', 'OUTSIDE']),
            date=DATE_1,
            group_loadings=pd.DataFrame({'Group': 1.0}, index=ASSETS))


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


def test_mpd_tracking_error_characterisation_goldens_and_risk_model_parity() -> None:
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
    expected_total = pd.Series(
        [0.043513676445772684, 0.045867442312087316,
         0.044379020046626944, 0.046352374699966455],
        index=covar_dates.rename(None),
        name='Tracking error')
    # rtol=1e-12, not 1e-15: the golden literals were produced on one machine, and the last
    # bits of the covariance products are BLAS/pandas-build dependent (observed 1.5e-15
    # relative drift on a different build, with legacy and delegated still agreeing to 0.0).
    pd.testing.assert_series_equal(legacy_total, expected_total, rtol=1e-12, atol=0.0)
    actual_total = model.compute_tre_history(
        benchmark_weights=benchmark_weights,
        portfolio_weights=strategy_weights,
        strict=False)
    pd.testing.assert_series_equal(actual_total, legacy_total, rtol=1e-12, atol=0.0)

    legacy_groups = legacy.compute_tracking_error_implied_by_covar(
        is_grouped=True,
        group_data=universe.group_data,
        group_order=universe.group_order)
    expected_groups = pd.DataFrame(
        [
            [0.043513676445772684, 0.05305230594209256, 0.015388323704411528,
             0.02343475902504267, 0.017030186248985182],
            [0.045867442312087316, 0.05592204063369915, 0.0162207174259667,
             0.024702404978773917, 0.017951392507890854],
            [0.044379020046626944, 0.04576692567903564, 0.0205052974216559,
             0.02590810091282049, 0.018827579299251294],
            [0.046352374699966455, 0.04780199485503095, 0.021417084659017645,
             0.0270601288630048, 0.01966476523040211],
        ],
        index=covar_dates.rename(None),
        columns=['Total', 'Equities', 'Bonds', 'Commodities', 'Alternatives'])
    pd.testing.assert_frame_equal(legacy_groups, expected_groups, rtol=1e-12, atol=0.0)
    actual_groups = model.compute_tre_history(
        benchmark_weights=benchmark_weights,
        portfolio_weights=strategy_weights,
        group_data=universe.group_data,
        strict=False)
    pd.testing.assert_frame_equal(actual_groups, legacy_groups, rtol=1e-12, atol=0.0)


def test_mpd_delegation_uses_asof_weights_for_off_grid_weight_dates() -> None:
    # Behaviour change vs qis 5.6.x, intended. The legacy implementation reindexed the
    # weight history to the exact covar dates before ffill, so weight rows dated off the
    # covar grid were dropped entirely: this setup returned 0.0 tracking error on every
    # date under the 5.6.1 implementation (verified against the released code). The
    # delegated as-of selection uses the latest weights known at each covar date.
    rng = np.random.default_rng(20260808)
    dates = pd.bdate_range('2024-01-01', '2024-03-29')
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal((len(dates), 3)), axis=0)),
        index=dates,
        columns=ASSETS)
    covar_dates = dates[[20, 40, 60]]
    covar_dict = {date: _covar(scale)
                  for date, scale in zip(covar_dates, [1.0, 1.1, 0.9])}
    weight_dates = covar_dates[[0, 1]] + pd.offsets.BDay(1)  # off the covar grid
    strategy_weights = pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]], index=weight_dates, columns=ASSETS)
    benchmark_weights = pd.DataFrame(
        [[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]], index=weight_dates, columns=ASSETS)
    strategy = backtest_model_portfolio(
        prices=prices, weights=strategy_weights, ticker='Off-grid strategy')
    benchmark = backtest_model_portfolio(
        prices=prices, weights=benchmark_weights, ticker='Off-grid benchmark')

    actual = MultiPortfolioData(
        portfolio_datas=[strategy, benchmark],
        covar_dict=covar_dict).compute_tracking_error_implied_by_covar()
    expected = RiskModel(covar=covar_dict).compute_tre_history(
        benchmark_weights=benchmark_weights,
        portfolio_weights=strategy_weights,
        strict=False)
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=0.0)
    assert actual.iloc[0] == 0.0  # before the first weight observation
    assert bool((actual.iloc[1:] > 0.0).all())  # the 5.6.1 implementation reported 0.0 here


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


def test_tre_decomposition_matches_covar_te_for_independently_assembled_model() -> None:
    model = _consistent_factor_model()
    assets = model.covar[DATE_1].index
    benchmark = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=assets)
    portfolio = pd.Series([0.45, 0.10, 0.35, -0.05, 0.15], index=assets)
    decomposition = model.compute_tre_decomposition_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1)
    covar_te = model.compute_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1)
    np.testing.assert_allclose(
        decomposition['tracking_error'], covar_te, rtol=1e-12, atol=0.0)


def test_tre_decomposition_history_has_contract_columns() -> None:
    model = _consistent_factor_model()
    assets = model.covar[DATE_1].index
    benchmark = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=assets)
    portfolio = pd.Series([0.45, 0.10, 0.35, -0.05, 0.15], index=assets)
    actual = model.compute_tre_decomposition_history(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio)
    assert actual.columns.tolist() == ['tracking_error', 'factor_te', 'residual_te']
    for date in model.dates:
        expected = model.compute_tre_at_date(
            benchmark_weights=benchmark,
            portfolio_weights=portfolio,
            date=date)
        np.testing.assert_allclose(
            actual.loc[date, 'tracking_error'], expected, rtol=1e-12, atol=0.0)


def test_marginal_tre_euler_sum_and_factor_split_on_long_short_weights() -> None:
    model = _consistent_factor_model()
    assets = model.covar[DATE_1].index
    benchmark = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=assets)
    portfolio = pd.Series([0.55, -0.10, 0.40, -0.05, 0.20], index=assets)
    actual = model.compute_marginal_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1)
    tracking_error = model.compute_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1)
    np.testing.assert_allclose(actual['mcte'].sum(), tracking_error, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        actual['mcte_systematic'] + actual['mcte_residual'],
        actual['mcte'],
        rtol=1e-12,
        atol=1e-16)


def test_marginal_tre_zero_active_weights_returns_all_zero() -> None:
    model = _consistent_factor_model()
    weights = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10],
                        index=model.covar[DATE_1].index)
    actual = model.compute_marginal_tre_at_date(
        benchmark_weights=weights,
        portfolio_weights=weights,
        date=DATE_1)
    assert bool((actual == 0.0).all().all())


def test_marginal_tre_groups_are_additive_to_total_with_unassigned_visible() -> None:
    model = _consistent_factor_model()
    assets = model.covar[DATE_1].index
    benchmark = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=assets)
    portfolio = pd.Series([0.55, -0.10, 0.40, -0.05, 0.20], index=assets)
    groups = pd.Series(['Group 1', 'Group 1', 'Group 2', 'Group 2'], index=assets[:4])
    actual = model.compute_marginal_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1,
        group_data=groups)
    assert actual.index.tolist() == ['Total', 'Group 1', 'Group 2', UNASSIGNED_GROUP]
    pd.testing.assert_series_equal(
        actual.drop(index='Total').sum(axis=0),
        actual.loc['Total'],
        rtol=1e-12,
        atol=1e-16,
        check_names=False)


def test_decomposition_covariance_only_model_names_missing_factor_field() -> None:
    weights = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
    with pytest.raises(ValueError, match="factor_loadings"):
        _model().compute_tre_decomposition_at_date(
            benchmark_weights=weights,
            portfolio_weights=weights,
            date=DATE_1)


def test_marginal_tre_covariance_only_model_returns_total_column_only() -> None:
    benchmark = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
    portfolio = pd.Series([0.4, 0.4, 0.2], index=ASSETS)
    actual = _model().compute_marginal_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=portfolio,
        date=DATE_1)
    assert actual.columns.tolist() == ['mcte']


def test_decomposition_public_method_routes_through_strict_aligner() -> None:
    with pytest.raises(ValueError, match=r"portfolio_weights.*OUTSIDE"):
        _consistent_factor_model().compute_tre_decomposition_at_date(
            benchmark_weights=pd.Series({'X0': 1.0}),
            portfolio_weights=pd.Series({'X0': 0.8, 'OUTSIDE': 0.2}),
            date=DATE_1)
