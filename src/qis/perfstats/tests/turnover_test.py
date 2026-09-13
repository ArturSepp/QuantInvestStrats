import pandas as pd
import pytest

from qis.perfstats.turnover import TurnoverComputationType, compute_turnover
from qis.portfolio.portfolio_data import AttributionMetric, PortfolioData


def _turnover_inputs():
    index = pd.bdate_range('2025-01-06', periods=3)
    columns = ['A', 'B']
    units = pd.DataFrame(
        [[1.0, 1.0], [2.0, 0.0], [1.0, 1.0]],
        index=index,
        columns=columns,
    )
    unit_notional = pd.DataFrame(
        [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
        index=index,
        columns=columns,
    )
    nav = pd.Series(100.0, index=index, name='Portfolio')
    input_weights = pd.DataFrame(
        [[0.1, 0.2], [0.2, 0.0], [0.1, 0.2]],
        index=index,
        columns=columns,
    )
    return units, unit_notional, nav, input_weights


def test_compute_turnover_supports_target_and_executed_conventions() -> None:
    units, unit_notional, nav, input_weights = _turnover_inputs()
    expected_nav = pd.DataFrame(
        [[None, None], [0.1, 0.2], [0.1, 0.2]],
        index=units.index,
        columns=units.columns,
        dtype=float,
    )
    expected_gross = pd.DataFrame(
        [[None, None], [0.5, 1.0], [1.0 / 3.0, 2.0 / 3.0]],
        index=units.index,
        columns=units.columns,
        dtype=float,
    )
    expected_target = input_weights.diff().abs()

    actual_default = compute_turnover(
        units=units,
        unit_notional=unit_notional,
        nav=nav,
    )
    actual_gross = compute_turnover(
        computation_type=TurnoverComputationType.EXECUTED_NOTIONAL_GROSS,
        units=units,
        unit_notional=unit_notional,
    )
    actual_target = compute_turnover(
        computation_type=TurnoverComputationType.TARGET_WEIGHTS,
        input_weights=input_weights,
    )

    pd.testing.assert_frame_equal(actual_default, expected_nav)
    pd.testing.assert_frame_equal(actual_gross, expected_gross)
    pd.testing.assert_frame_equal(actual_target, expected_target)


def test_compute_turnover_supports_volatility_normalized_weights() -> None:
    _, _, _, input_weights = _turnover_inputs()
    vols = pd.DataFrame(
        [[0.20, 0.10], [0.25, 0.12], [0.30, 0.15]],
        index=input_weights.index,
        columns=input_weights.columns,
    )
    expected = pd.DataFrame(
        [[None, None], [0.025, 0.024], [0.030, 0.030]],
        index=input_weights.index,
        columns=input_weights.columns,
        dtype=float,
    )

    actual = compute_turnover(
        computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
        input_weights=input_weights,
        vols=vols,
    )
    portfolio = PortfolioData(
        nav=pd.Series(100.0, index=input_weights.index, name='Portfolio'),
        prices=pd.DataFrame(1.0, index=input_weights.index, columns=input_weights.columns),
        weights=input_weights,
        input_weights=input_weights,
    )
    actual_from_portfolio = portfolio.get_turnover(
        turnover_computation_type=(
            TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS
        ),
        vols=vols,
        roll_period=None,
        add_total=False,
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_from_portfolio, expected)


def test_volatility_normalized_turnover_requires_aligned_annualized_vols() -> None:
    _, _, _, input_weights = _turnover_inputs()

    with pytest.raises(TypeError, match='vols must be a pandas DataFrame'):
        compute_turnover(
            computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
            input_weights=input_weights,
        )

    misaligned_vols = pd.DataFrame(
        0.20,
        index=input_weights.index,
        columns=list(reversed(input_weights.columns)),
    )
    with pytest.raises(ValueError, match='columns must exactly match'):
        compute_turnover(
            computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
            input_weights=input_weights,
            vols=misaligned_vols,
        )

    misaligned_vols = pd.DataFrame(
        0.20,
        index=input_weights.index.shift(1, freq='B'),
        columns=input_weights.columns,
    )
    with pytest.raises(ValueError, match='index must exactly match'):
        compute_turnover(
            computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
            input_weights=input_weights,
            vols=misaligned_vols,
        )

    negative_vols = pd.DataFrame(
        0.20,
        index=input_weights.index,
        columns=input_weights.columns,
    )
    negative_vols.iloc[1, 0] = -0.01
    with pytest.raises(ValueError, match='must not contain negative'):
        compute_turnover(
            computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
            input_weights=input_weights,
            vols=negative_vols,
        )

    warmup_vols = negative_vols.abs()
    warmup_vols.iloc[1, 0] = float('nan')
    actual = compute_turnover(
        computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
        input_weights=input_weights,
        vols=warmup_vols,
    )
    assert pd.isna(actual.iloc[1, 0])


def test_portfolio_turnover_uses_portfolio_default_and_legacy_mapping() -> None:
    units, unit_notional, nav, input_weights = _turnover_inputs()
    qis_default = PortfolioData(
        nav=nav,
        prices=pd.DataFrame(1.0, index=units.index, columns=units.columns),
        weights=input_weights,
        input_weights=input_weights,
        units=units,
        turnover_unit_notional=unit_notional,
    )
    portfolio = PortfolioData(
        nav=nav,
        prices=pd.DataFrame(1.0, index=units.index, columns=units.columns),
        weights=input_weights,
        input_weights=input_weights,
        units=units,
        turnover_unit_notional=unit_notional,
        turnover_computation_type=TurnoverComputationType.EXECUTED_NOTIONAL_GROSS,
    )
    expected_gross = compute_turnover(
        computation_type=TurnoverComputationType.EXECUTED_NOTIONAL_GROSS,
        units=units,
        unit_notional=unit_notional,
    )
    expected_nav = compute_turnover(
        computation_type=TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
        units=units,
        unit_notional=unit_notional,
        nav=nav,
    )

    actual_nav = qis_default.get_turnover(roll_period=None, add_total=False)
    actual_gross = portfolio.get_turnover(roll_period=None, add_total=False)
    with pytest.warns(DeprecationWarning):
        actual_target = portfolio.get_turnover(
            roll_period=None,
            add_total=False,
            is_unit_based_traded_volume=False,
        )

    pd.testing.assert_frame_equal(actual_nav, expected_nav)
    pd.testing.assert_frame_equal(actual_gross, expected_gross)
    pd.testing.assert_frame_equal(actual_target, input_weights.diff().abs())


def test_volatility_normalized_attribution_excludes_executed_position_drift() -> None:
    units, unit_notional, nav, _ = _turnover_inputs()
    constant_weights = pd.DataFrame(
        [[0.60, 0.40]] * len(units.index),
        index=units.index,
        columns=units.columns,
    )
    vols = pd.DataFrame(0.20, index=units.index, columns=units.columns)
    portfolio = PortfolioData(
        nav=nav,
        prices=unit_notional,
        weights=constant_weights,
        input_weights=constant_weights,
        units=units,
    )

    attribution = portfolio.get_performance_attribution_data(
        attribution_metric=AttributionMetric.VOL_ADJUSTED_TURNOVER,
        vols=vols,
    )

    pd.testing.assert_series_equal(
        attribution,
        pd.Series(0.0, index=units.columns),
    )


def test_compute_turnover_warns_and_returns_nan_for_zero_gross_exposure() -> None:
    units, unit_notional, _, _ = _turnover_inputs()
    units.loc[units.index[1]] = 0.0

    with pytest.warns(RuntimeWarning, match='gross exposure is zero'):
        actual = compute_turnover(
            computation_type=TurnoverComputationType.EXECUTED_NOTIONAL_GROSS,
            units=units,
            unit_notional=unit_notional,
        )

    assert actual.loc[units.index[1]].isna().all()
