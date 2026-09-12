import pandas as pd
import pytest

from qis.perfstats.turnover import TurnoverComputationType, compute_turnover
from qis.portfolio.portfolio_data import PortfolioData


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


def test_compute_turnover_supports_all_three_conventions() -> None:
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
