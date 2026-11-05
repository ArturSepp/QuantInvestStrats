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


@pytest.mark.parametrize(
    'computation_type',
    (
        TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
        TurnoverComputationType.EXECUTED_NOTIONAL_GROSS,
    ),
)
@pytest.mark.parametrize('invalid_notional', (-1.0, float('inf'), float('-inf')))
def test_compute_turnover_rejects_invalid_unit_notionals_without_mutation(
        computation_type: TurnoverComputationType,
        invalid_notional: float,
        ) -> None:
    """Reject invalid observed notionals while leaving the caller's mixed panel unchanged."""
    units, unit_notional, nav, _ = _turnover_inputs()
    unit_notional = unit_notional.astype('Float64')
    unit_notional.iloc[0, 1] = pd.NA
    unit_notional.iloc[1, 0] = invalid_notional
    original = unit_notional.copy(deep=True)

    with pytest.raises(ValueError, match='unit_notional must not contain'):
        compute_turnover(
            computation_type=computation_type,
            units=units,
            unit_notional=unit_notional,
            nav=nav,
        )

    pd.testing.assert_frame_equal(unit_notional, original)


@pytest.mark.filterwarnings('error')
def test_compute_turnover_rejects_notional_before_zero_denominator_warning() -> None:
    """Validate the numerator before a later zero denominator can emit its warning."""
    units, unit_notional, nav, _ = _turnover_inputs()
    unit_notional.iloc[1, 0] = -1.0
    nav.iloc[1] = 0.0

    with pytest.raises(ValueError, match='unit_notional must not contain'):
        compute_turnover(
            computation_type=TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
            units=units,
            unit_notional=unit_notional,
            nav=nav,
        )


@pytest.mark.parametrize('invalid_volatility', (float('inf'), float('-inf')))
def test_compute_turnover_rejects_infinite_volatility_without_mutation(
        invalid_volatility: float,
        ) -> None:
    """Reject either signed infinity while retaining nullable warm-up observations."""
    _, _, _, input_weights = _turnover_inputs()
    vols = pd.DataFrame(0.20, index=input_weights.index, columns=input_weights.columns)
    vols = vols.astype('Float64')
    vols.iloc[0, 1] = pd.NA
    vols.iloc[1, 0] = invalid_volatility
    original = vols.copy(deep=True)

    with pytest.raises(ValueError, match='vols must not contain'):
        compute_turnover(
            computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
            input_weights=input_weights,
            vols=vols,
        )

    pd.testing.assert_frame_equal(vols, original)


def test_compute_turnover_preserves_nullable_missing_and_zero_scales() -> None:
    """Keep missing scales unavailable and accept exact zero in both affected calculations."""
    units, unit_notional, nav, input_weights = _turnover_inputs()
    unit_notional = unit_notional.astype('Float64')
    unit_notional.iloc[1, 0] = 0.0
    unit_notional.iloc[1, 1] = pd.NA
    vols = pd.DataFrame(0.20, index=input_weights.index, columns=input_weights.columns)
    vols = vols.astype('Float64')
    vols.iloc[1, 0] = 0.0
    vols.iloc[1, 1] = pd.NA

    actual_notional = compute_turnover(
        computation_type=TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
        units=units,
        unit_notional=unit_notional,
        nav=nav,
    )
    actual_volatility = compute_turnover(
        computation_type=TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
        input_weights=input_weights,
        vols=vols,
    )

    expected_notional = pd.DataFrame(
        [[pd.NA, pd.NA], [0.0, pd.NA], [0.1, 0.2]],
        index=units.index,
        columns=units.columns,
        dtype='Float64',
    )
    expected_volatility = pd.DataFrame(
        [[pd.NA, pd.NA], [0.0, pd.NA], [0.02, 0.04]],
        index=input_weights.index,
        columns=input_weights.columns,
        dtype='Float64',
    )
    pd.testing.assert_frame_equal(actual_notional, expected_notional)
    pd.testing.assert_frame_equal(actual_volatility, expected_volatility)


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


def test_portfolio_turnover_requires_explicit_notional_for_negative_prices() -> None:
    """Require a valid contract notional when return prices are not valid unit notionals."""
    units, unit_notional, nav, input_weights = _turnover_inputs()
    negative_prices = unit_notional.multiply(-1.0)
    default_notional = PortfolioData(
        nav=nav,
        prices=negative_prices,
        weights=input_weights,
        input_weights=input_weights,
        units=units,
    )
    explicit_notional = PortfolioData(
        nav=nav,
        prices=negative_prices,
        weights=input_weights,
        input_weights=input_weights,
        units=units,
        turnover_unit_notional=unit_notional,
    )

    with pytest.raises(ValueError, match='unit_notional must not contain'):
        default_notional.get_turnover(roll_period=None, add_total=False)

    actual = explicit_notional.get_turnover(roll_period=None, add_total=False)
    expected = pd.DataFrame(
        [[None, None], [0.1, 0.2], [0.1, 0.2]],
        index=units.index,
        columns=units.columns,
        dtype=float,
    )
    pd.testing.assert_frame_equal(actual, expected)


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


@pytest.mark.filterwarnings('error::RuntimeWarning')
@pytest.mark.parametrize('computation_type', tuple(TurnoverComputationType))
@pytest.mark.parametrize(
    ('dtype', 'holdings'),
    [('uint8', [10, 5, 8]), ('UInt8', [10, 5, 8]), ('Int8', [10, 5, 8]),
     ('uint64', [10, 5, 8]), ('UInt64', [10, 5, 8]), ('int8', [10, 5, 8]),
     ('Int8', [1, -127, 1])],
)
def test_compute_turnover_promotes_integer_holdings_and_targets(
        computation_type: TurnoverComputationType,
        dtype: str,
        holdings: list[int],
        ) -> None:
    """Value whole-unit trades identically through the engine and PortfolioData."""
    index = pd.date_range('2025-01-01', periods=3, tz='UTC', name='date')
    units = pd.DataFrame({'A': holdings}, index=index, dtype=dtype)
    targets = pd.DataFrame({'A': [1, 0, 1]}, index=index, dtype=dtype)
    notionals = pd.DataFrame(2.0, index=index, columns=['A'])
    nav = pd.Series(100.0, index=index, name='Portfolio')
    vols = pd.DataFrame(0.2, index=index, columns=['A'])
    originals = [value.copy(deep=True) for value in (units, targets, notionals, vols)]
    portfolio = PortfolioData(
        nav=nav, prices=notionals, units=units, input_weights=targets,
        turnover_unit_notional=notionals,
    )

    actual = compute_turnover(
        computation_type, units=units, unit_notional=notionals, nav=nav,
        input_weights=targets, vols=vols,
    )
    delegated = portfolio.get_turnover(
        turnover_computation_type=computation_type, vols=vols,
        roll_period=None, add_total=False,
    )

    # Python integers give mathematical changes, independently of pandas' storage arithmetic.
    # The signed fixture also hits abs(-128), which cannot remain in an Int8 result.
    expected_values = [None]
    for previous, current in zip(holdings, holdings[1:]):
        if computation_type == TurnoverComputationType.TARGET_WEIGHTS:
            value = 1
        elif computation_type == TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS:
            value = 0.2
        elif computation_type == TurnoverComputationType.EXECUTED_NOTIONAL_NAV:
            value = abs(current - previous) * 2 / 100
        else:
            value = abs(current - previous) / abs(current)
        expected_values.append(value)
    nullable = dtype[0].isupper()
    expected_dtype = (
        'Int64' if nullable and computation_type == TurnoverComputationType.TARGET_WEIGHTS
        else 'Float64' if nullable else 'float64'
    )
    expected = pd.DataFrame(
        {'A': expected_values}, index=index.copy(deep=True), dtype=expected_dtype,
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(delegated, expected)
    for value, original in zip((units, targets, notionals, vols), originals):
        pd.testing.assert_frame_equal(value, original)


@pytest.mark.parametrize('dtype', ['uint8', 'UInt8', 'Int8'])
@pytest.mark.parametrize(
    ('computation_type', 'expected_values'),
    [(TurnoverComputationType.EXECUTED_NOTIONAL_NAV, [None, 0.5, 0.3]),
     (TurnoverComputationType.EXECUTED_NOTIONAL_GROSS, [None, 1.0, 0.375])],
)
def test_compute_turnover_promotes_integer_notional_products(
        dtype: str,
        computation_type: TurnoverComputationType,
        expected_values: list[float | None],
        ) -> None:
    """Use widened holdings for valuation too, not just for their differences."""
    index = pd.date_range('2025-01-01', periods=3)
    units = pd.DataFrame({'A': [10, 5, 8]}, index=index, dtype=dtype)
    notionals = pd.DataFrame(100, index=index, columns=['A'], dtype=dtype)
    actual = compute_turnover(
        computation_type, units=units, unit_notional=notionals,
        nav=pd.Series(1000.0, index=index),
    )
    # A five-unit sale trades 500; the remaining book is also 500, not uint8(500)=244.
    expected = pd.DataFrame(
        {'A': expected_values}, index=index,
        dtype='Float64' if dtype[0].isupper() else 'float64',
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_compute_turnover_preserves_mixed_storage_and_missing_observations() -> None:
    """Promote columns positionally without casting float neighbors or filling missing trades."""
    index = pd.date_range('2025-01-01', periods=3, name='date')
    specifications = [
        (('asset', 'same'), [10, 5, 8], 'UInt8', [None, 5, 3], 'Int64'),
        (('asset', 'same'), [0.5, 0.25, 0.75], 'float32', [None, 0.25, 0.5], 'float32'),
        (('asset', 'nullable'), [0.5, 0.25, 0.75], 'Float32', [None, 0.25, 0.5], 'Float32'),
        (('mask', 'gap'), [1, None, 0], 'Int8', [None, None, None], 'Int64'),
        (('mask', 'absent'), [None] * 3, 'UInt8', [None] * 3, 'Int64'),
        (('storage', 'sparse'), [1, 0, 1], pd.SparseDtype('uint8', 0),
         [float('nan'), 1.0, 1.0], pd.SparseDtype('float64', 0)),
    ]
    data = pd.concat([
        pd.Series(values, index=index, name=name, dtype=dtype)
        for name, values, dtype, _, _ in specifications
    ], axis=1)
    original = data.copy(deep=True)
    expected = pd.concat([
        pd.Series(values, index=index.copy(deep=True), name=name, dtype=dtype)
        for name, _, _, values, dtype in specifications
    ], axis=1)

    actual = compute_turnover(TurnoverComputationType.TARGET_WEIGHTS, input_weights=data)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize('dtype', ['int64', 'Int64', 'uint64', 'UInt64'])
def test_compute_turnover_preserves_wide_integer_unit_changes(dtype: str) -> None:
    """Do not erase a one-unit change by converting large integer levels to float first."""
    index = pd.date_range('2025-01-01', periods=3)
    targets = pd.DataFrame({'A': [2**53, 2**53 + 1, 2**53 + 2]}, index=index, dtype=dtype)
    expected = pd.DataFrame(
        {'A': [None, 1, 1]}, index=index,
        dtype='Int64' if dtype[0].isupper() else 'float64',
    )

    actual = compute_turnover(TurnoverComputationType.TARGET_WEIGHTS, input_weights=targets)

    pd.testing.assert_frame_equal(actual, expected)
