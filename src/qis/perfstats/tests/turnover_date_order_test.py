"""Regression tests for chronological turnover calculations.

Turnover differences represent changes between dated observations, so physical row order must not
change their values. These tests cover every public computation mode and reject ambiguous dated
inputs before arithmetic while preserving caller-owned objects.
"""

import pandas as pd
import pytest

from qis.perfstats.turnover import TurnoverComputationType, compute_turnover


def _turnover_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Return chronological inputs with independently calculable turnover."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", name="date")
    units = pd.DataFrame({"A": [1.0, 3.0, 2.0]}, index=index)
    unit_notional = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=index)
    nav = pd.Series([100.0, 200.0, 100.0], index=index, name="Portfolio")
    input_weights = pd.DataFrame({"A": [0.0, 1.0, 0.0]}, index=index)
    vols = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=index)
    return units, unit_notional, nav, input_weights, vols


def _compute(
    computation_type: TurnoverComputationType,
    units: pd.DataFrame,
    unit_notional: pd.DataFrame,
    nav: pd.Series,
    input_weights: pd.DataFrame,
    vols: pd.DataFrame,
) -> pd.DataFrame:
    """Call the public turnover engine with the complete deterministic input set."""
    return compute_turnover(
        computation_type=computation_type,
        units=units,
        unit_notional=unit_notional,
        nav=nav,
        input_weights=input_weights,
        vols=vols,
    )


def _assert_unchanged(
    value: pd.DataFrame | pd.Series,
    original: pd.DataFrame | pd.Series,
) -> None:
    """Compare one caller-owned pandas input with its snapshot."""
    if isinstance(value, pd.Series):
        assert isinstance(original, pd.Series)
        pd.testing.assert_series_equal(value, original)
    else:
        assert isinstance(original, pd.DataFrame)
        pd.testing.assert_frame_equal(value, original)


@pytest.mark.parametrize(
    ("computation_type", "expected_values"),
    (
        (TurnoverComputationType.TARGET_WEIGHTS, [None, 1.0, 1.0]),
        (TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS, [None, 0.2, 0.3]),
        (TurnoverComputationType.EXECUTED_NOTIONAL_NAV, [None, 0.2, 0.3]),
        (TurnoverComputationType.EXECUTED_NOTIONAL_GROSS, [None, 2.0 / 3.0, 0.5]),
    ),
    ids=("target", "volatility-normalized", "executed-nav", "executed-gross"),
)
def test_compute_turnover_uses_chronological_differences(
    computation_type: TurnoverComputationType,
    expected_values: list[float | None],
) -> None:
    """Make each turnover convention invariant to independent dated-row permutations."""
    units, unit_notional, nav, input_weights, vols = _turnover_inputs()
    # Distinct permutations prevent pandas alignment from masking row-order dependence.
    units = units.iloc[[0, 2, 1]]
    unit_notional = unit_notional.iloc[[1, 0, 2]]
    nav = nav.iloc[[2, 1, 0]]
    input_weights = input_weights.iloc[[2, 0, 1]]
    vols = vols.iloc[[1, 2, 0]]
    originals = tuple(
        value.copy(deep=True) for value in (units, unit_notional, nav, input_weights, vols)
    )

    actual = _compute(
        computation_type,
        units,
        unit_notional,
        nav,
        input_weights,
        vols,
    )

    # Keep the chronological oracle literal and independent of production normalization.
    expected = pd.DataFrame(
        {"A": expected_values},
        index=pd.DatetimeIndex(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            name="date",
        ),
        dtype=float,
    )
    pd.testing.assert_frame_equal(actual, expected)
    for value, original in zip((units, unit_notional, nav, input_weights, vols), originals):
        _assert_unchanged(value, original)


@pytest.mark.parametrize("invalid_kind", ("duplicate", "nat"))
@pytest.mark.parametrize(
    ("input_name", "computation_type"),
    (
        ("input_weights", TurnoverComputationType.TARGET_WEIGHTS),
        ("vols", TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS),
        ("units", TurnoverComputationType.EXECUTED_NOTIONAL_NAV),
        ("unit_notional", TurnoverComputationType.EXECUTED_NOTIONAL_GROSS),
        ("nav", TurnoverComputationType.EXECUTED_NOTIONAL_NAV),
    ),
)
def test_compute_turnover_rejects_ambiguous_dated_inputs_without_mutation(
    input_name: str,
    computation_type: TurnoverComputationType,
    invalid_kind: str,
) -> None:
    """Reject duplicate or missing dates on each applicable input before computing turnover."""
    units, unit_notional, nav, input_weights, vols = _turnover_inputs()
    if invalid_kind == "duplicate":
        invalid_index = pd.DatetimeIndex(
            [units.index[0], units.index[0], units.index[2]],
            name="date",
        )
        message = f"{input_name} index must not contain duplicate dates"
    else:
        invalid_index = pd.DatetimeIndex(
            [units.index[0], pd.NaT, units.index[2]],
            name="date",
        )
        message = f"{input_name} index must not contain NaT"
    if input_name == "units":
        units.index = invalid_index
    elif input_name == "unit_notional":
        unit_notional.index = invalid_index
    elif input_name == "nav":
        nav.index = invalid_index
    elif input_name == "input_weights":
        input_weights.index = invalid_index
    elif input_name == "vols":
        vols.index = invalid_index
    else:
        raise AssertionError(f"unsupported turnover input: {input_name}")
    originals = tuple(
        value.copy(deep=True) for value in (units, unit_notional, nav, input_weights, vols)
    )

    with pytest.raises(ValueError, match=message):
        _compute(
            computation_type,
            units,
            unit_notional,
            nav,
            input_weights,
            vols,
        )

    for value, original in zip((units, unit_notional, nav, input_weights, vols), originals):
        _assert_unchanged(value, original)
