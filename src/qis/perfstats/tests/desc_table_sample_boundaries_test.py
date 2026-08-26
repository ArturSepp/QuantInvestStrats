"""Regression coverage for descriptive-table sample-size boundaries.

``compute_desc_table`` combines statistics with different minimum useful sample sizes. A single
observation defines its mean and extrema but not its sample standard deviation; two observations
define the displayed sample moments but remain too small for a stable normality result. These
tests require each statistic to decide eligibility independently, without warnings or loss of
still-defined neighboring results.

The central fixture places one, two, seven, eight, nineteen, twenty, and zero observations in one
twenty-month panel. Its samples are symmetric around one and have independently calculated sums
of squared deviations, sign counts, moments, quantiles, last values, and ranks. Both ordinary
``float64``/``np.nan`` and nullable ``Float64``/``pd.NA`` representations must produce the exact
same display table. Separate controls cover zero-row rejection, a valid dated zero-column panel,
named Series/DataFrame consistency, monthly annualization, formatting, warnings, and ownership.
"""

import warnings
from typing import Protocol, cast

import numpy as np
import pandas as pd
import pytest

# qis
import qis.perfstats.desc_table as desc_table_module
from qis.perfstats.desc_table import DescTableType


class _DescTableModuleProtocol(Protocol):
    """Typed test-side interface for the public function exercised below."""

    def compute_desc_table(
        self,
        *,
        df: pd.DataFrame | pd.Series,
        desc_table_type: DescTableType,
        var_format: str = "{:.2f}",
        annualize_vol: bool = False,
        is_add_tstat: bool = False,
        norm_variable_display_type: str = "{:.1f}",
    ) -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=20, freq="ME")

_ONE_OBSERVATION = "One Observation"
_TWO_OBSERVATIONS = "Two Observations"
_SEVEN_OBSERVATIONS = "Seven Observations"
_EIGHT_OBSERVATIONS = "Eight Observations"
_NINETEEN_OBSERVATIONS = "Nineteen Observations"
_TWENTY_OBSERVATIONS = "Twenty Observations"
_ALL_MISSING = "All Missing"

_ASSETS = (
    _ONE_OBSERVATION,
    _TWO_OBSERVATIONS,
    _SEVEN_OBSERVATIONS,
    _EIGHT_OBSERVATIONS,
    _NINETEEN_OBSERVATIONS,
    _TWENTY_OBSERVATIONS,
    _ALL_MISSING,
)

_IMPLEMENTED_MODES = tuple(mode for mode in DescTableType if mode is not DescTableType.NONE)

_EXPECTED_VALUES: dict[str, tuple[str, ...]] = {
    "Avg": ("1.00", "1.00", "1.00", "1.00", "1.00", "1.00", "nan"),
    "Std": ("nan", "1.41", "2.16", "2.93", "5.63", "6.37", "nan"),
    "Positive": ("100.0%", "50.0%", "57.1%", "50.0%", "52.6%", "50.0%", "nan%"),
    "Skew": ("nan", "0.0", "0.0", "0.0", "0.0", "0.0", "nan"),
    "Kurt": ("nan", "-2.0", "-1.2", "-1.4", "-1.2", "-1.3", "nan"),
    "P-val": ("nan", "nan", "nan", "nan", "nan", "0.08", "nan"),
    "Last": ("1.00", "2.00", "4.00", "5.00", "10.00", "11.00", "nan"),
    "Rank": ("100%", "100%", "100%", "100%", "100%", "100%", "nan%"),
    "Min": ("1.00", "0.00", "-2.00", "-3.00", "-8.00", "-9.00", "nan"),
    "-1std": ("1.00", "0.32", "-1.04", "-1.88", "-5.12", "-5.96", "nan"),
    "Median": ("1.00", "1.00", "1.00", "1.00", "1.00", "1.00", "nan"),
    "+1std": ("1.00", "1.68", "3.04", "3.88", "7.12", "7.96", "nan"),
    "Max": ("1.00", "2.00", "4.00", "5.00", "10.00", "11.00", "nan"),
}

_EXPECTED_MODE_COLUMNS: dict[DescTableType, tuple[str, ...]] = {
    DescTableType.SHORT: ("Avg", "Std"),
    DescTableType.AVG_WITH_POSITIVE_PROB: ("Positive",),
    DescTableType.WITH_POSITIVE_PROB: ("Avg", "Std", "Positive"),
    DescTableType.WITH_KURTOSIS: ("Avg", "Std", "Skew", "Kurt"),
    DescTableType.WITH_NORMAL_PVAL: ("Avg", "Std", "Skew", "Kurt", "P-val"),
    DescTableType.WITH_SCORE: ("Avg", "Std", "Last", "Rank"),
    DescTableType.EXTENSIVE: (
        "Avg",
        "Std",
        "Skew",
        "Kurt",
        "Min",
        "-1std",
        "Median",
        "+1std",
        "Max",
    ),
    DescTableType.SKEW_KURTOSIS: ("Skew", "Kurt"),
    DescTableType.WITH_MEDIAN: ("Avg", "Std", "Median", "Skew", "Kurt"),
}


def _place_values(values: tuple[float, ...], positions: tuple[int, ...]) -> tuple[float, ...]:
    """Place a finite sample at selected positions in the common date range.

    Args:
        values: Chronologically ordered finite observations.
        positions: Zero-based positions receiving those observations.

    Returns:
        Twenty-value tuple with NaN at every unselected position.
    """
    placed_values = [np.nan] * len(_DATES)
    for position, value in zip(positions, values):
        placed_values[position] = value
    return tuple(placed_values)


def _mixed_returns(*, nullable: bool) -> pd.DataFrame:
    """Create the mixed sample-cardinality panel in deliberate column order.

    The finite samples are symmetric around one. Their sums of squared deviations are 2, 28,
    60, 570, and 770 for sample sizes two, seven, eight, nineteen, and twenty. Missing values
    occur at the interior, boundaries, and tails so eligibility depends only on the observed
    count rather than physical placement.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Twenty-month return panel containing all relevant column states simultaneously.
    """
    nineteen_positions = tuple(position for position in range(len(_DATES)) if position != 9)
    returns = pd.DataFrame(
        {
            _ONE_OBSERVATION: _place_values((1.0,), (9,)),
            _TWO_OBSERVATIONS: _place_values((0.0, 2.0), (0, 19)),
            _SEVEN_OBSERVATIONS: _place_values(
                (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0), tuple(range(7))
            ),
            _EIGHT_OBSERVATIONS: _place_values(
                (-3.0, -2.0, -1.0, 0.0, 2.0, 3.0, 4.0, 5.0), tuple(range(12, 20))
            ),
            _NINETEEN_OBSERVATIONS: _place_values(
                tuple(float(value) for value in range(-8, 11)), nineteen_positions
            ),
            _TWENTY_OBSERVATIONS: tuple(float(value) for value in range(-9, 1))
            + tuple(float(value) for value in range(2, 12)),
            _ALL_MISSING: (np.nan,) * len(_DATES),
        },
        index=_DATES,
    )
    if nullable:
        return returns.astype(pd.Float64Dtype())
    return returns


def _expected_table(desc_table_type: DescTableType) -> pd.DataFrame:
    """Build the complete expected table from independently specified strings.

    Args:
        desc_table_type: Implemented descriptive-table mode under test.

    Returns:
        Expected schema, row order, and display strings for the mixed panel.
    """
    return pd.DataFrame(
        {column: _EXPECTED_VALUES[column] for column in _EXPECTED_MODE_COLUMNS[desc_table_type]},
        index=pd.Index(_ASSETS),
    )


def _compute_without_warnings(
    data: pd.DataFrame | pd.Series,
    desc_table_type: DescTableType,
    *,
    annualize_vol: bool = False,
    is_add_tstat: bool = False,
    var_format: str = "{:.2f}",
    norm_variable_display_type: str = "{:.1f}",
) -> pd.DataFrame:
    """Call the public function while treating every emitted warning as a failure.

    Args:
        data: Series or DataFrame supplied to the public function.
        desc_table_type: Descriptive-table mode under test.
        annualize_vol: Whether to display annualized volatility.
        is_add_tstat: Whether to append the existing optional t-statistic.
        var_format: Display format for level statistics.
        norm_variable_display_type: Display format for moments and the t-statistic.

    Returns:
        Formatted descriptive-statistics table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=desc_table_type,
            var_format=var_format,
            annualize_vol=annualize_vol,
            is_add_tstat=is_add_tstat,
            norm_variable_display_type=norm_variable_display_type,
        )


def _zero_row_input(input_shape: str) -> pd.DataFrame | pd.Series:
    """Build a zero-row pandas object with the requested declared shape.

    Args:
        input_shape: Named Series, one-column frame, multi-column frame, or zero-column frame.

    Returns:
        Empty pandas object retaining its declared labels.
    """
    if input_shape == "named-series":
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float, name="Asset")
    columns_by_shape: dict[str, list[str]] = {
        "one-column-frame": ["Asset"],
        "multi-column-frame": ["Asset A", "Asset B"],
        "zero-column-frame": [],
    }
    return pd.DataFrame(index=pd.DatetimeIndex([]), columns=columns_by_shape[input_shape])


# =============================================================================
# Mixed-panel sample-cardinality regressions
# =============================================================================


@pytest.mark.parametrize("desc_table_type", _IMPLEMENTED_MODES)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_applies_each_statistic_minimum_independently(
    desc_table_type: DescTableType, nullable: bool
) -> None:
    """Preserve every defined statistic while leaving undersized results missing.

    The mixed panel checks all relevant physical-column states in one call. In particular, a
    one-point column retains its mean and score even though sample volatility and moments are
    undefined; normality remains missing through nineteen points and becomes ``0.08`` at twenty.

    Args:
        desc_table_type: Implemented reporting mode under test.
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    returns = _mixed_returns(nullable=nullable)
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(returns, desc_table_type)

    pd.testing.assert_frame_equal(actual, _expected_table(desc_table_type))
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Zero-row input contract and zero-column control
# =============================================================================


@pytest.mark.parametrize("desc_table_type", _IMPLEMENTED_MODES)
@pytest.mark.parametrize("annualize_vol", (False, True), ids=("periodic", "annualized"))
def test_compute_desc_table_rejects_zero_rows_consistently_across_modes(
    desc_table_type: DescTableType, annualize_vol: bool
) -> None:
    """Raise one descriptive error before reducers or frequency inference run.

    Args:
        desc_table_type: Implemented reporting mode under test.
        annualize_vol: Whether the rejected call would otherwise infer a frequency.
    """
    returns = _zero_row_input("one-column-frame")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="^data must contain at least one observation$"):
            _DESC_TABLE_MODULE.compute_desc_table(
                df=returns,
                desc_table_type=desc_table_type,
                annualize_vol=annualize_vol,
            )


@pytest.mark.parametrize(
    "input_shape",
    ("named-series", "multi-column-frame", "zero-column-frame"),
)
def test_compute_desc_table_rejects_every_zero_row_pandas_shape(input_shape: str) -> None:
    """Apply the zero-row contract independently of the declared Series/DataFrame shape.

    Args:
        input_shape: Empty pandas shape whose labels must not alter the public error.
    """
    returns = _zero_row_input(input_shape)
    original_returns = returns.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="^data must contain at least one observation$"):
            _DESC_TABLE_MODULE.compute_desc_table(
                df=returns,
                desc_table_type=DescTableType.SHORT,
            )

    if isinstance(returns, pd.DataFrame):
        pd.testing.assert_frame_equal(returns, cast(pd.DataFrame, original_returns))
    else:
        pd.testing.assert_series_equal(returns, cast(pd.Series, original_returns))


@pytest.mark.parametrize("desc_table_type", _IMPLEMENTED_MODES)
def test_compute_desc_table_allows_dated_panels_without_asset_columns(
    desc_table_type: DescTableType,
) -> None:
    """Return the selected empty schema when dates exist but no asset column is declared.

    Args:
        desc_table_type: Implemented reporting mode whose complete schema is expected.
    """
    returns = pd.DataFrame(index=_DATES)

    actual = _compute_without_warnings(returns, desc_table_type)

    assert actual.empty
    assert actual.index.equals(returns.columns)
    assert tuple(actual.columns) == _EXPECTED_MODE_COLUMNS[desc_table_type]


# =============================================================================
# Pandas-shape, formatting, and annualization controls
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_preserves_named_series_dataframe_consistency(nullable: bool) -> None:
    """Return the same one-row normality table for equivalent named pandas inputs.

    Args:
        nullable: Whether the selected one-point history uses nullable ``Float64`` storage.
    """
    frame = _mixed_returns(nullable=nullable).filter(items=[_ONE_OBSERVATION])
    series = cast(pd.Series, frame[_ONE_OBSERVATION])
    original_frame = frame.copy(deep=True)
    original_series = series.copy(deep=True)

    frame_result = _compute_without_warnings(frame, DescTableType.WITH_NORMAL_PVAL)
    series_result = _compute_without_warnings(series, DescTableType.WITH_NORMAL_PVAL)

    expected = _expected_table(DescTableType.WITH_NORMAL_PVAL).filter(
        items=[_ONE_OBSERVATION], axis="index"
    )
    pd.testing.assert_frame_equal(frame_result, expected)
    pd.testing.assert_frame_equal(series_result, expected)
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(series, original_series)


def test_compute_desc_table_preserves_custom_formats_at_sample_boundaries() -> None:
    """Apply caller formats to defined and undefined statistics without warning.

    A two-point sample has mean one, sample standard deviation ``sqrt(2)``, zero skewness, and
    excess kurtosis ``-2``. The custom formats must affect those values while the normality
    result remains the established ``nan`` representation.
    """
    returns = _mixed_returns(nullable=False).filter(items=[_TWO_OBSERVATIONS])

    actual = _compute_without_warnings(
        returns,
        DescTableType.WITH_NORMAL_PVAL,
        var_format="{:.3f}",
        norm_variable_display_type="{:.2f}",
    )

    expected = pd.DataFrame(
        {
            "Avg": ("1.000",),
            "Std": ("1.414",),
            "Skew": ("0.00",),
            "Kurt": ("-2.00",),
            "P-val": ("nan",),
        },
        index=pd.Index([_TWO_OBSERVATIONS]),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_compute_desc_table_preserves_monthly_annualization_and_tstat_formula() -> None:
    """Keep existing annualization and optional t-statistics for eligible samples.

    Monthly frequency multiplies each mean by 12 and each sample standard deviation by
    ``sqrt(12)``. The displayed t-statistic remains the accepted ``12 * mean / annualized std``;
    this regression establishes sample eligibility without changing that separate convention.
    """
    returns = _mixed_returns(nullable=False)

    actual = _compute_without_warnings(
        returns,
        DescTableType.SHORT,
        annualize_vol=True,
        is_add_tstat=True,
    )

    expected = pd.DataFrame(
        {
            "Avg": ("1.00", "1.00", "1.00", "1.00", "1.00", "1.00", "nan"),
            "Std An": ("nan", "4.90", "7.48", "10.14", "19.49", "22.05", "nan"),
            "T-stat": ("nan", "2.4", "1.6", "1.2", "0.6", "0.5", "nan"),
        },
        index=pd.Index(_ASSETS),
    )
    pd.testing.assert_frame_equal(actual, expected)
