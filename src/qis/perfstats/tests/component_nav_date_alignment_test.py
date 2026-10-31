"""Date-pairing regressions for the component-NAV display adjustment.

For the literal two-year histories below, an independent 60-digit decimal calculation gives
gamma = 1.0375083463026575 from portfolio growth 1.5 and component growths 1.1 and 1.2.
The dated adjusted levels, not just their annual-return sum, detect a misplaced interior factor.
These tests retain the existing first-stored-portfolio-row anchor and do not define a new join
or duplicate-date policy for inputs outside the matching unique timestamp grid.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.returns import (
    adjust_component_navs_to_portfolio,
    compute_pa_return,
    portfolio_navs_to_additive,
)


def _navs(dtype: str = "float64") -> tuple[pd.Series, pd.DataFrame]:
    """Build independent portfolio/component axes with the same three observation dates."""
    dates = pd.date_range("2020-01-01", periods=3, freq="YS", name="date")
    portfolio = pd.Series([100.0, 120.0, 150.0], index=dates.copy(deep=True), dtype=dtype)
    components = pd.DataFrame(
        {"Carry": [100.0, 105.0, 110.0], "Price": [100.0, 110.0, 120.0]},
        index=dates.copy(deep=True),
        dtype=dtype,
    )
    components.columns.name = "component"
    return portfolio, components


def _adjusted_reference(dtype: str = "float64") -> pd.DataFrame:
    """Return literal dated levels from the independent compounding calculation."""
    return pd.DataFrame(
        [
            [100.0, 100.0],
            [108.94661349212132, 114.13454746793661],
            [118.41256117483974, 129.17733946346152],
        ],
        index=pd.date_range("2020-01-01", periods=3, freq="YS", name="date"),
        columns=pd.Index(["Carry", "Price"], name="component"),
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("order", [[2, 1, 0], [0, 2, 1]], ids=["descending", "nonmonotone"])
def test_adjust_component_navs_to_portfolio_pairs_factors_with_component_dates(dtype, order):
    """Keep dated values and annual-return reconciliation when only components are reordered."""
    portfolio, components = _navs(dtype)
    components = components.iloc[order, ::-1]
    before_portfolio = portfolio.copy(deep=True)
    before_components = components.copy(deep=True)
    expected = _adjusted_reference(dtype).reindex(
        index=components.index.copy(deep=True), columns=components.columns.copy(deep=True)
    )
    # Building the expectation must not change the caller's labels or frequency metadata.
    pd.testing.assert_frame_equal(components, before_components, check_exact=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = adjust_component_navs_to_portfolio(portfolio, components)

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(compute_pa_return(actual)), 0.2245750502963005, rtol=1e-13, atol=1e-13
    )
    pd.testing.assert_series_equal(portfolio, before_portfolio, check_exact=True)
    pd.testing.assert_frame_equal(components, before_components, check_exact=True)


def test_adjust_component_navs_to_portfolio_matches_instants_without_replacing_axis_metadata():
    """Match timezone-equivalent timestamps and different datetime units exactly."""
    portfolio, components = _navs()
    portfolio.index = portfolio.index.tz_localize("UTC").as_unit("ns")
    components.index = components.index.tz_localize("UTC").tz_convert("America/New_York")
    components.index.name = "component_date"
    components = components.iloc[[2, 0, 1]]
    expected = _adjusted_reference().iloc[[2, 0, 1]]
    expected.index = components.index.copy(deep=True)

    actual = adjust_component_navs_to_portfolio(portfolio, components)

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_adjust_component_navs_to_portfolio_keeps_mixed_missing_columns_independent(dtype):
    """Apply dated factors without filling or coupling ragged and all-missing components."""
    portfolio, components = _navs(dtype)
    components = pd.DataFrame(
        {
            "Complete": [100.0, 105.0, 110.0],
            "Leading": [np.nan, 105.0, 110.0],
            "Trailing": [100.0, 105.0, np.nan],
            "Missing": [np.nan, np.nan, np.nan],
        },
        index=components.index.copy(deep=True),
        dtype=dtype,
    )
    years = 731.0 / 365.25
    # Native PA returns use the full index span, including each ragged column's missing dates.
    gross = np.array([1.1, 110.0 / 105.0, 1.05]) ** (1.0 / years)
    gamma = (1.0 + (1.5 ** (1.0 / years) - 1.0) / 4.0) / np.mean(gross)
    factors = np.array([1.0, gamma ** (366.0 / 365.25), gamma**years])
    expected_values = components.to_numpy(dtype=float, na_value=np.nan) * factors[:, None]
    expected = pd.DataFrame(
        expected_values,
        index=components.index.copy(deep=True),
        columns=components.columns.copy(deep=True),
        dtype=dtype,
    ).iloc[[2, 0, 1]]
    components = components.iloc[[2, 0, 1]]
    original = components.copy(deep=True)

    # Endpoint fallback warnings are existing behavior, not a new missing-data policy.
    with pytest.warns(UserWarning):
        actual = adjust_component_navs_to_portfolio(portfolio, components)

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-13, atol=1e-12)
    pd.testing.assert_frame_equal(components, original, check_exact=True)


def test_adjust_component_navs_to_portfolio_and_wrapper_retain_the_existing_time_origin():
    """Date pairing must not silently normalize the portfolio's first stored date."""
    portfolio, components = _navs()
    portfolio = portfolio.iloc[::-1]
    expected = _adjusted_reference() / 1.0764778288621795
    components = components.iloc[[1, 2, 0]]

    actual = adjust_component_navs_to_portfolio(portfolio, components)
    pd.testing.assert_frame_equal(
        actual, expected.reindex(components.index), rtol=1e-13, atol=1e-12
    )

    # pandas 2.x can sort a DatetimeIndex outer join even with sort=False.
    grouped = pd.concat([portfolio.rename("Portfolio"), components], axis=1, sort=False).reindex(
        portfolio.index
    )
    pd.testing.assert_index_equal(grouped.index, portfolio.index)
    original = grouped.copy(deep=True)
    wrapped = portfolio_navs_to_additive(grouped, "Portfolio")
    expected_wrapper = pd.concat([portfolio.rename("Portfolio"), expected], axis=1, sort=True)
    pd.testing.assert_frame_equal(wrapped, expected_wrapper, rtol=1e-13, atol=1e-12)
    pd.testing.assert_frame_equal(grouped, original, check_exact=True)


@pytest.mark.parametrize(
    "boundary",
    [
        "different-date",
        "duplicate-portfolio",
        "duplicate-components",
        "NaT-portfolio",
        "NaT-components",
        "mixed-timezone-awareness",
    ],
)
def test_adjust_component_navs_to_portfolio_retains_other_calendar_handling(boundary):
    """Do not replace established positional behavior with a new join or rejection policy."""
    portfolio, components = _navs()
    expected = _adjusted_reference()
    alternate_dates = pd.DatetimeIndex(["2020-01-01", "2021-06-01", "2022-01-01"])
    if boundary == "different-date":
        components.index = alternate_dates
    elif boundary.startswith("duplicate"):
        target = portfolio if boundary.endswith("portfolio") else components
        target.index = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2022-01-01"])
        if target is portfolio:
            expected.iloc[1] = [105.0, 110.0]
    elif boundary.startswith("NaT"):
        target = portfolio if boundary.endswith("portfolio") else components
        target.index = pd.DatetimeIndex(["2020-01-01", pd.NaT, "2022-01-01"])
        if target is portfolio:
            expected.iloc[1] = np.nan
    else:
        components.index = components.index.tz_localize("UTC")
    expected.index = components.index.copy(deep=True)

    actual = adjust_component_navs_to_portfolio(portfolio, components)

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-13, atol=1e-12)


def test_adjust_component_navs_to_portfolio_does_not_admit_a_shorter_component_calendar():
    """Preserve the existing shape error rather than implicitly joining unequal calendars."""
    portfolio, components = _navs()

    with pytest.raises(ValueError, match="shape"):
        adjust_component_navs_to_portfolio(portfolio, components.iloc[[0, 2]])
