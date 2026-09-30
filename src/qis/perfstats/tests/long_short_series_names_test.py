"""Regression tests for name-independent long-short relative NAV construction.

Series names are display metadata, so absent or duplicate names must not control role selection.
The boundary case also covers the documented union-and-forward-fill behavior for nullable inputs.
"""

import pandas as pd
import pytest

import qis


@pytest.mark.parametrize(
    ("long_name", "short_name"),
    [(None, None), ("asset", "asset"), ("long", "short")],
)
def test_long_short_relative_nav_ignores_series_names(long_name, short_name) -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    long_price = pd.Series([100.0, 110.0, 121.0], index=index, name=long_name)
    short_price = pd.Series([100.0, 100.0, 100.0], index=index, name=short_name)
    long_before = long_price.copy(deep=True)
    short_before = short_price.copy(deep=True)

    actual = qis.long_short_to_relative_nav(long_price=long_price, short_price=short_price)

    # Derive the expected strategy from each leg before combining their returns.
    long_returns = long_price.pct_change(fill_method=None).fillna(0.0)
    short_returns = short_price.pct_change(fill_method=None).fillna(0.0)
    expected = (1.0 + long_returns - short_returns).cumprod().rename(0)
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(long_price, long_before)
    pd.testing.assert_series_equal(short_price, short_before)


def test_long_short_relative_nav_aligns_nullable_gapped_prices() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    long_price = pd.Series([100.0, 110.0, pd.NA, 121.0], index=index, dtype="Float64")
    short_price = pd.Series([100.0, 105.0, 105.0], index=index[[0, 2, 3]], dtype="Float64")

    actual = qis.long_short_to_relative_nav(long_price=long_price, short_price=short_price)

    # The public contract aligns the union first and carries each observed price through gaps.
    aligned = pd.concat([long_price, short_price], axis=1, sort=True).ffill()
    expected_returns = aligned.iloc[:, 0].pct_change(fill_method=None).fillna(0.0) - aligned.iloc[
        :, 1
    ].pct_change(fill_method=None).fillna(0.0)
    expected = (1.0 + expected_returns).cumprod().rename(0)
    pd.testing.assert_series_equal(actual, expected)
