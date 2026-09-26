"""Fully missing rows of ``qis.portfolio_returns_to_nav`` follow ``qis.to_portfolio_returns``.

A date on which every contribution is missing has a missing portfolio return in both functions,
rather than a zero return in one and a missing return in the other. The NAV starts at one on
the first row, is carried flat through an interior missing row, and ends at the last date with
an observed contribution, as ``qis.returns_to_nav`` does for trailing missing returns. Expected
paths are hand products.
"""

import numpy as np
import pandas as pd

import qis


_DATES = pd.date_range('2024-01-31', periods=6, freq='ME')


def _contributions() -> pd.DataFrame:
    """Per-asset contributions with missing first, interior and trailing rows."""
    return pd.DataFrame({'a': [np.nan, 0.01, 0.02, np.nan, -0.01, np.nan],
                         'b': [np.nan, 0.03, np.nan, np.nan, 0.02, np.nan]}, index=_DATES)


def test_fully_missing_rows_are_missing_in_both_aggregations() -> None:
    """The aggregate return matches to_portfolio_returns on every row."""
    contributions = _contributions()
    weights = pd.DataFrame(1.0, index=_DATES, columns=contributions.columns)
    portfolio_returns = qis.to_portfolio_returns(weights=weights, returns=contributions)

    nav = qis.portfolio_returns_to_nav(returns=contributions)

    expected = pd.Series([1.0, 1.04, 1.04 * 1.02, 1.04 * 1.02, 1.04 * 1.02 * 1.01],
                         index=_DATES[:5])
    pd.testing.assert_series_equal(nav, expected, check_exact=False, rtol=1e-14,
                                   check_names=False, check_freq=False)
    assert portfolio_returns.iloc[[0, 3, 5]].isna().all()
    reference = qis.returns_to_nav(returns=portfolio_returns.fillna({_DATES[0]: 0.0}),
                                   init_period=None)
    pd.testing.assert_series_equal(nav, reference, check_names=False, check_freq=False)


def test_first_row_contribution_is_discarded() -> None:
    """With init_period=1 the NAV is one on the first row whether or not it is observed."""
    contributions = _contributions().fillna({'a': 0.0, 'b': 0.0}).iloc[1:3]

    nav = qis.portfolio_returns_to_nav(returns=contributions)

    np.testing.assert_allclose(nav.to_numpy(), [1.0, 1.02], rtol=1e-14)
