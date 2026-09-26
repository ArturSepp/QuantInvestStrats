"""Keyword and fill contracts of ``qis.to_returns``.

An unknown keyword is reported rather than silently ignored, so a misspelt ``is_log_return``
cannot return simple returns unnoticed. The forward-fill policy ``ffill_nans`` applies when
prices are resampled onto the ``freq`` grid; input already on that grid keeps its missing
observations, as documented, and ``freq=None`` fills them.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis


_MONTH_ENDS = pd.date_range('2024-01-31', periods=4, freq='ME')


def test_unknown_keyword_is_reported_with_a_suggestion() -> None:
    """A misspelt keyword warns and names the intended argument."""
    prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=_MONTH_ENDS)

    with pytest.warns(UserWarning, match="is_log_return.*is_log_returns"):
        returns = qis.to_returns(prices=prices, is_log_return=True)

    np.testing.assert_allclose(returns.to_numpy()[1:], [0.1, -0.1, 0.1], atol=1e-12)


def test_known_keywords_do_not_warn() -> None:
    """The documented arguments pass without a warning."""
    prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=_MONTH_ENDS)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        qis.to_returns(prices=prices, is_log_returns=True, freq=None, drop_first=True)


def test_fill_policy_on_and_off_the_freq_grid() -> None:
    """A month-end gap is kept on the month-end grid, filled with freq=None or when resampling."""
    prices = pd.Series([100.0, np.nan, 110.0, 99.0], index=_MONTH_ENDS, name='asset')
    daily = prices.reindex(pd.date_range('2024-01-01', '2024-04-30', freq='D'))
    daily.iloc[0] = 100.0

    on_grid = qis.to_returns(prices=prices, freq='ME')
    native = qis.to_returns(prices=prices, freq=None)
    resampled = qis.to_returns(prices=daily, freq='ME')

    assert on_grid.iloc[1:3].isna().all() and abs(on_grid.iloc[3] + 0.1) < 1e-12
    np.testing.assert_allclose(native.to_numpy()[1:], [0.0, 0.1, -0.1], atol=1e-12)
    np.testing.assert_allclose(resampled.to_numpy()[1:], [0.0, 0.1, -0.1], atol=1e-12)
