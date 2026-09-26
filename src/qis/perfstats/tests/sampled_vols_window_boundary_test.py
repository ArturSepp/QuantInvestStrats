"""Each return belongs to exactly one volatility window of ``qis.compute_sampled_vols``.

A return dated on a window boundary closes the window that ends there; it must not also open
the next window. The price path below has a 20% jump on Wednesday 31 January 2024, a month-end
that is a business day. February's volatility must be computed from the 21 February returns
alone, so the jump cannot enter it.
"""

import numpy as np
import pandas as pd

import qis
from qis.utils.dates import split_df_by_freq


def _prices_with_month_end_jump() -> pd.Series:
    """Business-day prices with alternating 1% moves and a 20% jump on 31 January 2024."""
    dates = pd.bdate_range('2024-01-01', '2024-04-30')
    returns = pd.Series(np.where(np.arange(len(dates)) % 2 == 0, 0.01, -0.01), index=dates)
    returns.iloc[0] = 0.0
    returns.loc[pd.Timestamp('2024-01-31')] = 0.20
    return 100.0 * (1.0 + returns).cumprod().rename('asset')


def test_boundary_return_is_counted_in_one_window_only() -> None:
    """February's volatility excludes the return dated 31 January."""
    prices = _prices_with_month_end_jump()
    returns = prices.pct_change()

    vols = qis.compute_sampled_vols(prices=prices, freq_vol='ME')

    february = returns.loc['2024-02-01':'2024-02-29'].to_numpy()
    march = returns.loc['2024-03-01':'2024-03-29'].to_numpy()
    assert len(february) == 21 and len(march) == 21
    expected = np.sqrt(252.0) * np.array([np.std(february, ddof=1), np.std(march, ddof=1)])
    np.testing.assert_allclose(vols.loc[['2024-02-29', '2024-03-31']].to_numpy(), expected,
                               rtol=1e-12)


def test_right_closed_split_partitions_the_observations() -> None:
    """With inclusive='right' consecutive windows share no observation."""
    returns = _prices_with_month_end_jump().pct_change().to_frame()

    windows = split_df_by_freq(df=returns, freq='ME', include_start_date=False,
                               include_end_date=False, inclusive='right')

    labels = [window.index for window in windows.values()]
    for left, right in zip(labels[:-1], labels[1:]):
        assert left.intersection(right).empty
    assert labels[0][0] == pd.Timestamp('2024-02-01')

    both = split_df_by_freq(df=returns, freq='ME', include_start_date=False,
                            include_end_date=False)
    assert list(both.values())[0].index[0] == pd.Timestamp('2024-01-31')
