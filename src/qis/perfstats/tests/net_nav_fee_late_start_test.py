"""A NAV column that starts late runs its own fee account from its first observation.

``qis.compute_net_navs_ex_perf_man_fees`` must not let the missing gross returns before a
column's first NAV enter the fee recursion. The column is missing before its first NAV, equals
one there, and matches the net NAV of the same column passed on its own observed range.
"""

import numpy as np
import pandas as pd

import qis


_DATES = pd.date_range('2020-12-31', periods=7, freq='QE')


def _gross_navs() -> pd.DataFrame:
    """Two gross NAV histories; the second starts two quarters later."""
    return pd.DataFrame({'Early': [100.0, 104.0, 99.0, 108.0, 112.0, 101.0, 118.0],
                         'Late': [np.nan, np.nan, 50.0, 55.0, 53.0, 60.0, 66.0]}, index=_DATES)


def test_late_column_matches_its_own_history() -> None:
    """Fees accrue from the late column's first NAV, as for that column alone."""
    navs = _gross_navs()
    original = navs.copy(deep=True)

    net = qis.compute_net_navs_ex_perf_man_fees(navs=navs, man_fee=0.02, perf_fee=0.2,
                                                perf_fee_frequency='YE')

    for column in navs.columns:
        alone = qis.compute_net_navs_ex_perf_man_fees(navs=navs[column].dropna(), man_fee=0.02,
                                                      perf_fee=0.2, perf_fee_frequency='YE')
        pd.testing.assert_series_equal(net[column].dropna(), alone, check_freq=False,
                                       check_names=False)
    assert net['Late'].iloc[:2].isna().all() and net['Late'].iloc[2] == 1.0
    assert net['Late'].iloc[2:].notna().all()
    pd.testing.assert_frame_equal(navs, original)


def test_late_series_without_fees_is_the_rebased_gross_nav() -> None:
    """With zero fees a late-starting Series is its gross NAV divided by its first level."""
    late = _gross_navs()['Late']

    net = qis.compute_net_navs_ex_perf_man_fees(navs=late, man_fee=0.0, perf_fee=0.0)

    expected = late / 50.0
    pd.testing.assert_series_equal(net, expected, check_exact=False, rtol=1e-14, check_freq=False)
