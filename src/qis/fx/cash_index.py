"""Currency cash accounts with monthly resets, independently of measurement dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qis.fx.fx_rates_data import FxRatesData


def compute_monthly_cash_index(
    data: FxRatesData, currency: str, dates: pd.DatetimeIndex,
) -> pd.Series:
    """Sample one ACT/365F cash index with calendar month-end opening resets.

    Annual quotes are as-of month-end, as-of aligned under the existing FX-data
    policy, never backfilled. Interest is simple within a month and reinvested at
    month-end. Intramonth observations sample that same account without resetting
    it. Normalization is one at the first month-end with an available rate.
    No term forward is repriced and no source quote is changed.

    Args:
        data: Supplied FX container with annual decimal domestic rates.
        currency: Existing currency ID in domestic_rates.
        dates: Ordered unique midnight valuation dates, in the source timezone.

    Returns:
        Cash levels on dates, missing before the first supported monthly opening.

    Raises:
        ValueError: Invalid dates or nonpositive cash growth.
    """
    rates = data.domestic_rates[currency]
    for index in (dates, rates.index):
        if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
                or not index.is_unique or not index.is_monotonic_increasing
                or not index.equals(index.normalize())):
            raise ValueError("Cash dates must be ordered unique midnight timestamps")
    result = pd.Series(np.nan, index=dates, name=currency, dtype=float)
    first = rates.first_valid_index()
    if dates.empty or first is None:
        return result
    start = pd.offsets.MonthEnd().rollforward(first)
    if start > dates[-1]:
        return result
    resets = pd.date_range(start, dates[-1], freq="ME")
    earned = data._get_period_cash_returns(resets, currency, "ME", False, 1)
    levels = (1.0 + earned).cumprod(skipna=False)
    opening_quotes = rates.reindex(resets, method="ffill")
    supported = dates[dates >= start]
    positions = resets.searchsorted(supported, side="right") - 1
    opening_dates = resets[positions]
    elapsed = (supported - opening_dates).days / 365.0
    gross = 1.0 + opening_quotes.iloc[positions].to_numpy() * elapsed
    # A missing new reset quote cannot erase interest already earned at that boundary.
    gross = np.where(elapsed == 0.0, 1.0, gross)
    if np.any(gross <= 0):
        raise ValueError("Cash gross factors must be strictly positive")
    result.loc[supported] = levels.iloc[positions].to_numpy() * gross
    return result
