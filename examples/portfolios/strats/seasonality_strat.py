"""Point-in-time calendar-month seasonality signals used by portfolio examples.

``compute_seasonal_signal`` estimates one ternary position per calendar month: +1 when the
historical 40th percentile is positive, -1 when the 60th percentile is negative, and 0
otherwise. ``compute_rolling_seasonal_signals`` refits that rule once per year using only the
preceding ``num_sample_years`` and records each decision at the prior month end. A decision at
*t* therefore applies over *[t, t+1]*.
"""

import numpy as np
import pandas as pd

import qis


def q60(x: pd.Series) -> float:
    """Return the 60th percentile of a return sample."""
    return x.quantile(0.6)


def q40(x: pd.Series) -> float:
    """Return the 40th percentile of a return sample."""
    return x.quantile(0.4)


def compute_seasonal_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Estimate a ternary position for every asset and calendar month.

    Args:
        prices: Daily price panel used as the estimation sample.

    Returns:
        DataFrame indexed by calendar month 1 through 12, with one position per asset.
    """
    prices = prices.asfreq('B', method='ffill')
    returns = qis.to_returns(prices, freq='ME', drop_first=True)
    returns['month'] = returns.index.month
    seasonal_returns = returns.groupby('month').agg(['mean', q40, q60])
    signals = {}
    for asset in prices.columns:
        asset_returns = seasonal_returns[asset]
        signal = np.where(
            asset_returns['q40'] > 0.0,
            1.0,
            np.where(asset_returns['q60'] < 0.0, -1.0, 0.0),
        )
        signals[asset] = pd.Series(signal, index=asset_returns.index).fillna(0.0)
    return pd.DataFrame.from_dict(signals, orient='columns')


def compute_rolling_seasonal_signals(
    prices: pd.DataFrame,
    num_sample_years: int = 25,
) -> pd.DataFrame:
    """Refit seasonality annually and timestamp positions at the prior month end.

    For an investment month in year ``Y``, the estimator sees prices from January ``Y-N``
    through December ``Y-1`` only. The January position is dated at the preceding December
    month end, the February position at January month end, and so on.

    Args:
        prices: Daily price panel with a ``DatetimeIndex``.
        num_sample_years: Number of complete prior calendar years in each estimation window.

    Returns:
        Monthly decision schedule. Row *t* is the position applied to the following month's
        return.

    Raises:
        ValueError: If the sample length is non-positive or no complete estimation window exists.
    """
    if num_sample_years <= 0:
        raise ValueError(f'num_sample_years must be positive, got {num_sample_years}')
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError('prices must have a DatetimeIndex')

    prices = prices.sort_index().asfreq('B', method='ffill').dropna(how='all')
    monthly_prices = prices.resample('ME').last()
    decisions = []
    for year in sorted(pd.unique(monthly_prices.index.year)):
        estimation_start = pd.Timestamp(year=year - num_sample_years, month=1, day=1)
        investment_start = pd.Timestamp(year=year, month=1, day=1)
        estimation_sample = prices.loc[
            (prices.index >= estimation_start) & (prices.index < investment_start)
        ]
        observed_years = pd.unique(estimation_sample.index.year)
        if len(observed_years) < num_sample_years:
            continue

        seasonal_signal = compute_seasonal_signal(prices=estimation_sample)
        investment_dates = monthly_prices.index[monthly_prices.index.year == year]
        for investment_date in investment_dates:
            position = monthly_prices.index.get_loc(investment_date)
            if position == 0 or investment_date.month not in seasonal_signal.index:
                continue
            decision_date = monthly_prices.index[position - 1]
            decisions.append(seasonal_signal.loc[investment_date.month].rename(decision_date))

    if not decisions:
        raise ValueError(
            f'no {num_sample_years}-year estimation window exists in '
            f'{prices.index[0]:%d%b%Y} to {prices.index[-1]:%d%b%Y}'
        )
    return pd.DataFrame(decisions).sort_index()
