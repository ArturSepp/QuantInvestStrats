"""
compute seasonal strategy using monthly returns
"""

import numpy as np
import pandas as pd
import qis as qis


def q60(x):
    return x.quantile(0.6)


def q40(x):
    return x.quantile(0.4)


def compute_seasonal_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    compute seasonal signal
    """
    prices = prices.asfreq('B', method='ffill')  # make B frequency
    returns = qis.to_returns(prices, freq='ME', drop_first=True)
    returns['month'] = returns.index.month
    seasonal_returns = returns.groupby('month').agg(['mean', q40, q60])
    signals = {}
    for asset in prices.columns:
        df = seasonal_returns[asset]
        signal = np.where(df['q40'] > 0.0, +1.0, np.where(df['q60'] < 0.0, -1.0, 0.0))
        signals[asset] = pd.Series(signal, index=df.index).fillna(0.0)
    signals = pd.DataFrame.from_dict(signals, orient='columns')
    return signals


def compute_rolling_seasonal_signals(prices: pd.DataFrame,
                                     num_sample_years: int = 25
                                     ) -> pd.DataFrame:
    rebalancing_years = list(np.arange(2000, 2025))
    monthly_returns = qis.to_returns(prices, freq='ME', drop_first=True, include_end_date=True)
    signals = []
    for year in rebalancing_years:
        # print(year)
        estimation_sample = prices.loc[f"{year-num_sample_years}":f"{year-1}", :]
        if year == 2024:
            print('here')
        # print(estimation_sample)
        investment_sample = monthly_returns.iloc[monthly_returns.index.year==year, :]
        investment_sample_index = investment_sample.index
        investment_sample_index = investment_sample_index.shift(-1, freq="ME")  # sift signal to end of previous month
        signal = compute_seasonal_signal(prices=estimation_sample)
        signal = signal.iloc[:len(investment_sample_index), :]  # drop last few mon
        # set signal of the investment sample with weight at the end of prior months
        signal = signal.set_index(investment_sample_index)
        # now move to start of the month
        signals.append(signal)
    signals = pd.concat(signals, axis=0)
    return signals
