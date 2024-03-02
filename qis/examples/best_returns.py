
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from enum import Enum
import yfinance as yf


def add_performance_metric(data: pd.DataFrame) -> List[str]:
    num_years = (data.index[-1] - data.index[0]).days / 365
    ratio = data.iloc[-1] / data.iloc[0]
    compounded_return_pa = ratio ** (1.0 / num_years) - 1
    print(compounded_return_pa)
    titles = [f"{key}: {v:0.0%} p.a." for key, v in compounded_return_pa.to_dict().items()]
    return titles


class WoType(Enum):
    BEST = 1
    WORST = 2
    BOTH = 3


def perf_wo_best_worst(prices: pd.Series,
                       freq: str = 'ME',
                       cut_off: int = 1,
                       wo_type: WoType = WoType.BEST,
                       ax: plt.Subplot = None
                       ) -> pd.DataFrame:

    returns = prices.pct_change()
    returns_a = returns.groupby(pd.Grouper(freq=freq))

    wo_best_returns = []
    wo_worst_returns = []
    for key, data in returns_a:
        data = data.sort_values()  # sort by max returns
        wo_best = data.copy()
        wo_best.iloc[-cut_off:] = 0.0
        wo_best_returns.append(wo_best.sort_index())  # sort back
        wo_worst = data.copy()
        wo_worst.iloc[:cut_off] = 0.0
        #wo_worst.iloc[-cut_off:] = 0.0
        wo_worst_returns.append(wo_worst.sort_index())  # sort back

    wo_best_returns = pd.concat(wo_best_returns, axis=0)
    wo_worst_returns = pd.concat(wo_worst_returns, axis=0)

    wo_best_perf = wo_best_returns.add(1.0).cumprod(axis=0).multiply(prices.iloc[0])
    wo_worst_perf = wo_worst_returns.add(1.0).cumprod(axis=0).multiply(prices.iloc[0])

    if wo_type == WoType.BEST:
        joint_data = pd.concat([prices, wo_best_perf.rename('W/O Best')], axis=1)

    elif wo_type == WoType.WORST:
        joint_data = pd.concat([prices, wo_best_perf.rename('W/O Best'), wo_worst_perf.rename('W/O Worst')], axis=1)

    else:
        joint_data = pd.concat([prices, wo_worst_perf.rename('W/O Best and Worst') ], axis=1)

    #     ppd.plot_prices(prices=joint_data)
    joint_data.columns = add_performance_metric(joint_data)
    sns.lineplot(data=joint_data, ax=ax)

    return joint_data


class UnitTests(Enum):
    PERF1 = 1


def run_unit_test(unit_test: UnitTests):

    ticker = 'SPY'
    prices = yf.download(ticker, start=None, end=None)['Adj Close'].rename(ticker)

    freq = 'ME'
    wo_type = WoType.BEST
    if unit_test == UnitTests.PERF1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        perf_wo_best_worst(prices=prices, freq=freq, wo_type=WoType.WORST, ax=ax)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PERF1

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
