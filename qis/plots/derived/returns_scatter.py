# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union
from enum import Enum

# qis
from qis.utils.df_melt import melt_scatter_data_with_xvar
import qis.perfstats.returns as ret
from qis.plots.scatter import plot_scatter
from qis.perfstats.config import ReturnTypes


def plot_returns_scatter(prices: pd.DataFrame,
                         benchmark: str = None,
                         benchmark_prices: Union[pd.Series, pd.DataFrame] = None,
                         freq: Optional[str] = 'QE',
                         order: int = 2,
                         ci: Optional[int] = 95,
                         add_45line: bool = False,
                         is_vol_norm: bool = False,
                         y_column: str = 'Strategy returns',
                         xlabel: str = None,
                         ylabel: str = 'returns',
                         var_format: str = '{:.1%}',
                         title: Union[str, None] = None,
                         add_hue_model_label: bool = True,
                         hue_name: str = 'hue',
                         return_type: ReturnTypes = ReturnTypes.RELATIVE,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:

    if benchmark_prices is None:
        price_data_full = prices
    else:
        if isinstance(benchmark_prices, pd.Series):  # use benchmark set by series
            price_data_full = pd.concat([benchmark_prices, prices], axis=1)
            benchmark = benchmark_prices.name
            benchmark_prices = None
        else:  # for df price data must be sries
            if not isinstance(prices, pd.Series):
                raise ValueError(f"must be series\n{prices}")
            price_data_full = pd.concat([prices, benchmark_prices], axis=1)

    returns = ret.to_returns(prices=price_data_full,
                             include_start_date=True,
                             include_end_date=True,
                             return_type=return_type,
                             freq=freq)
    if is_vol_norm:
        returns = returns.divide(np.nanstd(returns, axis=0), axis=1)

    if benchmark_prices is None:
        scatter_data = melt_scatter_data_with_xvar(df=returns,
                                                   xvar_str=benchmark,
                                                   y_column=y_column,
                                                   hue_name=hue_name)
    else:
        scatter_data = melt_scatter_data_with_xvar(df=returns,
                                                   xvar_str=str(prices.name),
                                                   y_column=y_column,
                                                   hue_name=hue_name)
        benchmark = y_column
        y_column = str(prices.name)

    fig = plot_scatter(df=scatter_data,
                       x=benchmark,
                       y=y_column,
                       xlabel=xlabel or benchmark,
                       ylabel=ylabel,
                       hue=hue_name,
                       xvar_format=var_format,
                       yvar_format=var_format,
                       add_universe_model_label=False,
                       add_universe_model_prediction=False,
                       add_universe_model_ci=False,
                       add_hue_model_label=add_hue_model_label,
                       add_45line=add_45line,
                       title=title,
                       order=order,
                       ci=ci,
                       ax=ax,
                       **kwargs)
    return fig


class UnitTests(Enum):
    RETURNS = 1
    RETURNS2 = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.RETURNS:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)
        plot_returns_scatter(prices=prices,
                             benchmark='SPY',
                             var_format='{:.2%}',
                             ax=ax,
                             **global_kwargs)

    elif unit_test == UnitTests.RETURNS2:

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        plot_returns_scatter(prices=prices[['SPY', 'TLT']],
                             benchmark='TLT',
                             y_column='benchmarks',
                             ylabel='SPY',
                             var_format='{:.2%}',
                             ax=ax,
                             **global_kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.RETURNS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
