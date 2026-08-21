"""
returns of one or more assets scattered against a benchmark's, with a fitted relationship.
``plot_returns_scatter`` is the only entry point: prices are converted to returns at ``freq``,
optionally divided by their own standard deviation under ``is_vol_norm``, melted into long form
and drawn by ``plot_scatter`` with a degree-``order`` fit and a ``ci`` confidence band.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union

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
            price_data_full = pd.concat([benchmark_prices, prices], axis=1, sort=True)
            benchmark = benchmark_prices.name
            benchmark_prices = None
        else:  # for df price data must be sries
            if not isinstance(prices, pd.Series):
                raise ValueError(f"must be series\n{prices}")
            price_data_full = pd.concat([prices, benchmark_prices], axis=1, sort=True)

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
