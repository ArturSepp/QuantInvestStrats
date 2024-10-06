import matplotlib.pyplot as plt
import pandas as pd
import qis as qis
import qis.plots.utils as put
from typing import Optional, Tuple


def generate_price_history_report(prices: pd.DataFrame,
                                  figsize: Tuple[float, float] = (8.3, 11.7),
                                  **kwargs
                                  ) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
    qis.plot_ra_perf_table(prices=prices,
                           title='Risk-Adjusted Performance',
                           ax=axs[0],
                           **kwargs)
    plot_price_history(prices=prices,
                       title='Price History',
                       ax=axs[1],
                       **kwargs)
    return fig


def plot_price_history(prices: pd.DataFrame,
                       title: str = None,
                       #date_format: str = '%d-%b-%y',
                       ax: plt.Subplot = None,
                       **kwargs
                       ) -> Optional[plt.Figure]:

    start_dates = {}
    end_dates = {}
    durations = {}
    for asset in prices.columns:
        price = prices[asset].dropna()
        if not price.empty:
            start_dates[asset] = price.index[0]
            end_dates[asset] = price.index[-1]
            durations[asset] = qis.get_time_to_maturity(maturity_time=price.index[-1],
                                                        value_time=price.index[0])

    start_dates = pd.Series(start_dates).rename('Start')
    end_dates = pd.Series(end_dates).rename('End')
    durations = pd.Series(durations).rename('Period')
    df = pd.concat([start_dates, end_dates, durations], axis=1)
    df = df.iloc[::-1]  # reverse index
    if ax is None:
        width, height, _, _ = put.calc_df_table_size(df=df, min_rows=len(df.index), min_cols=len(df.index)//2)
        fig, ax = plt.subplots(1, 1, figsize=(width, height), constrained_layout=True)
    else:
        fig = None

    ax.hlines(df.index, xmin=df['Start'], xmax=df['End'])

    # put.set_ax_tick_params(ax=ax)
    put.set_ax_ticks_format(ax=ax, **qis.update_kwargs(kwargs, dict()))

    if title is not None:
        put.set_title(ax=ax, title=title, **kwargs)
    ax.margins(x=0.015, y=0.015)
    return fig
