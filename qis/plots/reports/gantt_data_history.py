"""
plot gantt figure showing price data history
requires plotly package
"""

import numpy as np
import pandas as pd
from plotly import figure_factory as ff, graph_objects as go
from typing import List, Optional


def plot_price_history(prices: pd.DataFrame,
                       descriptions: Optional[pd.Series] = None,
                       dates_for_lines: Optional[List[pd.Timestamp]] = (pd.Timestamp('01Jan2005'), pd.Timestamp('01Jan2020'), ),
                       resource_name: str = 'Price',
                       file_path_to_save: Optional[str] = None
                       ) -> go.Figure:
    """
    plot gantt figure showing data history
    """
    dicts = []
    for asset in prices.columns:
        if descriptions is not None:
            description = f"{asset}, {descriptions.loc[asset]}"
        else:
            description = f"{asset}"

        # get data without nans
        asset_data = prices[asset].dropna()
        if asset_data.empty:
            data_start = np.nan
            data_end = np.nan
        else:
            data_start = asset_data.index[0]
            data_end = asset_data.index[-1]

        bbg_dict = dict(Task=description, Start=data_start, Finish=data_end, Resource=resource_name)
        dicts.append(bbg_dict)

    colors = {resource_name: 'rgb(0, 255, 100)',
              'NA': 'rgb(100, 100, 100)'}

    fig = ff.create_gantt(dicts, colors=colors, index_col='Resource', show_colorbar=True,
                          group_tasks=True)

    # add date lines
    if dates_for_lines is not None:
        for current_date in dates_for_lines:
            fig['layout']['shapes'] += tuple([
                {
                    'type': 'line',
                    'x0': current_date,
                    'y0': -1,
                    'x1': current_date,
                    'y1': len(dicts) - 1,
                    'line': {
                        'color': 'rgb(239, 90, 19)',
                        'width': 3,
                    },
                }
            ])

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
    )
    if file_path_to_save is not None:
        fig.write_image(file_path_to_save)
    return fig


def plot_backfill_history(backfill_data: pd.DataFrame,
                          bbg_data: pd.DataFrame,
                          futures_backfills: pd.DataFrame,
                          instrument_names: pd.Series = None,
                          dates_for_lines: Optional[List[pd.Timestamp]] = (pd.Timestamp('01Jan1962'),
                                                                           pd.Timestamp('01Jan1971'),
                                                                           pd.Timestamp('01Jan1992'),),
                          file_path_to_save: Optional[str] = None
                          ) -> go.Figure:
    """
    plot data history with backfills
    instrument_names aligned with backfill_data.columns
    """
    if instrument_names is not None:
        # descriptions = [f"{asset}, {instrument_names[asset]}" for asset in backfill_data.columns]
        descriptions = [f"{instrument_names[asset]}" for asset in backfill_data.columns]
    else:
        descriptions = [f"{asset}" for asset in backfill_data.columns]

    is_in_bb_data = np.isin(backfill_data.columns.to_list(), bbg_data.columns.to_list(), assume_unique=True)

    dicts = []
    is_add_cut_off = True  # to limit bloomberg backfill indicator to universes start
    for asset, description, in_bb in zip(backfill_data.columns, descriptions, is_in_bb_data):

        # 1 check backfill data
        if in_bb and np.all(np.isnan(backfill_data[asset])) == False:

            # get bloomberg time series start
            asset_data = bbg_data[asset].dropna()
            bbg_start = asset_data.index[0]

            if is_add_cut_off:
                const_data_start = backfill_data[asset].dropna().index[0]
                if bbg_start < const_data_start:
                    bbg_start = const_data_start

            bbg_dict = dict(Task=description, Start=bbg_start, Finish=asset_data.index[-1], Resource='Bloomberg')
            dicts.append(bbg_dict)

            # now check if it is art of universes
            non_nan_backfill_data = backfill_data[asset].dropna()
            if non_nan_backfill_data.index[0] < bbg_start:
                futures_backfill = futures_backfills[asset].dropna()
                if not futures_backfill.empty:
                    # first if future backfill
                    backfill_dict = dict(Task=description, Start=futures_backfill.index[0], Finish=bbg_start, Resource='Futures Backfill')
                    dicts.append(backfill_dict)
                    # then cash backfill
                    backfill_dict = dict(Task=description, Start=non_nan_backfill_data.index[0], Finish=futures_backfill.index[0], Resource='Cash Backfill')
                    dicts.append(backfill_dict)
                else:
                    backfill_dict = dict(Task=description, Start=non_nan_backfill_data.index[0], Finish=bbg_start, Resource='Cash Backfill')
                    dicts.append(backfill_dict)

        elif asset in bbg_data.columns:
            asset_data = bbg_data[asset].dropna()
            this2 = dict(Task=description, Start=asset_data.index[0], Finish=asset_data.index[-1], Resource='Bloomberg')
            dicts.append(this2)

        elif (asset in backfill_data.columns and np.all(backfill_data[asset].isnull()) is False):
            asset_data = backfill_data[asset].dropna()
            this2 = dict(Task=description, Start=asset_data.index[0], Finish=asset_data.index[-1], Resource='Cash Backfill')
            dicts.append(this2)

    colors = {'Cash Backfill': 'rgb(220, 0, 0)',
              'Futures Backfill': 'rgb(100, 100, 100)',
              'Bloomberg': 'rgb(0, 255, 100)',
              'NA': 'rgb(100, 100, 100)'}

    fig = ff.create_gantt(dicts, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)

    # add current line
    if dates_for_lines is not None:
        for current_date in dates_for_lines:
            fig['layout']['shapes'] += tuple([
                {
                    'type': 'line',
                    'x0': current_date,
                    'y0': -1,
                    'x1': current_date,
                    'y1': len(dicts)-1,
                    'line': {
                        'color': 'rgb(239, 90, 19)',
                        'width': 3,
                    },
                }
            ])

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
    )
    if file_path_to_save is not None:
        fig.write_image(file_path_to_save)
    return fig
