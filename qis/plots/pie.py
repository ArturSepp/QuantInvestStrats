"""
a pie chart of one column of a frame, wedges labelled by the index and sized by value.
``plot_pie`` is the only entry point: ``y_column`` picks the column and defaults to the first,
and ``autopct`` formats the percentage written on each wedge.
"""
# packages
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Union

# qis
import qis.plots.utils as put


def plot_pie(df: Union[pd.Series, pd.DataFrame],
             y_column: str = None,
             ylabel: str = '',
             title: str = None,
             colors: List[str] = None,
             legend_loc: Optional[str] = None,
             autopct: Optional[str] = '%.0f%%',
             ax: plt.Subplot = None,
             **kwargs
             ) -> Optional[plt.Figure]:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return fig

    if y_column is None and isinstance(df, pd.DataFrame):
        y_column = df.columns[0]
    if colors is None:
        colors = put.get_cmap_colors(n=len(df.index), **kwargs)

    df.plot.pie(y=y_column, autopct=autopct, colors=colors, ax=ax)

    if legend_loc is None:
        ax.legend().set_visible(False)

    if title is not None:
        put.set_title(ax=ax, title=title, **kwargs)

    ax.set_ylabel(ylabel)

    return fig
