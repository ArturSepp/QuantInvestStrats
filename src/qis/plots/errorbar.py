"""
point estimates with error bars, one series per column of a frame, drawn through ``ax.errorbar``.
``plot_errorbar`` is the only entry point: ``y_std_errors`` sets the bar half-widths as a scalar,
as a Series applied to every column, or as a frame read column by column, and ``exact`` overlays
a reference series as scatter points.
"""

# packages
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional
# qis
import qis.plots.utils as put


def _require_matching_error_labels(error_labels: pd.Index,
                                   estimate_labels: pd.Index,
                                   axis_name: str
                                   ) -> None:
    """Require one error axis to identify every estimate exactly once."""
    if error_labels.has_duplicates:
        raise ValueError(f'y_std_errors {axis_name} must not contain duplicate labels')
    if (len(error_labels) != len(estimate_labels)
            or not error_labels.isin(estimate_labels).all()
            or not estimate_labels.isin(error_labels).all()):
        raise ValueError(
            f'y_std_errors {axis_name} must contain exactly the same labels as df {axis_name}'
        )


def _align_y_std_errors(df: pd.DataFrame,
                        y_std_errors: Union[float, pd.Series, pd.DataFrame]
                        ) -> Union[float, pd.Series, pd.DataFrame]:
    """Validate and align labelled error magnitudes to plotted estimates."""
    if isinstance(y_std_errors, pd.Series):
        _require_matching_error_labels(y_std_errors.index, df.index, 'index')
        return y_std_errors.reindex(df.index)
    if isinstance(y_std_errors, pd.DataFrame):
        _require_matching_error_labels(y_std_errors.index, df.index, 'index')
        _require_matching_error_labels(y_std_errors.columns, df.columns, 'columns')
        return y_std_errors.reindex(index=df.index, columns=df.columns)
    return y_std_errors


def plot_errorbar(df: Union[pd.Series, pd.DataFrame],
                  y_std_errors: Union[float, pd.Series, pd.DataFrame] = 0.5,
                  exact: Union[pd.Series, pd.DataFrame] = None,  # can add exact solution
                  legend_title: str = None,
                  legend_loc: Optional[Union[str, bool]] = 'upper left',
                  xlabel: str = None,
                  ylabel: str = None,
                  var_format: Optional[str] = '{:.0f}',
                  title: Union[str, bool] = None,
                  fontsize: int = 10,
                  capsize: int = 10,
                  colors: List[str] = None,
                  exact_colors: Union[str, List[str]] = 'green',
                  marker: Optional[str] = 'o',
                  exact_marker: str = "v",
                  y_limits: Tuple[Optional[float], Optional[float]] = None,
                  add_zero_line: bool = False,
                  ax: plt.Subplot = None,
                  **kwargs
                  ) -> Optional[plt.Figure]:
    """Plot labelled point estimates with optional error and reference values.

    Args:
        df: Point estimates, plotted as one series per column.
        y_std_errors: Scalar error magnitude, a Series shared across columns, or a DataFrame with
            one error series per estimate column. Pandas errors must have unique labels that match
            the estimate rows exactly; DataFrame errors must also match the estimate columns.
        exact: Optional exact values overlaid as scatter points.

    Returns:
        A newly created figure, or ``None`` when drawing on a supplied axis.

    Raises:
        ValueError: If labelled errors contain duplicate, missing, or extra row or column labels.
    """

    if not df.empty:
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif not isinstance(df, pd.DataFrame):
            raise TypeError(f"unsupported data type {type(df)}")

        # Validate before creating a figure so rejected labels leave no open pyplot state.
        y_std_errors = _align_y_std_errors(df=df, y_std_errors=y_std_errors)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return fig

    columns = df.columns

    if colors is None:
        colors = put.get_n_colors(n=len(columns), **kwargs)

    for idx, column in enumerate(columns):
        if isinstance(y_std_errors, pd.DataFrame):
            yerr = y_std_errors[column].to_numpy()
        elif isinstance(y_std_errors, pd.Series):
            yerr = y_std_errors.to_numpy()
        else:
            yerr = y_std_errors

        ax.errorbar(x=df.index, y=df[column].to_numpy(), yerr=yerr, color=colors[idx], fmt=marker, capsize=capsize)

    if exact is not None:  # add exact as scatter points

        if isinstance(exact, pd.Series):
            exact = exact.to_frame()

        labels = columns.to_list()
        markers = [marker] * len(columns)
        if isinstance(exact_colors, str):
            exact_colors = [exact_colors] * len(columns)
        for idx1, column in enumerate(exact.columns):
            for idx, index in enumerate(df.index):
                put.add_scatter_points(ax=ax,
                                       label_x_y=[(index, exact.loc[index, column])],
                                       color=exact_colors[idx1],
                                       marker=exact_marker, **kwargs)
            labels = labels + [column]
            colors = colors + [exact_colors[idx1]]
            markers = markers + [exact_marker]
    else:
        labels = columns
        markers = [marker]*len(columns)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize)

    if legend_loc is not None:
        put.set_legend(ax=ax,
                       markers=markers,
                       labels=labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       legend_title=legend_title,
                       handlelength=0,
                       fontsize=fontsize,
                       **kwargs)

    else:
        ax.legend().set_visible(False)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)

    if var_format is not None:
        put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=None, yvar_format=var_format, **kwargs)
    else:
        put.set_ax_ticks_format(ax=ax, fontsize=fontsize, **kwargs)
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)
    put.set_spines(ax=ax, **kwargs)

    if add_zero_line:
        ax.axhline(0, color='black', lw=1)

    return fig
