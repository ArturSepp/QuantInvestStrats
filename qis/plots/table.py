"""
a DataFrame rendered as a matplotlib table, so it composes into a page beside the charts.

``plot_df_table`` lays out and colours; it does not compute. Values arrive already formatted as
strings unless ``var_format`` is given. The figure size is built from ``row_height`` and the
first entry of ``col_widths`` (``column_width`` by default), then passed to ``plt.subplots`` as
a figsize, so the units are inches; ``first_column_width`` sets the leading cell, not the figure.
``plot_df_table_with_ci`` merges a frame of estimates with a frame of confidence intervals into
one cell string and colours the cells by the estimate.

Colouring is what makes the grid readable: ``heatmap_columns`` shades down a column,
``special_rows_colors`` picks out a total row, ``rows_edge_lines`` separates groups, and the
``set_cells_facecolor`` family applies the same to an already-built table. Shared arguments are
in ``qis/docs/plotting_kwargs.md``. A numeric frame coloured wholesale is ``qis.plots.heatmap``.
"""
# packages
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table as Table
from typing import List, Tuple, Optional, Literal, Union

# qis
import qis.plots.utils as put
import qis.utils.df_str as dfs

ROW_HIGHT = 0.625  # cm
COLUMN_WIDTH = 2.0  # cm ?
FIRST_COLUMN_WIDTH = 3.0


def plot_df_table(df: Union[pd.DataFrame, pd.Series],
                  add_index_as_column: bool = True,
                  column_width: float = COLUMN_WIDTH,
                  row_height: float = ROW_HIGHT,
                  first_column_width: Optional[float] = FIRST_COLUMN_WIDTH,
                  first_row_height: float = None,
                  col_widths: List[float] = None,  # can pass as cols
                  rotation_for_columns_headers: int = None,
                  rotation_for_text: int = None,
                  transpose: bool = False,
                  index_column_name: str = ' ',
                  fontsize: int = 10,
                  header_color: str = '#40466e',
                  header_text_color: str = 'w',
                  row_colors: List[str] = ('#f1f1f2', 'w'),
                  edge_color: str = 'lightgray',
                  bbox: Tuple[float] = (0, 0, 1, 1),
                  header_column_id: int = 0,
                  header_row_id: int = 0,
                  left_aligned_first_col: bool = False,
                  var_format: str = None,  # '{:.2f}' to convert numerical data to str
                  title: str = None,
                  heatmap_columns: List[int] = None,
                  heatmap_rows: List[int] = None,
                  heatmap_rows_columns: Tuple[Tuple[int, int], Tuple[int, int]] = None,  # row[0]: row[1], column[0]:column[1]
                  cmap: str = 'RdYlGn',
                  special_rows_colors: List[Tuple[int, str]] = None,
                  special_columns_colors: List[Tuple[int, str]] = None,
                  data_colors: List[Tuple[float, float, float]] = None,
                  diagonal_color: str = None,
                  rows_edge_lines: List[int] = None,
                  rows_edge_color: str = 'blue',
                  columns_edge_lines: List[Tuple[int, str]] = None,
                  bold_font: bool = False,
                  linewidth: float = 0.5,  # table borders
                  alpha: float = 1.0,
                  emply_column_names: bool = False,
                  ax: plt.Subplot = None,
                  **kwargs
                  ) -> Optional[plt.Figure]:
    """
    render a DataFrame as a matplotlib table, with optional per-cell colouring.

    A table drawn as a figure rather than as text, so it composes into a multi-panel factsheet
    page beside the charts it summarises and exports into the same PDF. Values are formatted
    before they arrive, so this function lays out and colours; it does not compute.

    The colouring arguments are what make it more than a grid. ``heatmap_columns`` shades down a
    column so ranks are visible without reading the numbers; ``special_rows_colors`` picks out a
    total or a benchmark row; ``rows_edge_lines`` separates groups.

    Arguments shared with every ``plot_*`` function are documented in
    ``qis/docs/plotting_kwargs.md``.

    Args:
        df: values to render, already formatted as strings where display matters
        add_index_as_column: draw the index as the leading column
        column_width: width of a data column, in the table's own units
        row_height: height of a data row
        first_column_width: width of the leading column, usually wider because it holds names
        first_row_height: height of the header row. None follows ``row_height``
        col_widths: explicit width per column, overriding ``column_width``
        rotation_for_columns_headers: rotation in degrees of the header text
        rotation_for_text: rotation in degrees of the cell text
        transpose: swap rows and columns before rendering
        index_column_name: header of the index column. The default is a space, which leaves
            the corner cell visually empty without collapsing it
        header_color: fill colour of the header row
        header_text_color: text colour of the header row
        row_colors: colours cycled down the data rows, giving the banded look
        edge_color: colour of the cell borders
        bbox: (x0, y0, width, height) of the table within the axis, in axis coordinates
        header_column_id: index of the column treated as a header
        header_row_id: index of the row treated as a header
        left_aligned_first_col: left-align the leading column, which reads better for names
        heatmap_columns: positions of columns to colour by value down the column
        heatmap_rows: positions of rows to colour by value across the row
        heatmap_rows_columns: (rows, columns) block coloured on its own joint scale
        cmap: colour map for the heatmap regions
        special_rows_colors: (position, colour) pairs overriding the banding for a row
        special_columns_colors: (position, colour) pairs overriding the banding for a column
        data_colors: explicit colour per cell, same shape as the data
        diagonal_color: fill for the leading diagonal, for a correlation or transition matrix
        rows_edge_lines: positions after which to draw a horizontal separator
        rows_edge_color: colour of those separators
        columns_edge_lines: positions after which to draw a vertical separator
        bold_font: render all text bold
        linewidth: width of the cell borders
        alpha: cell opacity
        emply_column_names: blank the header text while keeping the header row, for a table
            whose columns are labelled by an adjacent panel

    Returns:
        the figure drawn on, or None when ``ax`` was supplied or ``df`` is empty
    """
    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return None

    df = df.copy()  # data object will be changed
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if transpose:
        t_data = df.T
        if add_index_as_column:
            t_data.columns = t_data.iloc[0, :]  # rename columns by original index
            df = t_data.drop(labels=t_data.index[0], axis=0)
        else:
            t_data.columns = df.index  # rename columns using original index
            df = t_data

    if add_index_as_column:
        df.insert(0, column=index_column_name, value=df.index)

    if col_widths is None:
        col_widths = [column_width for _ in df]
    else:
        first_column_width = col_widths[0]

    # allocate size
    size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array([col_widths[0], row_height])

    if ax is None:  # create new axis
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    else:  # add table to existing axis
        fig = None
        ax.axis('off')

    if first_column_width is None:
        first_column_width = col_widths[0]
    #else:
    #    col_widths[0] = first_column_width # after this change

    if var_format is not None:
        df = dfs.df_to_str(df=df, var_format=var_format)

    if emply_column_names is False:
        col_labels = df.columns.to_list()
    else:
        col_labels = [df.columns[0]] + [''] * (len(df.columns) - 1)

    mpl_table = ax.table(cellText=df.to_numpy(),
                         bbox=bbox,
                         colLabels=col_labels,
                         colLoc='center')

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(fontsize)

    weight = 'bold' if bold_font else 'normal'
    for k, cell in mpl_table._cells.items():
        cell.set_linewidth(0.5)

        if k[0] == header_row_id or k[1] < header_column_id:
            cell.set_text_props(weight=weight, color=header_text_color)
            cell.set_facecolor(header_color)
            if first_row_height is not None:
                cell.set_height(first_row_height)

            if rotation_for_columns_headers is not None:
                cell.get_text().set_rotation(rotation_for_columns_headers)  # set rotation
                # cell.set_height(first_column_width / 15)  # aling widths

        elif k[0] > header_row_id:
            cell.set_facecolor(row_colors[k[0] % len(row_colors) ])

        if k[1] == 0:
            width = first_column_width
        else:
            width = col_widths[k[1]]
        cell.set_width(width)
        if rotation_for_text is not None:
            cell.get_text().set_rotation(rotation_for_text)  # set rotation

    if heatmap_columns is not None:
        for heatmap_column in heatmap_columns:
            column_data = df[df.columns[heatmap_column]]
            if isinstance(column_data, pd.DataFrame):
                raise ValueError(f"dublicated columns: {column_data.columns}")
            x_array = dfs.series_to_numeric(ds=column_data)
            colors = put.compute_heatmap_colors(a=x_array, cmap=cmap)
            for k, cell in mpl_table._cells.items():
                if k[0] > 0 and k[1] == heatmap_column:  # skip first row
                    cell.set_facecolor(colors[k[0]-1])
                    cell.set_alpha(alpha)

    if heatmap_rows is not None:
        for heatmap_row in heatmap_rows:
            row_data = df.iloc[heatmap_row, 1:]  # exclude first row
            x_array = dfs.series_to_numeric(ds=row_data)
            colors = put.compute_heatmap_colors(a=x_array, cmap=cmap)
            for k, cell in mpl_table._cells.items():
                if k[1] > 0 and k[0] == heatmap_row+1:  # heatmap_row is not counting first headers
                    cell.set_facecolor(colors[k[1]-1])
                    cell.set_alpha(alpha)

    if heatmap_rows_columns is not None:   # row[0]: row[1], column[0]:column[1]
        # set colors on all data
        row_start, row_end = heatmap_rows_columns[0][0], heatmap_rows_columns[0][1]
        col_start, col_end = heatmap_rows_columns[1][0], heatmap_rows_columns[1][1]

        if add_index_as_column:
            col_start = col_start + 1
            col_end = col_end + 1
        else:
            col_start = col_start - 1

        data_extract = df.iloc[row_start: row_end, col_start: col_end]
        a = dfs.df_to_numeric(df=data_extract)
        colors = put.compute_heatmap_colors(a=a, cmap=cmap)
        for row_idx in range(len(data_extract.index)):
            col_idx = 0
            for k, cell in mpl_table._cells.items():
                if k[0] == row_start + row_idx + 1 and k[1] >= col_start and k[1] < col_end:
                    cell.set_facecolor(colors[row_idx][col_idx])
                    cell.set_alpha(alpha)
                    col_idx += 1

    if special_rows_colors is not None:
        for special_rows_color in special_rows_colors:
            set_cells_facecolor(mpl_table,
                                row=special_rows_color[0],
                                color=special_rows_color[1],
                                alpha=0.5*alpha,
                                bold_font=bold_font)

    if special_columns_colors is not None:
        for special_columns_color in special_columns_colors:
            set_cells_facecolor(mpl_table,
                                column=special_columns_color[0],
                                color=special_columns_color[1],
                                alpha=0.5*alpha,
                                bold_font=bold_font,
                                header_row_id=header_row_id)

    if diagonal_color is not None:
        set_diag_cells_facecolor(mpl_table, color=diagonal_color)

    if data_colors is not None:
        set_data_colors(mpl_table, header_column_id=0 if add_index_as_column else -1,
                        data_colors=data_colors)

    set_row_edge_color(mpl_table, row=None, color=edge_color)

    if left_aligned_first_col:
        set_align_for_column(mpl_table, col=0, align='left')
    else:
        set_align_for_column(mpl_table, col=0, align='right')

    if rows_edge_lines is not None or columns_edge_lines is not None:
        ax.axis(xmin=0, xmax=df.shape[1], ymin=df.shape[0], ymax=-1)  # need to reset axis to match rows position

    if rows_edge_lines is not None:
        for rows_edge_line in rows_edge_lines:
            ax.axhline(y=rows_edge_line, color=rows_edge_color, alpha=0.5*alpha, lw=0.75)

    if columns_edge_lines is not None:
        for columns_edge_line in columns_edge_lines:
            ax.axvline(x=columns_edge_line[0], color=columns_edge_line[1], alpha=0.5*alpha, lw=0.75)
            # set_column_edge_color(mpl_table, column=columns_edge_line[0], color=columns_edge_line[1])

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    return fig


def plot_df_table_with_ci(df: pd.DataFrame,
                          df_ci: pd.DataFrame,
                          var_format: str = '{:.2f}',
                          ax: plt.Subplot = None,
                          is_add_heatmap: bool = True,
                          axis: Literal[None, 0, 1] = 1,  # heatmap by column
                          **kwargs
                          ) -> plt.Figure:
    """
    table with ci
    """
    table_str = dfs.df_with_ci_to_str(df=df,
                                      df_ci=df_ci,
                                      var_format=var_format)
    if is_add_heatmap:
        data_colors = put.compute_heatmap_colors(a=df.to_numpy(), axis=axis)
    else:
        data_colors = None

    fig = plot_df_table(df=table_str,
                        data_colors=data_colors,
                        ax=ax,
                        **kwargs)
    return fig


def set_row_edge_color(table: Table,
                       row: int = None,
                       color: str = 'slategray'
                       ) -> None:
    for k, cell in table._cells.items():
        if row is None:
            cell.set_edgecolor(color)
        else:
            if k[0] == row:
                cell.set_edgecolor(color)


def set_column_edge_color(table: Table,
                          column: int = None,
                          color: str = 'slategray'
                          ) -> None:
    for k, cell in table._cells.items():
        if column is None:
            cell.set_edgecolor(color)
        else:
            if k[1] == column:
                cell.set_edgecolor(color)
                cell.visible_edges = "L"


def set_cells_facecolor(table: Table,
                        header_row_id: int = 0,
                        row: int = None,
                        column: int = None,
                        color: str = 'slategray',
                        alpha: float = 1.0,
                        bold_font: bool = False
                        ) -> None:

    for k, cell in table._cells.items():
        if row is None and column is None:
            cell.set_facecolor(color)
            cell.set_alpha(alpha)
            if bold_font:
                txt = cell.get_text()
                txt.set_fontweight("bold")

        elif column is None:  # set solor to row
            if k[0] == row:
                cell.set_facecolor(color)
                cell.set_alpha(alpha)
                if bold_font:
                    txt = cell.get_text()
                    txt.set_fontweight("bold")

        elif row is None:  # set solor to column
            if k[1] == column and k[0] > header_row_id:
                cell.set_facecolor(color)
                cell.set_alpha(alpha)
                if bold_font:
                    txt = cell.get_text()
                    txt.set_fontweight("bold")


def set_diag_cells_facecolor(table: Table,
                             color: str = 'slategray',
                             bold_font: bool = False,
                             alpha: float = 1.0,
                             ) -> None:
    for k, cell in table._cells.items():
        if k[0] == k[1] and k[0] > 0:
            cell.set_facecolor(color)
            cell.set_alpha(alpha)
            if bold_font:
                txt = cell.get_text()
                txt.set_fontweight("bold")


def set_data_colors(table: Table,
                    data_colors: List[Tuple[float, float, float]],
                    header_row_id: int = 0,
                    header_column_id = 0,
                    bold_font: bool = False,
                    alpha: float = 1.0
                    ) -> None:
    for k, cell in table._cells.items():
        if k[1] > header_column_id and k[0] > header_row_id:
            cell.set_facecolor(data_colors[k[0]-1][k[1]-1])
            cell.set_alpha(alpha)
            if bold_font:
                txt = cell.get_text()
                txt.set_fontweight("bold")


def set_align_for_column(table: Table,
                         col: int,
                         align: str = 'left'
                         ) -> None:
    cells = [key for key in table._cells if key[1] == col]
    for cell in cells:
        table.properties()["celld"][cell]._loc = align
