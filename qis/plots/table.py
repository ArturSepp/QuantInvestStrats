"""
plot df as table
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table as Table
from enum import Enum
from typing import List, Tuple, Optional, Literal

# qis
import qis.plots.utils as put
import qis.utils.df_str as dfs

ROW_HIGHT = 0.625  # cm
COLUMN_WIDTH = 2.0  # cm ?
FIRST_COLUMN_WIDTH = 3.0


def plot_df_table(df: pd.DataFrame,
                  add_index_as_column: bool = True,
                  column_width: float = COLUMN_WIDTH,
                  first_column_width: Optional[float] = FIRST_COLUMN_WIDTH,
                  row_height: float = ROW_HIGHT,
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
                  columns_edge_lines: List[Tuple[int, str]] = None,
                  bold_font: bool = False,
                  linewidth: float = 0.5,  # table borders
                  alpha: float = 1.0,
                  emply_column_names: bool = False,
                  ax: plt.Subplot = None,
                  **kwargs
                  ) -> Optional[plt.Figure]:
    """
    plot dataframe as maotplotlib table
    """
    df = df.copy()  # data object will be changed
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

    if first_column_width is None:
        first_column_width = col_widths[0]
    #else:
    #    col_widths[0] = first_column_width # after this change

    if ax is None:  # create new axis
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    else:  # add table to existing axis
        fig = None
        ax.axis('off')

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
            x_array = dfs.series_to_numeric(ds=column_data)
            colors = put.compute_heatmap_colors(a=x_array, cmap=cmap)
            for k, cell in mpl_table._cells.items():
                if k[0] > 0 and k[1] == heatmap_column:  # skip first row
                    cell.set_facecolor(colors[k[0]-1])

    if heatmap_rows is not None:
        for heatmap_row in heatmap_rows:
            row_data = df.iloc[heatmap_row, 1:]  # exclude first row
            x_array = dfs.series_to_numeric(ds=row_data)
            colors = put.compute_heatmap_colors(a=x_array, cmap=cmap)
            for k, cell in mpl_table._cells.items():
                if k[1] > 0 and k[0] == heatmap_row+1:  # heatmap_row is not counting first headers
                    cell.set_facecolor(colors[k[1]-1])

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
            ax.axhline(y=rows_edge_line, color='black', alpha=0.5*alpha)

    if columns_edge_lines is not None:
        for columns_edge_line in columns_edge_lines:
            ax.axvline(x=columns_edge_line[0], color=columns_edge_line[1], alpha=0.5*alpha)
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
            if bold_font:
                txt = cell.get_text()
                txt.set_fontweight("bold")

        elif column is None:  # set solor to row
            if k[0] == row:
                cell.set_facecolor(color)
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
                             bold_font: bool = False
                             ) -> None:
    for k, cell in table._cells.items():
        if k[0] == k[1] and k[0] > 0:
            cell.set_facecolor(color)
            if bold_font:
                txt = cell.get_text()
                txt.set_fontweight("bold")


def set_data_colors(table: Table,
                    data_colors: np.ndarray,
                    header_row_id: int = 0,
                    header_column_id = 0,
                    bold_font: bool = False
                    ) -> None:
    for k, cell in table._cells.items():
        if k[1] > header_column_id and k[0] > header_row_id:
            cell.set_facecolor(data_colors[k[0]-1][k[1]-1])
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


class UnitTests(Enum):
    TABLE = 0


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.TABLE:

        cars = {'Brand': ['Honda Civic', 'Toyota Corolla', 'Ford Focus', 'Audi A4'],
                'Price': [220.0, 250.0, 270.0, 35.0],
                'Engine': [175.0, 300.0, 100.0, 500.0],
                'Speed': [200.0, 150.0, 200.0, 175.0]}

        data = pd.DataFrame.from_dict(cars)
        data = data.set_index('Brand', drop=False)
        print(data)
        data['Price'] = data['Price']
        data['Engine'] = data['Engine']
        plot_df_table(df=data, heatmap_columns=[2], bold_font=False)
        plot_df_table(df=data, heatmap_rows_columns=((0, len(data.index)), (3, 4)))

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.TABLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
