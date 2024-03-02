"""
plot returns heatmap table by monthly and annual
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from matplotlib.ticker import FuncFormatter
from typing import Optional

# qis
import qis.plots.utils as put
import qis.perfstats.cond_regression as cre
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs, BenchmarkReturnsQuantilesRegime


def plot_scatter_regression(prices: pd.DataFrame,
                            regime_benchmark_str: str,
                            regime_params: BenchmarkReturnsQuantileRegimeSpecs = BenchmarkReturnsQuantileRegimeSpecs(),
                            drop_benchmark: bool = True,
                            x_var_format: str = '{:.0%}',
                            y_var_format: str = '{:.0%}',
                            beta_format: str = '{:.1f}',
                            xlabel: Optional[str] = None,
                            ylabel: Optional[str] = None,
                            title: Optional[str] = None,
                            is_asset_detailed: bool = False,
                            is_print_summary: bool = False,
                            add_last_date: bool = False,
                            ax: plt.Subplot = None,
                            **kwargs
                            ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    regime_classifier = BenchmarkReturnsQuantilesRegime(regime_params=regime_params)

    regmodel_out_dict = {}
    estimated_params = cre.estimate_cond_regression(prices=prices,
                                                    benchmark=regime_benchmark_str,
                                                    drop_benchmark=drop_benchmark,
                                                    regime_params=regime_params,
                                                    is_print_summary=is_print_summary,
                                                    regmodel_out_dict=regmodel_out_dict)

    markers = put.get_n_markers(n=len(estimated_params.index))
    colors = put.get_n_colors(n=len(estimated_params.index))
    lines = []
    for (asset, pandas_out), marker, color_ in zip(regmodel_out_dict.items(), markers, colors):

        pandas_out = pandas_out.sort_values(regime_benchmark_str)  # sort for line plot
        sns.scatterplot(x=regime_benchmark_str,
                        y=asset,
                        hue=pandas_out[BenchmarkReturnsQuantilesRegime.REGIME_COLUMN],
                        data=pandas_out,
                        palette=list(regime_classifier.get_regime_ids_colors().values()),
                        hue_order=list(regime_classifier.get_regime_ids_colors().keys()),
                        marker=marker,
                        legend=None,
                        ax=ax)  # marker=marker

        if is_asset_detailed: # will be single strategy
            palette = list(regime_classifier.get_regime_ids_colors().values())
            legend = True
            line_marker = None

            for regime, color in regime_classifier.get_regime_ids_colors().items():
                lines.append((f"{regime} beta={beta_format.format(estimated_params.loc[asset, regime])}",
                              {'color': color, 'linestyle': '-', 'marker': line_marker}))
        else:
            palette = [color_] * len(regime_classifier.get_regime_ids_colors().keys())
            legend = None
            line_marker = marker
            lines.append((asset, {'color': color_, 'linestyle': '-', 'marker': line_marker}))

        sns.lineplot(x=regime_benchmark_str,
                     y=cre.PREDICTION,
                     hue=pandas_out[BenchmarkReturnsQuantilesRegime.REGIME_COLUMN],
                     data=pandas_out,
                     hue_order=list(regime_classifier.get_regime_ids_colors().keys()),
                     palette=palette,
                     marker=line_marker,
                     legend=legend, ax=ax)

    if add_last_date:
        label_x_y = {}
        for asset, df in regmodel_out_dict.items():
            x = df[regime_benchmark_str].iloc[-1]
            y = df[asset].iloc[-1]
            label = f"Last {df.index[-1].strftime('%d-%b-%Y')}: x={x_var_format.format(x)}, y={x_var_format.format(y)}"
            label_x_y[label] = (x, y)
        if len(prices.columns) == 2:
            colors = [df[cre.COLOR_COLUMN].iloc[-1]]  # last color
        else:
            colors = colors
        put.add_scatter_points(ax=ax, label_x_y=label_x_y, colors=colors, **kwargs)

    put.set_legend(ax=ax, lines=lines, **kwargs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: x_var_format.format(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: y_var_format.format(y)))

    if xlabel is None:
        xlabel = f"x={regime_benchmark_str}"
    if ylabel is None:
        if len(prices.columns) == 2:
            ylabel = f"y = {asset}"
        else:
            ylabel = 'Assets'
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, **kwargs)

    put.set_spines(ax=ax, **kwargs)

    return fig


class UnitTests(Enum):
    SCATTER_ALL = 1
    SCATTER_1 = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()
    regime_params = BenchmarkReturnsQuantileRegimeSpecs(freq='QE')

    if unit_test == UnitTests.SCATTER_ALL:
        plot_scatter_regression(prices=prices,
                                regime_benchmark_str='SPY',
                                regime_params=regime_params,
                                add_last_date=True,
                                is_asset_detailed=False)

    elif unit_test == UnitTests.SCATTER_1:
        plot_scatter_regression(prices=prices[['SPY', 'TLT']],
                                regime_benchmark_str='SPY',
                                regime_params=regime_params,
                                add_last_date=True,
                                is_asset_detailed=True)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.SCATTER_1

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
