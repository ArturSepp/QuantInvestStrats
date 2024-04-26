from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from enum import Enum

# qis
import qis.utils.df_cut as dfc
from qis.perfstats.config import RegimeData, PerfParams
from qis.perfstats.regime_classifier import RegimeClassifier, BenchmarkReturnsQuantileRegimeSpecs, \
    BenchmarkReturnsQuantilesRegime, VolQuantileRegimeSpecs, BenchmarkVolsQuantilesRegime, compute_bnb_regimes_pa_perf_table
import qis.plots.bars as pba
import qis.plots.boxplot as bxp


def plot_regime_data(regime_classifier: RegimeClassifier,
                     regime_data_to_plot: RegimeData = RegimeData.REGIME_SHARPE,
                     drop_sharpe_from_labels: bool = False,  # only leave the regime names
                     x_rotation: int = 0,
                     add_bar_values: bool = True,
                     title: Optional[str] = 'Conditional Excess Sharpe ratio',
                     var_format: str = '{:.1f}',
                     bbox_to_anchor: Optional[Tuple[float, float]] = (1.0, 0.95),
                     fontsize: int = 10,
                     is_top_totals: bool = True,
                     is_add_totals: bool = True,
                     is_use_vbar: bool = False,
                     reverse_columns: bool = True,
                     legend_loc: Optional[str] = 'upper right',
                     ax: plt.Subplot = None,
                     **kwargs
                     ) -> plt.Figure:

    regimes_pa_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(**kwargs)
    data = regime_datas[regime_data_to_plot]
    data = data.dropna(how='all', axis=0)  # remove data wit all rows nans

    if drop_sharpe_from_labels:
        data.columns = [x.replace(' Sharpe', '') for x in data.columns]

    regime_colors = list(regime_classifier.get_regime_ids_colors().values())

    if is_add_totals:
        totals = np.nansum(data.to_numpy(), axis=1)
    else:
        totals = None

    if is_use_vbar:
        fig = pba.plot_vbars(df=data,
                             x_rotation=x_rotation,
                             colors=regime_colors,
                             var_format=var_format,
                             add_bar_values=add_bar_values,
                             title=title,
                             totals=totals,
                             fontsize=fontsize,
                             bbox_to_anchor=bbox_to_anchor,
                             xmin_shift=-0.05,
                             x_step=0.5,
                             ax=ax,
                             **kwargs)
    else:
        fig = pba.plot_bars(df=data,
                            x_rotation=x_rotation,
                            colors=regime_colors,
                            xvar_format=var_format,
                            add_bar_values=add_bar_values,
                            title=title,
                            totals=totals,
                            is_top_totals=is_top_totals,
                            fontsize=fontsize,
                            bbox_to_anchor=bbox_to_anchor,
                            legend_loc=legend_loc,
                            reverse_columns=reverse_columns,
                            ax=ax,
                            **kwargs)
    return fig


def plot_regime_boxplot(regime_classifier: RegimeClassifier,
                        prices: pd.DataFrame,
                        benchmark: str,
                        hue_var_name: str = 'Asset',
                        x_index_var_name: str = 'Regime',
                        ylabel: str = 'avg regime return',
                        x_rotation: int = 0,
                        legend_loc: Optional[str] = 'upper left',
                        yvar_format: str = '{:.0%}',
                        meanline: bool = True,
                        ax: plt.Subplot = None,
                        **kwargs
                        ) -> plt.Figure:

    df = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices, benchmark=benchmark).iloc[1:, :]
    regime_ids_colors = regime_classifier.get_regime_ids_colors()
    df = df.set_index(regime_classifier.REGIME_COLUMN, drop=True) # boxplot b regime as hue
    df = dfc.sort_index_by_hue(df=df, hue_order=list(regime_ids_colors.keys()))
    fig = bxp.df_boxplot_by_hue_var(df=df,
                                    hue_var_name=hue_var_name,
                                    x_index_var_name=x_index_var_name,
                                    y_var_name=ylabel,
                                    add_zero_line=True,
                                    meanline=meanline,
                                    is_heatmap_colors=False,
                                    labels=prices.columns,
                                    legend_loc=legend_loc,
                                    x_rotation=x_rotation,
                                    yvar_format=yvar_format,
                                    ax=ax,
                                    **kwargs)
    return fig


def add_bnb_regime_shadows(ax: plt.Subplot,
                           data_df: pd.DataFrame = None,
                           benchmark: str = None,
                           pivot_prices: pd.Series = None,
                           regime_params: BenchmarkReturnsQuantileRegimeSpecs = None,
                           is_force_lim: bool = True,
                           alpha: float = 0.3,
                           **kwargs
                           ) -> None:

    if regime_params is None:
        regime_params = BenchmarkReturnsQuantileRegimeSpecs()
    regime_classifier = BenchmarkReturnsQuantilesRegime(regime_params=regime_params)

    if pivot_prices is not None:
        benchmark = pivot_prices.name
    elif benchmark is not None and data_df is not None:
        if benchmark in data_df.columns:
            pivot_prices = data_df[benchmark]
        else:
            raise KeyError(f"{benchmark} not in {data_df.columns}")
    else:
        raise ValueError(f"need pivot_prices or benchmark")

    regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(prices=pivot_prices,
                                                                          benchmark=benchmark,
                                                                          **regime_params._asdict())

    regime_id_color = regime_classifier.class_data_to_colors(regime_data=regime_ids[RegimeClassifier.REGIME_COLUMN])

    # fill in the first date before the class date
    price_data_index = pivot_prices.index

    ax.axvspan(xmin=price_data_index[0],
               xmax=regime_ids.index[0],
               alpha=alpha,
               color=regime_id_color.loc[regime_id_color.index[0]], lw=0)
    for date_1, date in zip(regime_ids.index[:-1], regime_ids.index[1:]):
        ax.axvspan(xmin=date_1, xmax=date, alpha=alpha, color=regime_id_color[date], lw=0)
    if is_force_lim:
        ax.set_xlim([price_data_index[0], regime_ids.index[-1]])


class UnitTests(Enum):
    BNB_REGIME = 1
    VOL_REGIME = 2
    BNB_REGIME_SHADOWS = 3
    BNB_PERF_TABLE = 4
    AVG_PLOT = 5


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    kwargs = dict(var_format='{:.1f}')

    perf_params = PerfParams()

    if unit_test == UnitTests.BNB_REGIME:
        regime_params = BenchmarkReturnsQuantileRegimeSpecs()
        regime_classifier = BenchmarkReturnsQuantilesRegime(regime_params=regime_params)
        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices, benchmark='SPY')
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(prices=prices,
                                                                                        benchmark='SPY',
                                                                                        perf_params=perf_params)
        print(f"regime_means:\n{cond_perf_table}")
        print(f"regime_pa:\n{regime_datas}")

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), tight_layout=True)

        plot_regime_data(regime_classifier=regime_classifier,
                         drop_sharpe_from_labels=True,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         title='(A) Weekly roll',
                         is_use_vbar=True,
                         is_add_totals=False,
                         add_bar_values=False,
                         fontsize=8,
                         ncol=3,
                         bbox_to_anchor=(0.5, 1.12),
                         pad=15,
                         ax=ax,
                         **kwargs)

        plot_regime_data(regime_classifier=regime_classifier,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         is_use_vbar=False,
                         bbox_to_anchor=None,
                         **kwargs)

    elif unit_test == UnitTests.VOL_REGIME:
        regime_params = VolQuantileRegimeSpecs()
        perf_params = PerfParams()
        regime_classifier = BenchmarkVolsQuantilesRegime(regime_params=regime_params)
        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices, benchmark='SPY')
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(prices=prices,
                                                                                   benchmark='SPY',
                                                                                   perf_params=perf_params)
        print(f"regime_means:\n{cond_perf_table}")
        print(f"regime_pa:\n{regime_datas}")

        plot_regime_data(regime_classifier=regime_classifier,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         is_use_vbar=True,
                         **kwargs)

        plot_regime_data(regime_classifier=regime_classifier,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         is_use_vbar=False,
                         **kwargs)

    elif unit_test == UnitTests.BNB_REGIME_SHADOWS:
        import qis.plots.time_series as pts
        with sns.axes_style('white'):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
            pts.plot_time_series(df=prices, ax=ax)

            add_bnb_regime_shadows(ax=ax,
                                   data_df=prices,
                                   benchmark='SPY',
                                   regime_params=BenchmarkReturnsQuantileRegimeSpecs(),
                                   perf_params=PerfParams())

    elif unit_test == UnitTests.BNB_PERF_TABLE:
        df = compute_bnb_regimes_pa_perf_table(prices=prices,
                                               benchmark='SPY',
                                               regime_params=BenchmarkReturnsQuantileRegimeSpecs(),
                                               perf_params=PerfParams())
        print(df)

    elif unit_test == UnitTests.AVG_PLOT:
        regime_params = VolQuantileRegimeSpecs()
        perf_params = PerfParams()
        regime_classifier = BenchmarkVolsQuantilesRegime(regime_params=regime_params)

        with sns.axes_style('white'):
            fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
            plot_regime_boxplot(regime_classifier=regime_classifier,
                                prices=prices,
                                benchmark='SPY',
                                perf_params=perf_params,
                                ax=ax,
                                **kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.BNB_REGIME

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
