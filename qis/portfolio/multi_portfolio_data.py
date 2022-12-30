"""
core implementation of baktest report focus on strategy and comparison vs benchmark stategy
"""

# built in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional

# qis
import qis.utils.dates as da
import qis.utils.struct_ops as sop
import qis.utils.df_to_str as dfs
import qis.utils.df_groups as dfg
import qis.perfstats.returns as ret
import qis.perfstats.perf_table as rpt
import qis.perfstats.drawdowns as cdr
from qis.perfstats.config import PerfParams, PerfStat, RegimeData
import qis.perfstats.regime_classifier as rcl
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs

# plots
import qis.plots.table as ptb
import qis.plots.time_series as pts
import qis.plots.derived.prices as ppd
import qis.plots.derived.returns_heatmap as rhe
import qis.plots.derived.perf_table as ppt
import qis.plots.derived.returns_scatter as prs
import qis.plots.derived.correlations as pco
from qis.plots.bars import plot_bars

# internal
from qis.portfolio.portfolio_data import PortfolioData, AttributionMetric


PERF_PARAMS = PerfParams(freq='W-WED')
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='M')


@dataclass
class MultiPortfolioData:
    """
    data structure to unify multi portfolio reporting
    """
    portfolio_datas: List[PortfolioData]
    benchmark_prices: pd.DataFrame = None

    def __post_init__(self):
        self.set_navs(freq=None)  # default frequency is freq of backtests, can be non for strats at different freqs

    def set_navs(self, freq: Optional[str] = None):
        navs = []
        for portfolio in self.portfolio_datas:
            navs.append(portfolio.get_portfolio_nav())
        self.navs = pd.concat(navs, axis=1)

        if freq is not None:
            self.navs = self.navs.asfreq(freq=freq, method='ffill')

        if self.benchmark_prices is not None:
            self.benchmark_prices = self.benchmark_prices.reindex(index=self.navs.index, method='ffill')

    def _set_benchmark_prices(self, benchmark_prices: pd.DataFrame) -> None:
        self.benchmark_prices = benchmark_prices
        self.set_navs(freq=None)

    """
    data get methods
    """
    def get_navs(self,
                 benchmark: str = None,
                 time_period: da.TimePeriod = None
                 ) -> pd.DataFrame:
        """
        get portfolio navs
        """
        navs = self.navs
        if benchmark is not None:
            navs = pd.concat([self.benchmark_prices[benchmark], navs], axis=1).fillna(method='ffill')
        if time_period is not None:
            navs = time_period.locate(navs)
        return navs

    def get_benchmark_price(self,
                            benchmark: str,
                            time_period: da.TimePeriod = None
                            ) -> pd.Series:
        price = self.benchmark_prices[benchmark]
        if time_period is not None:
            price = time_period.locate(price)
        return price

    def get_ac_navs(self,
                    portfolio_idx: int = 0,
                    benchmark: str = None,
                    time_period: da.TimePeriod = None
                    ) -> pd.DataFrame:
        prices = self.portfolio_datas[portfolio_idx].get_ac_navs(time_period=time_period)
        if benchmark is not None:
            benchmark_price = self.get_benchmark_price(benchmark=self.benchmark_prices.columns[0],
                                                       time_period=time_period)
            prices = pd.concat([prices, benchmark_price], axis=1)
        return prices

    """
    plot methods
    """
    def add_regime_shadows(self, ax: plt.Subplot, benchmark: str, index: pd.Index = None,
                           regime_params: BenchmarkReturnsQuantileRegimeSpecs=REGIME_PARAMS
                           ) -> None:
        pivot_prices = self.benchmark_prices[benchmark]
        if index is not None:
            pivot_prices = pivot_prices.reindex(index=index, method='ffill')
        rcl.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)

    def plot_nav(self,
                 time_period: da.TimePeriod = None,
                 benchmark: str = None,
                 perf_params: PerfParams = PERF_PARAMS,
                 regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                 ax: plt.Subplot = None,
                 **kwargs) -> None:

        if ax is None:
            fig, ax = plt.subplots()

        prices = self.get_navs(time_period=time_period)
        ppd.plot_prices(prices=prices,
                        perf_params=perf_params,
                        ax=ax,
                        **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=prices.index, regime_params=regime_params)

    def plot_periodic_returns(self,
                              time_period: da.TimePeriod = None,
                              heatmap_freq: str = 'A',
                              date_format: str = '%Y',
                              is_transpose: bool = True,
                              title: str = None,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:
        prices = self.get_navs(time_period=time_period)
        rhe.plot_periodic_returns_table_from_prices(prices=prices,
                                                    freq=heatmap_freq,
                                                    ax=ax,
                                                    title=title,
                                                    date_format=date_format,
                                                    is_transpose=is_transpose,
                                                    **kwargs)

    def plot_performance_bars(self,
                              time_period: da.TimePeriod = None,
                              perf_column: PerfStat = PerfStat.SHARPE,
                              perf_params: PerfParams = PERF_PARAMS,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:

        prices = self.get_navs(time_period=time_period)
        ppt.plot_ra_perf_bars(prices=prices,
                              perf_column=perf_column,
                              perf_params=perf_params,
                              title=f"{perf_column.to_str()}: {da.get_time_period(prices).to_str()}",
                              ax=ax,
                              **kwargs)

    def plot_corr_table(self,
                        time_period: da.TimePeriod = None,
                        freq: str = 'W-WED',
                        ax: plt.Subplot = None,
                        **kwargs) -> None:
        prices = self.get_navs(time_period=time_period)
        if len(prices.columns) > 1:
            pco.plot_corr_table(prices=prices,
                                x_rotation=90,
                                freq=freq,
                                title=f'Correlation of {freq} returns',
                                ax=ax, **kwargs)

    def plot_drawdowns(self,
                       time_period: da.TimePeriod = None,
                       regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                       benchmark: str = None,
                       ax: plt.Subplot = None,
                       **kwargs) -> None:
        if len(self.portfolio_datas) == 1 and benchmark is not None:
            prices = self.get_navs(time_period=time_period, benchmark=benchmark)
        else:
            prices = self.get_navs(time_period=time_period)
        cdr.plot_drawdown(prices=prices, ax=ax, **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=prices.index, regime_params=regime_params)

    def plot_rolling_time_under_water(self,
                                      time_period: da.TimePeriod = None,
                                      regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                      benchmark: str = None,
                                      ax: plt.Subplot = None,
                                      **kwargs) -> None:
        if len(self.portfolio_datas) == 1 and benchmark is not None:
            prices = self.get_navs(time_period=time_period, benchmark=benchmark)
        else:
            prices = self.get_navs(time_period=time_period)
        cdr.plot_rolling_time_under_water(prices=prices, ax=ax, **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=prices.index, regime_params=regime_params)

    def plot_ra_perf_table(self,
                           time_period: da.TimePeriod = None,
                           perf_params: PerfParams = PERF_PARAMS,
                           ax: plt.Subplot = None,
                           **kwargs) -> None:
        benchmark = self.benchmark_prices.columns[0]
        prices = self.get_navs(benchmark=benchmark, time_period=time_period)
        ppt.plot_ra_perf_table_benchmark(prices=prices,
                                         benchmark=benchmark,
                                         perf_params=perf_params,
                                         perf_columns=rpt.BENCHMARK_TABLE_COLUMNS,
                                         title=f"RA performance table: {da.get_time_period(prices).to_str()}",
                                         rotation_for_columns_headers=0,
                                         ax=ax,
                                         **kwargs)

    def plot_ac_ra_perf_table(self,
                              benchmark_price: pd.Series,
                              time_period: da.TimePeriod = None,
                              perf_params: PerfParams = PERF_PARAMS,
                              ax: plt.Subplot = None,
                              **kwargs) -> None:
        strategy_prices = []
        ac_prices = []
        rows_edge_lines = [len(self.portfolio_datas)]
        for portfolio in self.portfolio_datas:
            portfolio_name = str(portfolio.nav.name)
            prices_ = portfolio.get_ac_navs(time_period=time_period)
            strategy_prices.append(prices_[portfolio_name])
            ac_prices_ = prices_.drop(portfolio_name, axis=1)
            ac_prices_.columns = [f"{portfolio_name}-{x}" for x in ac_prices_.columns]
            ac_prices.append(ac_prices_)
            rows_edge_lines.append(sum(rows_edge_lines)+len(ac_prices_.columns))
        strategy_prices = pd.concat(strategy_prices, axis=1)
        ac_prices = pd.concat(ac_prices, axis=1)
        prices = pd.concat([benchmark_price.reindex(index=strategy_prices.index, method='ffill'), strategy_prices, ac_prices], axis=1)
        ppt.plot_ra_perf_table_benchmark(prices=prices,
                                         benchmark=str(benchmark_price.name),
                                         perf_params=perf_params,
                                         perf_columns=rpt.BENCHMARK_TABLE_COLUMNS,
                                         drop_benchmark=True,
                                         rows_edge_lines=rows_edge_lines,
                                         title=f"RA performance table by Asset Group: {da.get_time_period(prices).to_str()}",
                                         rotation_for_columns_headers=0,
                                         row_height=0.5,
                                         ax=ax,
                                         **kwargs)

    def plot_exposures(self,
                       benchmark: str = None,
                       regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                       time_period: da.TimePeriod = None,
                       var_format: str = '{:.0%}',
                       ax: plt.Subplot = None,
                       **kwargs) -> None:
        exposures = []
        for portfolio in self.portfolio_datas:
            exposures.append(portfolio.get_exposures(time_period=time_period).sum(axis=1).rename(portfolio.nav.name))
        exposures = pd.concat(exposures, axis=1)
        pts.plot_time_series(df=exposures,
                             var_format=var_format,
                             legend_line_type=pts.LegendLineType.AVG_NONNAN_LAST,
                             title='Portfolio net exposures',
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=exposures.index, regime_params=regime_params)

    def plot_instrument_pnl_diff(self,
                                 portfolio_idx1: int = 0,
                                 portfolio_idx2: int = 1,
                                 is_grouped: bool = True,
                                 benchmark: str = None,
                                 regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                 time_period: da.TimePeriod = None,
                                 var_format: str = '{:.0%}',
                                 ax: plt.Subplot = None,
                                 **kwargs) -> None:
        pnl_inst1 = self.portfolio_datas[portfolio_idx1].instrument_pnl
        pnl_inst2 = self.portfolio_datas[portfolio_idx2].instrument_pnl
        df1_, df2_ = pnl_inst1.align(other=pnl_inst2, join='outer', axis=None)
        df1_, df2_ = df1_.fillna(0.0), df2_.fillna(0.0)
        diff = df1_.subtract(df2_)

        if is_grouped:
            diff = dfg.agg_df_by_groups_ax1(diff,
                                            group_data=self.portfolio_datas[portfolio_idx1].group_data,
                                            agg_func=np.nansum,
                                            total_column=f"{self.portfolio_datas[portfolio_idx1].nav.name}-{self.portfolio_datas[portfolio_idx2].nav.name}",
                                            group_order=self.portfolio_datas[portfolio_idx1].group_order)
        if time_period is not None:
            diff = time_period.locate(diff)
        diff = diff.cumsum(axis=0)

        pts.plot_time_series(df=diff,
                             var_format=var_format,
                             legend_line_type=pts.LegendLineType.LAST,
                             title=f"Cumulative p&l diff {self.portfolio_datas[portfolio_idx1].nav.name}-{self.portfolio_datas[portfolio_idx2].nav.name}",
                             ax=ax,
                             **sop.update_kwargs(kwargs, dict(legend_loc='lower left')))
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=diff.index, regime_params=regime_params)

    def plot_exposures_diff(self,
                            portfolio_idx1: int = 0,
                            portfolio_idx2: int = 1,
                            benchmark: str = None,
                            regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                            time_period: da.TimePeriod = None,
                            var_format: str = '{:.0%}',
                            ax: plt.Subplot = None,
                            **kwargs) -> None:
        exposures1 = self.portfolio_datas[portfolio_idx1].get_exposures(is_grouped=True, time_period=time_period, add_total=False)
        exposures2 = self.portfolio_datas[portfolio_idx2].get_exposures(is_grouped=True, time_period=time_period, add_total=False)
        diff = exposures1.subtract(exposures2)
        pts.plot_time_series(df=diff,
                             var_format=var_format,
                             legend_line_type=pts.LegendLineType.AVG_NONNAN_LAST,
                             title=f"Net exposure diff {self.portfolio_datas[portfolio_idx1].nav.name}-{self.portfolio_datas[portfolio_idx2].nav.name}",
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=diff.index, regime_params=regime_params)

    def plot_turnover(self,
                      roll_period: int = 260,
                      benchmark: str = None,
                      time_period: da.TimePeriod = None,
                      regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                      var_format: str = '{:.0%}',
                      ax: plt.Subplot = None,
                      **kwargs) -> None:

        turnover = []
        for portfolio in self.portfolio_datas:
            turnover.append(portfolio.get_turnover(roll_period=None, is_agg=True).rename(portfolio.nav.name))
        turnover = pd.concat(turnover, axis=1)
        if roll_period is not None:
            turnover = turnover.rolling(roll_period).sum()
        if time_period is not None:
            turnover = time_period.locate(turnover)

        pts.plot_time_series(df=turnover,
                             var_format=var_format,
                             y_limits=(0.0, None),
                             legend_line_type=pts.LegendLineType.AVG_NONNAN_LAST,
                             title='Annualized daily Turnover',
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=turnover.index, regime_params=regime_params)

    def plot_costs(self,
                   roll_period: int = 260,
                   benchmark: str = None,
                   time_period: da.TimePeriod = None,
                   regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                   var_format: str = '{:.2%}',
                   ax: plt.Subplot = None,
                   **kwargs) -> None:
        costs = []
        for portfolio in self.portfolio_datas:
            costs.append(portfolio.get_costs(roll_period=None, is_agg=True).rename(portfolio.nav.name))
        costs = pd.concat(costs, axis=1)
        if roll_period is not None:
            costs = costs.rolling(roll_period).sum()
        if time_period is not None:
            costs = time_period.locate(costs)
        pts.plot_time_series(df=costs,
                             var_format=var_format,
                             y_limits=(0.0, None),
                             legend_line_type=pts.LegendLineType.AVG_NONNAN_LAST,
                             title='Annualized daily Costs %',
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=costs.index, regime_params=regime_params)

    def plot_factor_betas(self,
                          benchmark_prices: pd.DataFrame,
                          benchmark: str = None,
                          time_period: da.TimePeriod = None,
                          regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                          var_format: str = '{:,.2f}',
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> None:
        factor_exposures = []
        for portfolio in self.portfolio_datas:
            factor_exposure = portfolio.compute_portfolio_benchmark_betas(benchmark_prices=benchmark_prices,
                                                                          time_period=time_period)
            factor_exposure.columns = [f"{portfolio.nav.name}-{x} beta" for x in factor_exposure.columns]
            factor_exposures.append(factor_exposure)
        factor_exposures = pd.concat(factor_exposures, axis=1)
        pts.plot_time_series(df=factor_exposures,
                             var_format=var_format,
                             legend_line_type=pts.LegendLineType.AVG_NONNAN_LAST,
                             title='Factor betas',
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, benchmark=benchmark, index=factor_exposures.index, regime_params=regime_params)

    def plot_nav_with_dd(self,
                         time_period: da.TimePeriod = None,
                         perf_params: PerfParams = PERF_PARAMS,
                         axs: List[plt.Subplot] = None,
                         **kwargs
                         ) -> None:
        prices = self.get_navs(time_period=time_period)
        if self.benchmark_prices is not None:
            regime_benchmark_str = self.benchmark_prices.columns[0]
        else:
            regime_benchmark_str = None
        ppd.plot_prices_with_dd(prices=prices,
                                perf_params=perf_params,
                                regime_benchmark_str=regime_benchmark_str,
                                axs=axs,
                                **kwargs)

    def plot_returns_scatter(self,
                             benchmark: str,
                             time_period: da.TimePeriod = None,
                             freq: str = 'Q',
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> None:
        prices = self.get_navs(benchmark=benchmark, time_period=time_period)
        local_kwargs = sop.update_kwargs(kwargs=kwargs,
                                         new_kwargs={'weight': 'bold',
                                                     #'alpha_an_factor': 52.0,
                                                     'x_rotation': 0,
                                                     'first_color_fixed': False,
                                                     'ci': None})
        prs.plot_returns_scatter(prices=prices,
                                 benchmark=benchmark,
                                 freq=freq,
                                 order=2,
                                 title=f"Scatterplot of {freq}-returns vs {benchmark}",
                                 ax=ax,
                                 **local_kwargs)

    def plot_performance_attribution(self,
                                     portfolio_ids: List[int] = (0, ),
                                     time_period: da.TimePeriod = None,
                                     attribution_metric: AttributionMetric = AttributionMetric.PNL,
                                     ax: plt.Subplot = None,
                                     **kwargs
                                     ) -> None:
        datas = []
        for portfolio_id in portfolio_ids:
            datas.append(self.portfolio_datas[portfolio_id].get_performance_data(attribution_metric=attribution_metric,
                                                                                 time_period=time_period))
        data = pd.concat(datas, axis=1)
        data = data.sort_values(data.columns[0], ascending=False)
        kwargs = sop.update_kwargs(kwargs=kwargs,
                                         new_kwargs={'ncol': len(data.columns),
                                                     'legend_loc': 'upper center',
                                                     'bbox_to_anchor': (0.5, 1.05),
                                                     'x_rotation': 90})
        data = data.replace({0.0: np.nan}).dropna()
        plot_bars(df=data,
                  show_y_axis=True,
                  title=f"{attribution_metric.title}",
                  stacked=False,
                  ax=ax,
                  **kwargs)

    def plot_performance_periodic_table(self,
                                        portfolio_id: int = 0,
                                        time_period: da.TimePeriod = None,
                                        attribution_metric: AttributionMetric = AttributionMetric.INST_PNL,
                                        freq: str = 'M',
                                        ax: plt.Subplot = None,
                                        **kwargs
                                        ) -> None:

        inst_returns = self.portfolio_datas[portfolio_id].get_attribution_table_by_instrument(time_period=time_period)

        total_return = inst_returns.sum(axis=1).rename('Sum Total')

        strategy_nav = self.portfolio_datas[portfolio_id].get_portfolio_nav(time_period=time_period)
        nav_return = ret.to_returns(prices=strategy_nav, freq=freq, drop_first=True).rename('Nav Total')
        returns = pd.concat([inst_returns, total_return, nav_return], axis=1)
        total_returns = ((1.0+returns).cumprod(axis=0).iloc[-1, :]-1.0).rename('Period Total')
        returns = returns.append(total_returns)

        rhe.plot_periodic_returns_table(returns=returns,
                                        title=f"{strategy_nav.name} Attribution by Instrument",
                                        freq=freq,
                                        ax=ax,
                                        **kwargs)

    def plot_composite_table(self,
                             portfolio_id: int = 0,
                             date_format: str = '%b-%y',
                             time_period: da.TimePeriod = None,
                             perf_params: PerfParams = PERF_PARAMS,
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> plt.Figure:
        """
        plot performqances with weights
        """
        perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS
        strategy_nav = self.portfolio_datas[portfolio_id].get_portfolio_nav(time_period=time_period)
        prices = pd.concat([strategy_nav, self.portfolio_datas[portfolio_id].get_instruments_navs(time_period=time_period)], axis=1)
        ra_perf_table = rpt.compute_ra_perf_table_with_benchmark(prices=prices,
                                                                 benchmark=str(strategy_nav.name),
                                                                 perf_params=perf_params)
        data = pd.DataFrame(data=prices.columns, index=ra_perf_table.index, columns=['Assets'])
        for perf_column in perf_columns:
            data[perf_column.to_str()] = dfs.series_to_str(ds=ra_perf_table[perf_column.to_str()],
                                                           var_format=perf_column.to_format(**kwargs))

        df = self.portfolio_datas[portfolio_id].weights
        if time_period is not None:
            df = time_period.locate(df)

        exposures = dfs.timeseries_df_to_str(df=df,
                                             freq='Q',
                                             var_format='{:.2%}',
                                             date_format=date_format,
                                             is_transpose=True)
        joint = pd.concat([data, exposures], axis=1)

        fig = ptb.plot_df_table(df=joint,
                                add_index_as_column=False,
                                is_transposed=False,
                                ax=ax,
                                **kwargs)
        return fig

    def plot_regime_data(self,
                         benchmark: str,
                         is_grouped: bool = False,
                         portfolio_idx: int = 0,
                         regime_data_to_plot: RegimeData = RegimeData.REGIME_SHARPE,
                         time_period: da.TimePeriod = None,
                         var_format: Optional[str] = None,
                         is_conditional_sharpe: bool = True,
                         perf_params: PerfParams = PERF_PARAMS,
                         regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                         legend_loc: Optional[str] = 'upper center',
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> None:
        if is_grouped:
            prices = self.get_ac_navs(portfolio_idx=portfolio_idx, benchmark=benchmark, time_period=time_period)
            title = f"Sharpe ratio decomposition by Asset Group to {benchmark} Bear/Normal/Bull regimes"
        else:
            prices = self.get_navs(benchmark=benchmark, time_period=time_period)
            title = f"Sharpe ratio decomposition by Strategies to {benchmark} Bear/Normal/Bull regimes"
        if var_format is None:
            if regime_data_to_plot == RegimeData.REGIME_SHARPE:
                var_format = '{:.2f}'
            else:
                var_format = '{:.2%}'
        regime_classifier = rcl.BenchmarkReturnsQuantilesRegime(regime_params=regime_params)
        rcl.plot_regime_data(regime_classifier=regime_classifier,
                             prices=prices,
                             benchmark=benchmark,
                             is_conditional_sharpe=is_conditional_sharpe,
                             regime_data_to_plot=regime_data_to_plot,
                             var_format=var_format,
                             regime_params=regime_params,
                             legend_loc=legend_loc,
                             perf_params=perf_params,
                             title=title,
                             ax=ax,
                             **kwargs)
