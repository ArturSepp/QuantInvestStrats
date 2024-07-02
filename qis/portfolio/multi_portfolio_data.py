"""
core implementation of baktest report focus on strategy and comparison vs benchmark stategy
"""

# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# qis
import qis as qis
from qis import PerfParams, PerfStat, RegimeData, BenchmarkReturnsQuantileRegimeSpecs, TimePeriod, RollingPerfStat
import qis.utils.struct_ops as sop
import qis.utils.df_groups as dfg
import qis.perfstats.returns as ret
import qis.perfstats.perf_stats as rpt
import qis.perfstats.regime_classifier as rcl

# plots
import qis.plots.time_series as pts
import qis.plots.derived.prices as ppd
import qis.plots.derived.returns_heatmap as rhe
import qis.plots.derived.perf_table as ppt
import qis.plots.derived.returns_scatter as prs
import qis.plots.derived.drawdowns as cdr
from qis.portfolio.portfolio_data import PortfolioData, AttributionMetric


PERF_PARAMS = PerfParams(freq='W-WED')
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='ME')


@dataclass
class MultiPortfolioData:
    """
    data structure to unify multi portfolio reporting
    """
    portfolio_datas: List[PortfolioData]
    benchmark_prices: Union[pd.DataFrame, pd.Series] = None

    def __post_init__(self):
        # default frequency is freq of backtests, can be non for strats at different freqs
        self.set_navs(freq=None)
        if self.benchmark_prices is not None and isinstance(self.benchmark_prices, pd.Series):
            self.benchmark_prices = self.benchmark_prices.to_frame()

    def set_navs(self, freq: Optional[str] = None) -> None:
        navs = []
        for portfolio in self.portfolio_datas:
            navs.append(portfolio.get_portfolio_nav())
        self.navs = pd.concat(navs, axis=1)

        if freq is not None:
            self.navs = self.navs.asfreq(freq=freq, method='ffill')

        if self.benchmark_prices is not None:
            self.benchmark_prices = self.benchmark_prices.reindex(index=self.navs.index, method='ffill')

    def set_benchmark_prices(self, benchmark_prices: pd.DataFrame, freq: Optional[str] = None) -> None:
        # can pass benchmark prices here
        self.benchmark_prices = benchmark_prices
        self.set_navs(freq=freq)

    """
    data get methods
    """
    def get_navs(self,
                 benchmark: str = None,
                 time_period: TimePeriod = None
                 ) -> pd.DataFrame:
        """
        get portfolio navs
        """
        navs = self.navs
        if benchmark is not None:
            navs = pd.concat([self.benchmark_prices[benchmark], navs], axis=1).ffill()
        if time_period is not None:
            navs = time_period.locate(navs)
        return navs

    def get_benchmark_price(self,
                            benchmark: str,
                            time_period: TimePeriod = None
                            ) -> pd.Series:
        price = self.benchmark_prices[benchmark]
        if time_period is not None:
            price = time_period.locate(price)
        return price

    def get_group_navs(self,
                       portfolio_idx: int = 0,
                       benchmark: str = None,
                       time_period: TimePeriod = None,
                       is_add_group_total: bool = False
                       ) -> pd.DataFrame:
        prices = self.portfolio_datas[portfolio_idx].get_group_navs(time_period=time_period, is_add_group_total=is_add_group_total)
        if benchmark is not None:
            benchmark_price = self.get_benchmark_price(benchmark=self.benchmark_prices.columns[0],
                                                       time_period=time_period)
            prices = pd.concat([prices, benchmark_price], axis=1)
        return prices

    """
    plot methods
    """
    def add_regime_shadows(self,
                           ax: plt.Subplot,
                           regime_benchmark: str,
                           index: pd.Index = None,
                           regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS
                           ) -> None:
        """
        add regime shadows using regime_benchmark
        """
        pivot_prices = self.benchmark_prices[regime_benchmark]
        if index is not None:
            pivot_prices = pivot_prices.reindex(index=index, method='ffill')
        qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)

    def plot_nav(self,
                 time_period: TimePeriod = None,
                 regime_benchmark: str = None,
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
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=prices.index, regime_params=regime_params)

    def plot_rolling_perf(self,
                          rolling_perf_stat: RollingPerfStat = RollingPerfStat.SHARPE,
                          regime_benchmark: str = None,
                          time_period: TimePeriod = None,
                          sharpe_rolling_window: int = 260,
                          sharpe_freq: Optional[str] = None,
                          sharpe_title: Optional[str] = None,
                          legend_stats: pts.LegendStats = pts.LegendStats.AVG_LAST,
                          var_format: str = '{:.2f}',
                          regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> plt.Figure:

        # do not use start end dates here so the sharpe will be continuous with different time_period
        if ax is None:
            fig, ax = plt.subplots()

        prices = self.get_navs(time_period=time_period)
        if sharpe_freq is None:
            sharpe_freq = pd.infer_freq(index=prices.index)

        sharpe_title = sharpe_title or f"{sharpe_rolling_window}-period rolling Sharpe ratio for {sharpe_freq}-returns"
        kwargs = qis.update_kwargs(kwargs, dict(roll_freq=sharpe_freq))
        fig = ppd.plot_rolling_perf_stat(prices=prices,
                                         rolling_perf_stat=rolling_perf_stat,
                                         time_period=time_period,
                                         roll_periods=sharpe_rolling_window,
                                         legend_stats=legend_stats,
                                         trend_line=None, #qis.TrendLine.ZERO_SHADOWS,
                                         var_format=var_format,
                                         title=sharpe_title,
                                         ax=ax,
                                         **kwargs)

        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=prices.index,
                                    regime_params=regime_params)
        return fig
    
    def plot_periodic_returns(self,
                              time_period: TimePeriod = None,
                              heatmap_freq: str = 'YE',
                              date_format: str = '%Y',
                              transpose: bool = True,
                              title: str = None,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:
        prices = self.get_navs(time_period=time_period)
        rhe.plot_periodic_returns_table(prices=prices,
                                        freq=heatmap_freq,
                                        ax=ax,
                                        title=title,
                                        date_format=date_format,
                                        transpose=transpose,
                                        **kwargs)

    def plot_performance_bars(self,
                              time_period: TimePeriod = None,
                              perf_column: PerfStat = PerfStat.SHARPE_RF0,
                              perf_params: PerfParams = PERF_PARAMS,
                              title: str = None,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:
        prices = self.get_navs(time_period=time_period)
        ppt.plot_ra_perf_bars(prices=prices,
                              perf_column=perf_column,
                              perf_params=perf_params,
                              title=title or f"{perf_column.to_str()}: {qis.get_time_period(prices).to_str()}",
                              ax=ax,
                              **kwargs)

    def plot_corr_table(self,
                        time_period: TimePeriod = None,
                        freq: str = 'W-WED',
                        ax: plt.Subplot = None,
                        **kwargs) -> None:
        prices = self.get_navs(time_period=time_period)
        if len(prices.columns) > 1:
            qis.plot_returns_corr_table(prices=prices,
                                        x_rotation=90,
                                        freq=freq,
                                        title=f'Correlation of {freq} returns',
                                        ax=ax,
                                        **kwargs)

    def plot_drawdowns(self,
                       time_period: TimePeriod = None,
                       regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                       regime_benchmark: str = None,
                       ax: plt.Subplot = None,
                       **kwargs) -> None:
        if len(self.portfolio_datas) == 1 and regime_benchmark is not None:
            prices = self.get_navs(time_period=time_period, benchmark=regime_benchmark)
        else:
            prices = self.get_navs(time_period=time_period)
        cdr.plot_rolling_drawdowns(prices=prices, ax=ax, **kwargs)
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=prices.index, regime_params=regime_params)

    def plot_rolling_time_under_water(self,
                                      time_period: TimePeriod = None,
                                      regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                      regime_benchmark: str = None,
                                      ax: plt.Subplot = None,
                                      **kwargs) -> None:
        if len(self.portfolio_datas) == 1 and regime_benchmark is not None:
            prices = self.get_navs(time_period=time_period, benchmark=regime_benchmark)
        else:
            prices = self.get_navs(time_period=time_period)
        cdr.plot_rolling_time_under_water(prices=prices, ax=ax, **kwargs)
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=prices.index, regime_params=regime_params)

    def get_ra_perf_table(self,
                          benchmark: str = None,
                          time_period: TimePeriod = None,
                          drop_benchmark: bool = False,
                          is_convert_to_str: bool = True,
                          perf_params: PerfParams = PERF_PARAMS,
                          perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS,
                          **kwargs
                          ) -> pd.DataFrame:
        if benchmark is None:
            benchmark = self.benchmark_prices.columns[0]
        prices = self.get_navs(benchmark=benchmark, time_period=time_period)
        ra_perf_table = ppt.get_ra_perf_benchmark_columns(prices=prices,
                                                          benchmark=benchmark,
                                                          drop_benchmark=drop_benchmark,
                                                          is_convert_to_str=is_convert_to_str,
                                                          perf_params=perf_params,
                                                          perf_columns=perf_columns,
                                                          **kwargs)
        return ra_perf_table

    def plot_ra_perf_table(self,
                           benchmark: str = None,
                           time_period: TimePeriod = None,
                           drop_benchmark: bool = False,
                           perf_params: PerfParams = PERF_PARAMS,
                           perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS,
                           ax: plt.Subplot = None,
                           **kwargs
                           ) -> None:
        if benchmark is None:
            benchmark = self.benchmark_prices.columns[0]
        prices = self.get_navs(benchmark=benchmark, time_period=time_period)
        ra_perf_title = f"RA performance table for {perf_params.freq_vol}-freq returns with beta to {benchmark}: {qis.get_time_period(prices).to_str()}"
        ppt.plot_ra_perf_table_benchmark(prices=prices,
                                         benchmark=benchmark,
                                         perf_params=perf_params,
                                         perf_columns=perf_columns,
                                         drop_benchmark=drop_benchmark,
                                         title=ra_perf_title,
                                         rotation_for_columns_headers=0,
                                         ax=ax,
                                         **kwargs)

    def plot_ac_ra_perf_table(self,
                              benchmark_price: pd.Series,
                              time_period: TimePeriod = None,
                              perf_params: PerfParams = PERF_PARAMS,
                              perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS,
                              is_grouped: bool = True,
                              ax: plt.Subplot = None,
                              **kwargs) -> None:
        if is_grouped:  # otherwise tables look too bad
            add_ac = True
            drop_benchmark = True
        else:
            add_ac = False
            drop_benchmark = False
        strategy_prices = []
        ac_prices = []
        rows_edge_lines = [len(self.portfolio_datas)]
        for portfolio in self.portfolio_datas:
            portfolio_name = str(portfolio.nav.name)
            navs_ = portfolio.get_portfolio_nav(time_period=time_period)  # navs include costs while group navs are cost free
            ac_prices_ = portfolio.get_group_navs(time_period=time_period, is_add_group_total=False)
            strategy_prices.append(navs_)
            if add_ac:
                ac_prices_.columns = [f"{portfolio_name}-{x}" for x in ac_prices_.columns]
                ac_prices.append(ac_prices_)
                rows_edge_lines.append(sum(rows_edge_lines)+len(ac_prices_.columns))
        strategy_prices = pd.concat(strategy_prices, axis=1)
        benchmark_price = benchmark_price.reindex(index=strategy_prices.index, method='ffill')
        if add_ac:  # otherwise tables look too bad
            ac_prices = pd.concat(ac_prices, axis=1)
            prices = pd.concat([benchmark_price, strategy_prices, ac_prices],axis=1)
        else:
            prices = pd.concat([strategy_prices, benchmark_price], axis=1)

        ra_perf_title = f"RA performance table for {perf_params.freq_vol}-freq returns with beta to {benchmark_price.name}: {qis.get_time_period(prices).to_str()}"
        ppt.plot_ra_perf_table_benchmark(prices=prices,
                                         benchmark=str(benchmark_price.name),
                                         perf_params=perf_params,
                                         perf_columns=perf_columns,
                                         drop_benchmark=drop_benchmark,
                                         rows_edge_lines=rows_edge_lines,
                                         title=ra_perf_title,
                                         rotation_for_columns_headers=0,
                                         row_height=0.5,
                                         ax=ax,
                                         **kwargs)

    def plot_exposures(self,
                       benchmark: str = None,
                       regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                       time_period: TimePeriod = None,
                       var_format: str = '{:.0%}',
                       ax: plt.Subplot = None,
                       **kwargs
                       ) -> None:
        exposures = []
        for portfolio in self.portfolio_datas:
            exposures.append(portfolio.get_weights(time_period=time_period).sum(axis=1).rename(portfolio.nav.name))
        exposures = pd.concat(exposures, axis=1)
        pts.plot_time_series(df=exposures,
                             var_format=var_format,
                             legend_stats=pts.LegendStats.AVG_NONNAN_LAST,
                             title='Portfolio net exposures',
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=benchmark, index=exposures.index, regime_params=regime_params)

    def plot_instrument_pnl_diff(self,
                                 portfolio_idx1: int = 0,
                                 portfolio_idx2: int = 1,
                                 is_grouped: bool = True,
                                 benchmark: str = None,
                                 regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                 time_period: TimePeriod = None,
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
                             legend_stats=pts.LegendStats.LAST_NONNAN,
                             title=f"Cumulative p&l diff {self.portfolio_datas[portfolio_idx1].nav.name}-{self.portfolio_datas[portfolio_idx2].nav.name}",
                             ax=ax,
                             **sop.update_kwargs(kwargs, dict(legend_loc='lower left')))
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=benchmark, index=diff.index, regime_params=regime_params)

    def plot_exposures_diff(self,
                            portfolio_idx1: int = 0,
                            portfolio_idx2: int = 1,
                            benchmark: str = None,
                            regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                            time_period: TimePeriod = None,
                            var_format: str = '{:.0%}',
                            ax: plt.Subplot = None,
                            **kwargs) -> None:
        exposures1 = self.portfolio_datas[portfolio_idx1].get_weights(is_grouped=True, time_period=time_period, add_total=False)
        exposures2 = self.portfolio_datas[portfolio_idx2].get_weights(is_grouped=True, time_period=time_period, add_total=False)
        diff = exposures1.subtract(exposures2)
        pts.plot_time_series(df=diff,
                             var_format=var_format,
                             legend_stats=pts.LegendStats.AVG_NONNAN_LAST,
                             title=f"Net exposure diff {self.portfolio_datas[portfolio_idx1].nav.name}-{self.portfolio_datas[portfolio_idx2].nav.name}",
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=benchmark, index=diff.index, regime_params=regime_params)

    def plot_turnover(self,
                      benchmark: str = None,
                      time_period: TimePeriod = None,
                      regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                      turnover_rolling_period: Optional[int] = 260,
                      turnover_title: Optional[str] = None,
                      var_format: str = '{:.0%}',
                      ax: plt.Subplot = None,
                      **kwargs) -> None:

        turnover = []
        for portfolio in self.portfolio_datas:
            turnover.append(portfolio.get_turnover(roll_period=None, is_agg=True).rename(portfolio.nav.name))
        turnover = pd.concat(turnover, axis=1)
        if turnover_rolling_period is not None:
            turnover = turnover.rolling(turnover_rolling_period).sum()
        if time_period is not None:
            turnover = time_period.locate(turnover)
        
        freq = pd.infer_freq(turnover.index)
        turnover_title = turnover_title or f"{turnover_rolling_period}-period rolling {freq}-freq Turnover"
        pts.plot_time_series(df=turnover,
                             var_format=var_format,
                             y_limits=(0.0, None),
                             legend_stats=pts.LegendStats.AVG_NONNAN_LAST,
                             title=turnover_title,
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=benchmark, index=turnover.index, regime_params=regime_params)

    def plot_costs(self,
                   cost_rolling_period: Optional[int] = 260,
                   cost_title: Optional[str] = None,
                   benchmark: str = None,
                   time_period: TimePeriod = None,
                   regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                   var_format: str = '{:.2%}',
                   is_norm_costs: bool = True,
                   ax: plt.Subplot = None,
                   **kwargs) -> None:
        costs = []
        for portfolio in self.portfolio_datas:
            costs.append(portfolio.get_costs(roll_period=None, is_agg=True, is_norm_costs=is_norm_costs).rename(portfolio.nav.name))
        costs = pd.concat(costs, axis=1)
        if cost_rolling_period is not None:
            costs = costs.rolling(cost_rolling_period).sum()
        if time_period is not None:
            costs = time_period.locate(costs)

        freq = pd.infer_freq(costs.index)
        cost_title = cost_title or f"{cost_rolling_period}-period rolling {freq}-freq Costs %"
        pts.plot_time_series(df=costs,
                             var_format=var_format,
                             y_limits=(0.0, None),
                             legend_stats=pts.LegendStats.AVG_NONNAN_LAST,
                             title=cost_title,
                             ax=ax,
                             **kwargs)
        if benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=benchmark, index=costs.index, regime_params=regime_params)

    def plot_factor_betas(self,
                          benchmark_prices: pd.DataFrame,
                          regime_benchmark: str = None,
                          time_period: TimePeriod = None,
                          regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                          beta_freq: str = 'B',
                          factor_beta_span: int = 260,
                          var_format: str = '{:,.2f}',
                          factor_beta_title: Optional[str] = None,
                          axs: List[plt.Subplot] = None,
                          **kwargs
                          ) -> None:
        """
        plot benchmarks betas by factor exposures
        """
        factor_exposures = {factor: [] for factor in benchmark_prices.columns}
        for portfolio in self.portfolio_datas:
            factor_exposure = portfolio.compute_portfolio_benchmark_betas(benchmark_prices=benchmark_prices,
                                                                          beta_freq=beta_freq,
                                                                          factor_beta_span=factor_beta_span,
                                                                          time_period=time_period)
            for factor in factor_exposure.columns:
                factor_exposures[factor].append(factor_exposure[factor].rename(portfolio.nav.name))

        if axs is None:
            fig, axs = plt.subplots(len(benchmark_prices.columns), 1, figsize=(12, 12), tight_layout=True)

        for idx, factor in enumerate(benchmark_prices.columns):
            factor_exposure = pd.concat(factor_exposures[factor], axis=1)
            if factor_beta_title is not None:
                factor_beta_title = f"{factor_beta_title} to {factor}"
            else:
                factor_beta_title = factor_beta_title or f"{factor_beta_span}-span rolling Beta of {beta_freq}-freq returns to {factor}"
            pts.plot_time_series(df=factor_exposure,
                                 var_format=var_format,
                                 legend_stats=pts.LegendStats.AVG_NONNAN_LAST,
                                 title=f"{factor_beta_title}",
                                 ax=axs[idx],
                                 **kwargs)
            if regime_benchmark is not None:
                self.add_regime_shadows(ax=axs[idx],
                                        regime_benchmark=regime_benchmark,
                                        index=factor_exposure.index,
                                        regime_params=regime_params)

    def plot_nav_with_dd(self,
                         time_period: TimePeriod = None,
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
                             time_period: TimePeriod = None,
                             freq: str = 'QE',
                             order: int = 2,
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> None:
        prices = self.get_navs(benchmark=benchmark, time_period=time_period)
        local_kwargs = sop.update_kwargs(kwargs=kwargs,
                                         new_kwargs={'weight': 'bold',
                                                     'x_rotation': 0,
                                                     'first_color_fixed': False,
                                                     'ci': None,
                                                     'markersize': 6})
        prs.plot_returns_scatter(prices=prices,
                                 benchmark=benchmark,
                                 freq=freq,
                                 order=order,
                                 title=f"Scatterplot of {freq}-freq returns vs {benchmark}",
                                 ax=ax,
                                 **local_kwargs)

    def plot_performance_attribution(self,
                                     portfolio_ids: List[int] = (0, ),
                                     time_period: TimePeriod = None,
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
        qis.plot_bars(df=data,
                      skip_y_axis=True,
                      title=f"{attribution_metric.title}",
                      stacked=False,
                      ax=ax,
                      **kwargs)

    def plot_performance_periodic_table(self,
                                        portfolio_id: int = 0,
                                        time_period: TimePeriod = None,
                                        freq: str = 'YE',
                                        ax: plt.Subplot = None,
                                        **kwargs
                                        ) -> None:

        inst_returns = self.portfolio_datas[portfolio_id].get_attribution_table_by_instrument(time_period=time_period)
        inst_navs = ret.returns_to_nav(returns=inst_returns, init_period=None)
        strategy_nav = self.portfolio_datas[portfolio_id].get_portfolio_nav(time_period=time_period)
        prices = pd.concat([inst_navs, strategy_nav], axis=1).dropna()
        rhe.plot_periodic_returns_table(prices=prices,
                                        title=f"{strategy_nav.name} Attribution by Instrument",
                                        freq=freq,
                                        ax=ax,
                                        **kwargs)

    def plot_regime_data(self,
                         benchmark: str,
                         is_grouped: bool = False,
                         portfolio_idx: int = 0,
                         regime_data_to_plot: RegimeData = RegimeData.REGIME_SHARPE,
                         time_period: TimePeriod = None,
                         var_format: Optional[str] = None,
                         is_conditional_sharpe: bool = True,
                         is_use_vbar: bool = False,
                         perf_params: PerfParams = PERF_PARAMS,
                         regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                         title: str = None,
                         legend_loc: Optional[str] = 'upper center',
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> None:
        if is_grouped:
            prices = self.get_group_navs(portfolio_idx=portfolio_idx, benchmark=benchmark, time_period=time_period)
        else:
            prices = self.get_navs(benchmark=benchmark, time_period=time_period)
        if var_format is None:
            if regime_data_to_plot == RegimeData.REGIME_SHARPE:
                var_format = '{:.2f}'
            else:
                var_format = '{:.2%}'

        title = title or f"Sharpe ratio split to {str(benchmark)} Bear/Normal/Bull {regime_params.freq}-freq regimes"
        regime_classifier = rcl.BenchmarkReturnsQuantilesRegime(regime_params=regime_params)
        qis.plot_regime_data(regime_classifier=regime_classifier,
                             prices=prices,
                             benchmark=benchmark,
                             is_conditional_sharpe=is_conditional_sharpe,
                             regime_data_to_plot=regime_data_to_plot,
                             var_format=var_format,
                             is_use_vbar=is_use_vbar,
                             regime_params=regime_params,
                             legend_loc=legend_loc,
                             perf_params=perf_params,
                             title=title,
                             ax=ax,
                             **kwargs)

    def compute_brinson_attribution(self,
                                    strategy_idx: int = 0,
                                    benchmark_idx: int = 1,
                                    freq: Optional[str] = None,
                                    total_column: str = 'Total Sum',
                                    time_period: TimePeriod = None,
                                    is_exclude_interaction_term: bool = True
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        strategy_pnl = self.portfolio_datas[strategy_idx].get_attribution_table_by_instrument(time_period=time_period, freq=freq)
        strategy_weights = self.portfolio_datas[strategy_idx].get_weights(time_period=time_period, freq=freq, is_input_weights=False)

        benchmark_pnl = self.portfolio_datas[benchmark_idx].get_attribution_table_by_instrument(time_period=time_period, freq=freq)
        benchmark_weights = self.portfolio_datas[benchmark_idx].get_weights(time_period=time_period, freq=freq, is_input_weights=False)

        asset_class_data = self.portfolio_datas[strategy_idx].group_data
        group_order = self.portfolio_datas[strategy_idx].group_order

        totals_table, active_total, grouped_allocation_return, grouped_selection_return, grouped_interaction_return = \
            qis.compute_brinson_attribution_table(benchmark_pnl=benchmark_pnl,
                                                  strategy_pnl=strategy_pnl,
                                                  strategy_weights=strategy_weights,
                                                  benchmark_weights=benchmark_weights,
                                                  asset_class_data=asset_class_data,
                                                  group_order=group_order,
                                                  total_column=total_column,
                                                  is_exclude_interaction_term=is_exclude_interaction_term)
        return totals_table, active_total, grouped_allocation_return, grouped_selection_return, grouped_interaction_return

    def plot_brinson_attribution(self,
                                 strategy_idx: int = 0,
                                 benchmark_idx: int = 1,
                                 freq: Optional[str] = None,
                                 axs: List[plt.Subplot] = (None, None, None, None, None),
                                 total_column: str = 'Total Sum',
                                 time_period: TimePeriod = None,
                                 is_exclude_interaction_term: bool = True,
                                 **kwargs
                                 ) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure, plt.Figure]:

        totals_table, active_total, grouped_allocation_return, grouped_selection_return, grouped_interaction_return = \
            self.compute_brinson_attribution(strategy_idx=strategy_idx,
                                             benchmark_idx=benchmark_idx,
                                             freq=freq,
                                             total_column=total_column,
                                             time_period=time_period,
                                             is_exclude_interaction_term=is_exclude_interaction_term)

        fig_table, fig_active_total, fig_ts_alloc, fig_ts_sel, fig_ts_inters = qis.plot_brinson_attribution_table(
            totals_table=totals_table,
            active_total=active_total,
            grouped_allocation_return=grouped_allocation_return,
            grouped_selection_return=grouped_selection_return,
            grouped_interaction_return=grouped_interaction_return,
            total_column=total_column,
            is_exclude_interaction_term=is_exclude_interaction_term,
            axs=axs,
            **kwargs)

        return fig_table, fig_active_total, fig_ts_alloc, fig_ts_sel, fig_ts_inters

    def plot_weights_boxplot(self,
                             strategy_idx: int = 0,
                             benchmark_idx: int = 1,
                             freq: Optional[str] = 'ME',
                             time_period: TimePeriod = None,
                             is_grouped: bool = False,
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> None:
        strategy_weights = self.portfolio_datas[strategy_idx].get_weights(time_period=time_period, freq=freq,
                                                                          is_grouped=is_grouped, is_input_weights=False)
        benchmark_weights = self.portfolio_datas[benchmark_idx].get_weights(time_period=time_period, freq=freq,
                                                                            is_grouped=is_grouped, is_input_weights=False)
        dfs = {self.portfolio_datas[strategy_idx].ticker: strategy_weights,
               self.portfolio_datas[benchmark_idx].ticker: benchmark_weights}

        local_kwargs = qis.update_kwargs(kwargs, dict(ncol=2, legend_loc='upper center', showmedians=True,
                                                      yvar_format='{:.1%}'))
        qis.df_dict_boxplot_by_columns(dfs=dfs,
                                       hue_var_name='groups' if is_grouped else 'instruments',
                                       y_var_name='weights',
                                       ylabel='weights',
                                       ax=ax,
                                       **local_kwargs)
