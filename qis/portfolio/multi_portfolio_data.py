"""
core implementation of baktest report focus on strategy and comparison vs benchmark stategy
"""

# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
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

# default perf params
PERF_PARAMS = PerfParams(freq='W-WED')
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='ME')


@dataclass
class MultiPortfolioData:
    """
    data structure to unify multi portfolio reporting
    portfolio_datas: List[PortfolioData]
    benchmark_prices: Union[pd.DataFrame, pd.Series] = None  # Optional
    covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None  # annualised covariance matrix
                                                        for investable universe for computing tracking error
    """
    portfolio_datas: List[PortfolioData]
    benchmark_prices: Union[pd.DataFrame, pd.Series] = None
    covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None
    navs: pd.DataFrame = None   # computed internally

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

    def set_benchmark_prices(self, benchmark_prices: pd.DataFrame) -> None:
        # can pass benchmark prices here
        if isinstance(benchmark_prices, pd.Series):
            benchmark_prices = benchmark_prices.to_frame()
        else:
            benchmark_prices = benchmark_prices
        self.benchmark_prices = benchmark_prices.reindex(index=self.navs.index, method='ffill')

    """
    data get methods
    """
    def get_navs(self,
                 add_benchmarks_to_navs: bool = False,
                 benchmark: str = None,
                 time_period: TimePeriod = None
                 ) -> pd.DataFrame:
        """
        get portfolio navs
        double check that benchmark is not part of portfolio
        """
        navs = self.navs
        if benchmark is not None:
            if benchmark not in navs.columns:
                navs = pd.concat([self.benchmark_prices[benchmark].reindex(index=navs.index).ffill(), navs], axis=1)
        elif add_benchmarks_to_navs:
            benchmarks = self.benchmark_prices.reindex(index=navs.index).ffill()
            for benchmark in benchmarks.columns:
                if benchmark not in navs.columns:
                    navs = pd.concat([navs, benchmarks[benchmark]], axis=1)

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
        prices = self.portfolio_datas[portfolio_idx].get_group_navs(time_period=time_period,
                                                                    is_add_group_total=is_add_group_total)
        if benchmark is not None:
            benchmark_price = self.get_benchmark_price(benchmark=self.benchmark_prices.columns[0],
                                                       time_period=time_period)
            prices = pd.concat([prices, benchmark_price], axis=1)
        return prices

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

    def get_aligned_weights(self,
                            strategy_idx: int = 0,
                            benchmark_idx: int = 1,
                            freq: Optional[str] = None,
                            is_input_weights: bool = True,
                            time_period: TimePeriod = None,
                            is_grouped: bool = False,
                            **kwargs
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        strategy_weights = self.portfolio_datas[strategy_idx].get_weights(time_period=time_period, freq=freq,
                                                                          is_input_weights=is_input_weights,
                                                                          is_grouped=is_grouped)
        benchmark_weights = self.portfolio_datas[benchmark_idx].get_weights(time_period=time_period, freq=freq,
                                                                            is_input_weights=is_input_weights,
                                                                            is_grouped=is_grouped)
        tickers_union = qis.merge_lists_unique(list1=strategy_weights.columns.to_list(),
                                               list2=benchmark_weights.columns.to_list())
        # replace with ac order of benchmark
        if is_grouped and self.portfolio_datas[benchmark_idx].group_order is not None:
            tickers_union = self.portfolio_datas[benchmark_idx].group_order
        strategy_weights = strategy_weights.reindex(columns=tickers_union)
        benchmark_weights = benchmark_weights.reindex(columns=tickers_union)
        return strategy_weights, benchmark_weights

    def get_aligned_turnover(self,
                             strategy_idx: int = 0,
                             benchmark_idx: int = 1,
                             turnover_rolling_period: Optional[int] = 260,
                             freq_turnover: Optional[str] = 'B',
                             time_period: TimePeriod = None,
                             is_grouped: bool = False,
                             **kwargs
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        strategy_turnover = self.portfolio_datas[strategy_idx].get_turnover(time_period=time_period, freq=freq_turnover,
                                                                            roll_period=turnover_rolling_period,
                                                                            add_total=False, is_grouped=is_grouped)
        benchmark_turnover = self.portfolio_datas[benchmark_idx].get_turnover(time_period=time_period,
                                                                              freq=freq_turnover,
                                                                              roll_period=turnover_rolling_period,
                                                                              add_total=False, is_grouped=is_grouped)
        tickers_union = qis.merge_lists_unique(list1=strategy_turnover.columns.to_list(),
                                               list2=benchmark_turnover.columns.to_list())
        # replace with ac order of benchmark
        if is_grouped and self.portfolio_datas[benchmark_idx].group_order is not None:
            tickers_union = self.portfolio_datas[benchmark_idx].group_order
        strategy_turnover = strategy_turnover.reindex(columns=tickers_union)
        benchmark_turnover = benchmark_turnover.reindex(columns=tickers_union)
        return strategy_turnover, benchmark_turnover

    def compute_tracking_error_implied_by_covar(self,
                                                strategy_idx: int = 0,
                                                benchmark_idx: int = 1
                                                ) -> pd.Series:
        """
        compute Ex ante  tracking error =
        (strategy_weights - strategy_weights) @ covar @ (strategy_weights - strategy_weights).T
        """
        if self.covar_dict is None:
            raise ValueError(f"must pass covar_dict")
        strategy_weights = self.portfolio_datas[strategy_idx].get_weights(freq=None, is_input_weights=True)
        benchmark_weights = self.portfolio_datas[benchmark_idx].get_weights(freq=None, is_input_weights=True)
        covar_index = list(self.covar_dict.keys())
        investable_assets = self.covar_dict[covar_index[0]].columns.to_list()
        strategy_weights = strategy_weights.reindex(index=covar_index, columns=investable_assets).ffill().fillna(0.0)
        benchmark_weights = benchmark_weights.reindex(index=covar_index, columns=investable_assets).ffill().fillna(0.0)

        weight_diffs = benchmark_weights - strategy_weights
        tracking_error = {}
        for date, pd_covar in self.covar_dict.items():
            w = weight_diffs.loc[date]
            tracking_error[date] = np.sqrt(w @ pd_covar @ w.T)
        tracking_error = pd.Series(tracking_error, name='Tracking error')
        return tracking_error

    def compute_tracking_error_table(self,
                                     strategy_idx: int = 0,
                                     benchmark_idx: int = 1,
                                     freq: Optional[str] = 'B',
                                     time_period: TimePeriod = None,
                                     annualization_factor: float = 260,
                                     is_unit_based_traded_volume: bool = True,
                                     **kwargs
                                     ) -> pd.DataFrame:
        """
        compute realised tracking error = mean(P&L diff) / std(P&L diff)
        """
        strategy_pnl = self.portfolio_datas[strategy_idx].get_attribution_table_by_instrument(time_period=time_period,
                                                                                              freq=freq)
        benchmark_pnl = self.portfolio_datas[benchmark_idx].get_attribution_table_by_instrument(time_period=time_period,
                                                                                                freq=freq)

        tickers_union = qis.merge_lists_unique(list1=strategy_pnl.columns.to_list(),
                                               list2=benchmark_pnl.columns.to_list())
        strategy_pnl = strategy_pnl.reindex(columns=tickers_union)
        benchmark_pnl = benchmark_pnl.reindex(columns=tickers_union).reindex(index=strategy_pnl.index)
        pnl_diff = strategy_pnl.subtract(benchmark_pnl)

        # strategy_weights = self.portfolio_datas[strategy_idx].get_weights(time_period=time_period, freq=None, is_input_weights=True)
        strategy_turnover = self.portfolio_datas[strategy_idx].get_turnover(time_period=time_period, freq=None,
                                                                            roll_period=None, add_total=False)
        strategy_cost = self.portfolio_datas[strategy_idx].get_costs(time_period=time_period, freq=freq,
                                                                     roll_period=None,
                                                                     add_total=False,
                                                                     is_unit_based_traded_volume=is_unit_based_traded_volume)
        strategy_ticker = self.portfolio_datas[strategy_idx].ticker

        # benchmark_weights = self.portfolio_datas[benchmark_idx].get_weights(time_period=time_period, freq=None, is_input_weights=True)
        benchmark_turnover = self.portfolio_datas[benchmark_idx].get_turnover(time_period=time_period, freq=None,
                                                                              roll_period=None, add_total=False)
        benchmark_cost = self.portfolio_datas[benchmark_idx].get_costs(time_period=time_period, freq=freq,
                                                                       roll_period=None,
                                                                       add_total=False,
                                                                       is_unit_based_traded_volume=is_unit_based_traded_volume)
        benchmark_ticker = self.portfolio_datas[benchmark_idx].ticker

        # compute stats
        total_strategy_perf = strategy_pnl.cumsum(0).iloc[-1, :].rename(f"{strategy_ticker} total perf")
        total_benchmark_perf = benchmark_pnl.cumsum(0).iloc[-1, :].rename(f"{benchmark_ticker} total perf")
        total_diff = total_strategy_perf.subtract(total_benchmark_perf).rename(
            f"{strategy_ticker}-{benchmark_ticker} total perf")

        tre = pd.Series(np.nanmean(pnl_diff, axis=0) / np.nanstd(pnl_diff, axis=0), index=tickers_union, name='TRE')

        tre_table = pd.concat([total_diff, tre,
                               total_strategy_perf, total_benchmark_perf,
                               annualization_factor * strategy_turnover.mean(0).rename(f"{strategy_ticker} an turnover"),
                               annualization_factor * benchmark_turnover.mean(0).rename(f"{benchmark_ticker} an turnover"),
                               annualization_factor * strategy_cost.mean(0).rename(f"{strategy_ticker} an cost"),
                               annualization_factor * benchmark_cost.mean(0).rename(f"{benchmark_ticker} an cost"),
                               ], axis=1)

        return tre_table

    # """
    # plot methods
    # """
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
                 add_benchmarks_to_navs: bool = False,
                 ax: plt.Subplot = None,
                 **kwargs) -> None:

        if ax is None:
            fig, ax = plt.subplots()

        prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
        ppd.plot_prices(prices=prices,
                        perf_params=perf_params,
                        ax=ax,
                        **kwargs)
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=prices.index,
                                    regime_params=regime_params)

    def plot_rolling_perf(self,
                          rolling_perf_stat: RollingPerfStat = RollingPerfStat.SHARPE,
                          regime_benchmark: str = None,
                          time_period: TimePeriod = None,
                          sharpe_rolling_window: int = 260,
                          freq_sharpe: Optional[str] = None,
                          sharpe_title: Optional[str] = None,
                          legend_stats: pts.LegendStats = pts.LegendStats.AVG_LAST,
                          regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                          add_benchmarks_to_navs: bool = False,
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> plt.Figure:

        # do not use start end dates here so the sharpe will be continuous with different time_period
        if ax is None:
            fig, ax = plt.subplots()

        prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)

        kwargs = qis.update_kwargs(kwargs, dict(roll_freq=freq_sharpe))
        fig = ppd.plot_rolling_perf_stat(prices=prices,
                                         rolling_perf_stat=rolling_perf_stat,
                                         time_period=time_period,
                                         roll_periods=sharpe_rolling_window,
                                         legend_stats=legend_stats,
                                         title=sharpe_title,
                                         trend_line=None,  # qis.TrendLine.ZERO_SHADOWS,
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
                              add_benchmarks_to_navs: bool = False,
                              title: str = None,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:
        prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
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
                              add_benchmarks_to_navs: bool = False,
                              title: str = None,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:
        prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
        ppt.plot_ra_perf_bars(prices=prices,
                              perf_column=perf_column,
                              perf_params=perf_params,
                              title=title or f"{perf_column.to_str()}: {qis.get_time_period(prices).to_str()}",
                              ax=ax,
                              **kwargs)

    def plot_corr_table(self,
                        time_period: TimePeriod = None,
                        freq: str = 'W-WED',
                        add_benchmarks_to_navs: bool = False,
                        ax: plt.Subplot = None,
                        **kwargs) -> None:
        prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
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
                       dd_legend_type: qis.DdLegendType = qis.DdLegendType.SIMPLE,
                       add_benchmarks_to_navs: bool = False,
                       ax: plt.Subplot = None,
                       **kwargs) -> None:
        prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
        cdr.plot_rolling_drawdowns(prices=prices, dd_legend_type=dd_legend_type, ax=ax, **kwargs)
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=prices.index,
                                    regime_params=regime_params)

    def plot_rolling_time_under_water(self,
                                      time_period: TimePeriod = None,
                                      regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                      regime_benchmark: str = None,
                                      dd_legend_type: qis.DdLegendType = qis.DdLegendType.SIMPLE,
                                      add_benchmarks_to_navs: bool = False,
                                      ax: plt.Subplot = None,
                                      **kwargs) -> None:
        prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
        cdr.plot_rolling_time_under_water(prices=prices, dd_legend_type=dd_legend_type, ax=ax, **kwargs)
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=prices.index,
                                    regime_params=regime_params)

    def plot_ra_perf_table(self,
                           benchmark: str = None,
                           drop_benchmark: bool = False,
                           add_benchmarks_to_navs: bool = False,
                           time_period: TimePeriod = None,
                           perf_params: PerfParams = PERF_PARAMS,
                           perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS,
                           strategy_idx: int = 0,
                           benchmark_idx: int = 1,
                           add_turnover: bool = False,
                           ax: plt.Subplot = None,
                           **kwargs
                           ) -> pd.DataFrame:
        if benchmark is None:
            benchmark = self.benchmark_prices.columns[0]
        prices = self.get_navs(time_period=time_period, benchmark=benchmark, add_benchmarks_to_navs=add_benchmarks_to_navs)
        if add_benchmarks_to_navs:
            drop_benchmark = False
        ra_perf_title = f"RA performance table for {perf_params.freq_vol}-freq returns with beta to {benchmark}: " \
                        f"{qis.get_time_period(prices).to_str()}"

        if add_turnover:
            turnover = self.get_turnover(time_period=time_period, **kwargs)
            turnover = turnover.mean(axis=0).to_frame('Turnover')
            df_to_add = qis.df_to_str(turnover, var_format='{:,.0%}')
        else:
            df_to_add = None

        fig, ra_perf_table = ppt.plot_ra_perf_table_benchmark(prices=prices,
                                                              benchmark=benchmark,
                                                              perf_params=perf_params,
                                                              perf_columns=perf_columns,
                                                              drop_benchmark=drop_benchmark,
                                                              title=ra_perf_title,
                                                              rotation_for_columns_headers=0,
                                                              df_to_add=df_to_add,
                                                              ax=ax,
                                                              **kwargs)
        return ra_perf_table

    def plot_ac_ra_perf_table(self,
                              benchmark_price: pd.Series,
                              add_benchmarks_to_navs: bool = False,
                              time_period: TimePeriod = None,
                              perf_params: PerfParams = PERF_PARAMS,
                              perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS,
                              is_grouped: bool = True,
                              ax: plt.Subplot = None,
                              **kwargs) -> None:

        if add_benchmarks_to_navs:
            prices = self.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
            drop_benchmark = False
            rows_edge_lines = [len(self.portfolio_datas)]
        else:
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
                if add_ac and ac_prices_ is not None:
                    ac_prices_.columns = [f"{portfolio_name}-{x}" for x in ac_prices_.columns]
                    ac_prices.append(ac_prices_)
                    rows_edge_lines.append(sum(rows_edge_lines)+len(ac_prices_.columns))
            strategy_prices = pd.concat(strategy_prices, axis=1)

            benchmark_price = benchmark_price.reindex(index=strategy_prices.index, method='ffill')
            if benchmark_price.name not in strategy_prices.columns:
                prices = pd.concat([strategy_prices, benchmark_price], axis=1)
            else:
                prices = strategy_prices
            if add_ac:  # otherwise tables look too bad
                ac_prices = pd.concat(ac_prices, axis=1)
                prices = pd.concat([prices, ac_prices], axis=1)

        ra_perf_title = f"RA performance table for {perf_params.freq_vol}-freq returns with beta to " \
                        f"{benchmark_price.name}: {qis.get_time_period(prices).to_str()}"
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

    def plot_nav_with_dd(self,
                         time_period: TimePeriod = None,
                         perf_params: PerfParams = PERF_PARAMS,
                         axs: List[plt.Subplot] = None,
                         **kwargs
                         ) -> None:
        prices = self.get_navs(time_period=time_period)
        if self.benchmark_prices is not None:
            regime_benchmark = self.benchmark_prices.columns[0]
        else:
            regime_benchmark = None
        ppd.plot_prices_with_dd(prices=prices,
                                perf_params=perf_params,
                                regime_benchmark=regime_benchmark,
                                axs=axs,
                                **kwargs)

    def plot_exposures(self,
                       regime_benchmark: str = None,
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
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=exposures.index, regime_params=regime_params)

    def plot_instrument_pnl_diff(self,
                                 portfolio_idx1: int = 0,
                                 portfolio_idx2: int = 1,
                                 is_grouped: bool = True,
                                 regime_benchmark: str = None,
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
            diff = dfg.agg_df_by_groups_ax1(
                diff,
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
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=diff.index, regime_params=regime_params)

    def plot_exposures_diff(self,
                            portfolio_idx1: int = 0,
                            portfolio_idx2: int = 1,
                            regime_benchmark: str = None,
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
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=diff.index, regime_params=regime_params)
    
    def get_turnover(self,
                     time_period: TimePeriod = None,
                     turnover_rolling_period: Optional[int] = 12,
                     freq_turnover: Optional[str] = 'ME',
                     is_unit_based_traded_volume: bool = True,
                     **kwargs
                     ):
        turnover = []
        for portfolio in self.portfolio_datas:
            turnover.append(portfolio.get_turnover(roll_period=turnover_rolling_period, freq=freq_turnover, is_agg=True,
                                                   is_unit_based_traded_volume=is_unit_based_traded_volume).rename(
                portfolio.nav.name))
        turnover = pd.concat(turnover, axis=1)
        if time_period is not None:
            turnover = time_period.locate(turnover)
        return turnover
        
    def plot_turnover(self,
                      regime_benchmark: str = None,
                      time_period: TimePeriod = None,
                      regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                      turnover_rolling_period: Optional[int] = 260,
                      freq_turnover: Optional[str] = 'B',
                      var_format: str = '{:.0%}',
                      is_unit_based_traded_volume: bool = True,
                      ax: plt.Subplot = None,
                      **kwargs) -> None:
        
        turnover = self.get_turnover(turnover_rolling_period=turnover_rolling_period,
                                     freq_turnover=freq_turnover,
                                     is_unit_based_traded_volume=is_unit_based_traded_volume,
                                     time_period=time_period)
        freq = pd.infer_freq(turnover.index)
        turnover_title = f"{turnover_rolling_period}-period rolling {freq}-freq Turnover"
        pts.plot_time_series(df=turnover,
                             var_format=var_format,
                             y_limits=(0.0, None),
                             legend_stats=pts.LegendStats.AVG_NONNAN_LAST,
                             title=turnover_title,
                             ax=ax,
                             **kwargs)
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=turnover.index, regime_params=regime_params)

    def plot_costs(self,
                   cost_rolling_period: Optional[int] = 260,
                   freq_cost: Optional[str] = 'B',
                   cost_title: Optional[str] = None,
                   regime_benchmark: str = None,
                   time_period: TimePeriod = None,
                   regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                   var_format: str = '{:.2%}',
                   is_unit_based_traded_volume: bool = True,
                   ax: plt.Subplot = None,
                   **kwargs) -> None:
        costs = []
        for portfolio in self.portfolio_datas:
            costs.append(portfolio.get_costs(roll_period=cost_rolling_period, freq=freq_cost, is_agg=True,
                                             is_unit_based_traded_volume=is_unit_based_traded_volume).rename(portfolio.nav.name))
        costs = pd.concat(costs, axis=1)
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
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=costs.index, regime_params=regime_params)

    def plot_factor_betas(self,
                          benchmark_prices: pd.DataFrame,
                          regime_benchmark: str = None,
                          time_period: TimePeriod = None,
                          regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                          freq_beta: str = 'B',
                          factor_beta_span: int = 260,
                          var_format: str = '{:,.2f}',
                          axs: List[plt.Subplot] = None,
                          **kwargs
                          ) -> None:
        """
        plot benchmarks betas by factor exposures
        """
        factor_exposures = {factor: [] for factor in benchmark_prices.columns}
        for portfolio in self.portfolio_datas:
            factor_exposure = portfolio.compute_portfolio_benchmark_betas(benchmark_prices=benchmark_prices,
                                                                          freq_beta=freq_beta,
                                                                          factor_beta_span=factor_beta_span,
                                                                          time_period=time_period)
            for factor in factor_exposure.columns:
                factor_exposures[factor].append(factor_exposure[factor].rename(portfolio.nav.name))

        if axs is None:
            fig, axs = plt.subplots(len(benchmark_prices.columns), 1, figsize=(12, 12), tight_layout=True)

        for idx, factor in enumerate(benchmark_prices.columns):
            factor_exposure = pd.concat(factor_exposures[factor], axis=1)
            factor_beta_title = f"{factor_beta_span}-span rolling Beta of {freq_beta}-freq returns to {factor}"
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
            datas.append(self.portfolio_datas[portfolio_id].get_performance_attribution_data(attribution_metric=attribution_metric,
                                                                                             time_period=time_period))
        data = pd.concat(datas, axis=1)
        data = data.sort_values(data.columns[0], ascending=False)
        kwargs = sop.update_kwargs(kwargs=kwargs,
                                         new_kwargs={'ncols': len(data.columns),
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
                         add_benchmarks_to_navs: bool = False,
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
            prices = self.get_navs(benchmark=benchmark, add_benchmarks_to_navs=add_benchmarks_to_navs, time_period=time_period)
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
                                    group_data: Optional[pd.Series] = None,
                                    group_order: Optional[List[str]] = None,
                                    freq: Optional[str] = None,
                                    total_column: str = 'Total Sum',
                                    time_period: TimePeriod = None,
                                    is_exclude_interaction_term: bool = True
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        strategy_pnl = self.portfolio_datas[strategy_idx].get_attribution_table_by_instrument(time_period=time_period, freq=freq)
        strategy_weights = self.portfolio_datas[strategy_idx].get_weights(time_period=time_period, freq=freq, is_input_weights=False)

        benchmark_pnl = self.portfolio_datas[benchmark_idx].get_attribution_table_by_instrument(time_period=time_period, freq=freq)
        benchmark_weights = self.portfolio_datas[benchmark_idx].get_weights(time_period=time_period, freq=freq, is_input_weights=False)

        if group_data is None:
            group_data = self.portfolio_datas[strategy_idx].group_data
        if group_order is None:
            group_order = self.portfolio_datas[strategy_idx].group_order

        totals_table, active_total, grouped_allocation_return, grouped_selection_return, grouped_interaction_return = \
            qis.compute_brinson_attribution_table(benchmark_pnl=benchmark_pnl,
                                                  strategy_pnl=strategy_pnl,
                                                  strategy_weights=strategy_weights,
                                                  benchmark_weights=benchmark_weights,
                                                  asset_class_data=group_data,
                                                  group_order=group_order,
                                                  total_column=total_column,
                                                  is_exclude_interaction_term=is_exclude_interaction_term,
                                                  strategy_name=self.portfolio_datas[strategy_idx].ticker or 'Strategy',
                                                  benchmark_name=self.portfolio_datas[benchmark_idx].ticker or 'Benchmark')
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

        local_kwargs = qis.update_kwargs(kwargs, dict(ncols=2, legend_loc='upper center', showmedians=True,
                                                      yvar_format='{:.1%}'))
        qis.df_dict_boxplot_by_columns(dfs=dfs,
                                       hue_var_name='groups' if is_grouped else 'instruments',
                                       y_var_name='weights',
                                       ylabel='weights',
                                       ax=ax,
                                       **local_kwargs)

    def plot_group_exposures_and_pnl(self,
                                     regime_benchmark: str = None,
                                     time_period: TimePeriod = None,
                                     regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                     total_name: str = 'Total',
                                     exposures_freq: Optional[str] = 'W-WED',
                                     figsize: Tuple[float, float] = (8.3, 11.7),  # A4 for portrait
                                     **kwargs
                                     ) -> List[plt.Figure]:
        """
        report total exposures by portfolio
        """
        # aling to the first
        group_data = self.portfolio_datas[0].group_data
        group_order = self.portfolio_datas[0].group_order
        grouped_exposures_aggs = {}  # dict[portfolio, Dict[group, pd.Dataframe]]
        grouped_pnls_aggs = {}
        for portfolio_data in self.portfolio_datas:
            grouped_exposures_agg, grouped_exposures_by_inst = portfolio_data.get_grouped_long_short_exposures(time_period=time_period,
                                                                                                               exposures_freq=exposures_freq,
                                                                                                               total_name=total_name)
            grouped_pnls_agg, grouped_pnls_by_inst = portfolio_data.get_grouped_cum_pnls(time_period=time_period,
                                                                                         total_name=total_name)

            grouped_exposures_aggs[portfolio_data.ticker] = grouped_exposures_agg
            grouped_pnls_aggs[portfolio_data.ticker] = grouped_pnls_agg

        group_exposures_by_portfolio = {group: {} for group in group_order}
        for group in group_data:
            for ticker, df in grouped_exposures_aggs.items():
                group_exposures_by_portfolio[group].update({ticker: df[group][total_name]})

        group_pnl_by_portfolio = {group: {} for group in group_order}
        for group in group_data:
            for ticker, df in grouped_pnls_aggs.items():
                group_pnl_by_portfolio[group].update({ticker: df[group][total_name]})

        for group in group_order:
            group_exposures_by_portfolio[group] = pd.DataFrame.from_dict(group_exposures_by_portfolio[group], orient='columns')
            group_pnl_by_portfolio[group] = pd.DataFrame.from_dict(group_pnl_by_portfolio[group], orient='columns')

        figs = []
        for group, exposures_agg in group_exposures_by_portfolio.items():
            fig, axs = plt.subplots(2, 1, figsize=figsize, tight_layout=True)
            qis.set_suptitle(fig, f"Total Exposures and P&L by {group}")
            figs.append(fig)
            qis.plot_time_series(df=exposures_agg,
                                 var_format='{:,.0%}',
                                 legend_stats=qis.LegendStats.AVG_MIN_MAX_LAST,
                                 title=f"Exposure by {group}",
                                 ax=axs[0],
                                 **kwargs)
            qis.plot_time_series(df=group_pnl_by_portfolio[group],
                                 var_format='{:,.0%}',
                                 legend_stats=qis.LegendStats.LAST,
                                 title=f"Cumulative P&L by {group}",
                                 ax=axs[1],
                                 **kwargs)
            if regime_benchmark is not None:
                for ax in axs:
                    self.add_regime_shadows(ax=ax,
                                            regime_benchmark=regime_benchmark,
                                            index=exposures_agg.index,
                                            regime_params=regime_params)

        return figs

    def plot_tre_time_series(self,
                             strategy_idx: int = 0,
                             benchmark_idx: int = 1,
                             regime_benchmark: str = None,
                             regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                             time_period: TimePeriod = None,
                             title: Optional[str] = 'Tracking error',
                             var_format: str = '{:.2%}',
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> None:
        tre = self.compute_tracking_error_implied_by_covar(strategy_idx=strategy_idx, benchmark_idx=benchmark_idx)
        if time_period is not None:
            tre = time_period.locate(tre)
        pts.plot_time_series(df=tre,
                             var_format=var_format,
                             legend_stats=pts.LegendStats.AVG_NONNAN_LAST,
                             title=title,
                             ax=ax,
                             **kwargs)
        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, index=tre.index, regime_params=regime_params)
