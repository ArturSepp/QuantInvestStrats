
from __future__ import annotations

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit
from dataclasses import dataclass
from typing import Union, Dict, Any, Optional, Tuple, List, NamedTuple
from enum import Enum

# qis
import qis.file_utils as fu
import qis.plots.derived.regime_data
import qis.utils.dates as da
import qis.utils.df_groups as dfg
import qis.utils.struct_ops as sop
from qis.utils import EnumMap
from qis.perfstats.config import PerfParams, RegimeData
import qis.perfstats.returns as ret
import qis.perfstats.perf_stats as rpt
import qis.perfstats.regime_classifier as rcl
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs

import qis.models.linear.ewm_factors as ef
import qis.plots.time_series as pts
import qis.plots.stackplot as pst
import qis.plots.derived.prices as ppd
import qis.plots.derived.perf_table as ppt
import qis.plots.derived.returns_scatter as prs
import qis.plots.derived.returns_heatmap as rhe
from qis.plots.derived.returns_heatmap import plot_returns_heatmap


class MetricSpec(NamedTuple):
    title: str


class AttributionMetric(MetricSpec, Enum):
    PNL = MetricSpec(title='P&L Attribution, sum=portfolio performance')
    PNL_RISK = MetricSpec(title='P&L Risk Attribution, sum=100%')
    INST_PNL = MetricSpec(title='Instrument P&L')


@dataclass
class PortfolioData:
    nav: pd.Series  # nav is computed with all cost
    weights: pd.DataFrame = None  # weights of portfolio
    units: pd.DataFrame = None  # units of portfolio instruments
    prices: pd.DataFrame = None  # prices of portfolio universe
    instrument_pnl: pd.DataFrame = None  # include net pnl by intsrument
    realized_costs: pd.DataFrame = None  # realized trading costs by instrument
    input_weights: Union[np.ndarray, pd.DataFrame, Dict[str, float]] = None  # inputs to potfolio
    is_rebalancing: pd.Series = None  # optional rebal info
    tickers_to_names_map: Optional[Dict[str, str]] = None  # renaming of long tickers
    group_data: pd.Series = None  # for asset class grouping
    group_order: List[str] = None

    def __post_init__(self):

        if isinstance(self.nav, pd.DataFrame):
            self.nav = self.nav.iloc[:, 0]
        if self.prices is None:
            self.prices = self.nav.to_frame()
        if self.weights is None:  # default will be delta-1 portfolio of nav
            self.weights = pd.DataFrame(1.0, index=self.prices.index, columns=self.prices.columns)
        if self.units is None:  # default will be delta-1 portfolio of nav
            self.units = pd.DataFrame(1.0, index=self.prices.index, columns=self.prices.columns)
        if self.instrument_pnl is None:
            self.instrument_pnl = self.prices.pct_change().multiply(self.weights.shift(1)).fillna(0.0)
        if self.realized_costs is None:
            self.realized_costs = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        if self.group_data is None:  # use instruments as groups
            self.group_data = pd.Series(self.prices.columns, index=self.prices.columns)
        if self.group_order is None:
            self.group_order = list(self.group_data.unique())

    def save(self, ticker: str) -> None:
        datasets = dict(nav=self.nav, prices=self.prices, weights=self.weights, units=self.units,
                        instrument_pnl=self.instrument_pnl, realized_costs=self.realized_costs)
        if self.group_data is not None:
            datasets['group_data'] = self.group_data
        fu.save_df_dict_to_csv(datasets=datasets, file_name=ticker)
        print(f"saved portfolio data for {ticker}")

    @classmethod
    def load(cls, ticker: str) -> PortfolioData:
        dataset_keys = ['nav', 'prices', 'weights', 'units', 'instrument_pnl', 'realized_costs', 'group_data']
        datasets = fu.load_df_dict_from_csv(dataset_keys=dataset_keys, file_name=ticker)
        return cls(**datasets)

    def _set_group_data(self, group_data: pd.Series, group_order: List[str] = None) -> None:
        self.group_data = group_data
        self.group_order = group_order

    """
    NAV level getters
    """
    def get_portfolio_nav(self, time_period: da.TimePeriod = None) -> pd.Series:
        """
        get nav using consistent function for all return computations
        """
        if time_period is not None:
            nav_ = time_period.locate(self.nav)
        else:
            nav_ = self.nav
        return nav_

    def get_instruments_pnl(self,
                            time_period: da.TimePeriod = None,
                            is_compounded: bool = False
                            ) -> pd.DataFrame:

        pnl = self.instrument_pnl
        if is_compounded:
            pnl = np.expm1(pnl)
        if time_period is not None:
            pnl = time_period.locate(pnl)
        return pnl

    def get_instruments_navs(self,
                             time_period: da.TimePeriod = None,
                             constant_trade_level: bool = False
                             ) -> pd.DataFrame:
        pnl = self.get_instruments_pnl(time_period=time_period)
        navs = ret.returns_to_nav(returns=pnl, constant_trade_level=constant_trade_level)
        return navs

    def get_ac_navs(self,
                    time_period: da.TimePeriod = None,
                    constant_trade_level: bool = False
                    ) -> pd.DataFrame:
        grouped_ac_pnl = dfg.agg_df_by_groups_ax1(df=self.get_instruments_pnl(time_period=time_period),
                                                  group_data=self.group_data,
                                                  agg_func=np.sum,
                                                  total_column=str(self.nav.name),
                                                  group_order=self.group_order)
        ac_navs = ret.returns_to_nav(returns=grouped_ac_pnl, constant_trade_level=constant_trade_level)
        return ac_navs

    def get_exposures(self,
                      time_period: da.TimePeriod = None,
                      is_grouped: bool = False,
                      add_total: bool = True
                      ) -> pd.DataFrame:
        if is_grouped:
            exposures = dfg.agg_df_by_groups_ax1(df=self.weights,
                                                 group_data=self.group_data,
                                                 agg_func=np.nansum,
                                                 total_column=str(self.nav.name) if add_total else None,
                                                 group_order=self.group_order)
        else:
            exposures = self.weights
        if time_period is not None:
            exposures = time_period.locate(exposures)
        return exposures

    def get_turnover(self,
                     is_agg: bool = False,
                     is_grouped: bool = False,
                     time_period: da.TimePeriod = None,
                     roll_period: Optional[int] = 260,
                     add_total: bool = True,
                     freq: Optional[str] = None
                     ) -> Union[pd.DataFrame, pd.Series]:
        turnover = (self.units.diff(1).abs()).multiply(self.prices)
        abs_exposure = self.units.multiply(self.prices).abs().sum(axis=1)
        # turnover = turnover.divide(self.nav.to_numpy(), axis=0)
        turnover = turnover.divide(abs_exposure.to_numpy(), axis=0)
        if is_agg:
            turnover = pd.Series(np.nansum(turnover, axis=1), index=self.nav.index, name=self.nav.name)
        elif is_grouped or len(turnover.columns) > 10:  # agg by groups
            turnover = dfg.agg_df_by_groups_ax1(df=turnover,
                                                group_data=self.group_data,
                                                agg_func=np.nansum,
                                                total_column=str(self.nav.name) if add_total else None,
                                                group_order=self.group_order)
        else:
            if add_total:
                turnover = pd.concat([turnover.sum(axis=1).rename(self.nav.name), turnover], axis=1)

        if roll_period is not None:
            turnover = turnover.rolling(roll_period).sum()
        elif freq is not None:
            turnover = turnover.resample(freq).sum()
        if time_period is not None:
            turnover = time_period.locate(turnover)
        return turnover

    def get_costs(self,
                  is_agg: bool = False,
                  is_grouped: bool = False,
                  time_period: da.TimePeriod = None,
                  add_total: bool = True,
                  roll_period: Optional[int] = 260
                  ) -> Union[pd.DataFrame, pd.Series]:

        costs = self.realized_costs.divide(self.nav.to_numpy(), axis=0)
        if is_agg:
            costs = pd.Series(np.nansum(costs, axis=1), index=self.nav.index, name=self.nav.name)
        elif is_grouped:
            costs = dfg.agg_df_by_groups_ax1(costs,
                                             group_data=self.group_data,
                                             agg_func=np.nansum,
                                             total_column=str(self.nav.name) if add_total else None,
                                             group_order=self.group_order)
        else:
            if add_total:
                costs = pd.concat([costs.sum(axis=1).rename(self.nav.name), costs], axis=1)

        if roll_period is not None:
            costs = costs.rolling(roll_period).sum()
        if time_period is not None:
            costs = time_period.locate(costs)
        return costs

    def compute_mcap_participation(self,
                                   mcap: pd.DataFrame,
                                   trade_level: float = 100000000
                                   ) -> pd.DataFrame:
        exposure = (self.units.multiply(self.prices)).divide(self.nav.to_numpy(), axis=0)
        participation = trade_level*exposure.divide(mcap)
        return participation

    def compute_volume_participation(self,
                                     volumes: pd.DataFrame,
                                     trade_level: float = 100000000
                                     ) -> pd.DataFrame:
        turnover = self.get_turnover(is_agg=False)
        participation = trade_level*turnover.divide(volumes)
        return participation

    def compute_cumulative_attribution(self) -> pd.DataFrame:
        attribution = (self.prices.pct_change()).multiply(self.weights.shift(1))
        attribution = attribution.cumsum(axis=0)
        return attribution

    def compute_realized_pnl(self, time_period: da.TimePeriod = None) -> Tuple[pd.DataFrame, ...]:
        avg_costs, realized_pnl, mtm_pnl, trades = compute_realized_pnl(prices=self.prices.to_numpy(),
                                                                        units=self.units.to_numpy())
        avg_costs = pd.DataFrame(avg_costs, index=self.prices.index, columns=self.prices.columns)
        realized_pnl = pd.DataFrame(realized_pnl, index=self.prices.index, columns=self.prices.columns)
        mtm_pnl = pd.DataFrame(mtm_pnl, index=self.prices.index, columns=self.prices.columns)
        trades = pd.DataFrame(trades, index=self.prices.index, columns=self.prices.columns)
        if time_period is not None:
            avg_costs = time_period.locate(avg_costs)
            realized_pnl = time_period.locate(realized_pnl)
            mtm_pnl = time_period.locate(mtm_pnl)
            trades = time_period.locate(trades)
        realized_pnl = realized_pnl.cumsum(axis=0)
        total_pnl = realized_pnl.add(mtm_pnl)
        return avg_costs, realized_pnl, mtm_pnl, total_pnl, trades

    def compute_portfolio_benchmark_betas(self, benchmark_prices: pd.DataFrame,
                                          time_period: da.TimePeriod = None,
                                          freq: str = None,
                                          span: int = 65 # quarter
                                          ) -> pd.DataFrame:
        instrument_prices = self.prices
        benchmark_prices = benchmark_prices.reindex(index=instrument_prices.index, method='ffill')
        ewm_linear_model = ef.estimate_ewm_linear_model(x=ret.to_returns(prices=benchmark_prices, freq=freq),
                                                        y=ret.to_returns(prices=instrument_prices, freq=freq),
                                                        span=span,
                                                        is_x_correlated=True)
        exposures = self.get_exposures().reindex(index=instrument_prices.index, method='ffill')
        benchmark_betas = ewm_linear_model.compute_agg_factor_exposures(asset_exposures=exposures)
        benchmark_betas = benchmark_betas.replace({0.0: np.nan}).fillna(method='ffill')  # fillholidays
        if time_period is not None:
            benchmark_betas = time_period.locate(benchmark_betas)
        return benchmark_betas

    def compute_portfolio_benchmark_attribution(self, benchmark_prices: pd.DataFrame,
                                                time_period: da.TimePeriod = None,
                                                freq: str = 'B',
                                                span: int = 63 # quarter
                                                ) -> pd.DataFrame:
        portfolio_benchmark_betas = self.compute_portfolio_benchmark_betas(benchmark_prices=benchmark_prices,
                                                                           freq=freq, span=span)
        benchmark_prices = benchmark_prices.reindex(index=portfolio_benchmark_betas.index, method='ffill')
        x = ret.to_returns(prices=benchmark_prices, freq=freq)
        x_attribution = (portfolio_benchmark_betas.shift(1)).multiply(x)
        total_attrib = x_attribution.sum(1)
        total = self.get_portfolio_nav().reindex(index=total_attrib.index, method='ffill').pct_change()
        residual = np.subtract(total, total_attrib)
        # joint_attrib = pd.concat([x_attribution, total_attrib.rename('Total benchmarks'), residual.rename('Residual')], axis=1)
        joint_attrib = pd.concat([x_attribution, residual.rename('Residual')], axis=1)
        if time_period is not None:
            joint_attrib = time_period.locate(joint_attrib)
        joint_attrib = joint_attrib.cumsum(axis=0)
        return joint_attrib

    """
    ### instrument level getters
    """
    def get_instruments_returns(self,
                                time_period: da.TimePeriod = None
                                ) -> pd.DataFrame:
        returns = self.prices.pct_change()
        if time_period is not None:
            returns = time_period.locate(returns)
        return returns

    def get_instruments_periodic_returns(self,
                                         time_period: da.TimePeriod = None,
                                         freq: str = 'M'
                                         ) -> pd.DataFrame:
        returns = self.get_instruments_returns(time_period=time_period)
        prices = ret.returns_to_nav(returns=returns, init_period=None)
        returns_f = ret.to_returns(prices=prices, freq=freq)
        return returns_f

    def get_instruments_performance_attribution(self,
                                                time_period: da.TimePeriod = None,
                                                constant_trade_level: bool = False
                                                ) -> pd.Series:
        navs = self.get_instruments_navs(time_period=time_period, constant_trade_level=constant_trade_level)
        perf = ret.to_total_returns(prices=navs).rename(self.nav.name)
        return perf

    def get_instruments_pnl_risk_attribution(self,
                                             time_period: da.TimePeriod = None
                                             ) -> pd.DataFrame:
        pnl = self.get_instruments_pnl(time_period=time_period)
        portfolio_pnl = pnl.sum(axis=1)

        pnl_risk = np.nanstd(pnl.replace({0.0: np.nan}), axis=0)
        portfolio_pnl_risk = np.nanstd(portfolio_pnl.replace({0.0: np.nan}), axis=0)
        # pnl_risk_ratio = pnl_risk / portfolio_pnl_risk
        pnl_risk_ratio = pnl_risk / np.nansum(pnl_risk)

        data = pd.Series(pnl_risk_ratio, index=pnl.columns, name=self.nav.name)
        if self.tickers_to_names_map is not None:
            data = data.rename(index=self.tickers_to_names_map)
        return data

    def get_performance_data(self,
                             attribution_metric: AttributionMetric = AttributionMetric.PNL,
                             time_period: da.TimePeriod = None
                             ) -> Union[pd.DataFrame, pd.Series]:
        if attribution_metric == AttributionMetric.PNL:
            data = self.get_instruments_performance_attribution(time_period=time_period)
        elif attribution_metric == AttributionMetric.PNL_RISK:
            data = self.get_instruments_pnl_risk_attribution(time_period=time_period)
        elif attribution_metric == AttributionMetric.INST_PNL:
            data = self.get_instruments_navs(time_period=time_period)
        else:
            raise NotImplementedError(f"{attribution_metric}")

        return data

    def get_num_investable_instruments(self, time_period: da.TimePeriod = None) -> pd.Series:
        exposures = self.weights.replace({0.0: np.nan})
        count = np.sum(np.where(np.isfinite(exposures), 1.0, 0.0), axis=1)
        num_investable_instruments = pd.Series(count, index=exposures.index, name=self.nav.name)
        if time_period is not None:
            num_investable_instruments = time_period.locate(num_investable_instruments)
        return num_investable_instruments

    def get_instruments_performance_table(self,
                                          time_period: da.TimePeriod = None,
                                          portfolio_name: str = 'Attribution'
                                          ) -> pd.DataFrame:
        """
        using avg weight
        """
        insts_returns = self.get_instruments_returns(time_period=time_period)
        insts_return = ret.to_total_returns(prices=ret.returns_to_nav(returns=insts_returns))

        weight = self.weights
        if time_period is not None:
            weight = time_period.locate(weight)
        weight = weight.mean(axis=0)
        portf_return = insts_return.multiply(weight).replace({0.0: np.nan}).dropna()
        data = pd.concat([weight.rename('Weight'),
                          insts_return.rename('Asset'),
                          portf_return.rename(portfolio_name)],
                         axis=1).dropna()
        data = data.sort_values('Weight', ascending=False)
        if self.tickers_to_names_map is not None:
            data = data.rename(index=self.tickers_to_names_map)

        return data

    def get_attribution_table_by_instrument(self,
                                            time_period: da.TimePeriod = None,
                                            freq: str = 'M',
                                            ) -> pd.DataFrame:
        """
        using avg weight
        """
        returns_f = self.get_instruments_periodic_returns(time_period=time_period, freq=freq)
        weight = self.weights.reindex(index=returns_f.index, method='ffill').shift(1)
        # first row is None
        portf_return = returns_f.multiply(weight).iloc[1:, :]
        if self.tickers_to_names_map is not None:
            portf_return = portf_return.rename(columns=self.tickers_to_names_map)
        return portf_return

    """
    plotting methods
    """
    def plot_nav(self, time_period: da.TimePeriod = None, **kwargs) -> None:
        nav = self.get_portfolio_nav(time_period=time_period)
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(16, 12), tight_layout=True)
            ppd.plot_prices_with_dd(prices=nav,
                                    axs=axs,
                                    **kwargs)

    def plot_ra_perf_table(self,
                           benchmark_price: pd.Series = None,
                           is_grouped: bool = True,
                           time_period: da.TimePeriod = None,
                           perf_params: PerfParams = None,
                           title: str = None,
                           ax: plt.Subplot = None,
                           **kwargs) -> None:
        if is_grouped:
            prices = self.get_ac_navs(time_period=time_period)
            title = title or f"RA performance table by groups: {da.get_time_period(prices).to_str()}"
        else:
            prices = self.get_portfolio_nav(time_period=time_period)
            title = title or f"RA performance table: {da.get_time_period(prices).to_str()}"
        if benchmark_price is not None:
            prices = pd.concat([prices, benchmark_price.reindex(index=prices.index, method='ffill')], axis=1)
            ppt.plot_ra_perf_table_benchmark(prices=prices,
                                             benchmark=str(benchmark_price.name),
                                             perf_params=perf_params,
                                             perf_columns=rpt.BENCHMARK_TABLE_COLUMNS,
                                             title=title,
                                             rotation_for_columns_headers=0,
                                             special_rows_colors=[(1, 'deepskyblue'), (len(prices.columns), 'lavender')],
                                             column_header='Portfolio',
                                             ax=ax,
                                             **kwargs)
        else:
            ppt.plot_ra_perf_table(prices=prices,
                                   perf_params=perf_params,
                                   perf_columns=rpt.COMPACT_TABLE_COLUMNS,
                                   title=title,
                                   rotation_for_columns_headers=0,
                                   column_header='Portfolio',
                                   ax=ax,
                                   **kwargs)

    def plot_returns_scatter(self,
                             benchmark_price: pd.Series = None,
                             is_grouped: bool = True,
                             time_period: da.TimePeriod = None,
                             title: str = None,
                             freq: str = 'Q',
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> None:
        if is_grouped:
            prices = self.get_ac_navs(time_period=time_period)
            title = title or f"Scatterplot of {freq}-returns by groups vs {str(benchmark_price.name)}"
        else:
            prices = self.get_portfolio_nav(time_period=time_period)
            title = title or f"Scatterplot of {freq}-returns vs {str(benchmark_price.name)}"
        prices = pd.concat([prices, benchmark_price.reindex(index=prices.index, method='ffill')], axis=1)
        local_kwargs = sop.update_kwargs(kwargs=kwargs,
                                         new_kwargs={'weight': 'bold',
                                                     #'alpha_an_factor': 52.0,
                                                     'x_rotation': 0,
                                                     'first_color_fixed': False,
                                                     'ci': None})
        prs.plot_returns_scatter(prices=prices,
                                 benchmark=str(benchmark_price.name),
                                 freq=freq,
                                 order=2,
                                 title=title,
                                 ax=ax,
                                 **local_kwargs)

    def plot_monthly_returns_heatmap(self, time_period: da.TimePeriod = None,
                                     heatmap_freq: str = 'A',
                                     date_format: str = '%Y',
                                     ax: plt.Subplot = None,
                                     **kwargs
                                     ) -> None:
        plot_returns_heatmap(prices=self.get_portfolio_nav(time_period=time_period),
                             heatmap_column_freq='M',
                             is_add_annual_column=True,
                             is_inverse_order=True,
                             heatmap_freq=heatmap_freq,
                             date_format=date_format,
                             ax=ax,
                             **kwargs)

    def plot_periodic_returns(self,
                              benchmark_prices: pd.DataFrame = None,
                              is_grouped: bool = True,
                              time_period: da.TimePeriod = None,
                              heatmap_freq: str = 'A',
                              date_format: str = '%Y',
                              transpose: bool = True,
                              title: str = None,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:
        if is_grouped:
            prices = self.get_ac_navs(time_period=time_period)
            title = title or f"{heatmap_freq}-returns by groups"
        else:
            prices = self.get_portfolio_nav(time_period=time_period)
            title = title or f"{heatmap_freq}-returns"
        if benchmark_prices is not None:
            hline_rows = [len(prices.columns)]
            prices = pd.concat([prices, benchmark_prices.reindex(index=prices.index, method='ffill')], axis=1)
        else:
            hline_rows = None

        rhe.plot_periodic_returns_table(prices=prices,
                                                    freq=heatmap_freq,
                                                    ax=ax,
                                                    title=title,
                                                    date_format=date_format,
                                                    transpose=transpose,
                                                    hline_rows=hline_rows,
                                                    **kwargs)

    def plot_regime_data(self,
                         benchmark_price: pd.Series,
                         is_grouped: bool = True,
                         regime_data_to_plot: RegimeData = RegimeData.REGIME_SHARPE,
                         time_period: da.TimePeriod = None,
                         var_format: Optional[str] = None,
                         is_conditional_sharpe: bool = True,
                         legend_loc: Optional[str] = 'upper center',
                         title: str = None,
                         perf_params: PerfParams = None,
                         regime_params: BenchmarkReturnsQuantileRegimeSpecs = None,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:

        if is_grouped:
            prices = self.get_ac_navs(time_period=time_period)
            title = title or f"Sharpe ratio decomposition by groups to {str(benchmark_price.name)} Bear/Normal/Bull regimes"
        else:
            prices = self.get_portfolio_nav(time_period=time_period)
            title = title or f"Sharpe ratio decomposition to {str(benchmark_price.name)} Bear/Normal/Bull regimes"
        prices = pd.concat([benchmark_price.reindex(index=prices.index, method='ffill'), prices], axis=1)

        regime_classifier = rcl.BenchmarkReturnsQuantilesRegime(regime_params=regime_params)
        fig = qis.plots.derived.regime_data.plot_regime_data(regime_classifier=regime_classifier,
                                                             prices=prices,
                                                             benchmark=str(benchmark_price.name),
                                                             is_conditional_sharpe=is_conditional_sharpe,
                                                             regime_data_to_plot=regime_data_to_plot,
                                                             var_format=var_format or '{:.2f}',
                                                             legend_loc=legend_loc,
                                                             perf_params=perf_params,
                                                             title=title,
                                                             ax=ax,
                                                             **kwargs)
        return fig

    def plot_vol_regimes(self,
                         benchmark_price: pd.Series,
                         is_grouped: bool = True,
                         time_period: da.TimePeriod = None,
                         title: str = None,
                         regime_params: BenchmarkReturnsQuantileRegimeSpecs = None,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:

        if is_grouped:
            prices = self.get_ac_navs(time_period=time_period)
            title = title or f"{regime_params.freq}-returns by groups conditional on vols {str(benchmark_price.name)}"
        else:
            prices = self.get_portfolio_nav(time_period=time_period)
            title = title or f"{regime_params.freq}-returns conditional on vols {str(benchmark_price.name)}"
        prices = pd.concat([benchmark_price.reindex(index=prices.index, method='ffill'), prices], axis=1)

        regime_classifier = rcl.BenchmarkVolsQuantilesRegime(regime_params=rcl.VolQuantileRegimeSpecs(freq=regime_params.freq))
        fig = qis.plots.derived.regime_data.plot_regime_boxplot(regime_classifier=regime_classifier,
                                                                prices=prices,
                                                                benchmark=str(benchmark_price.name),
                                                                title=title,
                                                                ax=ax,
                                                                **kwargs)
        return fig

    def plot_contributors(self,
                          time_period: da.TimePeriod = None,
                          num_assets: int = 10,
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> None:
        prices = self.get_instruments_navs(time_period=time_period)
        ppt.plot_top_bottom_performers(prices=prices, num_assets=num_assets, ax=ax, **kwargs)

    def plot_pnl(self, time_period: da.TimePeriod = None) -> None:
        avg_costs, realized_pnl, mtm_pnl, total_pnl, trades = self.compute_realized_pnl(time_period=time_period)
        prices = self.prices
        if time_period is not None:
            prices = time_period.locate(prices)
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(5, 1, figsize=(10, 16), tight_layout=True)
            pts.plot_time_series(df=prices, legend_stats=pts.LegendStats.FIRST_AVG_LAST, title='prices', ax=axs[0])
            pts.plot_time_series(df=avg_costs, legend_stats=pts.LegendStats.FIRST_AVG_LAST, title='avg_costs', ax=axs[1])
            pts.plot_time_series(df=realized_pnl, legend_stats=pts.LegendStats.FIRST_AVG_LAST, title='realized_pnl', ax=axs[2])
            pts.plot_time_series(df=mtm_pnl, legend_stats=pts.LegendStats.FIRST_AVG_LAST, title='mtm_pnl', ax=axs[3])
            pts.plot_time_series(df=total_pnl, legend_stats=pts.LegendStats.FIRST_AVG_LAST, title='total_pnl', ax=axs[4])

    def plot_weights(self,
                     is_input_weights: bool = True,
                     columns: List[str] = None,
                     freq: str = 'W-WED',
                     bbox_to_anchor: Tuple[float, float] = (0.4, 1.14),
                     ax: plt.Subplot = None,
                     title: str = '',
                     **kwargs
                     ) -> None:
        if is_input_weights:
            weights = self.input_weights.copy()
        else:
            weights = self.weights.copy()
        if columns is not None:
            weights = weights[columns]
        weights = weights.resample(freq).last().fillna(method='ffill')
        pst.plot_stack(df=weights,
                       add_mean_levels=False,
                       is_use_bar_plot=True,
                       # is_yaxis_limit_01=True,
                       baseline='zero',
                       bbox_to_anchor=bbox_to_anchor,
                       title=title,
                       legend_stats=pst.LegendStats.FIRST_AVG_LAST,
                       var_format='{:.1%}',
                       ax=ax,
                       **kwargs)


@njit
def compute_realized_pnl(prices: np.ndarray,
                         units: np.ndarray
                         ) -> Tuple[np.ndarray, ...]:
    """
    pnl for long only positions, computes avg entry price and pnl by instrument
    """
    avg_costs = np.zeros_like(prices)
    realized_pnl = np.zeros_like(prices)
    mtm_pnl = np.zeros_like(prices)
    trades = np.zeros_like(prices)
    for idx, (price1, unit1) in enumerate(zip(prices, units)):
        if idx == 0:
            avg_costs[idx] = np.where(np.greater(unit1, 1e-16), price1, 0.0)
        else:
            unit0 = units[idx-1]
            avg_costs0 = avg_costs[idx-1]
            delta = unit1 - unit0
            is_purchase = np.greater(delta, 1e-16)
            is_sell = np.less(delta, -1e-16)
            realized_pnl[idx] = np.where(is_sell, -delta*(price1-avg_costs0), 0.0)
            avg_costs[idx] = np.where(is_purchase, np.true_divide(delta*price1+unit0*avg_costs0, unit1), avg_costs0)
            mtm_pnl[idx] = unit0*(price1 - avg_costs0) - realized_pnl[idx]
            trades[idx] = delta
    return avg_costs, realized_pnl, mtm_pnl, trades


class AllocationType(EnumMap):
    EW = 1
    FIXED_WEIGHTS = 2
    ERC = 3
    ERC_ALT = 4


@dataclass
class PortfolioInput:
    """
    define data inputs for portfolio construction
    """
    name: str
    weights: Union[np.ndarray, pd.DataFrame, Dict[str, float]]
    prices: pd.DataFrame = None  # mandatory but we set none for enumarators
    allocation_type: AllocationType = AllocationType.FIXED_WEIGHTS
    time_period: da.TimePeriod = None
    rebalance_freq: str = 'Q'
    regime_freq: str = 'M'
    returns_freq: str = 'M'
    ewm_lambda: float = 0.92
    target_vol: float = None

    def update(self, new: Dict[Any, Any]):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
