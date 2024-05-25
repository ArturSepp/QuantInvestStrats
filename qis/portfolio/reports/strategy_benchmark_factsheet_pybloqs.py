"""
factsheet for multi strategy report for cross sectional comparison of strategies
and generating sensitivities to parameters
see example in qis.examples.factheets.multi_strategy.py
in pybloqs\jinja\table.html in line 44 have this: {% for col_name, cell in row.items() %}
"""
# packages
import numpy as np
import pandas as pd
import pybloqs as p
import pybloqs.block.table_formatters as tf

# qis
import qis as qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantileRegimeSpecs
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.portfolio.reports.config import PERF_PARAMS, REGIME_PARAMS
from qis.portfolio.reports.config import KWARGS_SUPTITLE, KWARGS_TITLE, KWARGS_TEXT, KWARGS_FOOTNOTE


def generate_strategy_benchmark_factsheet_with_pyblogs(multi_portfolio_data: MultiPortfolioData,
                                                       strategy_idx: int = 0,  # strategy is multi_portfolio_data[strategy_idx]
                                                       benchmark_idx: int = 1,  # benchmark is multi_portfolio_data[benchmark_idx]
                                                       time_period: TimePeriod = None,
                                                       time_period_last: TimePeriod = None,
                                                       perf_params: PerfParams = PERF_PARAMS,
                                                       regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                                       benchmark: str = None,
                                                       backtest_name: str = None,
                                                       heatmap_freq: str = 'YE',
                                                       weight_freq: str = 'ME',
                                                       is_input_weights: bool = True,
                                                       fontsize: int = 4,
                                                       **kwargs
                                                       ) -> p.VStack:
    """
    this one is well suited for large backtests with over 100 strategies
    """
    if time_period is None:
        raise ValueError(f"pass non none time_period")
    if time_period_last is None:
        time_period_last = qis.get_time_period_shifted_by_years(time_period=time_period, n_years=1)

    if benchmark is None:
        benchmark = multi_portfolio_data.benchmark_prices.columns[0]

    # define axiliary vars
    strategy_name = multi_portfolio_data.portfolio_datas[strategy_idx].ticker
    benchmark_name = multi_portfolio_data.portfolio_datas[benchmark_idx].ticker
    asset_class_data = multi_portfolio_data.portfolio_datas[strategy_idx].group_data
    group_order = multi_portfolio_data.portfolio_datas[strategy_idx].group_order

    plot_kwargs = dict(fontsize=fontsize,
                       linewidth=0.5,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75)
    kwargs = qis.update_kwargs(kwargs, plot_kwargs)

    # start building blocks
    blocks = [p.Paragraph(f"{backtest_name}", **KWARGS_SUPTITLE)]

    # get strategy weights
    strategy_pnl = multi_portfolio_data.portfolio_datas[strategy_idx].get_attribution_table_by_instrument(time_period=time_period,
                                                                                                          freq=weight_freq,
                                                                                                          is_input_weights=is_input_weights)
    benchmark_pnl = multi_portfolio_data.portfolio_datas[benchmark_idx].get_attribution_table_by_instrument(time_period=time_period,
                                                                                                            freq=weight_freq,
                                                                                                            is_input_weights=is_input_weights)

    strategy_weights = multi_portfolio_data.portfolio_datas[strategy_idx].get_weights(time_period=time_period,
                                                                                      freq=weight_freq,
                                                                                      is_input_weights=is_input_weights)
    benchmark_weights = multi_portfolio_data.portfolio_datas[benchmark_idx].get_weights(time_period=time_period,
                                                                                        freq=weight_freq,
                                                                                        is_input_weights=is_input_weights)

    # weights
    strategy_w = strategy_weights.iloc[-1, :].rename(strategy_name)
    benchmark_w = benchmark_weights.iloc[-1, :].rename(benchmark_name)
    delta_w = (strategy_w.subtract(benchmark_w)).rename(f"diff")

    strategy_dw = (strategy_weights.iloc[-1, :].subtract(strategy_weights.iloc[-2, :])).rename(f"delta {strategy_name}")
    benchmark_dw = (benchmark_weights.iloc[-1, :].subtract(benchmark_weights.iloc[-2, :])).rename(f"delta {benchmark_name}")
    
    weights = pd.concat([strategy_w, benchmark_w, delta_w, strategy_dw, benchmark_dw], axis=1)
    weights_by_ac = qis.agg_df_by_groups(df=weights.T, group_data=asset_class_data, group_order=group_order).T

    # attribution
    strategy_attrib_last = strategy_pnl.iloc[-1, :].rename(f"{strategy_name} last")
    benchmark_attrib_last = benchmark_pnl.iloc[-1, :].rename(f"{benchmark_name} last")
    strategy_attrib_3m = strategy_pnl.rolling(3).sum().iloc[-1, :].rename(f"{strategy_name} 3m")
    benchmark_attrib_3m = benchmark_pnl.rolling(3).sum().iloc[-1, :].rename(f"{benchmark_name} 3m")
    strategy_attrib_12m = strategy_pnl.rolling(12).sum().iloc[-1, :].rename(f"{strategy_name} 12m")
    benchmark_attrib_12m = benchmark_pnl.rolling(12).sum().iloc[-1, :].rename(f"{benchmark_name} 12m")
    attribs = pd.concat([strategy_attrib_last, benchmark_attrib_last,
                         strategy_attrib_3m, benchmark_attrib_3m,
                         strategy_attrib_12m, benchmark_attrib_12m], axis=1)

    attribs_by_ac = qis.agg_df_by_groups(df=attribs.T, group_data=asset_class_data, group_order=group_order).T
    attribs.loc['Total', :] = np.nansum(attribs, axis=0)
    attribs_by_ac.loc['Total', :] = np.nansum(attribs_by_ac, axis=0)

    # 1. weights table
    b_weights = p.Block([p.Paragraph(f"Instrument weights for {strategy_weights.index[-1]:'%d%b%Y'}", **KWARGS_TITLE),
                          p.Block(weights,
                                  formatters=[
                                      tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                                      tf.FmtReplaceNaN(value=''),
                                      tf.FmtHeatmap(columns=['diff']),  # , max_color=(255,0,255)
                                      tf.FmtAddCellBorder(each=1.0,
                                                          columns=weights.columns.to_list(),
                                                          color=tf.colors.GREY,
                                                          apply_to_header_and_index=True)
                                  ]),
                          p.Paragraph(f"  ", **KWARGS_FOOTNOTE)],
                         **KWARGS_TEXT)
    blocks.append(b_weights)

    # 2. weights by ac table
    b_weights2 = p.Block([p.Paragraph(f"Asset class weights for {strategy_weights.index[-1]:'%d%b%Y'}", **KWARGS_TITLE),
                          p.Block(weights_by_ac,
                                  formatters=[
                                      tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                                      tf.FmtReplaceNaN(value=''),
                                      tf.FmtHeatmap(columns=['diff']),  # , max_color=(255,0,255)
                                      tf.FmtAddCellBorder(each=1.0,
                                                          columns=weights_by_ac.columns.to_list(),
                                                          color=tf.colors.GREY,
                                                          apply_to_header_and_index=True)
                                  ]),
                          p.Paragraph(f"  ", **KWARGS_FOOTNOTE)],
                         **KWARGS_TEXT)
    blocks.append(b_weights2)

    # 3. attrib table
    b_attribs = p.Block([p.Paragraph(f"Attributions {strategy_weights.index[-1]:'%d%b%Y'}", **KWARGS_TITLE),
                          p.Block(attribs,
                                  formatters=[
                                      tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                                      tf.FmtReplaceNaN(value=''),
                                      tf.FmtHeatmap(columns=[f"{strategy_name} last", f"{benchmark_name} last"], rows=attribs.index[:-1]),
                                      tf.FmtHeatmap(columns=[f"{strategy_name} 3m", f"{benchmark_name} 3m"], rows=attribs.index[:-1]),
                                      tf.FmtHeatmap(columns=[f"{strategy_name} 12m", f"{benchmark_name} 12m"], rows=attribs.index[:-1]),
                                      tf.FmtAddCellBorder(each=1.0,
                                                          columns=attribs.columns.to_list(),
                                                          color=tf.colors.GREY,
                                                          apply_to_header_and_index=True)
                                  ]),
                          p.Paragraph(f"  ", **KWARGS_FOOTNOTE)],
                         **KWARGS_TEXT)
    blocks.append(b_attribs)

    # 4. attrib by ac table
    b_attribs2 = p.Block([p.Paragraph(f"Attributions by ac {strategy_weights.index[-1]:'%d%b%Y'}", **KWARGS_TITLE),
                          p.Block(attribs_by_ac,
                                  formatters=[
                                      tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                                      tf.FmtReplaceNaN(value=''),
                                      tf.FmtHeatmap(columns=[f"{strategy_name} last", f"{benchmark_name} last"], rows=attribs_by_ac.index[:-1]),
                                      tf.FmtHeatmap(columns=[f"{strategy_name} 3m", f"{benchmark_name} 3m"], rows=attribs_by_ac.index[:-1]),
                                      tf.FmtHeatmap(columns=[f"{strategy_name} 12m", f"{benchmark_name} 12m"], rows=attribs_by_ac.index[:-1]),
                                      tf.FmtAddCellBorder(each=1.0,
                                                          columns=attribs_by_ac.columns.to_list(),
                                                          color=tf.colors.GREY,
                                                          apply_to_header_and_index=True)
                                  ]),
                          p.Paragraph(f"  ", **KWARGS_FOOTNOTE)],
                         **KWARGS_TEXT)
    blocks.append(b_attribs2)

    report = p.VStack(blocks, cascade_cfg=False)
    return report
