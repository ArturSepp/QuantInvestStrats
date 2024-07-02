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
from scipy import stats as stats
from typing import Optional, Dict

# qis
import qis as qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantileRegimeSpecs
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.portfolio.reports.config import PERF_PARAMS, REGIME_PARAMS
from qis.portfolio.reports.config import KWARGS_SUPTITLE, KWARGS_TITLE, KWARGS_TEXT, KWARGS_FOOTNOTE


def generate_strategy_benchmark_factsheet_with_pyblogs(multi_portfolio_data: MultiPortfolioData,
                                                       alphas: Optional[Dict[str, pd.DataFrame]] = None,  # to show signals
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

    if alphas is not None:
        alphas_last = []
        alpha_scores = []
        for key, alpha in alphas.items():
            last_alpha = alpha.iloc[-1, :].rename(f"{key}")
            last_score = qis.compute_last_score(df=alpha).rename(f"{key}-score%")
            alpha_scores.append(last_score)
            alphas_last.append(last_alpha)
        alphas_df = pd.concat([pd.concat(alphas_last, axis=1), pd.concat(alpha_scores, axis=1)], axis=1)

        #weights1 = pd.concat([weights, pd.concat(alphas_last, axis=1)], axis=1)
        #weights_by_ac = qis.agg_df_by_groups(df=weights1.T, group_data=asset_class_data, group_order=group_order).T

    else:
        alphas_df = None

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

    # tre
    prices = multi_portfolio_data.portfolio_datas[strategy_idx].prices
    prices = prices.reindex(columns=strategy_weights.columns)
    covar = qis.compute_ewm_covar(a=qis.to_returns(prices=prices, freq=weight_freq, is_log_returns=True).to_numpy(),
                                  span=12)
    covar *= 12
    inst_vol = np.sqrt(np.diag(covar))
    tre_vol = np.abs(delta_w.T) @ np.diag(inst_vol)
    tre_contrib = delta_w.T @ covar
    tre_table = last_alpha.copy().to_frame('alpha')
    tre_table['weight diff'] = delta_w
    tre_table['tre (vol)'] = tre_vol
    # tre_table['tre contrib'] = tre_contrib
    # tre_table['tre_rel %'] = tre_contrib / (delta_w.T @ covar @ delta_w)
    tre_table['IR per vol bp'] = delta_w*last_alpha / (inst_vol)
    tre_table.loc['Total', :] = np.nansum(tre_table, axis=0)
    tre_table.loc['Total', 'alpha'] = delta_w.T @ last_alpha
    tre_table.loc['Total', 'tre (vol)'] = np.sqrt(delta_w.T @ covar @ delta_w)

    print(tre_table)


    # 1. weights table
    b_weights1 = p.Block([p.Paragraph(f"Instrument weights for {strategy_weights.index[-1]:'%d%b%Y'}", **KWARGS_TITLE),
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
    # 1a alphas_df table
    if alphas_df is not None:
        # heatmap for scores
        formatters = [tf.FmtHeatmap(columns=[alphas_df.columns[idx]]) for idx in range(2*len(alphas.keys()))]
        b_weights2 = p.Block([p.Paragraph(f"Instrument alphas and alpha time series scores for {strategy_weights.index[-1]:'%d%b%Y'}", **KWARGS_TITLE),
                              p.Block(alphas_df,
                                      formatters=formatters + [
                                          tf.FmtPercent(n_decimals=0, apply_to_header_and_index=False),
                                          tf.FmtReplaceNaN(value=''),
                                          tf.FmtAddCellBorder(each=1.0,
                                                              columns=alphas_df.columns.to_list(),
                                                              color=tf.colors.GREY,
                                                              apply_to_header_and_index=True)
                                      ]),
                              p.Paragraph(f"  ", **KWARGS_FOOTNOTE)],
                             **KWARGS_TEXT)
        b_weights = p.HStack([b_weights1, b_weights2])
    else:
        b_weights = b_weights1
    blocks.append(b_weights)

    # 2. tre_table
    b_tre_table = p.Block([p.Paragraph(f"Instrument alphas and tracking error for {strategy_weights.index[-1]:'%d%b%Y'}", **KWARGS_TITLE),
                          p.Block(tre_table,
                                  formatters=[
                                      tf.FmtPercent(n_decimals=2, apply_to_header_and_index=False),
                                      tf.FmtReplaceNaN(value=''),
                                      tf.FmtHeatmap(columns=['diff']),  # , max_color=(255,0,255)
                                      tf.FmtAddCellBorder(each=1.0,
                                                          columns=tre_table.columns.to_list(),
                                                          color=tf.colors.GREY,
                                                          apply_to_header_and_index=True)
                                  ]),
                          p.Paragraph(f"  ", **KWARGS_FOOTNOTE)],
                         **KWARGS_TEXT)
    blocks.append(b_tre_table)

    # 3. weights by ac table
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
