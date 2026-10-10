"""Current README reports from the frozen synthetic market universe; no vendor inputs."""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd
import textwrap

import qis
from qis.datasets import generate_synthetic_universe
from tools.docs_analytics.style import gallery_preview


def focus_risk(figure):
    """Reflow six existing risk panels; retain every selected line's values and shading."""
    fragments = ['Independent ME-freq', 'Correlated ME-freq',
                 'Risk Attribution with Correlated', 'Correlated B-freq',
                 'Rolling 36-span beta', 'Portfolio EWM']
    selected = [next(ax for ax in figure.axes if ax.get_title().startswith(fragment))
                for fragment in fragments]
    snapshots = [(line, line.get_ydata().copy()) for ax in selected for line in ax.lines]
    figure.set_layout_engine(None)
    figure.set_size_inches(11, 11)
    for ax in list(figure.axes):
        if ax not in selected:
            figure.delaxes(ax)
    figure._suptitle.set_text('VolParity 63 | selected risk panels | synthetic data')
    figure._suptitle.set_fontsize(13)
    for index, ax in enumerate(selected):
        ax.set_position((0.08 + 0.49 * (index % 2), 0.70 - 0.30 * (index // 2), 0.38, 0.20))
        ax.set_title(textwrap.fill(ax.get_title().replace('\n', ' '), 49), fontsize=9)
        ax.tick_params(labelsize=8)
        legend = ax.get_legend()
        if legend:
            for label in legend.get_texts():
                label.set_fontsize(7)
    for line, values in snapshots:
        np.testing.assert_array_equal(line.get_ydata(), values)


def produce(spec: dict) -> dict:
    """Render eight current API outputs and independently reconcile linked attribution."""
    params = spec['parameters']
    universe = generate_synthetic_universe(
        start=params['start'], end=params['end'], seed=params['seed'],
        apply_quirks=params['apply_quirks'])
    prices = universe.prices[params['assets']]
    returns = qis.to_returns(prices, is_log_returns=True)
    groups = universe.group_data.loc[prices.columns]
    group_order = [group for group in universe.group_order if group in groups.values]
    portfolios = []
    schedules = {}
    for span in params['volatility_spans']:
        _, weights, _ = qis.compute_ra_returns(returns=returns, span=span, vol_target=0.1)
        weights = weights.div(weights.sum(axis=1), axis=0)
        weights = weights.resample(params['rebalancing_freq']).last().dropna()
        weights = weights.loc[weights.index < prices.index[-1]]
        label = f'VolParity {span}'
        schedules[label] = weights
    schedules['EqualWeight'] = pd.DataFrame(
        1.0 / len(prices.columns), index=weights.index, columns=prices.columns)
    for label, schedule in schedules.items():
        np.testing.assert_allclose(schedule.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        portfolio = qis.backtest_model_portfolio(
            prices=prices, weights=schedule, rebalancing_freq=None,
            weight_implementation_lag=params['weight_implementation_lag'],
            rebalancing_costs=params['rebalancing_costs'], ticker=label)
        portfolio.set_group_data(group_data=groups, group_order=group_order)
        portfolios.append(portfolio)
    strategy = portfolios[params['volatility_spans'].index(params['strategy_span'])]
    pair = qis.MultiPortfolioData(
        portfolio_datas=[strategy, portfolios[-1]], benchmark_prices=prices[['SEQ_US']])
    multi = qis.MultiPortfolioData(
        portfolio_datas=portfolios[:-1], benchmark_prices=prices[['SEQ_US']])
    navs = pd.concat([portfolio.nav for portfolio in portfolios], axis=1)
    period = qis.get_time_period(navs)
    gross = qis.backtest_model_portfolio(
        prices=prices, weights=schedules[strategy.ticker], rebalancing_freq=None,
        weight_implementation_lag=params['weight_implementation_lag'], rebalancing_costs=0.0)
    entry = prices.index.searchsorted(schedules[strategy.ticker].index[0])
    entry_date = prices.index[entry + params['weight_implementation_lag']]
    # Before the first target executes, the account holds its initial NAV in cash.
    cash = np.where(gross.nav.index < entry_date, gross.nav.iloc[0], 0.0)
    marked_value = (gross.units * gross.prices).sum(axis=1) + cash
    np.testing.assert_allclose(marked_value, gross.nav, atol=1e-10, rtol=1e-12)
    valuation_error = float((marked_value - gross.nav).abs().max())

    totals, active, allocation, selection, interaction = pair.compute_brinson_attribution(
        time_period=period, is_exclude_interaction_term=True, is_linked=True, is_net=False)
    gross_paths = []
    for portfolio in pair.portfolio_datas:
        pnl, _ = portfolio.get_brinson_inputs(freq=None, is_net=False)
        pnl = pnl.loc[(pnl.index > navs.index[0]) & (pnl.index <= navs.index[-1])]
        gross_paths.append(np.cumprod(1.0 + pnl.sum(axis=1).to_numpy()))
    reference_active = gross_paths[0] - gross_paths[1]
    linked_active = active.cumsum().sum(axis=1).to_numpy()
    np.testing.assert_allclose(linked_active, reference_active, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(totals.loc['Total Sum', 'Total\nActive'], reference_active[-1],
                               atol=1e-10, rtol=1e-10)
    assert not any('Interaction' in column for column in totals.columns)
    assert list(active.columns) == ['Allocation Total', 'Selection Total']

    common = dict(time_period=period, reporting_frequency=params['reporting_frequency'],
                  add_rates_data=False,
                  benchmark_prices=prices[['SEQ_US']], fontsize=6)
    figures = {}
    figures['readme_multi_asset.png'] = qis.factsheet(
        prices, benchmark='SEQ_US', factsheet_name='Synthetic multi-asset universe', **common)[0]
    pages = qis.factsheet(
        strategy, factsheet_name='Volatility parity | synthetic data',
        add_current_position_var_risk_sheet=True, **common)
    assert len(pages) == 2
    figures['readme_strategy.png'], figures['readme_strategy_risk.png'] = pages
    pages = qis.factsheet(
        pair, kind='strategy_benchmark', factsheet_name='VolParity 63 vs EqualWeight | synthetic',
        add_brinson_attribution=True, **common)
    assert len(pages) == 2
    figures['readme_strategy_benchmark.png'], figures['readme_brinson.png'] = pages
    brinson = pages[1]
    effect_axes = [ax for ax in brinson.axes if 'Effects' in ax.get_title()]
    assert len(effect_axes) == 4 and all(ax.patches for ax in effect_axes)
    active_ax = next(ax for ax in effect_axes
                     if ax.get_title() == 'Cumulative Active Attribution Effects')
    for column, line in zip(active.columns, active_ax.lines):
        np.testing.assert_allclose(line.get_ydata(), active[column].cumsum(), atol=1e-12)
    figures['readme_multi_strategy.png'] = qis.factsheet(
        multi, factsheet_name='Volatility span comparison | synthetic data',
        add_group_exposures_and_pnl=False, add_strategy_factsheets=False, **common)[0]

    # The README uses readable excerpts of the reports, retaining the plotted values.
    for name, title, detail in (
        ('multi_asset', 'Synthetic investment universe', 'Correlation of ME-freq'),
        ('strategy', 'Volatility parity | 63-observation estimator', 'Exposures ('),
        ('strategy_benchmark', 'Volatility parity vs equal weights', 'Two-sided Turnover'),
        ('multi_strategy', 'Volatility spans: 21 / 63 / 126 observations', 'Two-sided Turnover'),
    ):
        figure = figures[f'readme_{name}.png']
        gallery_preview(figure, title=title, detail=detail,
                        comparison_labels=[strategy.ticker, 'EqualWeight']
                        if name == 'strategy_benchmark' else None)
        nav_ax = next(ax for ax in figure.axes if ax.get_title(loc='left')
                      == 'Cumulative performance')
        nav_ax.yaxis.set_major_formatter(ScalarFormatter())
        nav_ax.set_title('Price index (initial = 1)' if name == 'multi_asset'
                         else 'NAV (initial = 1)', loc='left')
        if name == 'strategy':
            weights_ax = next(ax for ax in figure.axes
                              if ax.get_title(loc='left') == 'Monthly instrument weights')
            handles, labels = weights_ax.get_legend_handles_labels()
            labels = [label.split(':')[0].split(',')[0].strip() for label in labels]
            legend = weights_ax.legend(
                handles, labels, ncol=3, fontsize=9, loc='upper center',
                bbox_to_anchor=(0.5, -0.19), frameon=True, facecolor='white')
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            assert not legend.get_window_extent(renderer).overlaps(
                figure.texts[-1].get_window_extent(renderer))
    focus_risk(figures['readme_strategy_risk.png'])
    # Wrap current table headings and show small active effects without rounding to zero.
    table = brinson.axes[0].tables[0]
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            text = cell.get_text().get_text()
            for before, after in [('VolParity 63', 'VolParity\n63'),
                                  ('EqualWeight', 'Equal\nWeight'),
                                  ('Weight Ave', 'Weight\nAve'),
                                  ('Return Total', 'Return\nTotal')]:
                text = text.replace(before, after)
            cell.get_text().set_text(text)
        elif column > 0:
            cell.get_text().set_text(f'{totals.iloc[row - 1, column - 1]:.2%}')
    for ax in effect_axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        for label, line in zip(ax.get_legend().get_texts(), ax.lines):
            name = label.get_text().split(' = ')[0]
            label.set_text(f'{name} = {line.get_ydata()[-1]:.2%}')

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    fig.suptitle('VolParity 63 | positions and trading | synthetic data')
    strategy.plot_current_weights(ax=axs[0, 0], fontsize=9)
    strategy.plot_last_weights_change(ax=axs[0, 1], fontsize=9)
    axs[0, 1].set_title(textwrap.fill(axs[0, 1].get_title(), 55), fontsize=9)
    strategy.plot_turnover(ax=axs[1, 0], freq_turnover='ME', turnover_rolling_period=12,
                           is_agg=True, fontsize=9)
    costs = strategy.get_costs(roll_period=None, add_total=False)
    qis.plot_time_series(df=costs.sum(axis=1).cumsum().rename('Costs'), ax=axs[1, 1], fontsize=9,
                         title='Cumulative realised trading costs / NAV (additive)',
                         var_format='{:.2%}')
    figures['readme_strategy_positions.png'] = fig
    perf_params = qis.PerfParams(
        freq='ME', freq_reg='QE',
        rates_data=pd.Series(params['annual_cash_rate'], index=prices.index))
    figures['readme_performance_table.png'], performance = qis.plot_ra_perf_table_benchmark(
        prices=prices, benchmark='SEQ_US', perf_params=perf_params,
        perf_columns=[qis.PerfStat.PA_RETURN, qis.PerfStat.VOL, qis.PerfStat.SHARPE_RF0,
                      qis.PerfStat.SHARPE_EXCESS, qis.PerfStat.MAX_DD,
                      qis.PerfStat.ALPHA_AN, qis.PerfStat.BETA, qis.PerfStat.R2],
        title='Synthetic universe | monthly risk; quarterly regression against SEQ_US', fontsize=10)
    for figure in figures.values():
        figure.canvas.draw()
    return {
        'figures': figures,
        'tables': {'prices': prices, 'portfolio_navs': navs, 'performance': performance,
                   'strategy_target_weights': schedules[strategy.ticker],
                   'strategy_costs': costs, 'brinson_totals': totals, 'brinson_active': active,
                   'brinson_allocation': allocation, 'brinson_selection': selection},
        'checks': {'target_weights_sum_to_one': True, 'zero_cost_nav_reconciles_units': True,
                   'linked_attribution_matches_compounded_returns': True,
                   'interaction_folded_into_selection': True,
                   'brinson_regime_shading_present': True,
                   'brinson_plotted_effects_match_tables': True},
        'parameters': params,
        'presentation': 'Selected current report panels; Brinson table wraps headings and uses '
                        'two-decimal percentages; positions compose PortfolioData plots',
        'summary': {'sample_start': str(prices.index[0].date()),
                    'sample_end': str(prices.index[-1].date()),
                    'zero_cost_units_valuation_max_error': valuation_error,
                    'gross_linked_active_return': float(reference_active[-1]),
                    'linked_attribution_max_error': float(
                        np.max(np.abs(linked_active - reference_active))),
                    'brinson_effect_axes_with_regime_shading': len(effect_axes)},
    }
