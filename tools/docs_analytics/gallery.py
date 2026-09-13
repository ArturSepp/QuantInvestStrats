"""Four replacement gallery exhibits using the frozen offline market-panel fixture."""

import numpy as np
import pandas as pd

import qis
from qis.datasets import generate_synthetic_universe
from tools.docs_analytics.style import gallery_preview


def check_units_identity(portfolio) -> float:
    """Check a fully invested, zero-cost NAV by independently valuing the held units."""
    marked_value = (portfolio.units * portfolio.prices).sum(axis=1)
    np.testing.assert_allclose(marked_value, portfolio.nav, atol=1e-10, rtol=1e-12)
    return float((marked_value - portfolio.nav).abs().max())


def produce(spec: dict, *, full_reports: bool = False) -> dict:
    """Create focused previews, or complete reports, from shared portfolio objects."""
    params = spec['parameters']
    universe = generate_synthetic_universe(
        start=params['start'], end=params['end'], seed=params['seed'],
        apply_quirks=params['apply_quirks'],
    )
    prices = universe.prices.loc[:, params['assets']]
    benchmark = universe.benchmark_prices
    if (prices.isna().any().any() or not np.isfinite(prices).all().all()
            or (prices <= 0).any().any()):
        raise ValueError('Gallery instruments must have complete positive prices; no filling')
    portfolios = []
    for label, weights in params['weights'].items():
        np.testing.assert_allclose(sum(weights), 1.0, atol=1e-12, rtol=0)
        schedule = qis.generate_static_weights_schedule(
            prices=prices, weights=weights, rebalancing_freq=params['rebalancing_freq'])
        portfolio = qis.backtest_model_portfolio(
            prices=prices, weights=schedule, rebalancing_freq=None,
            rebalancing_costs=params['rebalancing_costs'],
            weight_implementation_lag=params['weight_implementation_lag'], ticker=label,
        )
        portfolio.set_group_data(
            group_data=universe.group_data.loc[prices.columns],
            group_order=[group for group in universe.group_order
                         if group in universe.group_data.loc[prices.columns].values],
        )
        portfolios.append(portfolio)
    navs = pd.concat([portfolio.get_portfolio_nav() for portfolio in portfolios], axis=1)
    if not np.isfinite(navs).all().all() or (navs <= 0).any().any():
        raise ValueError('Portfolio NAVs must be finite and positive')

    # Independent valuation checks the existing backtest; it does not replace its algorithm.
    gross = qis.backtest_model_portfolio(
        prices=prices, weights=portfolios[0].input_weights, rebalancing_freq=None,
        rebalancing_costs=0.0, weight_implementation_lag=params['weight_implementation_lag'],
        ticker='Zero-cost valuation check',
    )
    valuation_error = check_units_identity(gross)
    period = qis.get_time_period(df=prices)
    preset = qis.fetch_default_report_kwargs(
        time_period=period, reporting_frequency=qis.ReportingFrequency.MONTHLY,
        add_rates_data=False,
    )
    perf_params = preset['perf_params']
    if (params['reporting_frequency'] != 'monthly' or perf_params.freq != 'ME'
            or perf_params.return_type.name != 'LOG'
            or perf_params.sharpe_convention.name != 'PA'):
        raise ValueError('Preview labels require monthly LOG volatility and PA Sharpe')
    multi = qis.MultiPortfolioData(portfolio_datas=portfolios, benchmark_prices=benchmark)
    pair = qis.MultiPortfolioData(portfolio_datas=portfolios[:2], benchmark_prices=benchmark)
    common = {
        'benchmark_prices': benchmark, 'reporting_frequency': params['reporting_frequency'],
        'time_period': period, 'add_rates_data': False,
    }
    figures = {}

    def one(filename, data, title, **kwargs):
        pages = qis.factsheet(data, factsheet_name=title, **common, **kwargs)
        if len(pages) != 1:
            raise ValueError(f'{filename} must have exactly one page, received {len(pages)}')
        figures[filename] = pages[0]

    one('multi_asset.png', prices, 'Synthetic investment universe',
        benchmark=benchmark.columns[0])
    one('strategy.png', portfolios[0], 'Balanced allocation | synthetic data')
    one('strategy_vs_benchmark.png', pair, 'Balanced vs Defensive | synthetic data',
        kind='strategy_benchmark', add_brinson_attribution=False)
    one('multi_strategy.png', multi, 'Three allocations | synthetic data',
        add_group_exposures_and_pnl=False, add_strategy_factsheets=False)
    if not full_reports:
        layouts = {
            'multi_asset.png': ('Synthetic investment universe', 'Correlation of ME-freq'),
            'strategy.png': ('Balanced allocation', 'Exposures ('),
            'strategy_vs_benchmark.png': ('Balanced vs Defensive', 'Two-sided Turnover'),
            'multi_strategy.png': ('Three portfolio allocations', 'Correlation of ME-freq'),
        }
        for filename, (title, detail) in layouts.items():
            gallery_preview(
                figures[filename], title=title, detail=detail,
                comparison_labels=list(params['weights'])[:2]
                if filename == 'strategy_vs_benchmark.png' else None,
            )
    tables = {
        'prices': prices,
        'benchmark': benchmark,
        'portfolio_navs': navs,
        'performance': qis.compute_ra_perf_table(
            prices=pd.concat([navs, benchmark], axis=1), perf_params=perf_params),
        'balanced_weights': portfolios[0].get_weights(freq=None),
        'balanced_turnover': portfolios[0].get_turnover(roll_period=None),
        'balanced_costs': portfolios[0].get_costs(roll_period=None),
    }
    return {
        'figures': figures, 'tables': tables,
        'checks': {
            'positive_complete_prices': True, 'finite_positive_navs': True,
            'weights_sum_to_one': True, 'zero_cost_nav_reconciles_units': True,
        },
        'parameters': params,
        'presentation': 'complete reports' if full_reports else 'four focused report panels',
        'reporting_preset': {
            key: value for key, value in preset.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        'performance_conventions': {
            'volatility_returns': perf_params.return_type.name,
            'sharpe': perf_params.sharpe_convention.name,
            'freq': perf_params.freq, 'freq_vol': perf_params.freq_vol,
            'freq_drawdown': perf_params.freq_drawdown, 'freq_reg': perf_params.freq_reg,
            'annualisation': 'qis frequency convention: ME = 12 periods/year',
        },
        'summary': {
            'observations': len(prices),
            'sample_start': str(prices.index[0].date()),
            'sample_end': str(prices.index[-1].date()),
            'terminal_navs': {str(k): float(v) for k, v in navs.iloc[-1].items()},
            'zero_cost_units_valuation_max_error': valuation_error,
        },
    }
