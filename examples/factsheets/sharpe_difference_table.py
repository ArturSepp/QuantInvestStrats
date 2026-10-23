"""Generate a paired HAC Sharpe-difference table from MultiPortfolioData, offline.

Run ``python -m examples.factsheets.sharpe_difference_table`` from the repository root
to display the table. For PNG/CSV export using the configured QIS OUTPUT_PATH::

    from examples.factsheets.sharpe_difference_table import Locals, run_local
    table = run_local(Locals.EXPORT)

Pass ``output_path`` to override that directory. Pass ``multi_portfolio_data``,
``base_portfolio_id`` and ``time_period`` to use existing portfolio NAVs.
The first sampled month-end NAV is the return baseline; base IDs are zero-based.

The default portfolios are fixed, monthly-rebalanced allocations on the frozen synthetic
universe. NAVs include 0.10% proportional transaction costs and exclude management and
performance fees. Statistics use monthly simple NAV returns, with annualisation 12.00
inferred by QIS from freq='ME', ddof=0, Bartlett HAC with three monthly lags and a zero
cash hurdle. Omitting n_lags uses the default of 3; explicit n_lags=None opts into
QIS's automatic lag rule. No allocation is fitted to this sample.
The figure, console and CSV show two decimals; the returned DataFrame retains precision.
"""

from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

import qis
from qis import local_path
from qis.datasets.synthetic import generate_synthetic_prices


def build_example_portfolios() -> qis.MultiPortfolioData:
    """Backtest four illustrative allocations on the same synthetic daily price panel."""
    assets = ['SEQ_US', 'SBD_TSY', 'SCM_GLD']
    prices = generate_synthetic_prices(
        start='2004-12-31', end='2025-12-31', apply_quirks=False,
    )[assets]
    allocations = {
        'Balanced 60/40': [0.60, 0.40, 0.00],
        'Equity 80/20': [0.80, 0.20, 0.00],
        'Defensive 40/60': [0.40, 0.60, 0.00],
        'Equity/bonds/gold 50/40/10': [0.50, 0.40, 0.10],
    }
    portfolios = [
        qis.backtest_model_portfolio(
            prices=prices, weights=dict(zip(assets, weights)),
            rebalancing_freq='ME', rebalancing_costs=0.0010, ticker=name,
        )
        for name, weights in allocations.items()
    ]
    return qis.MultiPortfolioData(portfolio_datas=portfolios)


class Locals(Enum):
    PLOT = 1
    EXPORT = 2


def run_local(
        local: Locals,
        output_path: Optional[str] = None,
        multi_portfolio_data: Optional[qis.MultiPortfolioData] = None,
        base_portfolio_id: int = 0,
        time_period: Optional[qis.TimePeriod] = None,
) -> pd.DataFrame:
    """Display or export monthly Sharpe comparisons against a selected base portfolio.

    Args:
        local: PLOT displays the table; EXPORT writes a PNG and a formatted CSV.
        output_path: Explicit output directory. With EXPORT, None uses QIS OUTPUT_PATH.
            Supplying a directory with PLOT also selects export instead of display.
        multi_portfolio_data: Existing portfolio NAVs; None builds the synthetic example.
            Existing NAVs retain their own costs and fee basis, with no additional deductions.
        base_portfolio_id: Zero-based base portfolio position, omitted from table rows.
        time_period: Inclusive NAV window; None uses the full supplied history. Include
            the previous month-end NAV to retain the desired first monthly return.

    Returns:
        Numeric comparison DataFrame with full precision and sampling metadata in attrs.
        The displayed and exported table uses two decimals for floats and integer counts.
    """
    if local not in (Locals.PLOT, Locals.EXPORT):
        raise ValueError(local)
    if multi_portfolio_data is None:
        multi_portfolio_data = build_example_portfolios()
    if time_period is None:
        time_period = qis.get_time_period(multi_portfolio_data.get_navs())

    # Sample month-end NAVs; omitting af lets QIS infer 12.00 periods per year.
    # Include the previous month-end NAV in time_period to retain the first return.
    fig, table = qis.plot_sharpe_difference_table(
        multi_portfolio_data=multi_portfolio_data,
        base_portfolio_id=base_portfolio_id,
        time_period=time_period,
        freq='ME',
        first_column_width=4.0,
    )
    fig.set_size_inches(16.0, 0.5 * (len(table) + 1) + 0.8)
    fig.tight_layout(pad=1.2)
    display = qis.df_to_str(
        table, var_format='{:0.2f}',
        var_formats={'Observations': '{:.0f}', 'HAC lags': '{:.0f}'},
    )
    print(f"Base: {table.attrs['base_portfolio']}; portfolio minus base; "
          f"{table.attrs['sample_start']:%Y-%m-%d} to {table.attrs['sample_end']:%Y-%m-%d}")
    print(f"Monthly simple NAV returns; inferred annualisation {table.attrs['af']:0.2f}; "
          "no additional cash or fee adjustment.")
    print(display.to_string())
    try:
        if local == Locals.EXPORT or output_path is not None:
            output = local_path.get_output_path(output_path)
            qis.save_fig(fig, file_name='sharpe_difference_table', local_path=output)
            qis.save_df_to_csv(display, file_name='sharpe_difference_table', local_path=output)
            print(f'PNG and CSV saved to {output}')
        else:
            plt.show()
    finally:
        plt.close(fig)
    return table


if __name__ == '__main__':
    run_local(local=Locals.PLOT)
