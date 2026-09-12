"""Compare all turnover conventions for funded and leveraged Yahoo portfolios.

The example holds a fixed 60/40 allocation to SPY and TLT and rebalances monthly. The first
portfolio is 100% funded (60% + 40%); the second has 2x gross exposure (120% + 80%) and an
implicit cash borrowing of 100% of NAV. Financing costs and trading costs are omitted so the
turnover denominators are the only difference between the two cases.
"""
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

import qis


TICKERS = ['SPY', 'TLT']
START_DATE = '2015-12-31'
INITIAL_NAV = 1_000_000.0


def fetch_monthly_prices() -> pd.DataFrame:
    """Download adjusted Yahoo prices and sample them at month ends."""
    data = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        ignore_tz=True,
    )
    prices = data['Close']
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.reindex(columns=TICKERS).dropna()
    if prices.empty:
        raise ValueError('Yahoo returned no joint SPY/TLT price history')
    return prices.resample('ME').last().dropna()


def build_monthly_rebalanced_portfolio(
        prices: pd.DataFrame,
        gross_leverage: float,
        ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Create monthly NAV, post-trade units, and fixed target weights.

    Positions are rebalanced at each observed month end. Between month ends, the previous units
    and cash balance determine the next pre-trade NAV. The 2x portfolio borrows one NAV of cash;
    its financing return is deliberately zero in this turnover-only illustration.
    """
    target = gross_leverage * pd.Series([0.60, 0.40], index=prices.columns)
    target_weights = pd.DataFrame(
        [target.to_numpy()] * len(prices.index),
        index=prices.index,
        columns=prices.columns,
    )
    units = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    nav = pd.Series(index=prices.index, dtype=float, name=f'{gross_leverage:.0f}x NAV')

    cash = INITIAL_NAV
    previous_units = pd.Series(0.0, index=prices.columns)
    for date, current_prices in prices.iterrows():
        current_nav = cash + previous_units.dot(current_prices)
        current_units = current_nav * target.divide(current_prices)

        nav.loc[date] = current_nav
        units.loc[date] = current_units
        cash = current_nav - current_units.dot(current_prices)
        previous_units = current_units

    return nav, units, target_weights


def compute_convention_comparison(
        prices: pd.DataFrame,
        gross_leverage: float,
        ) -> pd.DataFrame:
    """Compute total monthly two-sided turnover under every convention."""
    nav, units, target_weights = build_monthly_rebalanced_portfolio(
        prices=prices,
        gross_leverage=gross_leverage,
    )
    results = {}
    for computation_type in qis.TurnoverComputationType:
        by_instrument = qis.compute_turnover(
            computation_type=computation_type,
            units=units,
            unit_notional=prices,
            nav=nav,
            input_weights=target_weights,
        )
        results[computation_type.value] = by_instrument.sum(axis=1, min_count=1)
    return pd.DataFrame(results)


def main() -> None:
    prices = fetch_monthly_prices()
    comparisons: Dict[str, pd.DataFrame] = {
        '100% funded: 60% SPY + 40% TLT': compute_convention_comparison(
            prices=prices,
            gross_leverage=1.0,
        ),
        '2x leverage: 120% SPY + 80% TLT': compute_convention_comparison(
            prices=prices,
            gross_leverage=2.0,
        ),
    }

    annualized_average = pd.DataFrame(
        {name: 12.0 * turnover.mean() for name, turnover in comparisons.items()}
    ).T
    print('Annualized average two-sided turnover:')
    print(annualized_average.to_string(float_format=lambda value: f'{value:.2%}'))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, tight_layout=True)
    fig.suptitle('Turnover conventions under different leverage')
    for ax, (name, turnover) in zip(axes, comparisons.items()):
        qis.plot_time_series(
            df=turnover.rolling(12).sum(),
            title=f'12-month rolling Two-sided Turnover — {name}',
            var_format='{:.1%}',
            y_limits=(0.0, None),
            ax=ax,
        )
    plt.show()


if __name__ == '__main__':
    main()
