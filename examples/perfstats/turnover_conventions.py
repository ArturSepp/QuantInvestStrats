"""Compare all turnover conventions for funded and leveraged Yahoo portfolios.

The example applies a simple six-month relative-strength tilt between SPY and TLT and rebalances
monthly. The first portfolio is 100% funded; the second applies the same weights at 2x gross
exposure and borrows 100% of NAV. Financing costs and trading costs are omitted so the turnover
definitions are the only difference between the cases.
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
    """Create monthly NAV, post-trade units, and tactical target weights.

    Positions are rebalanced at each observed month end. Between month ends, the previous units
    and cash balance determine the next pre-trade NAV. The 2x portfolio borrows one NAV of cash;
    its financing return is deliberately zero in this turnover-only illustration.
    The asset with the stronger trailing six-month return receives 70% and the other receives
    30%. The first six observations use 60/40. Scaling by ``gross_leverage`` keeps the same signal
    while changing only the capital leverage.
    """
    trailing_returns = prices.pct_change(6, fill_method=None)
    has_signal = trailing_returns.notna().all(axis=1)
    spy_is_stronger = trailing_returns['SPY'].ge(trailing_returns['TLT'])
    spy_weight = pd.Series(0.60, index=prices.index)
    spy_weight.loc[has_signal & spy_is_stronger] = 0.70
    spy_weight.loc[has_signal & ~spy_is_stronger] = 0.30
    target_weights = gross_leverage * pd.DataFrame(
        {'SPY': spy_weight, 'TLT': 1.0 - spy_weight},
        index=prices.index,
    )
    units = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    nav = pd.Series(index=prices.index, dtype=float, name=f'{gross_leverage:.0f}x NAV')

    cash = INITIAL_NAV
    previous_units = pd.Series(0.0, index=prices.columns)
    for date, current_prices in prices.iterrows():
        target = target_weights.loc[date]
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
    # The paper's sqrt(a) * sigma[t] input is annualized here with a=12.
    annualized_vols = (
        prices.pct_change(fill_method=None).ewm(span=12, adjust=False).std()
        * (12.0 ** 0.5)
    )
    results = {}
    for computation_type in qis.TurnoverComputationType:
        by_instrument = qis.compute_turnover(
            computation_type=computation_type,
            units=units,
            unit_notional=prices,
            nav=nav,
            input_weights=target_weights,
            vols=annualized_vols,
        )
        results[computation_type.value] = by_instrument.sum(axis=1, min_count=1)
    return pd.DataFrame(results)


def main() -> None:
    prices = fetch_monthly_prices()
    comparisons: Dict[str, pd.DataFrame] = {
        '100% funded tactical SPY/TLT': compute_convention_comparison(
            prices=prices,
            gross_leverage=1.0,
        ),
        '2x leveraged tactical SPY/TLT': compute_convention_comparison(
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
