"""
All-weather backtest with qis: 25% each SPY / TLT / TIP / GLD,
quarterly rebalancing, 10bp trading costs, NAV starting at 100.

pip install "qis[data]"
"""
import inspect
import sys

import pandas as pd
import yfinance as yf
import qis
from qis import PerfStat

TICKERS = ['SPY', 'TLT', 'TIP', 'GLD']
WEIGHTS = {t: 0.25 for t in TICKERS}
START, END = '2002-07-31', '2026-09-30'
INITIAL_NAV = 100.0
REBALANCING_COSTS = 0.0010  # 10bp per unit of traded notional
TICKER = 'All Weather'


def get_stat(table: pd.DataFrame, stat: PerfStat) -> float:
    """Read one PerfStat column from the qis perf table, tolerant to label conventions."""
    keys = [stat.to_str()] if hasattr(stat, 'to_str') else []
    keys += list(stat.value) if isinstance(stat.value, tuple) else [stat.value]
    keys.append(stat.name)
    for key in keys:
        if isinstance(key, str) and key in table.columns:
            return float(table[key].iloc[0])
    raise KeyError(f"{stat} not in perf table columns: {list(table.columns)}")


# 1. Yahoo prices; auto_adjust=True -> dividend-adjusted (total-return) closes.
#    yfinance's `end` is exclusive, so request one day past END.
prices = yf.download(TICKERS, start=START, end=pd.Timestamp(END) + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)['Close'][TICKERS]
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices = prices.loc[START:END]

# 2. A 4 x 25% portfolio only exists once all four ETFs trade (TIP: Dec-2003, GLD: Nov-2004).
first_common = prices.dropna().index[0]
if first_common > pd.Timestamp(START):
    print(f"note: not all ETFs trade on {START}; backtest starts {first_common:%Y-%m-%d}",
          file=sys.stderr)
prices = prices.loc[first_common:].ffill()

# 3. Backtest: hold units between quarter-end rebalances, 10bp costs on traded notional.
kwargs = dict(prices=prices, weights=WEIGHTS, rebalancing_freq='QE',
              rebalancing_costs=REBALANCING_COSTS, ticker=TICKER)
params = inspect.signature(qis.backtest_model_portfolio).parameters
if 'is_rebalanced_at_first_date' in params:   # invest on day one, not at the first quarter end
    kwargs['is_rebalanced_at_first_date'] = True
if 'is_output_portfolio_data' in params:      # older qis returns a NAV series otherwise
    kwargs['is_output_portfolio_data'] = True
portfolio = qis.backtest_model_portfolio(**kwargs)
nav = portfolio.get_portfolio_nav()
nav = INITIAL_NAV * nav / nav.iloc[0]         # rebase to 100 (stats are scale-invariant)
nav.name = TICKER

# 4. Statistics from the qis risk-adjusted performance table.
#    freq='B': daily returns for vol/Sharpe; no rates_data -> Sharpe with rf = 0.
perf_params = qis.PerfParams(freq='B')
table = qis.compute_ra_perf_table(prices=nav.to_frame(), perf_params=perf_params)

# 5. Save daily NAV and print results.
nav.rename_axis('date').rename('nav').to_frame().to_csv('nav.csv', date_format='%Y-%m-%d')

print(f"first_date: {nav.index[0]:%Y-%m-%d}")
print(f"last_date: {nav.index[-1]:%Y-%m-%d}")
print(f"terminal_nav: {nav.iloc[-1]:.4f}")
print(f"total_return: {get_stat(table, PerfStat.TOTAL_RETURN):.4f}")
print(f"annual_return: {get_stat(table, PerfStat.PA_RETURN):.4f}")
print(f"volatility: {get_stat(table, PerfStat.VOL):.4f}")
print(f"sharpe: {get_stat(table, PerfStat.SHARPE_RF0):.4f}")
print(f"max_drawdown: {get_stat(table, PerfStat.MAX_DD):.4f}")
