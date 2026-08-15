"""Run and report a long/flat momentum-event strategy on free SPY intraday bars.

Install the data extra with ``pip install "qis[data]"``. The strategy observes a 5-minute close,
submits a signed unit order, and receives a full fill only at the next observation. Its
instrument contribution returns are compounded back to timestamped NAV. Business-day closing
NAV and position size are plotted with the QIS time-axis formatter and passed to a multi-asset
factsheet.

Change ``INTERVAL`` to ``"1m"`` and ``PERIOD`` to ``"5d"`` for a denser recent sample supported
by yfinance.
"""

# packages
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# qis
import qis
from qis.discrete_portfolio import (
    DiscreteBacktestResult,
    DiscretePortfolioState,
    FullFillExecution,
    Order,
    backtest_discrete_portfolio,
)
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet

TICKER = 'SPY'
INTERVAL = '5m'
PERIOD = '1mo'
FAST_WINDOW = 12
SLOW_WINDOW = 36
INITIAL_CASH = 100_000.0


@dataclass
class MovingAverageMomentumStrategy:
    """Long/flat momentum strategy retaining only its observed price history.

    Attributes:
        ticker: Instrument traded by the strategy.
        fast_window: Number of observed bars in the fast moving average.
        slow_window: Number of observed bars in the slow moving average.
        observed_prices: Prices received one at a time through ``on_bar``.
        previous_signal: Most recent fully formed positive/non-positive momentum regime.
        order_number: Monotonic identifier used in the order ledger.
    """

    ticker: str
    fast_window: int = FAST_WINDOW
    slow_window: int = SLOW_WINDOW
    observed_prices: List[float] = field(default_factory=list)
    previous_signal: Optional[bool] = None
    order_number: int = 0

    def on_bar(
            self,
            timestamp: pd.Timestamp,
            prices: pd.Series,
            state: DiscretePortfolioState,
    ) -> List[Order]:
        """Submit one order only when the momentum regime changes."""
        price = float(prices[self.ticker])
        if not np.isfinite(price) or price <= 0.0:
            return []
        self.observed_prices.append(price)
        if len(self.observed_prices) < self.slow_window:
            return []

        fast_average = float(np.mean(self.observed_prices[-self.fast_window:]))
        slow_average = float(np.mean(self.observed_prices[-self.slow_window:]))
        momentum_score = fast_average / slow_average - 1.0
        signal = momentum_score > 0.0
        if self.previous_signal is not None and signal == self.previous_signal:
            return []
        self.previous_signal = signal

        target_units = float(np.floor(state.nav / price)) if signal else 0.0
        quantity = float(target_units - state.units[self.ticker])
        if quantity == 0.0:
            return []
        self.order_number += 1
        reason = 'positive_momentum' if signal else 'non_positive_momentum'
        return [
            Order(
                order_id=f'momentum-{self.order_number:04d}',
                decision_time=timestamp,
                ticker=self.ticker,
                quantity=quantity,
                reason=reason,
            )
        ]


def download_intraday_close(
        ticker: str = TICKER,
        interval: str = INTERVAL,
        period: str = PERIOD,
) -> pd.DataFrame:
    """Download and normalize one adjusted intraday close series from yfinance."""
    downloaded = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        raise ValueError(f'yfinance returned no {interval} bars for {ticker}')
    close = downloaded['Close']
    if isinstance(close, pd.DataFrame):
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    close = close.rename(ticker).dropna()
    if close.index.tz is not None:
        # Preserve exchange-local wall-clock labels while removing timezone metadata for plots.
        close.index = close.index.tz_localize(None)
    prices = close.to_frame()
    if len(prices.index) < SLOW_WINDOW + 1:
        raise ValueError(f'need at least {SLOW_WINDOW + 1} bars, received {len(prices.index)}')
    return prices


def to_timestamped_navs(
        result: DiscreteBacktestResult,
        prices: pd.DataFrame,
        ticker: str = TICKER,
) -> pd.DataFrame:
    """Reconstruct strategy NAV from contribution P&L and add a buy-and-hold path.

    Args:
        result: Discrete result enriched with ``PortfolioData``.
        prices: Intraday price panel used by the replay.
        ticker: Price column used for the buy-and-hold comparison.

    Returns:
        Intraday timestamped strategy and buy-and-hold NAV paths.

    Raises:
        ValueError: If prices are missing or reconstructed NAV does not reconcile.
        RuntimeError: If the reporting adapter was not applied to the result.
    """
    portfolio_data = result.portfolio_data
    if portfolio_data is None:
        raise RuntimeError('reporting adapter did not produce PortfolioData')
    if ticker not in prices.columns:
        raise ValueError(f'{ticker!r} is not present in prices')

    strategy_returns = portfolio_data.instrument_pnl.sum(axis=1)
    initial_nav = float(portfolio_data.nav.iloc[0])
    strategy_nav = initial_nav * (1.0 + strategy_returns).cumprod()
    if not np.allclose(
            strategy_nav, portfolio_data.nav, rtol=1e-11, atol=1e-8, equal_nan=True,
    ):
        raise ValueError('NAV reconstructed from instrument P&L does not reconcile')
    strategy_nav = strategy_nav.rename(portfolio_data.nav.name or 'intraday momentum')

    benchmark_prices = prices[ticker].reindex(strategy_nav.index).astype(float)
    if not np.all(np.isfinite(benchmark_prices)) or not np.all(benchmark_prices > 0.0):
        raise ValueError('buy-and-hold prices must be finite and positive')
    benchmark_nav = initial_nav * benchmark_prices.divide(float(benchmark_prices.iloc[0]))
    benchmark_nav = benchmark_nav.rename(f'{ticker} buy-and-hold')
    return pd.concat([strategy_nav, benchmark_nav], axis=1)


def to_daily_reporting_navs(timestamped_navs: pd.DataFrame) -> pd.DataFrame:
    """Convert intraday NAVs to aligned business-day closing marks for reporting."""
    daily_navs = timestamped_navs.resample('B').last().dropna(how='any')
    if len(daily_navs.index) < 3:
        raise ValueError('at least three daily NAV observations are required for the factsheet')
    return daily_navs


def plot_daily_navs_and_position(
        result: DiscreteBacktestResult,
        timestamped_navs: pd.DataFrame,
        interval: str = INTERVAL,
        ticker: str = TICKER,
) -> plt.Figure:
    """Plot business-day closing NAVs and signed position units on aligned panels."""
    portfolio_data = result.portfolio_data
    if portfolio_data is None:
        raise RuntimeError('reporting adapter did not produce PortfolioData')
    if ticker not in portfolio_data.units.columns:
        raise ValueError(f'{ticker!r} is not present in portfolio units')

    daily_navs = to_daily_reporting_navs(timestamped_navs)
    daily_position = portfolio_data.units[ticker].resample('B').last()
    daily_position = daily_position.reindex(daily_navs.index).rename(f'{ticker} position')
    if daily_position.isna().any():
        raise ValueError('daily position size does not align with reporting NAVs')

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(10.0, 8.0),
        gridspec_kw={'height_ratios': [2.0, 1.0]},
    )
    qis.plot_time_series(
        df=daily_navs,
        x_date_freq='B',
        date_format='%d-%b',
        legend_stats=qis.LegendStats.NONE,
        title=f'{ticker} momentum-event strategy: B reporting ({interval} execution)',
        ylabel='NAV',
        ax=axes[0],
    )
    qis.plot_time_series(
        df=daily_position,
        x_date_freq='B',
        date_format='%d-%b',
        legend_stats=qis.LegendStats.NONE,
        title='End-of-day position size',
        ylabel='Signed units',
        ax=axes[1],
    )
    axes[0].tick_params(axis='x', labelbottom=False)
    figure.align_ylabels(axes)
    figure.tight_layout()
    return figure


def generate_nav_factsheet(timestamped_navs: pd.DataFrame) -> plt.Figure:
    """Generate a daily multi-asset factsheet for strategy and buy-and-hold NAVs."""
    daily_navs = to_daily_reporting_navs(timestamped_navs)
    return generate_multi_asset_factsheet(
        prices=daily_navs,
        benchmark=f'{TICKER} buy-and-hold',
        perf_params=qis.PerfParams(freq='B'),
        regime_classifier=qis.BenchmarkReturnsQuantilesRegime(freq='B'),
        heatmap_freq='W-FRI',
        factsheet_name=f'{TICKER} intraday momentum-event strategy',
        min_trailing_obs=5,
        x_date_freq='B',
        date_format='%d-%b',
    )


def run_example(
        interval: str = INTERVAL,
        period: str = PERIOD,
        show: bool = True,
) -> DiscreteBacktestResult:
    """Download bars, run the strategy, and create the NAV plot and factsheet."""
    prices = download_intraday_close(interval=interval, period=period)
    result = backtest_discrete_portfolio(
        prices=prices,
        strategy=MovingAverageMomentumStrategy(ticker=TICKER),
        initial_cash=INITIAL_CASH,
        execution_model=FullFillExecution(transaction_cost_rate=0.0001),
        ticker='SPY intraday momentum',
    )
    portfolio_data = result.portfolio_data
    if portfolio_data is None:
        raise RuntimeError('reporting adapter did not produce PortfolioData')
    reconciliation = (
        portfolio_data.instrument_pnl.sum(axis=1)
        - portfolio_data.nav.pct_change(fill_method=None).fillna(0.0)
    )
    print(result.trade_ledger.to_string(index=False))
    print(f'max accounting reconciliation error: {reconciliation.abs().max():.3e}')

    timestamped_navs = to_timestamped_navs(result=result, prices=prices)
    plot_daily_navs_and_position(
        result=result,
        timestamped_navs=timestamped_navs,
        interval=interval,
    )
    generate_nav_factsheet(timestamped_navs=timestamped_navs)
    if show:
        plt.show()
    return result


if __name__ == '__main__':
    run_example()
