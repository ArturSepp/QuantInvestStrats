"""
Backtest two leveraged-ETF combinations and benchmark vs SPY:
  • 50/50 SSO/IEF (using 2x equity ETF)
  • 100/50 SPY/IEF financed at 3m UST + 100bp

Renders a multi-asset factsheet via ``qis.generate_multi_asset_factsheet``.
"""

# packages
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis
from enum import Enum
from qis import local_path

class Locals(Enum):
    FACTSHEET = 1


def run_local(local: Locals, output_path: str = None):
    """Run the example using YAML defaults or an explicit output directory."""
    if local != Locals.FACTSHEET:
        raise ValueError(local)
    output_path = local_path.get_output_path(output_path)

    # select tickers
    benchmark = 'SPY'
    tickers = [benchmark, 'SSO', 'IEF']

    # fetch prices
    prices = yf.download(
        tickers=tickers, start='2003-12-31', end=None, ignore_tz=True, auto_adjust=True,
        threads=False,
    )['Close'][tickers]
    prices = prices.asfreq('B', method='ffill').dropna()  # make B frequency
    if prices.empty:
        raise ValueError('Yahoo returned no overlapping SPY/SSO/IEF history; retry the download.')

    rebalancing_freq = 'B'  # each business day
    rebalancing_costs = 0.0010  # 10bp for rebalancing

    # 50/50 SSO/IEF
    unleveraged_portfolio = qis.backtest_model_portfolio(prices=prices[['SSO', 'IEF']],
                                                         weights={'SSO': 0.5, 'IEF': 0.5},
                                                         rebalancing_freq=rebalancing_freq,
                                                         rebalancing_costs=rebalancing_costs,
                                                         ticker='50/50 SSO/IEF').get_portfolio_nav()

    # leveraged is funded at 100bp + 3m UST
    funding_rate = (
        0.01
        + yf.download('^IRX', start='1999-12-31', end=None,
                      ignore_tz=True, auto_adjust=True)['Close']['^IRX'].dropna()
        / 100.0
    )
    leveraged_portfolio = qis.backtest_model_portfolio(prices=prices[['SPY', 'IEF']],
                                                       weights={'SPY': 1.0, 'IEF': 0.5},
                                                       rebalancing_freq=rebalancing_freq,
                                                       rebalancing_costs=rebalancing_costs,
                                                       funding_rate=funding_rate,
                                                       ticker='100/50 SPY/IEF').get_portfolio_nav()


    prices = pd.concat([prices, unleveraged_portfolio, leveraged_portfolio], axis=1)

    # generate report since SSO launch
    time_period = qis.TimePeriod('21Jun2006', '01Sep2023')
    fig = qis.generate_multi_asset_factsheet(prices=prices,
                                             benchmark=benchmark,
                                             time_period=time_period,
                                             **qis.fetch_default_report_kwargs(time_period=time_period))

    qis.save_fig(fig=fig, file_name="leveraged_fund_analysis", local_path=output_path)

    qis.save_figs_to_pdf(figs=[fig],
                         file_name="leveraged_fund_analysis", orientation='landscape',
                         local_path=output_path)

    plt.show()
    return funding_rate


if __name__ == '__main__':
    run_local(local=Locals.FACTSHEET)
