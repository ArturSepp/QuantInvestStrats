"""
a static allocation over a universe whose instruments do not all start on the same date.

``backtest_model_portfolio`` takes weights as given and never modifies them. An instrument with
no price on a rebalancing date cannot be traded, so its weight stays in the cash balance and the
portfolio runs under-invested until the instrument starts. That is the correct contract and it is
almost never the allocation the caller meant: a 50/25/25 book whose third instrument starts five
years into the panel is a 50/25 book with 25% in cash for those five years.

``qis.generate_static_weights_schedule`` is the step that fixes it, before the backtest rather
than inside it. It reads which instruments are priced on each rebalancing date and reallocates
the missing weight over them, preserving the total exposure of the specification rather than
forcing the row to one, so a book that is 90% invested by design stays 90% invested.

The panel is the seeded synthetic universe, whose ``SEQ_EM`` leg starts five years after the
other two. Run it with ``python -m qis.examples.portfolios.static_weight_with_missing_prices``.
No network, no data file.
"""
# packages
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
# qis / project
import qis as qis
from qis.datasets.synthetic import generate_synthetic_prices

TICKERS = ['SEQ_US', 'SBD_TSY', 'SEQ_EM']  # SEQ_EM starts five years into the panel
STATIC_WEIGHTS = {'SEQ_US': 0.5, 'SBD_TSY': 0.25, 'SEQ_EM': 0.25}
REBALANCING_FREQ = 'QE'
REBALANCING_COSTS = 0.0010  # fractional units: 10 bp


class LocalTest(Enum):
    CASH_RESIDUAL = 1  # what the static vector does on its own
    LIVE_UNIVERSE = 2  # the reallocated schedule
    PERFORMANCE = 3  # what the difference is worth


def fetch_prices() -> pd.DataFrame:
    """the seeded panel, restricted to two full histories and one late start."""
    return generate_synthetic_prices()[TICKERS]


def run_local_test(local_test: LocalTest) -> None:
    """
    run one case.

    Args:
        local_test: which case to run
    """
    prices = fetch_prices()
    inception = prices['SEQ_EM'].first_valid_index()

    if local_test == LocalTest.CASH_RESIDUAL:
        print(f"panel {prices.index[0]:%d%b%Y} to {prices.index[-1]:%d%b%Y}, "
              f"SEQ_EM prices from {inception:%d%b%Y}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                          weights=STATIC_WEIGHTS,
                                                          rebalancing_freq=REBALANCING_FREQ,
                                                          rebalancing_costs=REBALANCING_COSTS,
                                                          is_rebalanced_at_first_date=True,
                                                          ticker='Static vector')
        for warning in caught:
            print(f"warning: {warning.message}\n")
        is_rebalancing = portfolio_data.is_rebalancing
        realised = portfolio_data.weights.loc[is_rebalancing[is_rebalancing == True].index, :]
        invested = realised.sum(axis=1)
        print(f"invested fraction at the rebalancings before SEQ_EM starts: "
              f"{invested.loc[:inception].min():.2f} to {invested.loc[:inception].max():.2f}")
        print(f"invested fraction at the rebalancings after:                "
              f"{invested.loc[inception:].min():.2f} to {invested.loc[inception:].max():.2f}")
        print("\nThe missing 25% is not an error: it is held as cash, accruing funding_rate, "
              "which\ndefaults to zero. Nothing in the output says so except this weight sum.")

    elif local_test == LocalTest.LIVE_UNIVERSE:
        weights = qis.generate_static_weights_schedule(prices=prices,
                                                       weights=STATIC_WEIGHTS,
                                                       rebalancing_freq=REBALANCING_FREQ)
        window = qis.TimePeriod(start=inception - pd.Timedelta(days=200),
                                end=inception + pd.Timedelta(days=200))
        around = window.locate(weights)
        print(f"the schedule around the SEQ_EM start on {inception:%d%b%Y}:\n")
        print(around.round(4).to_string())
        print(f"\n0.50 / 0.25 becomes {0.50 / 0.75:.4f} / {0.25 / 0.75:.4f} while SEQ_EM has no "
              "price, and\nSEQ_EM is admitted at the first rebalancing on or after its first "
              "price, not before it.")

        under_invested = qis.generate_static_weights_schedule(prices=prices,
                                                              weights=STATIC_WEIGHTS,
                                                              rebalancing_freq=REBALANCING_FREQ,
                                                              is_rescale_to_live_universe=False)
        print(f"\nis_rescale_to_live_universe=False keeps the cash residual, with an explicit "
              f"0.0:\n{under_invested.loc[around.index[0], :].round(4).to_dict()}")

        # a book that is 90% invested by design must not be levered to 100% by the reallocation
        cash_book = {'SEQ_US': 0.5, 'SBD_TSY': 0.25, 'SEQ_EM': 0.15}
        preserved = qis.generate_static_weights_schedule(prices=prices, weights=cash_book,
                                                         rebalancing_freq=REBALANCING_FREQ)
        forced = qis.generate_static_weights_schedule(prices=prices, weights=cash_book,
                                                      rebalancing_freq=REBALANCING_FREQ,
                                                      is_preserve_total_exposure=False)
        print(f"\na 90%-invested book, before the start: total exposure "
              f"{preserved.loc[:inception, :].sum(axis=1).iloc[0]:.2f} preserved, "
              f"{forced.loc[:inception, :].sum(axis=1).iloc[0]:.2f} forced to one")

    elif local_test == LocalTest.PERFORMANCE:
        weights = qis.generate_static_weights_schedule(prices=prices,
                                                       weights=STATIC_WEIGHTS,
                                                       rebalancing_freq=REBALANCING_FREQ)
        rescaled = qis.backtest_model_portfolio(prices=prices,
                                                weights=weights,
                                                rebalancing_costs=REBALANCING_COSTS,
                                                ticker='Live universe')
        with warnings.catch_warnings():  # the cash residual is the point of this leg, not news
            warnings.simplefilter('ignore')
            residual = qis.backtest_model_portfolio(prices=prices,
                                                    weights=STATIC_WEIGHTS,
                                                    rebalancing_freq=REBALANCING_FREQ,
                                                    rebalancing_costs=REBALANCING_COSTS,
                                                    is_rebalanced_at_first_date=True,
                                                    ticker='Static vector')
        navs = pd.concat([residual.get_portfolio_nav(), rescaled.get_portfolio_nav()],
                         axis=1, sort=True)
        perf_params = qis.PerfParams(freq='ME')
        columns = [qis.PerfStat.PA_RETURN.to_str(), qis.PerfStat.VOL.to_str(),
                   qis.PerfStat.SHARPE_RF0.to_str(), qis.PerfStat.MAX_DD.to_str()]
        table = qis.compute_ra_perf_table(prices=navs, perf_params=perf_params)[columns]
        print(table.round(3).to_string())
        print(f"\nterminal nav {navs.iloc[-1, 0]:.1f} against {navs.iloc[-1, 1]:.1f}: the gap is "
              f"five years of\nholding a quarter of the book in unremunerated cash.")

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
            qis.plot_prices_with_dd(prices=navs, perf_params=perf_params, axs=axs,
                                   title='Static 50/25/25 with a late-starting instrument')
        plt.show()

    else:
        raise ValueError(f"unknown case: {local_test!r}")


if __name__ == '__main__':

    for case in LocalTest:
        print(f"\n===== {case.name} =====")
        run_local_test(local_test=case)
