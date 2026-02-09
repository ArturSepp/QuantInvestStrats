import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from qis.perfstats.returns import (to_zero_first_nonnan_returns, returns_to_nav, compute_sampled_vols,
                                   adjust_navs_to_portfolio_pa, compute_net_navs_ex_perf_man_fees,
                                   compute_asset_returns_dict)


class LocalTests(Enum):
    TO_ZERO_NONNAN = 1
    VOL_SAMPLE = 2
    ADJUST_PORTFOLIO_PA_RETURNS = 3
    NET_RETURN = 4
    ROLLING_RETURNS = 5
    ASSET_RETURNS_DICT = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import qis.plots.time_series as pts
    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if local_test == LocalTests.TO_ZERO_NONNAN:
        np.random.seed(2)  # freeze seed
        dates = pd.date_range(start='31Dec2020', end='07Jan2021', freq='B')
        n = 3
        returns = pd.DataFrame(data=np.random.normal(0.0, 0.01, (len(dates), n)),
                               index=dates,
                               columns=['x' + str(m + 1) for m in range(n)])

        returns.iloc[:, 0] = np.nan
        returns.iloc[:2, 1] = np.nan
        returns.iloc[:1, 2] = np.nan
        returns.iloc[3, 2] = np.nan

        print(f"returns:\n{returns}")

        returns1 = to_zero_first_nonnan_returns(returns=returns)
        print(f"zero_first_non_nan_returns=\n{returns1}")

        navs = returns_to_nav(returns=returns)
        print(f"navs with init_period = 1:\n{navs}")
        navs = returns_to_nav(returns=returns, init_period=None)
        print(f"navs with init_period = None:\n{navs}")

    elif local_test == LocalTests.VOL_SAMPLE:
        vols = compute_sampled_vols(prices=prices,
                                    freq_return='B',
                                    freq_vol='ME')
        print(vols)

    elif local_test == LocalTests.ADJUST_PORTFOLIO_PA_RETURNS:
        returns = prices.pct_change()

        portfolio_price = returns_to_nav(returns=returns.sum(1)).rename('portfolio')

        asset_prices_adj = adjust_navs_to_portfolio_pa(portfolio_nav=portfolio_price,
                                                       asset_prices=prices)

        asset_prices_adj.columns = [x + ' adjusted' for x in asset_prices_adj.columns]

        plot_data = pd.concat([prices.divide(prices.iloc[0, :], axis=1),
                               asset_prices_adj.divide(asset_prices_adj.iloc[0, :], axis=1),
                               portfolio_price], axis=1)
        pts.plot_time_series(df=plot_data,
                             var_format='{:.2f}',
                             title='Original vs Adjusted NAVs')
        print(asset_prices_adj)

    elif local_test == LocalTests.NET_RETURN:
        nav = prices['SPY'].dropna()
        print(nav)
        net_navs = compute_net_navs_ex_perf_man_fees(navs=nav)
        print(net_navs)

    elif local_test == LocalTests.ASSET_RETURNS_DICT:
        # Create test data with different frequency requirements
        # Some assets need daily returns, others weekly
        test_prices = prices[['SPY', 'TLT', 'GLD']].copy()

        # Define return frequencies for each asset
        returns_freqs = pd.Series({
            'SPY': 'B',  # Daily returns
            'TLT': 'W-WED',  # Weekly returns
            'GLD': 'ME'  # Monthly returns
        })

        print(f"\nReturns frequencies:\n{returns_freqs}")

        # Compute asset returns grouped by frequency
        asset_returns_dict = compute_asset_returns_dict(
            prices=test_prices,
            returns_freqs=returns_freqs,
            drop_first=False,
            is_first_zero=True,
            is_log_returns=True
        )

        print(f"\nReturns dictionary: {asset_returns_dict}")

        # Display returns for each frequency group
        for freq, returns_df in asset_returns_dict.items():
            print(f"\n{freq} frequency returns:")
            print(f"Shape: {returns_df.shape}")
            print(f"Columns: {returns_df.columns.tolist()}")
            print(f"First 10 rows:\n{returns_df.head(10)}")
            print(f"Last 5 rows:\n{returns_df.tail()}")

            # Verify first non-NaN return is zero (is_first_zero=True)
            first_nonnan = returns_df.apply(lambda x: x[x.notna()].iloc[0] if x.notna().any() else np.nan)
            print(f"First non-NaN returns (should be ~0): {first_nonnan.to_dict()}")

        # Reconstruct NAVs from returns to verify
        print("\nReconstructed NAVs from returns:")
        for freq, returns_df in asset_returns_dict.items():
            navs = returns_to_nav(returns_df, init_period=1)
            print(f"\n{freq} frequency NAVs tail:\n{navs.tail()}")

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ASSET_RETURNS_DICT)
