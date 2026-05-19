import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from qis.perfstats.returns import (to_zero_first_nonnan_returns, returns_to_nav, compute_sampled_vols,
                                   adjust_navs_to_portfolio_pa, compute_net_navs_ex_perf_man_fees,
                                   compute_asset_returns_dict,
                                   to_returns,
                                   to_quarterly_returns)


class LocalTests(Enum):
    TO_ZERO_NONNAN = 1
    VOL_SAMPLE = 2
    ADJUST_PORTFOLIO_PA_RETURNS = 3
    NET_RETURN = 4
    ROLLING_RETURNS = 5
    ASSET_RETURNS_DICT = 6
    QUARTERLY_RETURNS = 7
    TO_RETURNS_HOLIDAY_NAN = 8
    TO_RETURNS_MIXED_FREQUENCIES = 9


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import qis.plots.time_series as pts
    from qis.tests.price_data_test import load_etf_data
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

        portfolio_price = returns_to_nav(returns=returns.sum(axis=1)).rename('portfolio')

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

    elif local_test == LocalTests.QUARTERLY_RETURNS:
        # Case 1 (original): monthly input, fund_b ends Feb 2024 mid-Q1.
        # Q1 2024 should be valid for fund_a, NaN for fund_b.
        idx = pd.date_range('2023-01-31', '2024-03-31', freq='ME')
        df = pd.DataFrame({'fund_a': 0.01, 'fund_b': 0.01}, index=idx)
        df.loc['2024-03-31', 'fund_b'] = np.nan  # fund_b ends Feb 2024
        print("Case 1 — monthly with fund_b ending mid-Q1:")
        out1 = to_quarterly_returns(df)
        print(out1)
        assert pd.notna(out1.loc['2024-03-31', 'fund_a']), \
            "fund_a Q1 2024 should be valid"
        assert pd.isna(out1.loc['2024-03-31', 'fund_b']), \
            "fund_b Q1 2024 should be NaN"

        # Case 2 (regression): weekly W-FRI input. Stamps don't land on
        # calendar QE dates — earlier ``returns.reindex(QE).notna()`` check
        # masked everything. Should now produce a full quarterly history.
        idx_wk = pd.date_range('2018-11-09', '2026-04-17', freq='W-FRI')
        np.random.seed(0)
        weekly = pd.Series(np.random.normal(0.001, 0.01, len(idx_wk)),
                           index=idx_wk, name='weekly_fund')
        out2 = to_quarterly_returns(weekly)
        print(f"\nCase 2 — W-FRI weekly: {out2.notna().sum()} of {len(out2)} valid")
        assert out2.notna().sum() >= 25, \
            f"expected ~30 valid quarters from W-FRI, got {out2.notna().sum()}"

    elif local_test == LocalTests.TO_RETURNS_HOLIDAY_NAN:
        # Regression test for to_returns when daily input contains an
        # explicit NaN on a date that coincides with a resample target
        # (the W-FRI bucket end). Root cause was in df_asfreq, not in
        # to_returns itself, but the user-facing symptom surfaces here:
        # before the fix, the weekly return for the holiday week was NaN.
        # After the fix it matches df.resample('W-FRI').last().pct_change(),
        # which is the canonical bucket-method ground truth.
        #
        # Reported by Ben Richards: SPY had a valid close on Thu 2025-07-03
        # but an explicit NaN on Fri 2025-07-04 (US Independence Day).
        bday_index = pd.bdate_range('2025-06-30', '2025-07-18', freq='B')
        prices = pd.Series(
            [100, 101, 102, 103, np.nan, 104, 105, 106, 107, 108,
             109, 110, 111, 112, 113][:len(bday_index)],
            index=bday_index,
            name='SPY',
        )
        print("daily prices with NaN on Fri 2025-07-04 (US holiday):")
        print(prices)

        # Ground truth — resample bucket convention.
        bucket = (prices.resample('W-FRI').last()
                  .pct_change(fill_method=None).iloc[1:])
        # qis convention via to_returns(freq='W-FRI').
        qis_ret = to_returns(prices, freq='W-FRI', drop_first=True)
        print(f"\nbucket method  2025-07-11: {bucket.loc['2025-07-11']:+.6f}")
        print(f"qis.to_returns 2025-07-11: {qis_ret.loc['2025-07-11']:+.6f}")
        print(f"bucket method  2025-07-18: {bucket.loc['2025-07-18']:+.6f}")
        print(f"qis.to_returns 2025-07-18: {qis_ret.loc['2025-07-18']:+.6f}")

        assert np.isclose(qis_ret.loc['2025-07-11'],
                          bucket.loc['2025-07-11'], equal_nan=False), \
            (f"holiday-week return mismatch: "
             f"qis={qis_ret.loc['2025-07-11']}, "
             f"bucket={bucket.loc['2025-07-11']}")
        print("\n✓ qis.to_returns matches bucket method for holiday week")

        # Also confirm the dense-input case is unchanged (no NaN, full
        # business-day series) so we know the fix is scoped to NaN handling.
        clean = pd.Series(np.linspace(100, 113, len(bday_index)),
                          index=bday_index, name='clean')
        clean_bucket = (clean.resample('W-FRI').last()
                        .pct_change(fill_method=None).iloc[1:])
        clean_qis = to_returns(clean, freq='W-FRI', drop_first=True)
        assert np.allclose(clean_qis.values, clean_bucket.values), \
            "dense input behaviour changed — fix should be NaN-scoped"
        print("✓ dense-input behaviour unchanged (fix is NaN-scoped)")

    elif local_test == LocalTests.TO_RETURNS_MIXED_FREQUENCIES:
        # Cross-frequency regression test. The df_asfreq fix applies a
        # pre-reindex ffill to the input before the resample picks bucket
        # anchors. This branch exercises:
        #   - daily → weekly (W-FRI) with NaN at the Friday bucket
        #   - daily → monthly (ME) with NaN at month-end
        #   - daily → quarterly (QE) with NaN at quarter-end
        # In each case the qis output must equal the bucket method.
        bday_index = pd.bdate_range('2024-01-02', '2025-12-31', freq='B')
        np.random.seed(7)
        np_prices = 100.0 * np.cumprod(1.0 + np.random.normal(0.0005, 0.012,
                                                              len(bday_index)))
        prices = pd.Series(np_prices, index=bday_index, name='SPY')

        # Inject explicit NaNs on a set of holiday-like dates that
        # coincide with common bucket ends.
        holiday_dates = [
            pd.Timestamp('2024-03-29'),  # Good Friday — last Fri of Q1
            pd.Timestamp('2024-05-31'),  # Friday + month-end
            pd.Timestamp('2024-07-05'),  # Friday after Independence Day
            pd.Timestamp('2024-12-25'),  # Christmas (Wed) — month-end-ish
            pd.Timestamp('2025-07-04'),  # Friday Independence Day
            pd.Timestamp('2025-09-30'),  # Tue + quarter-end
        ]
        for d in holiday_dates:
            if d in prices.index:
                prices.loc[d] = np.nan
        injected = [d for d in holiday_dates if d in prices.index]
        print(f"injected NaN on: {[str(d.date()) for d in injected]}")

        for freq in ['W-FRI', 'ME', 'QE']:
            bucket = (prices.resample(freq).last()
                      .pct_change(fill_method=None).iloc[1:])
            qis_ret = to_returns(prices, freq=freq, drop_first=True)
            # Align on the intersection of indices to make comparison robust
            # to any boundary mismatch in the resampled outputs.
            common = bucket.index.intersection(qis_ret.index)
            assert len(common) > 0, f"empty intersection at freq={freq}"
            diff = (qis_ret.loc[common] - bucket.loc[common]).abs()
            max_diff = diff.max()
            print(f"freq={freq}: {len(common)} buckets, max |qis - bucket| = {max_diff:.2e}")
            assert max_diff < 1e-9, \
                f"freq={freq}: qis diverges from bucket method, max diff {max_diff}"
        print("\n✓ qis.to_returns matches bucket method across W-FRI / ME / QE")

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.TO_RETURNS_MIXED_FREQUENCIES)
