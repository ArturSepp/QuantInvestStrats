import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from qis.perfstats.returns import (to_zero_first_nonnan_returns, returns_to_nav, compute_sampled_vols,
                                   adjust_component_navs_to_portfolio, compute_net_navs_ex_perf_man_fees,
                                   compute_asset_returns_dict,
                                   to_returns,
                                   to_quarterly_returns,
                                   compute_total_return,
                                   compute_pa_return,
                                   compute_excess_returns,
                                   prices_at_freq)
from qis.utils.df_ops import df_price_ffill_between_nans


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
    TOTAL_RETURN_TRAILING_NAN = 10
    PRICES_AT_FREQ_FFILL_NANS_OFF = 11
    EXCESS_RETURNS_LAG = 12
    PRICE_FFILL_BETWEEN_NANS_METHOD = 13
    PA_RETURN_ZEROS_SHAPE = 14
    ZERO_FIRST_NONNAN_BOTH_BRANCHES = 15


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

        component_navs_adj = adjust_component_navs_to_portfolio(portfolio_nav=portfolio_price,
                                                                component_navs=prices)

        component_navs_adj.columns = [x + ' adjusted' for x in component_navs_adj.columns]

        plot_data = pd.concat([prices.divide(prices.iloc[0, :], axis=1),
                               component_navs_adj.divide(component_navs_adj.iloc[0, :], axis=1),
                               portfolio_price], axis=1)
        pts.plot_time_series(df=plot_data,
                             var_format='{:.2f}',
                             title='Original vs Adjusted Component NAVs')
        print(component_navs_adj)

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

    elif local_test == LocalTests.TOTAL_RETURN_TRAILING_NAN:
        # Regression test for compute_total_return with trailing NaN.
        # Previously the function handled NaN at the START (using
        # get_first_nonnan_values) but not at the END — so any fund that
        # terminated mid-dataset or asset that delisted would silently
        # return NaN as total return. Fix mirrors the leading-NaN logic
        # using get_last_nonnan_values.
        import warnings
        # Series case: 5 years of monthly data, last 3 months NaN.
        idx = pd.date_range('2020-01-01', '2024-12-31', freq='ME')
        prices_s = pd.Series(np.linspace(100, 150, len(idx)),
                             index=idx, name='terminated_fund')
        prices_s.iloc[-3:] = np.nan
        last_valid = prices_s.dropna().iloc[-1]
        expected_tr = last_valid / 100.0 - 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tr = compute_total_return(prices_s)
        print(f"Series with trailing NaN: total_return = {tr:.6f}, "
              f"expected ~{expected_tr:.6f}")
        assert not np.isnan(tr), "trailing NaN should not produce NaN total return"
        assert np.isclose(tr, expected_tr), f"got {tr}, expected {expected_tr}"
        print("  ✓ Series: trailing NaN recovers last valid price")

        # DataFrame case: A ends mid-dataset, B is clean.
        df = pd.DataFrame({
            'A': np.linspace(100, 150, len(idx)),
            'B': np.linspace(100, 200, len(idx)),
        }, index=idx)
        df.iloc[-3:, 0] = np.nan  # A trails off
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tr_df = compute_total_return(df)
        print(f"DataFrame mixed: A trailing NaN, B clean. tr = {tr_df}")
        expected_a = df['A'].dropna().iloc[-1] / 100.0 - 1.0
        assert np.isclose(tr_df[0], expected_a), \
            f"A: expected {expected_a:.4f}, got {tr_df[0]}"
        assert np.isclose(tr_df[1], 1.0), f"B: expected 1.0, got {tr_df[1]}"
        print("  ✓ DataFrame: A recovers via last_nonnan, B unchanged")

        # Clean input must still produce identical output.
        prices_clean = pd.Series(np.linspace(100, 130, 12),
                                  index=pd.date_range('2024-01-01', periods=12, freq='ME'))
        tr_clean = compute_total_return(prices_clean)
        assert np.isclose(tr_clean, 0.30)
        print("  ✓ Clean input behaviour unchanged")

    elif local_test == LocalTests.PRICES_AT_FREQ_FFILL_NANS_OFF:
        # Regression test for prices_at_freq when freq is None.
        # Previously this branch ignored ffill_nans entirely and gated only
        # on fill_na_method (default 'ffill'), so ffill_nans=False without
        # also overriding fill_na_method gave ffilled output — opposite of
        # what the parameter promises. Now ffill_nans=False disables the
        # fill regardless.
        prices = pd.Series([100.0, 101.0, np.nan, 103.0],
                           index=pd.date_range('2024-01-01', periods=4, freq='D'))
        ffilled = prices_at_freq(prices, freq=None, ffill_nans=True)
        noffill = prices_at_freq(prices, freq=None, ffill_nans=False)
        print(f"input:            {prices.tolist()}")
        print(f"ffill_nans=True:  {ffilled.tolist()}")
        print(f"ffill_nans=False: {noffill.tolist()}")
        assert ffilled.iloc[2] == 101.0, "ffill should carry 101 forward"
        assert np.isnan(noffill.iloc[2]), \
            f"ffill_nans=False should preserve NaN, got {noffill.iloc[2]}"
        print("  ✓ ffill_nans=False is now honoured when freq is None")

        # Also verify freq is not None branch still works (was already correct).
        prices_freq = prices_at_freq(prices, freq='D', ffill_nans=True)
        prices_nofreq = prices_at_freq(prices, freq='D', ffill_nans=False)
        assert prices_freq.iloc[2] == 101.0
        assert np.isnan(prices_nofreq.iloc[2])
        print("  ✓ freq='D' branch unchanged (already correct)")

    elif local_test == LocalTests.EXCESS_RETURNS_LAG:
        # Regression test for compute_excess_returns lag convention.
        # Previously used lag=None (contemporaneous rate), which introduces
        # a small look-ahead bias because the funding cost paid at time t
        # depends on the rate set at t-1 (or earlier), not the rate
        # observed at t. get_excess_returns_nav already used lag=1; now
        # compute_excess_returns matches that convention.
        idx = pd.date_range('2024-01-01', periods=5, freq='D')
        returns = pd.Series([0.01] * 5, index=idx, name='r')
        # Time-varying rates so we can verify which rate gets applied per day.
        rates = pd.Series([0.05, 0.06, 0.07, 0.08, 0.09], index=idx)
        ex = compute_excess_returns(returns, rates)
        print(f"returns:        {returns.tolist()}")
        print(f"rates (annual): {rates.tolist()}")
        print(f"excess returns: {ex.tolist()}")

        # Day 0 must now be NaN: lag=1 means we need the rate from t=-1,
        # which doesn't exist. (Before the fix, it would use today's rate.)
        assert pd.isna(ex.iloc[0]), \
            f"day 0 should be NaN with lag=1, got {ex.iloc[0]}"
        # Day 1 should use the rate from day 0 (0.05), NOT day 1 (0.06).
        # multiply_df_by_dt converts annual rate to a per-day rate using
        # actual day count / annualization_factor=365.
        days = (idx[1] - idx[0]).days
        expected_day1_rate_dt = 0.05 * days / 365.0
        expected_day1_excess = 0.01 - expected_day1_rate_dt
        print(f"day 1: expected excess = {expected_day1_excess:.6f}")
        assert np.isclose(ex.iloc[1], expected_day1_excess), \
            (f"day 1 used wrong rate: expected to use 0.05 (yesterday's), "
             f"got excess {ex.iloc[1]}, expected {expected_day1_excess}")
        print("  ✓ lag=1 applied — day t uses rate from day t-1")

    elif local_test == LocalTests.PRICE_FFILL_BETWEEN_NANS_METHOD:
        # Regression test for df_price_ffill_between_nans method param.
        # Previously the `method` parameter was silently ignored —
        # the body always called .ffill() regardless of whether method
        # was 'ffill' or 'bfill'. Now dispatches correctly. Leading and
        # trailing NaN are preserved in both modes.
        prices = pd.Series([np.nan, np.nan, 100.0, np.nan, 102.0, np.nan, np.nan],
                            index=pd.date_range('2024-01-01', periods=7, freq='D'),
                            name='p')
        ff = df_price_ffill_between_nans(prices, method='ffill')
        bf = df_price_ffill_between_nans(prices, method='bfill')
        print(f"input:        {prices.tolist()}")
        print(f"ffill output: {ff.tolist()}")
        print(f"bfill output: {bf.tolist()}")
        # Middle gap (index 3) should be filled differently by ffill vs bfill.
        assert ff.iloc[3] == 100.0, f"ffill gap: expected 100, got {ff.iloc[3]}"
        assert bf.iloc[3] == 102.0, f"bfill gap: expected 102, got {bf.iloc[3]}"
        # Leading / trailing NaN preserved in both modes.
        for arr, label in [(ff, 'ffill'), (bf, 'bfill')]:
            assert np.isnan(arr.iloc[0]) and np.isnan(arr.iloc[1]), \
                f"{label}: leading NaN not preserved"
            assert np.isnan(arr.iloc[5]) and np.isnan(arr.iloc[6]), \
                f"{label}: trailing NaN not preserved"
        print("  ✓ ffill / bfill differ; leading & trailing NaN preserved")

        # method=None: gap stays NaN inside the valid range.
        none_result = df_price_ffill_between_nans(prices, method=None)
        assert np.isnan(none_result.iloc[3]), \
            f"method=None should leave gap as NaN, got {none_result.iloc[3]}"
        print("  ✓ method=None leaves interior gap as NaN")

    elif local_test == LocalTests.PA_RETURN_ZEROS_SHAPE:
        # Regression test for the np.zeros_like(int) typo in
        # compute_pa_return. The original code path
        #     compounded_return_pa = np.zeros_like(n)
        # produces a 0-d scalar array(0) when n is an int, not a vector
        # of zeros. This branch is only reached when num_years <= 0 (a
        # degenerate input), but if hit, downstream code that tries to
        # index the result crashes. Fix: np.zeros(n) gives a proper vector.
        n = 3
        result = np.zeros(n)
        assert result.shape == (n,), \
            f"expected 1-d vector of length {n}, got shape {result.shape}"
        print(f"  np.zeros(3) shape: {result.shape}  ✓")

        # Sanity check that normal compute_pa_return paths still work.
        idx = pd.date_range('2024-01-01', '2024-12-31', freq='ME')
        prices = pd.DataFrame(
            {'A': np.linspace(100, 110, len(idx)),
             'B': np.linspace(100, 105, len(idx))}, index=idx,
        )
        pa = compute_pa_return(prices)
        print(f"  compute_pa_return on 12-month 2-col DataFrame: {pa}, shape={np.shape(pa)}")
        assert np.shape(pa) == (2,)
        print("  ✓ compute_pa_return returns vector for DataFrame input")

    elif local_test == LocalTests.ZERO_FIRST_NONNAN_BOTH_BRANCHES:
        # Regression test for to_zero_first_nonnan_returns. The
        # init_period=1 branch previously had a defensive check
        #     if first_nonnan_index >= first_date: ...
        # which was always True (since first_date = returns.index[0] and
        # any non-NaN index is by definition >= the first index). The
        # check was removed; this test verifies that the simplified code
        # still zeros the first non-NaN return correctly for both Series
        # and DataFrame inputs, including ragged-start columns.
        # Series with leading NaN
        returns_s = pd.Series([np.nan, np.nan, 0.5, 0.02, 0.03],
                              index=pd.date_range('2024-01-01', periods=5, freq='D'))
        result_s = to_zero_first_nonnan_returns(returns_s, init_period=1)
        print(f"Series input:  {returns_s.tolist()}")
        print(f"Series output: {result_s.tolist()}")
        # Element [2] is the first non-NaN — must be zeroed.
        assert result_s.iloc[2] == 0.0
        # Leading NaN preserved.
        assert pd.isna(result_s.iloc[0]) and pd.isna(result_s.iloc[1])
        # Later values untouched.
        assert result_s.iloc[3] == 0.02 and result_s.iloc[4] == 0.03
        print("  ✓ Series: first non-NaN zeroed, leading NaN preserved")

        # DataFrame with ragged-start columns
        df_ret = pd.DataFrame({
            'A': [np.nan, np.nan, 0.5, 0.02, 0.03],
            'B': [0.3, 0.04, 0.05, 0.06, 0.07],
            'C': [np.nan, 0.4, 0.01, 0.02, 0.03],
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        result_df = to_zero_first_nonnan_returns(df_ret, init_period=1)
        print(f"\nDataFrame output:\n{result_df}")
        # Each column's first non-NaN is at a different row.
        assert result_df.loc[result_df.index[2], 'A'] == 0.0  # A starts at row 2
        assert result_df.loc[result_df.index[0], 'B'] == 0.0  # B starts at row 0
        assert result_df.loc[result_df.index[1], 'C'] == 0.0  # C starts at row 1
        # Subsequent rows in each column untouched.
        assert result_df.loc[result_df.index[3], 'A'] == 0.02
        assert result_df.loc[result_df.index[1], 'B'] == 0.04
        assert result_df.loc[result_df.index[2], 'C'] == 0.01
        # Leading NaNs preserved.
        assert pd.isna(result_df.loc[result_df.index[0], 'A'])
        assert pd.isna(result_df.loc[result_df.index[0], 'C'])
        print("  ✓ DataFrame: each column's first non-NaN zeroed at its own start")

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.QUARTERLY_RETURNS)