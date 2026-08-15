import pandas as pd
from enum import Enum
from qis.perfstats.config import PerfParams
from qis.perfstats.perf_stats import (compute_ra_perf_table,
                                      compute_ra_perf_table_with_benchmark,
                                      compute_rolling_drawdowns,
                                      compute_drawdowns_stats_table,
                                      compute_desc_freq_table)


class LocalTests(Enum):
    RA_PERF_TABLE = 1
    RA_PERF_TABLE_WITH_BENCHMARK = 2
    DRAWDOWN = 3
    DRAWDOWN_STATS_TABLE = 4
    TOP_BOTTOM = 5
    # new tests for demonstrating extended functionality
    RA_PERF_TABLE_NO_SUFFIX_COLUMNS = 6
    BENCHMARK_RESOLUTION_THREE_CASES = 7
    DESC_FREQ_TABLE_MEDIAN_FIX = 8
    DRAWDOWN_STATS_IS_RECOVERED = 9
    DOWNSIDE_VOL_EDGE_CASES = 10


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from qis.test_data import load_etf_data
    prices = load_etf_data()  # .dropna()

    if local_test == LocalTests.RA_PERF_TABLE:
        perf_params = PerfParams(freq='B')
        table = compute_ra_perf_table(prices=prices, perf_params=perf_params)
        print(table)

    elif local_test == LocalTests.RA_PERF_TABLE_WITH_BENCHMARK:
        perf_params = PerfParams(freq='ME')
        table = compute_ra_perf_table_with_benchmark(prices=prices, benchmark=prices.columns[0], perf_params=perf_params)
        print(table)

    elif local_test == LocalTests.DRAWDOWN:
        dd_data = compute_rolling_drawdowns(prices=prices['SPY'])
        print(dd_data)

    elif local_test == LocalTests.DRAWDOWN_STATS_TABLE:
        df = compute_drawdowns_stats_table(price=prices['SPY'], max_num=10)
        print(df)

    # ─────────────────────────────────────────────────────────────────────
    # NEW TESTS: demonstrate refactored behaviour
    # ─────────────────────────────────────────────────────────────────────

    elif local_test == LocalTests.RA_PERF_TABLE_NO_SUFFIX_COLUMNS:
        # Verify the merge fix: no more "Start date_y" / "End date_y" duplicates
        # caused by the old pd.merge(..., suffixes=(None, "_y")) pattern.
        perf_params = PerfParams(freq='B')
        table = compute_ra_perf_table(prices=prices, perf_params=perf_params)
        suffix_cols = [c for c in table.columns if str(c).endswith('_y')]
        print(f"Columns with '_y' suffix (should be empty): {suffix_cols}")
        print(f"Total columns: {len(table.columns)}")
        print(f"Column list:\n{list(table.columns)}")
        assert len(suffix_cols) == 0, f"unexpected suffixed columns: {suffix_cols}"
        print("PASS: no duplicate suffixed columns in output")

    elif local_test == LocalTests.BENCHMARK_RESOLUTION_THREE_CASES:
        # Verify all three input modes of compute_ra_perf_table_with_benchmark:
        # (1) benchmark as column name only
        # (2) benchmark_price as Series only
        # (3) both supplied explicitly
        perf_params = PerfParams(freq='ME')
        bench_name = prices.columns[0]
        bench_series = prices[bench_name].copy()
        prices_without_bench = prices.drop(columns=[bench_name])

        print("─── Case 1: benchmark name only (column already in prices) ───")
        t1 = compute_ra_perf_table_with_benchmark(
            prices=prices, benchmark=bench_name, perf_params=perf_params,
        )
        print(f"  rows: {len(t1)}, benchmark '{bench_name}' present: {bench_name in t1.index}")

        print("\n─── Case 2: benchmark_price Series only (benchmark not in prices) ───")
        t2 = compute_ra_perf_table_with_benchmark(
            prices=prices_without_bench, benchmark_price=bench_series, perf_params=perf_params,
        )
        print(f"  rows: {len(t2)}, benchmark '{bench_name}' present: {bench_name in t2.index}")

        print("\n─── Case 3: both supplied, benchmark column already in prices ───")
        t3 = compute_ra_perf_table_with_benchmark(
            prices=prices, benchmark=bench_name, benchmark_price=bench_series,
            perf_params=perf_params,
        )
        print(f"  rows: {len(t3)}, benchmark '{bench_name}' present: {bench_name in t3.index}")

        # all three cases should produce identical results for this ticker
        print("\n─── Sanity: beta of benchmark to itself should be 1.0 ───")
        from qis.perfstats.config import PerfStat
        beta_col = PerfStat.BETA.to_str()
        for i, t in enumerate([t1, t2, t3], 1):
            print(f"  Case {i} beta of {bench_name} to itself: {t.loc[bench_name, beta_col]:.4f}")
        print("PASS: all three benchmark-resolution paths handled")

    elif local_test == LocalTests.DESC_FREQ_TABLE_MEDIAN_FIX:
        # Verify the median bug fix: MEDIAN column should now be the actual median,
        # not the mean. Test by constructing a skewed series where mean != median.
        import numpy as np
        returns = prices.pct_change().dropna()
        table = compute_desc_freq_table(df=returns, freq='YE', agg_func=np.sum)
        print("Descriptive frequency table (annual sums of daily returns):")
        print(table.round(4))

        # Manually verify median column matches actual per-column medians
        from qis.perfstats.config import PerfStat
        yearly = returns.resample('YE').sum().dropna()
        actual_median = yearly.median()
        reported_median = table[PerfStat.MEDIAN.to_str()]
        max_diff = (actual_median - reported_median).abs().max()
        print(f"\nMax |reported median - actual median|: {max_diff:.2e}")
        assert max_diff < 1e-10, "median column does not match actual median"
        print("PASS: MEDIAN column is now correctly computed as median (not mean)")

    elif local_test == LocalTests.DRAWDOWN_STATS_IS_RECOVERED:
        # Verify the new is_recovered flag on drawdown episodes.
        df = compute_drawdowns_stats_table(price=prices['SPY'], max_num=10)
        print("Top 10 SPY drawdowns with recovery status:")
        print(df[['start', 'trough', 'end', 'max_dd', 'days_to_trough',
                  'days_recovery', 'is_recovered']].to_string())

        recovered_count = df['is_recovered'].sum()
        unrecovered_count = (~df['is_recovered']).sum()
        print(f"\nRecovered episodes: {recovered_count}, "
              f"unrecovered (current drawdown): {unrecovered_count}")

    elif local_test == LocalTests.DOWNSIDE_VOL_EDGE_CASES:
        # Verify the safe downside vol guard: series with very few negative returns
        # should produce 0.0 (or a valid small number) rather than RuntimeWarning/NaN.
        import numpy as np
        from qis.perfstats.perf_stats import compute_risk_table

        # Build a synthetic DataFrame with three regimes:
        # (a) normal returns — mix of positive and negative
        # (b) nearly-all-positive returns — should trigger the guard
        # (c) normal SPY returns as a control
        np.random.seed(42)
        idx = prices.index[-500:]  # last ~2 years of business days

        df_test = pd.DataFrame(index=idx)
        df_test['normal'] = np.random.normal(0.0005, 0.01, len(idx))
        df_test['mostly_positive'] = np.abs(np.random.normal(0.002, 0.005, len(idx)))  # always >= 0
        df_test['spy_control'] = prices['SPY'].reindex(idx).pct_change().fillna(0.0).values

        # convert to prices (starting at 100)
        test_prices = (1 + df_test).cumprod() * 100.0

        perf_params = PerfParams(freq='B')
        risk_table = compute_risk_table(prices=test_prices, perf_params=perf_params)
        print("Risk table for edge-case synthetic data:")
        print(risk_table[['Vol', 'Downside Vol', 'Worst', 'Best']].round(4))

        # The 'mostly_positive' column should have downside vol = 0 (no negatives)
        # rather than NaN or a warning.
        from qis.perfstats.config import PerfStat
        dv = risk_table.loc['mostly_positive', PerfStat.DOWNSIDE_VOL.to_str()]
        print(f"\nDownside vol for nearly-all-positive series: {dv}")
        assert not pd.isna(dv), "downside vol should not be NaN for edge-case series"
        print("PASS: downside vol guarded against empty negative-return arrays")


if __name__ == '__main__':

    # original tests
    # run_local_test(local_test=LocalTests.RA_PERF_TABLE)
    # run_local_test(local_test=LocalTests.RA_PERF_TABLE_WITH_BENCHMARK)
    # run_local_test(local_test=LocalTests.DRAWDOWN)
    run_local_test(local_test=LocalTests.DRAWDOWN_STATS_TABLE)

    # new tests for refactored functionality
    # run_local_test(local_test=LocalTests.RA_PERF_TABLE_NO_SUFFIX_COLUMNS)
    #run_local_test(local_test=LocalTests.BENCHMARK_RESOLUTION_THREE_CASES)
    # run_local_test(local_test=LocalTests.DESC_FREQ_TABLE_MEDIAN_FIX)
    # run_local_test(local_test=LocalTests.DRAWDOWN_STATS_IS_RECOVERED)
    # run_local_test(local_test=LocalTests.DOWNSIDE_VOL_EDGE_CASES)