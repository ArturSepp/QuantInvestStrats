import pandas as pd
import numpy as np
from enum import Enum
import qis.utils.dates as da
from qis.utils.df_str import df_index_to_str
from qis.utils.df_freq import df_asfreq, df_resample_at_other_index, df_resample_at_freq, df_resample_at_int_index


class LocalTests(Enum):
    AS_FREQ = 1
    RESAMPLE = 2
    INT_INDEX = 3
    EXPLICIT_NAN_HOLIDAY = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    time_period = da.TimePeriod('1Jan2020', '05Jan2021')
    daily_index = time_period.to_pd_datetime_index(freq='B')
    df = pd.DataFrame(
        data=np.tile(np.array([1.0, 2.0]), (len(daily_index), 1)),
        index=daily_index,
        columns=['1', '2']
    )
    print(df)

    if local_test == LocalTests.AS_FREQ:
        freq_data = df_asfreq(df=df, freq='YE')
        print(freq_data)
        freq_data = df_asfreq(df=df, freq='YE', include_end_date=True)
        print(freq_data)
        print(type(freq_data.index))

        freq_data_s = df_index_to_str(freq_data)
        print(freq_data_s)
        print(type(freq_data_s.index))

    elif local_test == LocalTests.RESAMPLE:
        time_period1 = da.TimePeriod('1Jan2020', '1Jan2021')
        other_index = time_period1.to_pd_datetime_index(freq='QE')
        print(other_index)
        freq_data1 = df_resample_at_other_index(df=df, other_index=other_index, agg_func=np.nansum)
        print(freq_data1)
        print(freq_data1.index)

        freq_data2 = df_resample_at_freq(df=df, freq='QE', agg_func=np.nansum, include_end_date=True)
        print(freq_data2)
        print(freq_data2.index)

    elif local_test == LocalTests.INT_INDEX:
        df2 = df_resample_at_int_index(df=df.cumsum(axis=0), func=None, sample_size=21)
        print(f"func=None")
        print(df2)

    elif local_test == LocalTests.EXPLICIT_NAN_HOLIDAY:
        # Regression test for df_asfreq behaviour when df contains an
        # explicit NaN value on a date that coincides with a target
        # resample timestamp (e.g. yfinance returning NaN on a US holiday
        # Friday that also happens to be the W-FRI bucket end).
        #
        # Before the pre-reindex ffill was added, the W-FRI return on the
        # holiday week was NaN — reindex(method='ffill') looks at INDEX
        # LABELS not values, so it found the Friday label directly and
        # returned its NaN value. After the fix, the return matches the
        # canonical resample('W-FRI').last().pct_change() bucket method.
        bday_index = pd.bdate_range('2025-06-30', '2025-07-18', freq='B')
        prices = pd.Series(
            [100, 101, 102, 103, np.nan, 104, 105, 106, 107, 108,
             109, 110, 111, 112, 113][:len(bday_index)],
            index=bday_index,
            name='SPY',
        )
        print("input daily prices (NaN on Fri 2025-07-04 = US Independence Day):")
        print(prices)

        # Ground truth: pandas resample-and-bucket convention.
        bucket = (prices.resample('W-FRI').last()
                  .pct_change(fill_method=None).iloc[1:])
        print("\nresample('W-FRI').last().pct_change() (ground truth):")
        print(bucket)

        # qis convention via df_asfreq → reindex → ffill.
        weekly_prices = df_asfreq(df=prices, freq='W-FRI', fill_na_method='ffill')
        qis_returns = weekly_prices.pct_change(fill_method=None).iloc[1:]
        print("\ndf_asfreq + pct_change (qis convention):")
        print(qis_returns)

        # Assert match for the holiday week (2025-07-11). Before the fix,
        # qis returned NaN here. After the fix, both methods agree.
        assert np.isclose(qis_returns.loc['2025-07-11'],
                          bucket.loc['2025-07-11'],
                          equal_nan=False), \
            (f"holiday-week return mismatch: qis={qis_returns.loc['2025-07-11']}, "
             f"bucket={bucket.loc['2025-07-11']}")
        print("\n✓ qis convention matches resample bucket method for holiday week")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.EXPLICIT_NAN_HOLIDAY)