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
        df2 = df_resample_at_int_index(df=df.cumsum(0), func=None, sample_size=21)
        print(f"func=None")
        print(df2)


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.INT_INDEX)