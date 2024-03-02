"""
compute descriptive data table
"""
# packages
import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import skew, kurtosis, percentileofscore, normaltest
from enum import Enum

# qis
import qis.utils as qu
from qis.perfstats.config import PerfStat


class DescTableType(Enum):
    NONE = 0
    SHORT = 1
    AVG_WITH_POSITIVE_PROB = 2
    WITH_POSITIVE_PROB = 3
    WITH_KURTOSIS = 4
    WITH_NORMAL_PVAL = 5
    WITH_SCORE = 6
    EXTENSIVE = 7
    SKEW_KURTOSIS = 8
    WITH_MEDIAN = 9


def compute_desc_table(df: Union[pd.DataFrame, pd.Series],
                       desc_table_type: DescTableType = DescTableType.SHORT,
                       var_format: str = '{:.2f}',
                       annualize_vol: bool = False,
                       is_add_tstat: bool = False,
                       norm_variable_display_type: str = '{:.1f}',  # for t-stsat
                       **kwargs
                       ) -> pd.DataFrame:
    """
    data corresponds to matrix of returns with index = time and columns = tickers
    data can contain nan in columns
    output is index = tickers, columns = descriptive data
    data converted to str
    """
    if isinstance(df, pd.DataFrame):
        descriptive_table = pd.DataFrame(index=df.columns)
    elif isinstance(df, pd.Series):
        descriptive_table = pd.DataFrame(index=[df.name])
        df = df.to_frame()
    else:
        raise TypeError(f"unsupported data type = {type(df)}")

    data_np = df.to_numpy()
    mean = np.nanmean(data_np, axis=0)
    std = np.nanstd(data_np, ddof=1, axis=0)

    descriptive_table[PerfStat.AVG.to_str()] = [var_format.format(x) for x in mean]

    if annualize_vol:
        an_factor = qu.infer_an_from_data(data=df)
        vol = std * np.sqrt(an_factor)
        descriptive_table[PerfStat.STD_AN.to_str()] = [var_format.format(x) for x in vol]
    else:
        an_factor = 1.0
        vol = std
        descriptive_table[PerfStat.STD.to_str()] = [var_format.format(x) for x in std]

    if is_add_tstat:
        an_mean = an_factor * mean
        tstats = np.divide(an_mean, vol, where=np.greater(an_mean, 0.0))
        descriptive_table[PerfStat.T_STAT.to_str()] = [norm_variable_display_type.format(x) for x in tstats]

    nan_policy = 'omit'  # skip nans
    if desc_table_type == desc_table_type.SHORT:
        pass

    elif desc_table_type == desc_table_type.AVG_WITH_POSITIVE_PROB:
        descriptive_table = descriptive_table.drop(PerfStat.AVG.to_str(), axis=1)
        descriptive_table = descriptive_table.drop(PerfStat.STD.to_str(), axis=1)

        positive = np.where(np.greater(data_np, 0.0), 1.0, 0.0)
        prob = np.sum(positive, axis=0) / data_np.shape[0]
        descriptive_table[PerfStat.POSITIVE.to_str(short=True, short_n=True)] = ['{:.1%}'.format(x) for x in prob]

    elif desc_table_type == desc_table_type.WITH_POSITIVE_PROB:
        positive = np.where(np.greater(data_np, 0.0), 1.0, 0.0)
        prob = np.sum(positive, axis=0) / data_np.shape[0]
        descriptive_table[PerfStat.POSITIVE.to_str(short=True, short_n=True)] = ['{:.1%}'.format(x) for x in prob]

    elif desc_table_type == desc_table_type.WITH_KURTOSIS:
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in skew(data_np, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in kurtosis(data_np, axis=0, nan_policy=nan_policy)]

    elif desc_table_type == desc_table_type.WITH_NORMAL_PVAL:
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in skew(data_np, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in kurtosis(data_np, axis=0, nan_policy=nan_policy)]
        k2, ps = normaltest(a=data_np, axis=0, nan_policy='omit')
        descriptive_table[PerfStat.NORMTEST.to_str(short=True, short_n=True)] = ['{:.2f}'.format(x) for x in ps]

    elif desc_table_type == desc_table_type.SKEW_KURTOSIS:
        descriptive_table = descriptive_table.drop(PerfStat.AVG.to_str(), axis=1)
        descriptive_table = descriptive_table.drop(PerfStat.STD.to_str(), axis=1)
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in skew(data_np, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in kurtosis(data_np, axis=0, nan_policy=nan_policy)]
    elif desc_table_type == desc_table_type.WITH_SCORE:
        column_data = [df[column].dropna() for column in df.columns]
        percentiles = [percentileofscore(a=x, score=x.iloc[-1], kind='rank') for x in column_data]
        descriptive_table[PerfStat.LAST.to_str()] = [var_format.format(x.iloc[-1]) for x in column_data]
        descriptive_table[PerfStat.RANK.to_str()] = ['{:.0%}'.format(0.01*x) for x in percentiles]

    elif desc_table_type == desc_table_type.EXTENSIVE:
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in skew(df.values, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in kurtosis(df.values, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.MIN.to_str()] \
            = [var_format.format(x) for x in np.nanmin(df.values, axis=0)]
        descriptive_table[PerfStat.QUANT_M_1STD.to_str(short=True)]\
            = [var_format.format(x) for x in np.nanquantile(df.values, q=0.16, axis=0)]
        descriptive_table[PerfStat.MEDIAN.to_str(short=True)] \
            = [var_format.format(x) for x in np.nanmedian(df.values, axis=0)]
        descriptive_table[PerfStat.QUANT_P1_STD.to_str(short=True)] \
            = [var_format.format(x) for x in np.nanquantile(df.values, q=0.84, axis=0)]
        descriptive_table[PerfStat.MAX.to_str()] \
            = [var_format.format(x) for x in np.nanmax(df.values, axis=0)]

    elif desc_table_type == desc_table_type.WITH_MEDIAN:
        descriptive_table[PerfStat.MEDIAN.to_str(short=True)] \
            = [var_format.format(x) for x in np.nanmedian(df.values, axis=0)]
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in skew(df.values, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in kurtosis(df.values, axis=0, nan_policy=nan_policy)]

    else:
        raise TypeError(f"desc_table_type={desc_table_type} is not implemented")

    return descriptive_table


class UnitTests(Enum):
    TABLE = 1


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    returns = load_etf_data().dropna().asfreq('QE').pct_change()

    if unit_test == UnitTests.TABLE:
        df = compute_desc_table(df=returns,
                                desc_table_type=DescTableType.EXTENSIVE,
                                var_format='{:.2f}',
                                annualize_vol=True,
                                is_add_tstat=False)
        print(df)


if __name__ == '__main__':

    unit_test = UnitTests.TABLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
