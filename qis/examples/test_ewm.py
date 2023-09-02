
import functools
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from enum import Enum

import qis
import qis.plots as qp
import qis.models as qm


def get_test_data(n1: int = 1000, n2: int = 3, means: Union[float, np.ndarray] = 0.0) -> pd.DataFrame:
    np_data = np.random.normal(means, 1.0, (n1, n2))
    random_nans = np.random.binomial(size=(n1, n2), n=1, p=0.5)
    np_data = np.where(random_nans, np.nan, np_data)
    df = pd.DataFrame(data=np_data,
                      index=pd.date_range(start='1Jan2020', periods=n1, freq='D'),
                      columns=[f"x{m+1}" for m in range(n2)])
    return df


def time_comp_with_pandas_ewm(span: int = 31) -> None:

    big_df = get_test_data(n1=10000, n2=10000)
    np_data = big_df.to_numpy()
    init_value = np.zeros(np_data.shape[1])

    def pd_ewm():
        big_df.ewm(span=span, adjust=False).mean()

    def np_to_ewm():
        qm.ewm_recursion(a=np_data, span=span, init_value=init_value)

    def pd_to_ewm():
        qm.compute_ewm(data=big_df, span=span, init_value=init_value)

    n = 20
    print(f"pandas ewm: {timeit.Timer(functools.partial(pd_ewm)).timeit(n) / n:.3f}")
    print(f"np_to_ewm: {timeit.Timer(functools.partial(np_to_ewm)).timeit(n) / n:.3f}")
    print(f"pd_to_ewm: {timeit.Timer(functools.partial(pd_to_ewm)).timeit(n) / n:.3f}")


def vol_comp_with_pandas_ewm(span: int = 31) -> None:
    df = get_test_data(n1=10000, n2=1)
    vol_pd = df.ewm(span=span, adjust=False).std()
    vol_ewm1 = qm.compute_ewm_vol(data=df, span=span, mean_adj_type=qm.MeanAdjType.NONE)
    vol_ewm2 = qm.compute_ewm_vol(data=df, span=span, mean_adj_type=qm.MeanAdjType.EWMA)
    joint = pd.concat([vol_pd.iloc[:, 0].rename('Pandas std'),
                      vol_ewm1.iloc[:, 0].rename('Ewm no adj'),
                      vol_ewm2.iloc[:, 0].rename('Ewm with Ewm adj')],
                      axis=1)

    qp.plot_time_series(df=joint,
                        title='Vols comparision',
                        legend_loc='upper left',
                        trend_line=qp.TrendLine.AVERAGE,
                        var_format='{:.2f}')


def ewm_covar_tensor_nans():
    """
    test how nanas are treated in covar matrix estimation
    """
    n = 3
    size = 2000
    a = np.random.multivariate_normal(mean=np.zeros(n), cov=np.array([[1.0, 0.5, 0.5],
                                                                      [0.5, 1.0, 0.5],
                                                                      [0.5, 0.5, 1.0]]),
                                      size=size)
    print(a)

    print('without nans')
    covar_tensor_txy = qis.compute_ewm_covar_tensor(a=a, span=200)
    print(covar_tensor_txy)

    print('with nans')
    a[:200, 1] = np.nan  # uptonans
    a[size-200:, 2] = np.nan  # after nans
    covar_tensor_txy = qis.compute_ewm_covar_tensor(a=a, span=200, nan_backfill=qis.NanBackfill.NAN_FILL)
    print(covar_tensor_txy)

class UnitTests(Enum):
    TIME_COMP_WITH_PANDAS_EWM = 1
    VOL_COMP_WITH_PANDAS_EWM = 2
    EWMA_NP = 3
    EWMA_DF = 4
    EWMA_MEAN = 5
    EWMA_VOL = 6
    EWMA_BETA = 7
    EWMA_AUTO_CORR = 8
    EWMA_CORR_MATRIX = 9
    EWMA_COVAR_TENSOR_NANS = 10


def run_unit_test(unit_test: UnitTests):

    np.random.seed(1)

    if unit_test == UnitTests.TIME_COMP_WITH_PANDAS_EWM:
        time_comp_with_pandas_ewm()

    elif unit_test == UnitTests.VOL_COMP_WITH_PANDAS_EWM:
        vol_comp_with_pandas_ewm()

    elif unit_test == UnitTests.EWMA_COVAR_TENSOR_NANS:
        ewm_covar_tensor_nans()

    else:  # apply same data for these tests

        data = get_test_data(n1=10000, n2=3, means=np.array([-2.0, -1.0, 0.0]))
        ewm_lambda = 0.94
        ewm_lambda3 = np.array([0.94, 0.50, 0.10])
        plot_data, title = None, None

        if unit_test == UnitTests.EWMA_NP:
            ewm1 = qm.ewm_recursion(a=data.iloc[:, 0].to_numpy(dtype=np.double), ewm_lambda=ewm_lambda, init_value=0.0)
            print(ewm1)

            ewm2 = qm.ewm_recursion(a=data.to_numpy(), ewm_lambda=ewm_lambda, init_value=np.zeros(len(data.columns)))
            print(ewm2)

            ewm3 = qm.ewm_recursion(a=data.to_numpy(), ewm_lambda=ewm_lambda3, init_value=np.zeros(len(data.columns)))
            print(ewm3)

        elif unit_test == UnitTests.EWMA_DF:

            ewm_df1 = qm.compute_ewm(data=data.iloc[:, 0], ewm_lambda=ewm_lambda, init_type=qm.InitType.MEAN)
            print(ewm_df1)

            ewm_df1_np = qm.compute_ewm(data=data.iloc[:, 0].to_numpy(), ewm_lambda=ewm_lambda, init_type=qm.InitType.MEAN)
            print(ewm_df1_np)

            ewm_df2_np = qm.compute_ewm(data=data.to_numpy(), ewm_lambda=ewm_lambda, init_type=qm.InitType.MEAN)
            print(ewm_df2_np)

            ewm_df2 = qm.compute_ewm(data=data, ewm_lambda=ewm_lambda, init_type=qm.InitType.MEAN)
            print(ewm_df2)

            ewm_df3 = qm.compute_ewm(data=data, ewm_lambda=ewm_lambda3, init_type=qm.InitType.MEAN)
            print(ewm_df3)

            ewm_df3.columns = [x + ' ewm' for x in data.columns]
            plot_data = pd.concat([data, ewm_df3], axis=1)
            title = 'ewm'

        elif unit_test == UnitTests.EWMA_MEAN:

            datas = {'np1': data.iloc[:, 0].to_numpy(), 'np': data.to_numpy(), 'series': data.iloc[:, 0], 'df': data}

            for key, data_ in datas.items():
                print(key)
                ewm_mean = qm.compute_roll_mean(data=data,
                                                 mean_adj_type=qm.MeanAdjType.EWMA,
                                                 ewm_lambda=ewm_lambda)
                data_adjusted = data - ewm_mean
                ewm_data = qm.compute_ewm(data=data_adjusted,
                                              ewm_lambda=ewm_lambda)

            ewm_mean.columns = [x + ' ewm_mean' for x in data.columns]
            ewm_data.columns = [x + '-ewm_mean' for x in data.columns]
            plot_data = pd.concat([data, ewm_mean, ewm_data], axis=1)
            title = 'ewm-mean'

        elif unit_test == UnitTests.EWMA_VOL:

            datas = {'np1': data.iloc[:, 0].to_numpy(), 'np': data.to_numpy(), 'series': data.iloc[:, 0], 'df': data}

            for key, data_ in datas.items():
                print(key)
                ewm_data = qm.compute_ewm_vol(data=data_,
                                                  ewm_lambda=ewm_lambda,
                                                  mean_adj_type=qm.MeanAdjType.EWMA,
                                                  init_type=qm.InitType.MEAN)
            print(ewm_data)
            ewm_data.columns = [x + ' ewm vol' for x in data.columns]
            data.columns = [x + '^2' for x in data.columns]
            plot_data = pd.concat([np.power(data, 2), ewm_data], axis=1)
            title = 'ewm vol'

        elif unit_test == UnitTests.EWMA_BETA:

            cross_xy_types = [qm.CrossXyType.COVAR, qm.CrossXyType.BETA, qm.CrossXyType.CORR]

            for cross_xy_type in cross_xy_types:
                print(cross_xy_type)
                ewm_data = qm.compute_ewm_cross_xy(x_data=data[[data.columns[0]]],
                                                       y_data=data,
                                                       cross_xy_type=cross_xy_type,
                                                       ewm_lambda=ewm_lambda,
                                                       mean_adj_type=qm.MeanAdjType.EWMA,
                                                       init_type=qm.InitType.MEAN)
            ewm_data.columns = [x + ' ewm beta' for x in data.columns]
            plot_data = pd.concat([data, ewm_data], axis=1)
            title = 'ewm beta'

        elif unit_test == UnitTests.EWMA_AUTO_CORR:
            plot_data = qm.compute_dynamic_auto_corr(data=data,
                                                     ewm_lambda=ewm_lambda,
                                                     mean_adj_type=qm.MeanAdjType.NONE,
                                                     aggregation_type='median',
                                                     is_normalize=True)
            title = 'ewm auto-corr'

        else:
            return

        if plot_data is not None:
            qp.plot_time_series(df=plot_data,
                                title=title,
                                legend_loc='upper left',
                                trend_line=qp.TrendLine.AVERAGE,
                                var_format='{:.2f}')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.EWMA_COVAR_TENSOR_NANS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
