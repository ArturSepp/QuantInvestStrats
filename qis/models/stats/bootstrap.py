"""
implementation of bootsrap methods in numba
"""

import numpy as np
import pandas as pd
from enum import Enum
from numba import njit
from numba.typed import List
from typing import Union, Tuple, Dict
from statsmodels.tsa.ar_model import AutoReg

# qis
import qis.utils.np_ops as npo
import qis.perfstats.returns as ret


class BootsrapType(Enum):
    IID = 1
    STATIONARY = 2


class BootsrapOutput(Enum):
    SERIES_TO_DF = 1
    DF_TO_LIST_ARRAYS = 2


@njit
def set_seed(value):
    """
    set seed for numba space
    """
    np.random.seed(value)


@njit
def bootstrap_indices_iid(num_data_index: int,
                          num_samples: int = 10,
                          index_length: int = 1000,
                          seed: int = 1
                          ) -> np.ndarray:
    """
    generate iid bootstrap indices
    """
    np.random.seed(seed)
    set_seed(seed)
    bootstrapped_indices = np.zeros((index_length, num_samples), dtype=np.int64)
    for idx in np.arange(num_samples):
        previous_index, next_index = 0, 0
        while next_index < index_length-1:
            ran_indices = np.random.randint(low=0, high=num_data_index, size=num_data_index)
            next_index = np.minimum(previous_index+num_data_index, index_length-1)
            required_fill = next_index - previous_index
            bootstrapped_indices[previous_index:next_index, idx] = ran_indices[:required_fill]
            previous_index = next_index
    return bootstrapped_indices


@njit
def bootstrap_indices_stationary(num_data_index: int,
                                 num_samples: int = 10,
                                 num_bootstrap_index: int = 1000,
                                 block_size: int = 30,
                                 seed: int = 1
                                 ) -> np.ndarray:
    """
    generate stationary bootstrap indices
    """
    np.random.seed(seed)
    set_seed(seed)
    bootstrapped_indices = np.zeros((num_bootstrap_index, num_samples), dtype=np.int64)
    for idx in np.arange(num_samples):
        previous_index, next_index = 0, 0
        while next_index < num_bootstrap_index-1:
            start_index = np.random.randint(low=0, high=num_data_index) # random start of a block
            random_block_size = np.random.geometric(1.0/block_size)  # random block size, default size is 1
            end_index_in_data = np.minimum(start_index+random_block_size, num_data_index)  # end of sample
            proposed_fill = end_index_in_data-start_index  # how much we can increase based on index in data
            next_index = np.minimum(previous_index + proposed_fill, num_bootstrap_index)  # index for backfill indices
            implemented_fill = next_index-previous_index  # how much we will increase based on index in bootstrap indices
            bootstrapped_indices[previous_index:next_index, idx] = np.arange(start_index, start_index+implemented_fill)
            previous_index = next_index
    return bootstrapped_indices


@njit
def get_bootsrtap_data_list(data_np: np.ndarray,
                            bootstrapped_indices: np.ndarray
                            ) -> List:
    """
    map indices to data
    """
    bootstrap_sample = List()
    for index in np.transpose(bootstrapped_indices):
        bootstrapped_data = data_np[index, :]
        bootstrap_sample.append(bootstrapped_data)
    return bootstrap_sample


@njit
def get_bootsrtap_ar_data_list(residuals: np.ndarray,
                               intercept: np.ndarray,
                               beta: np.ndarray,
                               data0: np.ndarray,
                               bootstrapped_indices: np.ndarray,
                               is_positive: bool = True
                               ) -> List:
    """
    map indices to ar data
    """
    bootstrap_sample = List()
    for index in np.transpose(bootstrapped_indices):
        bootstrapped_resids = residuals[index, :]
        bootstrapped_data = np.zeros_like(bootstrapped_resids)
        y0 = data0*np.ones_like(beta)  # nb important for numba so it can map beta*y0
        for idx, resid in enumerate(bootstrapped_resids):
            y0 = intercept + beta*y0 + resid
            if is_positive:
                y0 = np.where(np.greater(y0, 0.0), y0, np.nanquantile(y0, 0.25))
            bootstrapped_data[idx] = y0

        bootstrap_sample.append(bootstrapped_data)
    return bootstrap_sample


def compute_ar_residuals(data: Union[pd.Series, pd.DataFrame]
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(data, pd.Series):
        data = data.to_frame()
    intercept, beta = np.zeros(len(data.columns)), np.zeros(len(data.columns))
    residuals = np.zeros((len(data.index)-1, len(data.columns)))
    for idx, column in enumerate(data.columns):
        ar_model = AutoReg(data[column].dropna(), lags=1).fit()
        intercept[idx], beta[idx] = ar_model.params[0], ar_model.params[1]
        data_np = data[column].to_numpy()
        y, y_1 = data_np[1:], data_np[:-1]
        residuals[:, idx] = y - (ar_model.params[1] * y_1 + ar_model.params[0])
    return residuals, intercept, beta


def generate_bootstrapped_indices(num_data_index: int,
                                  bootsrap_type: BootsrapType = BootsrapType.IID,
                                  num_samples: int = 10,
                                  index_length: int = 1000,
                                  block_size: int = 30,
                                  seed: int = 1
                                  ) -> np.ndarray:
    """
    wrapper for numba function
    """
    if bootsrap_type == BootsrapType.IID:
        bootstrapped_indices = bootstrap_indices_iid(num_data_index=num_data_index,
                                                     num_samples=num_samples,
                                                     index_length=index_length,
                                                     seed=seed)
    elif bootsrap_type == BootsrapType.STATIONARY:
        bootstrapped_indices = bootstrap_indices_stationary(num_data_index=num_data_index,
                                                            num_samples=num_samples,
                                                            num_bootstrap_index=index_length,
                                                            block_size=block_size,
                                                            seed=seed)
    else:
        raise ValueError(f"not implemented")

    return bootstrapped_indices


def bootstrap_data(data: Union[pd.Series, pd.DataFrame],
                   bootsrap_type: BootsrapType = BootsrapType.STATIONARY,
                   bootsrap_output: BootsrapOutput = BootsrapOutput.DF_TO_LIST_ARRAYS,
                   num_samples: int = 10,
                   index_length: int = 1000,
                   block_size: int = 30,
                   seed: int = 1,
                   bootstrapped_indices: np.ndarray = None
                   ) -> Union[List, pd.DataFrame]:
    """
    core data bootsrap
    """
    if bootstrapped_indices is None:
        bootstrapped_indices = generate_bootstrapped_indices(num_data_index=len(data.index),
                                                             bootsrap_type=bootsrap_type,
                                                             num_samples=num_samples,
                                                             index_length=index_length,
                                                             block_size=block_size,
                                                             seed=seed)

    if bootsrap_output == BootsrapOutput.DF_TO_LIST_ARRAYS:
        bootstrap_sample = get_bootsrtap_data_list(data_np=data.to_numpy(),
                                                   bootstrapped_indices=bootstrapped_indices)

    elif bootsrap_output == BootsrapOutput.SERIES_TO_DF:
        if not isinstance(data, pd.Series):
            raise ValueError(f"data must be series")

        bootstrap_sample = get_bootsrtap_data_list(data_np=npo.np_array_to_matrix(a=data.to_numpy(), n_col=1),
                                                   bootstrapped_indices=bootstrapped_indices)
        data = []
        for idx, sample in enumerate(bootstrap_sample):
            data.append(pd.DataFrame(sample, columns=[f"path_{idx+1}"]))
        bootstrap_sample = pd.concat(data, axis=1)

    else:
        raise ValueError(f"not implemented")

    return bootstrap_sample


def bootstrap_ar_process(data: Union[pd.Series, pd.DataFrame],
                         bootsrap_type: BootsrapType = BootsrapType.STATIONARY,
                         bootsrap_output: BootsrapOutput = BootsrapOutput.DF_TO_LIST_ARRAYS,
                         num_samples: int = 10,
                         index_length: int = 1000,
                         block_size: int = 30,
                         seed: int = 1,
                         bootstrapped_indices: np.ndarray = None
                         ) -> Union[List, pd.DataFrame]:

    residuals, intercept, beta = compute_ar_residuals(data=data)
    if bootstrapped_indices is None:
        bootstrapped_indices = generate_bootstrapped_indices(num_data_index=len(data.index),
                                                             bootsrap_type=bootsrap_type,
                                                             num_samples=num_samples,
                                                             index_length=index_length,
                                                             block_size=block_size,
                                                             seed=seed)

    if bootsrap_output == BootsrapOutput.DF_TO_LIST_ARRAYS:
        bootstrap_sample = get_bootsrtap_ar_data_list(residuals=residuals,
                                                      intercept=intercept,
                                                      beta=beta,
                                                      data0=np.nanmean(data, axis=0),
                                                      bootstrapped_indices=bootstrapped_indices)

    elif bootsrap_output == BootsrapOutput.SERIES_TO_DF:
        if not isinstance(data, pd.Series):
            raise ValueError(f"data must be series")

        bootstrap_sample = get_bootsrtap_ar_data_list(residuals=residuals,
                                                      intercept=intercept,
                                                      beta=beta,
                                                      data0=np.array(np.nanmean(data)),
                                                      bootstrapped_indices=bootstrapped_indices)

        data = []
        for idx, sample in enumerate(bootstrap_sample):
            data.append(pd.DataFrame(sample, columns=[f"path_{idx+1}"]))
        bootstrap_sample = pd.concat(data, axis=1)

    else:
        raise ValueError(f"not implemented")

    return bootstrap_sample


def bootstrap_price_data(prices: Union[pd.Series, pd.DataFrame],
                         bootsrap_type: BootsrapType = BootsrapType.STATIONARY,
                         bootsrap_output: BootsrapOutput = BootsrapOutput.DF_TO_LIST_ARRAYS,
                         num_samples: int = 10,
                         index_length: int = 1000,
                         block_size: int = 20,
                         is_log_returns: bool = False,
                         seed: int = 1,
                         bootstrapped_indices: np.ndarray = None,
                         init_to_end: bool = True
                         ) -> Union[List[np.ndarray], pd.DataFrame]:
    """
    bootstrap price data
    for pd.Dataframe use bootsrap_output = BootsrapOutput.DF_TO_LIST_ARRAYS to get list of nd.arrays
    block_size = 1 corresponds to iid sampling
    """
    returns = ret.to_returns(prices=prices, is_log_returns=is_log_returns, drop_first=True)

    bootstrap_returns = bootstrap_data(data=returns,
                                       bootsrap_type=bootsrap_type,
                                       bootsrap_output=bootsrap_output,
                                       num_samples=num_samples,
                                       index_length=index_length,
                                       block_size=block_size,
                                       seed=seed,
                                       bootstrapped_indices=bootstrapped_indices)

    if bootsrap_output == BootsrapOutput.DF_TO_LIST_ARRAYS:
        if init_to_end:
            init_value = prices.iloc[-1, :].to_numpy()
        else:
            init_value = prices.iloc[0, :].to_numpy()

        bootstrap_sample = List()
        for returns in bootstrap_returns:
            if is_log_returns:
                bootstrap_sample.append(ret.log_returns_to_nav(log_returns=returns, init_value=init_value))
            else:
                bootstrap_sample.append(ret.returns_to_nav(returns=returns, init_value=init_value))

    elif bootsrap_output == BootsrapOutput.SERIES_TO_DF:
        if init_to_end:
            init_value = prices[-1]*np.ones(num_samples)
        else:
            init_value = prices[0]*np.ones(num_samples)

        if is_log_returns:
            bootstrap_sample = ret.log_returns_to_nav(log_returns=bootstrap_returns, init_value=init_value)
        else:
            bootstrap_sample = ret.returns_to_nav(returns=bootstrap_returns, init_value=init_value)
    else:
        raise ValueError(f"not implemented")

    return bootstrap_sample


def bootstrap_price_fundamental_data(price_datas: Dict[str, Union[pd.Series, pd.DataFrame]],
                                     fundamental_datas: Dict[str, Union[pd.Series, pd.DataFrame]],
                                     bootsrap_type: BootsrapType = BootsrapType.STATIONARY,
                                     bootsrap_output: BootsrapOutput = BootsrapOutput.DF_TO_LIST_ARRAYS,
                                     num_samples: int = 10,
                                     index_length: int = 1000,
                                     block_size: int = 30,
                                     is_log_returns: bool = False,
                                     seed: int = 1,
                                     is_price_weighted_fundamentals: bool = False  # multiply by price_datas[0] price bootstrap
                                     ) -> Tuple[Dict[str, Union[pd.DataFrame, List]], Dict[str, Union[pd.DataFrame, List]]]:

    """
    price data in the first element must  be aligned with all fundamentals
    """
    prices = price_datas[list(price_datas.keys())[0]]
    for key, fundamental_data in fundamental_datas.items():
        assert all(prices.index == fundamental_data.index)
        if isinstance(prices, pd.DataFrame) and isinstance(fundamental_data, pd.DataFrame):
            assert all(prices.columns == fundamental_data.columns)
        elif isinstance(prices, pd.Series) and isinstance(fundamental_data, pd.Series):
            pass
        else:
            raise ValueError(f" data types not aligned")

    # very important to reduce lenth for returns and ar-1 bootstrap
    bootstrapped_indices = generate_bootstrapped_indices(num_data_index=len(prices.index)-1,
                                                         bootsrap_type=bootsrap_type,
                                                         num_samples=num_samples,
                                                         index_length=index_length,
                                                         block_size=block_size,
                                                         seed=seed)

    bootstrap_prices = {}
    for key, prices in price_datas.items():
        bootstrap_prices[key] = bootstrap_price_data(prices=prices,
                                                     bootsrap_type=bootsrap_type,
                                                     bootsrap_output=bootsrap_output,
                                                     num_samples=num_samples,
                                                     index_length=index_length,
                                                     block_size=block_size,
                                                     is_log_returns=is_log_returns,
                                                     seed=seed,
                                                     bootstrapped_indices=bootstrapped_indices)

    bootstrap_fundamentals = {}
    for key, fundamental_data in fundamental_datas.items():
        fundamentals = bootstrap_ar_process(data=fundamental_data,
                                            bootsrap_type=bootsrap_type,
                                            bootsrap_output=bootsrap_output,
                                            num_samples=num_samples,
                                            index_length=index_length,
                                            block_size=block_size,
                                            seed=seed,
                                            bootstrapped_indices=bootstrapped_indices)

        if is_price_weighted_fundamentals:
            prices = bootstrap_prices[list(bootstrap_prices.keys())[0]]
            for idx, (price, fund) in enumerate(zip(prices, fundamentals)):
                fundamentals[idx] = price*fund

        bootstrap_fundamentals[key] = fundamentals

    return bootstrap_prices, bootstrap_fundamentals


class UnitTests(Enum):
    DATA_LIST = 1
    DATA_SERIES = 2
    BTC = 3


def run_unit_test(unit_test: UnitTests):

    # simulate t*n data
    dates = pd.date_range(start='12/31/2018', end='12/31/2019', freq='B')
    n = 3
    mean = [-2.0, -1.0, 0.0]
    data = pd.DataFrame(data=np.random.normal(mean, 1.0, (len(dates), n)),
                        index=dates,
                        columns=['x'+str(m+1) for m in range(n)])
    print(data)

    if unit_test == UnitTests.DATA_LIST:
        bt_lists = bootstrap_data(data=data,
                                  bootsrap_type=BootsrapType.STATIONARY,
                                  bootsrap_output=BootsrapOutput.DF_TO_LIST_ARRAYS)
        for this in bt_lists:
            print(this)

    elif unit_test == UnitTests.DATA_SERIES:
        bt_data = bootstrap_data(data=data.iloc[:, 0],
                                 bootsrap_type=BootsrapType.STATIONARY,
                                 bootsrap_output=BootsrapOutput.SERIES_TO_DF)
        print(bt_data)

    elif unit_test == UnitTests.BTC:
        from legacy.crypto.data.apis.coinmetric_ import load_btc_price
        pivot_prices = load_btc_price().dropna()
        bt_data = bootstrap_price_data(prices=pivot_prices,
                                       bootsrap_type=BootsrapType.STATIONARY,
                                       bootsrap_output=BootsrapOutput.DF_TO_LIST_ARRAYS)
        print(bt_data)


if __name__ == '__main__':

    unit_test = UnitTests.BTC

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
