import numpy as np
import pandas as pd
from enum import Enum

# qis
import qis.utils.dates as da
import qis.perfstats.returns as ret
import qis.models.linear.ewm as ewm


def apply_pca(cmatrix: np.ndarray,
              is_max_sign_positive: bool = True,
              eigen_signs: np.ndarray = None
              ) -> (np.ndarray, np.ndarray):

    # from sample covar_model
    eig_vals, eig_vecs = np.linalg.eigh(cmatrix)

    # Make a list of (eigenvalue, eigenvector) tuples for sorting
    eig_pairs = [(eig_vals[i], eig_vecs[:, i]) for i in range(len(eig_vals))]

    # sort using numpy
    # inds = eig_vals.argsort()
    # print(inds)

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    # eig_pairs.sort()
    eig_pairs.reverse()

    # get back to ndarrays
    eigen_values = np.array([eig_pair[0] for eig_pair in eig_pairs])
    eigen_vectors = np.array([eig_pair[1] for eig_pair in eig_pairs])

    if is_max_sign_positive and eigen_signs is None:

        signed_eigen_vectors = eigen_vectors
        for idx, eigen_vector in enumerate(eigen_vectors):
            arg_max = np.argmax(np.abs(eigen_vector))
            if eigen_vector[arg_max] < 0.0:
                eigen_vector = - eigen_vector
                signed_eigen_vectors[idx] = eigen_vector
        eigen_vectors = signed_eigen_vectors

    elif eigen_signs is not None:

        # eigen_vectors = eigen_signs * eigen_vectors
        signed_eigen_vectors = eigen_vectors.copy()
        for idx, eigen_vector in enumerate(eigen_vectors):
            if np.sign(eigen_vector[0]) != eigen_signs[idx]:
               signed_eigen_vectors[idx] = - eigen_vector
        eigen_vectors = signed_eigen_vectors

    return eigen_values, eigen_vectors


def compute_pca_r2(cmatrix: np.ndarray, is_cumulative: bool = False) -> (np.ndarray, np.ndarray):
    eigen_values, _ = apply_pca(cmatrix=cmatrix)
    if is_cumulative:
        out = np.cumsum(eigen_values) / np.sum(eigen_values)
    else:
        out = eigen_values / np.sum(eigen_values)
    return out


def compute_data_pca_r2(data: pd.DataFrame,
                        freq: str = 'ME',
                        time_period: da.TimePeriod = None,
                        ewm_lambda: float = 0.94,
                        is_corr: bool = True
                        ) -> pd.DataFrame:

    corr_tensor_txy = ewm.compute_ewm_covar_tensor(a=data.to_numpy(),
                                                     ewm_lambda=ewm_lambda,
                                                     is_corr=is_corr)

    if time_period is None:
        time_period = da.get_time_period(df=data)
    sample_dates = time_period.to_pd_datetime_index(freq=freq)
    original_idx = pd.Series(range(len(data.index)), index=data.index)
    resampled_index = original_idx.reindex(index=sample_dates, method='ffill')

    pca_r2s = {}
    for date, date_idx in zip(resampled_index.index, resampled_index.to_numpy()):
        pca_r2s[date] = compute_pca_r2(cmatrix=corr_tensor_txy[date_idx])

    pca_r2s = pd.DataFrame.from_dict(pca_r2s,
                                     orient='index',
                                     columns=[f"PC{n+1}" for n in range(len(data.columns))])
    return pca_r2s


class UnitTests(Enum):
    PCA_R2 = 1


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()
    print(prices)
    returns = ret.to_returns(prices=prices)

    if unit_test == UnitTests.PCA_R2:
        pca_r2 = compute_data_pca_r2(data=returns,
                                     freq='YE',
                                     ewm_lambda=0.97)
        print(pca_r2)


if __name__ == '__main__':

    unit_test = UnitTests.PCA_R2

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)