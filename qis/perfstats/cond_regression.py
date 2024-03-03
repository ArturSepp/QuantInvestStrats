# buit in
import numpy as np
import pandas as pd
import statsmodels.api as sm
from enum import Enum
from typing import Dict

# qis
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs, BenchmarkReturnsQuantilesRegime


RESID_VAR_COLUMN = 'An var resid'
ALPHA_COLUMN = 'Alpha'
COLOR_COLUMN = 'Color'
PREDICTION = 'Prediction'
VAR = 'Var'


def estimate_cond_regression(prices: pd.DataFrame,
                             benchmark: str,
                             regime_params: BenchmarkReturnsQuantileRegimeSpecs = None,
                             is_print_summary: bool = False,
                             drop_benchmark: bool = True,  # in some regressions we need to keep the benchmark inside
                             min_period: int = 0,  # min number of return in y
                             is_add_alpha: bool = False,
                             regmodel_out_dict: Dict[str, pd.DataFrame] = None
                             ) -> pd.DataFrame:

    """
    estimate beta params for multiple assets
    sampled_returns_with_regime include regime columns
    assets may have nan, if all returns are non -> zero is returnded
    important is that benchmark is fist: if it fails there, the benchmark is has nons which is badly defined
    """
    if regime_params is None:
        regime_params = BenchmarkReturnsQuantileRegimeSpecs()
    regime_classifier = BenchmarkReturnsQuantilesRegime(regime_params=regime_params)
    sampled_returns_with_regime_id = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices,
                                                                                              benchmark=benchmark,
                                                                                              **regime_params._asdict())
    sampled_returns_with_regime_id = sampled_returns_with_regime_id.dropna(how='all', axis=0)
    regime_groups = sampled_returns_with_regime_id.groupby([BenchmarkReturnsQuantilesRegime.REGIME_COLUMN], observed=False)
    # create prediction matrix
    regression_datas = []
    for regime in regime_classifier.get_regime_ids():
        # regime-conditional benchmark data will be benchmark returns renamed by regime name
        benchmark_data = regime_groups.get_group((regime, ))[benchmark].rename(regime)
        regression_datas.append(benchmark_data)

    # now x is 3 column manrix with bear normal bull columns
    regression_matrix = pd.concat(regression_datas, axis=1).fillna(0)
    if is_add_alpha:
        regression_matrix_1 = sm.add_constant(regression_matrix).rename(columns={'const': ALPHA_COLUMN})
    else:
        regression_matrix_1 = regression_matrix

    # drop regime column to leave only assets
    if drop_benchmark:
        asset_returns = sampled_returns_with_regime_id.drop(columns=[BenchmarkReturnsQuantilesRegime.REGIME_COLUMN, benchmark])
    else:
        asset_returns = sampled_returns_with_regime_id.drop(columns=[BenchmarkReturnsQuantilesRegime.REGIME_COLUMN])

    regime_colors = regime_classifier.class_data_to_colors(sampled_returns_with_regime_id[BenchmarkReturnsQuantilesRegime.REGIME_COLUMN]).rename(COLOR_COLUMN)
    model_params = []
    for asset in asset_returns:  # fit model

        regression_matrix_a = regression_matrix_1
        y = asset_returns[asset]
        nan_ind = np.isnan(y.to_numpy())
        if np.all(nan_ind):
            if len(model_params) == 0:
                raise ValueError(f"the first y has all nans, which must be benchmark")
            else:
                estimated_model_params = pd.Series(data=np.nan, index=model_params[0].index)
        else:
            if np.sum(nan_ind == False) < min_period: # skip estimation
                estimated_model_params = pd.Series(data=np.nan, index=model_params[0].index)
            else:
                if np.any(nan_ind):
                    y = y.loc[nan_ind==False]
                # align with y index
                regression_matrix_a = regression_matrix_1.loc[y.index, :]
                model = sm.OLS(y, regression_matrix_a)
                estimated_model = model.fit()

                if is_print_summary:
                    print(estimated_model.summary())
                if regmodel_out_dict is not None:
                    prediction = estimated_model.predict(regression_matrix_1)
                    pandas_out = pd.concat([sampled_returns_with_regime_id[BenchmarkReturnsQuantilesRegime.REGIME_COLUMN],
                                            sampled_returns_with_regime_id[benchmark],
                                            regression_matrix,
                                            y,
                                            prediction.rename(PREDICTION),
                                            regime_colors
                                            ], axis=1)
                    regmodel_out_dict[asset] = pandas_out

                estimated_model_params = estimated_model.params.copy()
                estimated_model_params[RESID_VAR_COLUMN] = estimated_model.mse_resid

        # append to all assets params
        model_params.append(estimated_model_params.rename(asset))

    model_params = pd.concat(model_params, axis=1).T

    return model_params


def get_regime_regression_params(prices: pd.DataFrame,
                                 regime_params: BenchmarkReturnsQuantileRegimeSpecs,
                                 benchmark: str,
                                 is_print_summary: bool = True,
                                 drop_benchmark: bool = False,
                                 min_period: int = 0,
                                 ) -> pd.DataFrame:

    estimated_params = estimate_cond_regression(prices=prices,
                                                benchmark=benchmark,
                                                regime_params=regime_params,
                                                is_print_summary=is_print_summary,
                                                drop_benchmark=drop_benchmark,
                                                min_period=min_period)
    return estimated_params


class UnitTests(Enum):
    REGRESSION = 1


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    print(prices)

    if unit_test == UnitTests.REGRESSION:

        estimated_params = get_regime_regression_params(prices=prices,
                                                        regime_params=BenchmarkReturnsQuantileRegimeSpecs(),
                                                        benchmark='SPY',
                                                        drop_benchmark=True,
                                                        is_print_summary=True)
        print(estimated_params)


if __name__ == '__main__':

    unit_test = UnitTests.REGRESSION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
