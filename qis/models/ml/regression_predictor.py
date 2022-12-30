"""
factor prediction
estimate factor model:

y_t = B^T * x_t

x_t is vector of predictors
y_t is vector of tradable factors
B is factor loading
the goal is to compute the prediction of y_t given y_t and estimated B
factors loadings B are not stored

x_t and y_t are of type vardata with defined normalization

x_t and y_t most typicall with freq='B'
model is updated as freq >= 'B' say month
the data used is daily up ast period = monthth and forecasted using daily freq for current months
"""
# built in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import NamedTuple, Optional, Tuple, Union, Dict, Any

# sklearn
import xgboost as xgb
from sklearn.linear_model import ElasticNet, Lasso, Ridge, MultiTaskLasso
from sklearn.multioutput import MultiOutputRegressor
from group_lasso import GroupLasso

# qis
import qis.utils.np_ops as npo
import qis.utils.sampling as sut
import qis.models.ml.var_data as vd
import qis.models.linear.ewm as ewm
from qis.models.ml.var_data import VarData


class RegModel(Enum):
    LASSO = 1
    MULTI_TASK_LASSO = 2
    GROUP_LASSO = 3
    RIDGE = 4
    ELASTIC_NET = 5
    XGBOOST = 6


class PredictionType(Enum):
    SIMPLE = 1
    Y_WITH_ALPHA = 2


def estimate_y_prediction(x_data: pd.DataFrame,  # factor
                          y_data: pd.DataFrame,  # yvars variable
                          alpha: float = 0.001,
                          test_period_x: Union[pd.DataFrame, np.ndarray] = None,
                          reg_model: RegModel = RegModel.LASSO,
                          fit_intercept: bool = True,
                          estimation_ewm_lambda: Optional[float] = 0.7,
                          is_output_params: bool = False,
                          xgboost_dict: Dict[str, Any] = None
                          ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    estimate time series ml y_{t} = beta * x_{t}
    estimation_ewm_lambda defines weithing of the ml
    if test data is pased it is assumed that test_period_x_{t} is updated obesrvations for predicition of y
    prediction is not weighted with lambda
    """
    X = x_data.to_numpy()
    y = y_data.to_numpy()

    if y.ndim == 1:  # make it column array, -1 is refered as the num of elements tin original row vector
        y = y.reshape((-1, ))
    else:  # maltivar display is not possible
        is_display_stats = False

    if reg_model == RegModel.LASSO:
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept)

    elif reg_model == RegModel.MULTI_TASK_LASSO:
        model = MultiTaskLasso(alpha=alpha, fit_intercept=fit_intercept)

    elif reg_model == RegModel.GROUP_LASSO:
        model = GroupLasso(groups=None,
                           group_reg=5,
                           l1_reg=alpha,
                           frobenius_lipschitz=True,
                           scale_reg="inverse_group_size",
                           subsampling_scheme=1,
                           supress_warning=True,
                           n_iter=1000,
                           tol=1e-3)

    elif reg_model == RegModel.RIDGE:
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    elif reg_model == RegModel.ELASTIC_NET:
        model = ElasticNet(alpha=alpha, fit_intercept=fit_intercept)

    elif reg_model == RegModel.XGBOOST:
        base_model = xgb.XGBRegressor(**xgboost_dict)
        model = MultiOutputRegressor(base_model)
    else:
        raise ValueError(f"{reg_model} not implemented")

    if estimation_ewm_lambda is not None:
        weights = npo.compute_expanding_power(n=len(x_data.index),
                                              power_lambda=np.sqrt(estimation_ewm_lambda),
                                              is_reverse=True)
        X = np.dot(np.diag(weights), X)
        y = np.dot(np.diag(weights), y)

    model.fit(X, y)
    if is_output_params:
        # shape of params = (len(y.columns), len(x.columns)
        # transpose to report as factor * assets
        if reg_model == RegModel.XGBOOST:
            params = 0.0
        elif reg_model == RegModel.GROUP_LASSO:
            params = model.coef_
        else:
            params = np.transpose(model.coef_)
        output_params = pd.DataFrame(data=params, index=x_data.columns, columns=y_data.columns)
    else:
        output_params = None

    if isinstance(test_period_x, pd.DataFrame):
        xtest = test_period_x.to_numpy()
        ypred = model.predict(xtest)
        ypred = pd.DataFrame(data=ypred, index=test_period_x.index, columns=y_data.columns)

    elif isinstance(test_period_x, np.ndarray):
        if test_period_x.ndim == 1:  # nd.array need to be transposed to row arras
            if test_period_x is None:
                xtest = X[-1].reshape((1, -1))  #  take last value (single sample) in x and reshape to row array
            else:
                xtest = test_period_x.reshape((1, -1))
        else:
            xtest = test_period_x
        ypred = model.predict(xtest)

    else:
        ypred = None

    return ypred, output_params


class RegressionPredictor:

    class MethodParams(NamedTuple):
        model_update_freq: str = 'M'
        data_freq: Optional[str] = 'B' # default
        alpha: float = 0.001
        fit_intercept: bool = False
        x_lag: Optional[int] = 1  # defines lag estimation
        roll_period: int = 5*12 # defines sample period for model estimation
        reg_model: RegModel = RegModel.LASSO
        prediction_ewm_lambda: Optional[float] = 0.90
        xgboost_dict: Dict[str, Any] = None

    def __init__(self,
                 x_data: VarData,
                 y_data: VarData,
                 method_params: MethodParams):

        vd.validate_vardata(data=x_data)
        vd.validate_vardata(data=y_data)

        # todo: ensure alignment of X and Y
        self.X = x_data.to_var_data()
        self.Y = y_data.to_var_data()
        self.method_params = method_params

    def compute_prediction(self,
                           fit_intercept: Optional[bool] = None,
                           is_output_params: bool = False
                           ) -> Tuple[pd.DataFrame, Optional[Dict[pd.Timestamp, pd.DataFrame]]]:
        """
        estimate ml at given freq
        """
        if fit_intercept is None:
            fit_intercept = self.method_params.fit_intercept

        x_lag = self.method_params.x_lag

        train_live_samples = sut.split_to_train_live_samples(ts_index=self.X.index,
                                                             model_update_freq=self.method_params.model_update_freq,
                                                             roll_period=self.method_params.roll_period)

        agg_x = self.X
        agg_y = self.Y

        ypred_pds = []
        if is_output_params:
            output_params_dict = {}
        else:
            output_params_dict = None

        for date, validation_period in train_live_samples.train_live_dates.items():
            x_data = validation_period.train.locate(agg_x)
            y_data = validation_period.train.locate(agg_y)
            test_period_x = validation_period.live.locate(agg_x)

            if x_lag is not None and x_lag > 0:  # shift backward x var
                x_data = x_data.shift(x_lag).iloc[x_lag:, :]
                y_data = y_data.iloc[x_lag:, :]

            ypred, output_params = estimate_y_prediction(x_data=x_data,
                                                         y_data=y_data,
                                                         alpha=self.method_params.alpha,
                                                         fit_intercept=fit_intercept,
                                                         test_period_x=test_period_x,
                                                         reg_model=self.method_params.reg_model,
                                                         estimation_ewm_lambda=self.method_params.prediction_ewm_lambda,
                                                         is_output_params=is_output_params,
                                                         xgboost_dict=self.method_params.xgboost_dict)
            ypred_pds.append(ypred)
            if is_output_params:
                output_params_dict[date] = output_params

        y_prediction = pd.concat(ypred_pds, axis=0)
        y_prediction = y_prediction.loc[~y_prediction.index.duplicated(keep='last')]
        y_prediction = y_prediction.reindex(index=self.Y.index, method='ffill') # just in case

        return y_prediction, output_params_dict

    def compute_alpha_prediction(self,
                                 ewm_lambda: float = 0.94,
                                 prediction_type: PredictionType = PredictionType.Y_WITH_ALPHA,
                                 fit_intercept: bool = False,
                                 is_output_params: bool = False
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[pd.Timestamp, pd.DataFrame]]]:

        y_prediction, output_params_dict = self.compute_prediction(fit_intercept=fit_intercept,
                                                                   is_output_params=is_output_params)

        if prediction_type == PredictionType.SIMPLE:
            alpha_prediction = y_prediction
            ewm_r2 = alpha_prediction
        elif prediction_type == PredictionType.Y_WITH_ALPHA:
            alpha_prediction, ewm_r2 = ewm.compute_ewm_alpha_r2(y_data=self.Y, y_prediction=y_prediction, ewm_lambda=ewm_lambda)
            alpha_prediction = alpha_prediction.add(y_prediction)

        else:
            raise TypeError(f"not implemented")

        return alpha_prediction, ewm_r2, output_params_dict


class UnitTests(Enum):
    TEST = 1


def run_unit_test(unit_test: UnitTests):

    # data
    from qis.data.yf_data import load_etf_data
    prices = load_etf_data().dropna()
    returns = prices.asfreq('W-WED', method='ffill').pct_change()

    if unit_test == UnitTests.TEST:

        # define ml model
        x_data = VarData(data=returns.iloc[:, 0].to_frame(),
                         data_type=vd.DataType.RETURN,
                         normalization_type=vd.NormalizationType.INVERSE_VOL,
                         smoothing_type=vd.SmoothingType.EWMA)

        y_data = VarData(data=returns.iloc[:, 1:],
                         data_type=vd.DataType.RETURN,
                         normalization_type=vd.NormalizationType.INVERSE_VOL,
                         smoothing_type=vd.SmoothingType.EWMA)

        method_params = RegressionPredictor.MethodParams()
        regression_predictor = RegressionPredictor(x_data=x_data,
                                                   y_data=y_data,
                                                   method_params=method_params)

        print(regression_predictor.X)
        print(regression_predictor.Y)

        y_prediction = regression_predictor.compute_prediction()
        print(y_prediction)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.TEST

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
