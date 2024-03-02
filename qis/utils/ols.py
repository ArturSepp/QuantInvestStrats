"""
statsmodels wrappers for ols analysis
"""
# packages
import warnings
import numpy as np
import pandas as pd
from statsmodels import api as sm
from statsmodels.regression.linear_model import RegressionResults as RegModel
from typing import Tuple


def fit_ols(x: np.ndarray,
            y: np.ndarray,
            order: int = 1,
            fit_intercept: bool = True
            ) -> RegModel:
    """
    fit regression model
    """
    cond = np.logical_and(np.isfinite(x), np.isfinite(y))
    x, y = x[cond], y[cond]
    x1 = get_ols_x(x=x, order=order, fit_intercept=fit_intercept)
    reg_model = sm.OLS(y, x1).fit()
    return reg_model


def estimate_ols_alpha_beta(x: np.ndarray,
                            y: np.ndarray,
                            fit_intercept: bool = True
                            ) -> Tuple[float, float, float]:
    try:
        reg_model = fit_ols(x=x, y=y, order=1, fit_intercept=fit_intercept)
    except:
        warnings.warn(f"problem with x={x}, y={y}")
        return 0.0, 0.0, 0.0
    if fit_intercept:
        alpha = reg_model.params.iloc[0]
        beta = reg_model.params.iloc[1]
    else:
        alpha = 0.0
        beta = reg_model.params.iloc[0]
    r2 = reg_model.rsquared
    return alpha, beta, r2


def estimate_alpha_beta_paired_dfs(x: pd.DataFrame,
                                   y: pd.DataFrame,
                                   fit_intercept: bool = True
                                   ) -> Tuple[pd.Series, pd.Series]:
    """
    ols for paired x and y dfs, default axis=0
    """
    # align:
    x = x.dropna()
    y = y.dropna()
    y = y.loc[x.index, x.columns]
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    n_col = len(x.columns)
    alphas, betas = np.zeros(n_col), np.zeros(n_col)
    for idx in np.arange(n_col):
        alphas[idx], betas[idx], _ = estimate_ols_alpha_beta(x=x_np[:, idx],
                                                             y=y_np[:, idx],
                                                             fit_intercept=fit_intercept)
    alphas = pd.Series(alphas, index=x.columns)
    betas = pd.Series(betas, index=x.columns)
    return alphas, betas


def get_ols_x(x: np.ndarray, order: int, fit_intercept: bool = True) -> np.ndarray:
    """
    compute powers of x
    """
    if order == 0:
        x = np.ones_like(x)
        fit_intercept = False
    elif order == 1:
        x = x
    elif order == 2:
        x = np.column_stack((x, np.square(x)))
    elif order == 3:
        x2 = np.square(x)
        x = np.column_stack((x, x2, x*x2))
    elif order == 4:
        x2 = np.square(x)
        x = np.column_stack((x, x2, x*x2, x2*x2))
    else:
        raise ValueError(f"order = {order} is not implemnted")

    if fit_intercept:
        x = sm.add_constant(x)
    return x


def reg_model_params_to_str(reg_model: RegModel,
                            order: int,
                            r2_only: bool = False,
                            beta_format: str = '{0:+0.2f}',
                            alpha_format: str = '{0:+0.2f}',
                            fit_intercept: bool = True,
                            alpha_an_factor: float = None,
                            **kwargs
                            ) -> str:
    try:
        r2 = f", R\N{SUPERSCRIPT TWO}={reg_model.rsquared:.0%}"
    except:
        r2 = f", R\N{SUPERSCRIPT TWO}=0.0%"

    if r2_only:
        text_str = f" R\N{SUPERSCRIPT TWO}={reg_model.rsquared:.0%}"
    else:
        if fit_intercept:
            if alpha_an_factor is not None:
                # alpha = '{:+0.0%}'.format(alpha_an_factor*reg_model.params[0])
                alpha = '{:+0.0%}'.format(np.expm1(alpha_an_factor * reg_model.params[0]))
            else:
                alpha = alpha_format.format(reg_model.params[0])
            idx1 = 1
        else:
            alpha = ''
            idx1 = 0

        if order == 1:
            text_str = 'y=' + beta_format.format(reg_model.params[idx1]) + 'X' \
                       + alpha \
                       + f', R\N{SUPERSCRIPT TWO}=' + '{0:.0%}'.format(reg_model.rsquared)

        elif order == 2:
            if fit_intercept:  # with intercept
                text_str = 'y=' + beta_format.format(reg_model.params[idx1+1]) + 'X' + f'\N{SUPERSCRIPT TWO}' \
                            + beta_format.format(reg_model.params[idx1]) + 'X' \
                            + alpha + r2
            else :  # without intercept
                text_str = 'y=' + beta_format.format(reg_model.params[idx1+1]) + 'X' + f'\N{SUPERSCRIPT TWO}' \
                            + beta_format.format(reg_model.params[idx1]) + 'X' \
                            + alpha \
                            + f', R\N{SUPERSCRIPT TWO}=' + '{0:.0%}'.format(reg_model.rsquared)

        elif order == 3:
            try:
                text_str = 'y=' + beta_format.format(reg_model.params[idx1+2]) + 'x' + f'\N{SUPERSCRIPT THREE}' \
                           + beta_format.format(reg_model.params[idx1+1]) + 'x' + f'\N{SUPERSCRIPT TWO}' \
                           + beta_format.format(reg_model.params[idx1]) + 'x' \
                           + alpha \
                           + f', R\N{SUPERSCRIPT TWO}=' + '{0:.0%}'.format(reg_model.rsquared)
            except:
                text_str = 'model cannot be estimated'
        else:
            raise TypeError(f"order = {order} is not implemnted")

    return text_str
