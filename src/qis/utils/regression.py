"""
statsmodels OLS wrappers: fit, extract alpha and beta, and render the fitted equation.

``fit_multivariate_ols`` regresses a Series on the columns of a frame and returns, in that order,
the prediction, the parameters and a formatted label of the fitted equation; ``fit_ols`` is the
array form. ``estimate_ols_alpha_beta`` reduces a fit to alpha, beta, R² and the conventional alpha
p-value, returning zeros with a warning rather than raising when the fit fails.
``estimate_ols_alpha_beta_hac`` returns the same point estimates together with a Bartlett-kernel
HAC standard error, p-value and confidence interval for alpha. ``reg_model_params_to_str`` formats
the fitted equation for a chart legend, and annualises the intercept as expm1(a α) when
``alpha_an_factor`` is passed. ``newey_west_lag_rule`` supplies the opt-in Newey-West rule of thumb
for callers that do not want to select a fixed Bartlett lag count.

Every fit runs through ``filter_x_y`` first: rows where any of x or y is non-finite are dropped,
so a prediction is indexed on the surviving rows, not on the full input. ``order`` is a
polynomial degree in a single x, not a second regressor - ``get_ols_x`` stacks x, x², x³ and x⁴
up to order 4. These are static, full-sample fits; time-varying betas are in ``ewm.py``.
"""
# packages
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from statsmodels import api as sm
from statsmodels.regression.linear_model import RegressionResults as RegModel
from typing import Tuple, Union


@dataclass(frozen=True)
class OlsAlphaBetaHacResult:
    """Generic OLS alpha/beta estimates with HAC inference for the intercept.

    Attributes:
        alpha: OLS intercept.
        beta: OLS slope on the first explanatory variable.
        r_squared: Conventional OLS coefficient of determination.
        alpha_pvalue: Normal-reference p-value using the HAC intercept standard error.
        alpha_hac_se: Bartlett-kernel HAC standard error of the intercept.
        alpha_confidence_interval: Lower and upper HAC confidence bounds for the intercept.
    """

    alpha: float
    beta: float
    r_squared: float
    alpha_pvalue: float
    alpha_hac_se: float
    alpha_confidence_interval: tuple[float, float]


def newey_west_lag_rule(nobs: int) -> int:
    """Return the Newey-West (1994) rule-of-thumb Bartlett lag count.

    Args:
        nobs: Positive number of observations in the estimation sample.

    Returns:
        The floor of ``4 * (nobs / 100) ** (2 / 9)``.

    Raises:
        ValueError: If ``nobs`` is not positive.
    """
    if nobs < 1:
        raise ValueError(f'nobs must be positive, got {nobs}')
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def fit_multivariate_ols(x: pd.DataFrame,
                         y: pd.Series,
                         fit_intercept: bool = True,
                         verbose: bool = True,
                         beta_format: str = '{0:+0.2f}',
                         alpha_format: str = '{0:+0.2f}',
                         ) -> Tuple[pd.Series, pd.Series, str]:
    """
    ordinary least squares of y on the columns of x, with a formatted summary of the fit.

    Rows where any variable is missing are dropped before fitting, so the prediction is indexed on
    the rows that survived rather than on the full input.

    Args:
        x: explanatory variables, one column per regressor
        y: dependent variable, aligned with ``x``
        fit_intercept: include an intercept, reported under the name ``'intercept'``
        verbose: print the statsmodels summary
        beta_format: format for the slope coefficients in the returned label
        alpha_format: format for the intercept in the returned label

    Returns:
        the fitted parameters indexed by regressor name, the prediction indexed by the retained
        dates, and a formatted label of the fitted equation for a chart legend
    """

    x_, y_, cond = filter_x_y(x=x.to_numpy(), y=y.to_numpy())

    xname = x.columns.to_list()
    if fit_intercept:
        x_ = sm.add_constant(x_)
        xname = ['intercept'] + xname

    fitted_model = sm.OLS(y_, x_).fit()
    prediction = pd.Series(fitted_model.predict(x_), index=y.index[cond])
    if verbose:
        print(fitted_model.summary(yname=y.name or 'y', xname=xname))

    params = pd.Series(fitted_model.params, index=xname)
    try:
        r2 = f", R\N{SUPERSCRIPT TWO}={fitted_model.rsquared:.0%}"
    except (AttributeError, ValueError):
        r2 = ", R\N{SUPERSCRIPT TWO}=0.0%"
    if fit_intercept:
        reg_label = (
            "y="
            + alpha_format.format(params.iloc[0])
            + "".join([
                f"{beta_format.format(x)}*{key}"
                for key, x in params.iloc[1:].to_dict().items()
            ])
            + r2
        )
    else:
        reg_label = (
            "y="
            + "".join([
                f"{beta_format.format(x)}*{key}"
                for key, x in params.to_dict().items()
            ])
            + r2
        )
    return prediction, params, reg_label


def fit_ols(x: np.ndarray,
            y: np.ndarray,
            order: int = 1,
            fit_intercept: bool = True
            ) -> RegModel:
    """
    fit regression model
    """
    x, y, cond = filter_x_y(x=x, y=y)
    x1 = get_ols_x(x=x, order=order, fit_intercept=fit_intercept)
    reg_model = sm.OLS(y, x1).fit()
    return reg_model


def estimate_ols_alpha_beta_hac(
        x: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        hac_lags: int = 3,
        confidence_level: float = 0.95,
) -> OlsAlphaBetaHacResult:
    """Estimate OLS alpha/beta with Bartlett-kernel HAC inference for alpha.

    The point estimates and R² come from ordinary least squares with an intercept. Inference for
    alpha uses statsmodels' heteroskedasticity and autocorrelation-consistent covariance with the
    small-sample correction and a normal reference distribution. The requested lag count is capped
    at one less than the number of retained observations.

    Args:
        x: Explanatory variable; non-finite observations are removed jointly with ``y``.
        y: Dependent variable aligned with ``x``.
        hac_lags: Maximum number of Bartlett-kernel autocovariance lags.
        confidence_level: Two-sided confidence level for the alpha interval.

    Returns:
        Generic OLS estimates and HAC inference without performance-reporting labels.

    Raises:
        ValueError: If the HAC settings are invalid, estimation fails, or outputs are non-finite.
    """
    if hac_lags < 0:
        raise ValueError('hac_lags must be non-negative')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError('confidence_level must be between zero and one')
    try:
        reg_model = fit_ols(x=x, y=y)
        effective_hac_lags = min(hac_lags, int(reg_model.nobs) - 1)
        robust_model = reg_model.get_robustcov_results(
            cov_type='HAC',
            maxlags=effective_hac_lags,
            use_correction=True,
            use_t=False,
        )
        alpha = float(robust_model.params[0])
        beta = float(robust_model.params[1])
        r_squared = float(reg_model.rsquared)
        alpha_pvalue = float(robust_model.pvalues[0])
        alpha_hac_se = float(robust_model.bse[0])
        alpha_confidence_interval_array = np.asarray(
            robust_model.conf_int(alpha=1.0 - confidence_level),
            dtype=float,
        )[0]
    except Exception as exception:
        raise ValueError('OLS HAC estimation failed') from exception
    if not np.isfinite([
            alpha,
            beta,
            alpha_pvalue,
            alpha_hac_se,
            *alpha_confidence_interval_array,
    ]).all():
        raise ValueError('OLS HAC estimation produced non-finite outputs')
    alpha_confidence_interval = (
        float(alpha_confidence_interval_array[0]),
        float(alpha_confidence_interval_array[1]),
    )
    return OlsAlphaBetaHacResult(
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        alpha_pvalue=alpha_pvalue,
        alpha_hac_se=alpha_hac_se,
        alpha_confidence_interval=alpha_confidence_interval,
    )


def filter_x_y(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.ndim == 1:  # x is 1-dimensional
        cond = np.logical_and(np.isfinite(x), np.isfinite(y))
    else:
        cond = np.logical_and(np.isfinite(x[:, 0]), np.isfinite(y))
        for idx in np.arange(1, x.shape[1]):
            cond = np.logical_and(cond, np.isfinite(x[:, idx]))
    x, y = x[cond], y[cond]
    return x, y, cond


def estimate_ols_alpha_beta(x: Union[np.ndarray, pd.Series, pd.DataFrame],
                            y: Union[np.ndarray, pd.Series],
                            order: int = 1,
                            fit_intercept: bool = True
                            ) -> Tuple[float, float, float, float]:
    try:
        reg_model = fit_ols(x=x, y=y, order=order, fit_intercept=fit_intercept)
    except Exception:
        warnings.warn(f"problem with x={x}, y={y}")
        return 0.0, 0.0, 0.0, 0.0
    if fit_intercept:
        if isinstance(reg_model.params, pd.Series):
            alpha = reg_model.params.iloc[0]
            beta = reg_model.params.iloc[1]
            alpha_pvalue = reg_model.pvalues.iloc[0]
        else:
            alpha = reg_model.params[0]
            beta = reg_model.params[1]
            alpha_pvalue = reg_model.pvalues[0]
    else:
        alpha = 0.0
        alpha_pvalue = 0.0
        if isinstance(reg_model.params, pd.Series):
            beta = reg_model.params.iloc[0]
        else:
            beta = reg_model.params[0]
    r2 = reg_model.rsquared
    return alpha, beta, r2, alpha_pvalue


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
    except (AttributeError, ValueError):
        r2 = ", R\N{SUPERSCRIPT TWO}=0.0%"

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
            text_str = (
                'y='
                + beta_format.format(reg_model.params[idx1])
                + 'X'
                + alpha
                + ', R\N{SUPERSCRIPT TWO}='
                + '{0:.0%}'.format(reg_model.rsquared)
            )

        elif order == 2:
            if fit_intercept:  # with intercept
                text_str = (
                    'y='
                    + beta_format.format(reg_model.params[idx1+1])
                    + 'X\N{SUPERSCRIPT TWO}'
                    + beta_format.format(reg_model.params[idx1])
                    + 'X'
                    + alpha
                    + r2
                )
            else:  # without intercept
                text_str = (
                    'y='
                    + beta_format.format(reg_model.params[idx1+1])
                    + 'X\N{SUPERSCRIPT TWO}'
                    + beta_format.format(reg_model.params[idx1])
                    + 'X'
                    + alpha
                    + ', R\N{SUPERSCRIPT TWO}='
                    + '{0:.0%}'.format(reg_model.rsquared)
                )

        elif order == 3:
            try:
                text_str = (
                    'y='
                    + beta_format.format(reg_model.params[idx1+2])
                    + 'x\N{SUPERSCRIPT THREE}'
                    + beta_format.format(reg_model.params[idx1+1])
                    + 'x\N{SUPERSCRIPT TWO}'
                    + beta_format.format(reg_model.params[idx1])
                    + 'x'
                    + alpha
                    + ', R\N{SUPERSCRIPT TWO}='
                    + '{0:.0%}'.format(reg_model.rsquared)
                )
            except (AttributeError, IndexError, ValueError):
                text_str = 'model cannot be estimated'
        else:
            raise TypeError(f"order = {order} is not implemented")

    return text_str
