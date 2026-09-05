"""
statsmodels OLS wrappers: fit, extract alpha and beta, and render the fitted equation.

``fit_multivariate_ols`` regresses a Series on the columns of a frame and returns, in that order,
the prediction, the parameters and a formatted label of the fitted equation; ``fit_ols`` is the
array form. ``estimate_ols_alpha_beta`` reduces a fit to alpha, beta, R² and the conventional alpha
p-value, returning zeros with a warning rather than raising when the fit fails.
``estimate_ols_alpha_beta_hac`` returns the same point estimates together with a Bartlett-kernel
HAC standard error, p-value and confidence interval for alpha. ``estimate_hac_mean`` is the
constant-only case: a sample mean with the same Bartlett HAC inference and the one-parameter
small-sample correction, for a return series that has no regressor.
``estimate_ewma_alpha_beta_hac`` fits several dependent series on one common regressor using
exponentially weighted least squares and returns their joint Bartlett-HAC covariance.
``reg_model_params_to_str`` formats the fitted equation for a chart legend, and annualises the
intercept as expm1(a α) when
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
from scipy.stats import norm
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


@dataclass(frozen=True)
class EwmaAlphaBetaHacResult:
    """Joint EWMA-WLS alpha/beta estimates with Bartlett-HAC inference.

    Attributes:
        alpha: Weighted intercepts indexed by dependent-series name.
        beta: Weighted slopes indexed by dependent-series name.
        r_squared: Weighted coefficients of determination.
        alpha_pvalue: Normal-reference two-sided p-values for the intercepts.
        alpha_hac_se: Bartlett-HAC standard errors for the intercepts.
        alpha_confidence_interval: Intercept confidence bounds in ``Lower`` and ``Upper`` columns.
        parameter_covariance: Joint HAC covariance with ``(equation, parameter)`` axes.
        weights: Unnormalised EWMA objective weights aligned with the retained rows.
        ewm_lambda: Per-row EWMA decay, equal to ``1 - 2 / (span + 1)``.
        nobs: Number of rows retained in the common finite sample.
        effective_nobs: Kish effective sample size of ``weights``.
        hac_lags: Actual Bartlett lag count used after the sample-length cap.
        confidence_level: Two-sided confidence level used for the intercept intervals.
    """

    alpha: pd.Series
    beta: pd.Series
    r_squared: pd.Series
    alpha_pvalue: pd.Series
    alpha_hac_se: pd.Series
    alpha_confidence_interval: pd.DataFrame
    parameter_covariance: pd.DataFrame
    weights: pd.Series
    ewm_lambda: float
    nobs: int
    effective_nobs: float
    hac_lags: int
    confidence_level: float


@dataclass(frozen=True)
class HacMeanResult:
    """Sample mean with Bartlett-kernel HAC inference.

    Attributes:
        mean: Sample mean of the retained finite observations.
        hac_se: Bartlett-kernel HAC standard error of the mean with the ``T/(T-1)`` correction.
        pvalue: Normal-reference two-sided p-value of the mean against zero.
        confidence_interval: Lower and upper HAC confidence bounds for the mean.
        nobs: Number of retained finite observations.
    """

    mean: float
    hac_se: float
    pvalue: float
    confidence_interval: tuple[float, float]
    nobs: int


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


def estimate_ewma_alpha_beta_hac(
        x: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: pd.DataFrame,
        span: float = 36.0,
        hac_lags: int = 3,
        confidence_level: float = 0.95,
) -> EwmaAlphaBetaHacResult:
    """Fit joint EWMA-weighted alpha/beta regressions with Bartlett-HAC inference.

    For retained row ``t`` in a sample of length ``T``, the WLS objective weight is
    ``lambda ** (T - 1 - t)``, where ``lambda = 1 - 2 / (span + 1)``. The last retained row
    therefore has weight one. All equations share the same design, weights and common finite
    sample. Their HAC score vectors are stacked before covariance estimation, preserving the
    cross-equation covariance needed for exact linear attribution contrasts.

    The Bartlett covariance uses the same ``T / (T - k)`` small-sample correction as
    ``statsmodels.WLS(...).get_robustcov_results(cov_type='HAC', use_correction=True)`` with
    ``k=2``. P-values and confidence intervals use a normal reference distribution. This
    low-level estimator attaches no calendar meaning to rows and does not sort them: the caller
    controls their order and frequency.

    Args:
        x: One common explanatory series. A pandas index must exactly equal the index of ``y``.
        y: Dependent series, one uniquely named equation per column.
        span: EWMA span, strictly greater than one and with Kish effective size above two.
        hac_lags: Maximum number of Bartlett-kernel autocovariance lags.
        confidence_level: Two-sided confidence level for the alpha intervals.

    Returns:
        Joint EWMA-WLS point estimates, alpha inference, covariance and sample metadata.

    Raises:
        TypeError: If ``y`` is not a DataFrame or ``hac_lags`` is not an integer.
        ValueError: If inputs are misaligned or invalid, the common sample is too short, the
            weighted design is singular, or estimation produces non-finite outputs.
    """
    if not isinstance(y, pd.DataFrame):
        raise TypeError(f'y must be a pandas DataFrame, got {type(y).__name__}')
    if y.shape[1] == 0:
        raise ValueError('y must contain at least one dependent series')
    if not y.columns.is_unique:
        raise ValueError('y columns must be unique')
    if isinstance(hac_lags, (bool, np.bool_)) or not isinstance(
            hac_lags, (int, np.integer)
    ):
        raise TypeError(f'hac_lags must be an integer, got {hac_lags!r}')
    if hac_lags < 0:
        raise ValueError(f'hac_lags must be non-negative, got {hac_lags}')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f'confidence_level must be between zero and one, got {confidence_level}'
        )
    if isinstance(span, (bool, np.bool_)) or not isinstance(
            span, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f'span must be a finite number greater than one, got {span!r}')
    span_value = float(span)
    if not np.isfinite(span_value) or span_value <= 1.0:
        raise ValueError(f'span must be a finite number greater than one, got {span!r}')

    if isinstance(x, (pd.Series, pd.DataFrame)) and not x.index.equals(y.index):
        raise ValueError('x and y pandas indexes must be identical')
    try:
        x_values = np.asarray(x, dtype=float)
        y_values = y.to_numpy(dtype=float)
    except (TypeError, ValueError) as exception:
        raise ValueError('x and y must contain numeric values') from exception
    if x_values.ndim == 2 and x_values.shape[1] == 1:
        x_values = x_values[:, 0]
    if x_values.ndim != 1:
        raise ValueError(f'x must be one-dimensional, got shape {x_values.shape}')
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError(
            f'x and y must have the same number of rows, got {x_values.shape[0]} and '
            f'{y_values.shape[0]}'
        )

    finite_rows = np.isfinite(x_values) & np.isfinite(y_values).all(axis=1)
    x_values = x_values[finite_rows]
    y_values = y_values[finite_rows]
    retained_index = y.index[finite_rows]
    nobs = int(x_values.shape[0])
    n_parameters = 2
    if nobs <= n_parameters:
        raise ValueError(
            f'EWMA-WLS HAC estimation requires at least three common finite rows, got {nobs}'
        )

    ewm_lambda = 1.0 - 2.0 / (span_value + 1.0)
    weights_array = np.power(
        ewm_lambda,
        np.arange(nobs - 1, -1, -1, dtype=float),
    )
    weight_sum = float(weights_array.sum())
    effective_nobs = float(np.square(weight_sum) / np.square(weights_array).sum())
    if not np.isfinite(effective_nobs) or effective_nobs <= n_parameters:
        raise ValueError(
            'EWMA weights must have Kish effective sample size above two, '
            f'got {effective_nobs:.6g}'
        )

    design = np.column_stack((np.ones(nobs, dtype=float), x_values))
    weighted_design = np.sqrt(weights_array)[:, None] * design
    if np.linalg.matrix_rank(weighted_design) < n_parameters:
        raise ValueError('EWMA-WLS design is singular; x must vary in the retained sample')
    try:
        bread = np.linalg.inv(design.T @ (weights_array[:, None] * design))
        parameters = bread @ design.T @ (weights_array[:, None] * y_values)
    except np.linalg.LinAlgError as exception:
        raise ValueError('EWMA-WLS estimation failed') from exception

    residuals = y_values - design @ parameters
    scores = np.einsum(
        'ti,tj->tji',
        weights_array[:, None] * design,
        residuals,
    ).reshape(nobs, -1)
    meat = scores.T @ scores
    effective_hac_lags = min(int(hac_lags), nobs - 1)
    for lag in range(1, effective_hac_lags + 1):
        kernel_weight = 1.0 - lag / (effective_hac_lags + 1.0)
        lagged_cross_product = scores[lag:].T @ scores[:-lag]
        meat += kernel_weight * (lagged_cross_product + lagged_cross_product.T)
    joint_bread = np.kron(np.eye(y.shape[1]), bread)
    parameter_covariance_array = (
        nobs
        / (nobs - n_parameters)
        * joint_bread
        @ meat
        @ joint_bread
    )
    parameter_covariance_array = 0.5 * (
        parameter_covariance_array + parameter_covariance_array.T
    )

    weighted_mean = np.sum(weights_array[:, None] * y_values, axis=0) / weight_sum
    weighted_tss = np.sum(
        weights_array[:, None] * np.square(y_values - weighted_mean),
        axis=0,
    )
    if np.any(weighted_tss <= 0.0):
        constant_columns = y.columns[weighted_tss <= 0.0].tolist()
        raise ValueError(f'y series must vary in the retained sample, got {constant_columns}')
    r_squared_array = 1.0 - np.sum(
        weights_array[:, None] * np.square(residuals),
        axis=0,
    ) / weighted_tss

    alpha_array = parameters[0]
    beta_array = parameters[1]
    alpha_positions = np.arange(0, 2 * y.shape[1], 2)
    alpha_variances = np.diag(parameter_covariance_array)[alpha_positions]
    variance_tolerance = 100.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(np.diag(parameter_covariance_array)))),
    )
    if np.any(alpha_variances < -variance_tolerance):
        raise ValueError('EWMA-WLS HAC estimation produced negative alpha variances')
    alpha_hac_se_array = np.sqrt(np.maximum(alpha_variances, 0.0))
    alpha_z_scores = np.divide(
        np.abs(alpha_array),
        alpha_hac_se_array,
        out=np.full_like(alpha_array, np.inf),
        where=alpha_hac_se_array > 0.0,
    )
    alpha_z_scores[(alpha_hac_se_array == 0.0) & (alpha_array == 0.0)] = 0.0
    alpha_pvalue_array = 2.0 * norm.sf(alpha_z_scores)
    critical_value = float(norm.ppf(0.5 + 0.5 * confidence_level))
    lower = alpha_array - critical_value * alpha_hac_se_array
    upper = alpha_array + critical_value * alpha_hac_se_array

    numeric_outputs = np.concatenate((
        parameters.ravel(),
        r_squared_array,
        alpha_pvalue_array,
        alpha_hac_se_array,
        lower,
        upper,
        parameter_covariance_array.ravel(),
    ))
    if not np.isfinite(numeric_outputs).all():
        raise ValueError('EWMA-WLS HAC estimation produced non-finite outputs')

    equation_index = pd.Index(y.columns, name='equation')
    parameter_index = pd.MultiIndex.from_product(
        [equation_index, ['Intercept', 'Beta']],
        names=['equation', 'parameter'],
    )
    return EwmaAlphaBetaHacResult(
        alpha=pd.Series(alpha_array, index=equation_index, name='Alpha'),
        beta=pd.Series(beta_array, index=equation_index, name='Beta'),
        r_squared=pd.Series(r_squared_array, index=equation_index, name='R-squared'),
        alpha_pvalue=pd.Series(alpha_pvalue_array, index=equation_index, name='Alpha p-value'),
        alpha_hac_se=pd.Series(alpha_hac_se_array, index=equation_index, name='Alpha HAC SE'),
        alpha_confidence_interval=pd.DataFrame(
            {'Lower': lower, 'Upper': upper},
            index=equation_index,
        ),
        parameter_covariance=pd.DataFrame(
            parameter_covariance_array,
            index=parameter_index,
            columns=parameter_index,
        ),
        weights=pd.Series(weights_array, index=retained_index, name='EWMA weight'),
        ewm_lambda=ewm_lambda,
        nobs=nobs,
        effective_nobs=effective_nobs,
        hac_lags=effective_hac_lags,
        confidence_level=confidence_level,
    )


def estimate_hac_mean(
        y: Union[np.ndarray, pd.Series],
        hac_lags: int = 3,
        confidence_level: float = 0.95,
) -> HacMeanResult:
    """Estimate a sample mean with Bartlett-kernel HAC inference.

    The mean is the intercept of a regression on a constant only, so the covariance is the
    Bartlett HAC covariance of the demeaned series with the one-parameter small-sample correction
    ``T/(T-1)`` and a normal reference distribution. Use it for the mean of a return series that
    has no regressor, such as a total-return component; ``estimate_ols_alpha_beta_hac`` is the
    two-parameter case with a benchmark. The requested lag count is capped at one less than the
    number of retained observations.

    Args:
        y: Observations; non-finite values are removed.
        hac_lags: Maximum number of Bartlett-kernel autocovariance lags.
        confidence_level: Two-sided confidence level for the interval.

    Returns:
        The mean, its HAC standard error, p-value, interval and the retained sample size.

    Raises:
        ValueError: If the HAC settings are invalid, fewer than two finite observations remain,
            estimation fails, or outputs are non-finite.
    """
    if hac_lags < 0:
        raise ValueError(f'hac_lags must be non-negative, got {hac_lags}')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f'confidence_level must be between zero and one, got {confidence_level}')
    values = np.asarray(y, dtype=float).ravel()
    values = values[np.isfinite(values)]
    nobs = int(values.shape[0])
    if nobs < 2:
        raise ValueError(
            f'HAC mean estimation requires at least two finite observations, got {nobs}'
        )
    try:
        reg_model = sm.OLS(values, np.ones((nobs, 1), dtype=float)).fit()
        robust_model = reg_model.get_robustcov_results(
            cov_type='HAC',
            maxlags=min(hac_lags, nobs - 1),
            use_correction=True,
            use_t=False,
        )
        mean = float(robust_model.params[0])
        hac_se = float(robust_model.bse[0])
        pvalue = float(robust_model.pvalues[0])
        interval = np.asarray(robust_model.conf_int(alpha=1.0 - confidence_level), dtype=float)[0]
    except Exception as exception:
        raise ValueError('HAC mean estimation failed') from exception
    if not np.isfinite([mean, hac_se, pvalue, *interval]).all():
        raise ValueError('HAC mean estimation produced non-finite outputs')
    return HacMeanResult(
        mean=mean,
        hac_se=hac_se,
        pvalue=pvalue,
        confidence_interval=(float(interval[0]), float(interval[1])),
        nobs=nobs,
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
