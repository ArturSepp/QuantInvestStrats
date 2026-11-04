"""Explicit-family HAC tests of means, regression alphas and paired Sharpe differences.

Mean/alpha calls fit one complete sample with ``estimate_hac_regression(span=None)``.
The intercept tests H0: mean=0 or H0: alpha=0 against a two-sided normal reference.
Bartlett HAC uses T/(T-k), with k=1 for means and k=1+number of factors for alphas.
Holm/BHY corrections apply to the whole supplied family; confidence intervals remain
unadjusted and pointwise. Inputs are fractional simple returns with caller-supplied
funding and costs. No cash subtraction, resampling, filling or sample selection occurs.

Columns are rescaled by their positive maximum absolute observation before estimation
to make numerical feasibility independent of units. Intercepts, SEs and bounds are then
restored to response units. Optional annual display multiplies these by af. Paired Sharpe
calls delegate to ``sharpe_difference_test`` on the same complete sample for every candidate
versus one fixed benchmark, preserving its uncorrected HAC and sqrt(af) display convention.
Annual display leaves statistics and p-values unchanged. This module does not select factors,
estimate cross-sectional prices of risk, or correct an unrecorded research search.

``InterceptInference`` selects the intercept standard error of the mean and alpha tests:
the equal-weight Bartlett sandwich (default) or the AR(1)-prewhitened Bartlett variance of
``qis.utils.hac_intercept_prewhitening``, which adds a ``rho`` column. The paired Sharpe
test has no such switch.
"""

# packages
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Dict, Optional
from scipy.stats import norm

# qis
from qis.perfstats import multiple_testing, sharpe_inference
from qis.perfstats.multiple_testing import MultipleTestingMethod
from qis.utils import regression
from qis.utils.hac_intercept_prewhitening import DEGENERACY_RTOL, estimate_prewhitened_intercepts
from qis.utils.rolling_regression import _validate_calendar


class InterceptInference(Enum):
    """Standard-error estimator for mean and alpha (intercept) hypotheses.

    Attributes:
        BARTLETT_HAC: Equal-weight Bartlett sandwich with the T/(T-k) correction.
        PREWHITENED_AR1: Scalar AR(1) prewhitening of each intercept influence series with
            a recoloured Bartlett variance; requires hac_lags <= T-2 and reports rho.
    """

    BARTLETT_HAC = 'bartlett_hac'
    PREWHITENED_AR1 = 'prewhitened_ar1'


@dataclass(frozen=True)
class SignificanceResult:
    """A complete hypothesis family with explicit inference and sample conventions.

    Attributes:
        table: Hypothesis-indexed estimate, hac_se, statistic, pvalue_raw,
            pvalue_adjusted, reject, ci_lower_pointwise and ci_upper_pointwise columns.
            Estimates/SEs/bounds share the display units; statistics and p-values do not
            change with annual display. Table attrs label units and pointwise intervals.
            Sharpe tables also include sharpe_candidate and sharpe_benchmark, with estimate
            equal to candidate minus benchmark (subject to the paired test's zero convention).
            Prewhitened mean/alpha tables add the fitted AR(1) coefficient rho.
        metadata: Family/correction settings; test, zero null and alternative; normal
            reference; sample dates and size; calendar; factor columns; HAC lags,
            inference and confidence level; simple-return/funding/cost descriptions; af
            and output units. Sharpe tables record the fixed benchmark name and population-moment
            convention. A custom expected_index is copied, not inferred from observations.
    """

    table: pd.DataFrame
    metadata: Dict[str, object]


def _real_setting(value: float, name: str) -> float:
    """Normalize a finite real scalar without accepting bool or numeric strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number, got {value!r}')
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exception:
        raise ValueError(f'invalid {name}, got {value!r}') from exception
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return result


def _finite_return_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate numeric, labelled returns without sorting, filling or deleting rows."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f'{name} must be a nonempty DataFrame')
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].tolist()
        raise ValueError(f'{name} columns must be unique, got {duplicates!r}')
    invalid_labels = [label for label in frame.columns
                      if not multiple_testing._valid_hypothesis_label(label)]
    if invalid_labels:
        raise ValueError(f'{name} has missing or empty column labels: {invalid_labels!r}')
    invalid_types = [label for label, dtype in frame.dtypes.items()
                     if (not pd.api.types.is_numeric_dtype(dtype)
                         or pd.api.types.is_complex_dtype(dtype)
                         or pd.api.types.is_bool_dtype(dtype))]
    if invalid_types:
        raise ValueError(f'{name} must contain real numeric returns: {invalid_types!r}')
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    invalid = ~np.isfinite(values).all(axis=0)
    if invalid.any():
        raise ValueError(f'{name} must be complete and finite; invalid columns: '
                         f'{frame.columns[invalid].tolist()!r}')
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def _compute_intercept_significance(
        returns: pd.DataFrame,
        factors: Optional[pd.DataFrame],
        family_id: str,
        return_basis: str,
        cost_basis: str,
        sampling_frequency: Optional[str],
        expected_index: Optional[pd.DatetimeIndex],
        hac_lags: Optional[int],
        confidence_level: float,
        method: MultipleTestingMethod,
        significance_level: float,
        af: Optional[float],
        inference: InterceptInference,
) -> SignificanceResult:
    """Validate a common sample, fit once, and assemble its intercept hypothesis family."""
    if not isinstance(inference, InterceptInference):
        raise ValueError(f'inference must be an InterceptInference member, got {inference!r}')
    prewhitened = inference is InterceptInference.PREWHITENED_AR1
    for name, value in (('return_basis', return_basis), ('cost_basis', cost_basis)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonblank description, got {value!r}')
    confidence_level = _real_setting(confidence_level, 'confidence_level')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f'confidence_level must be between zero and one, got {confidence_level}')
    annualised = af is not None
    display_factor = 1.0 if af is None else _real_setting(af, 'af')
    if display_factor <= 0.0:
        raise ValueError(f'af must be positive, got {af!r}')
    y = _finite_return_frame(returns, 'returns')
    if sampling_frequency is None and expected_index is None:
        raise ValueError('declare sampling_frequency or expected_index explicitly')
    if sampling_frequency is not None and not isinstance(sampling_frequency, str):
        raise ValueError('sampling_frequency must be a string or None')
    _validate_calendar(y.index, sampling_frequency, expected_index)
    x = None if factors is None else _finite_return_frame(factors, 'factors')
    if x is not None:
        if not x.index.equals(y.index):
            raise ValueError('returns and factors must have identical indexes in the same order')
        overlap = y.columns.intersection(x.columns, sort=False)
        if not overlap.empty:
            raise ValueError(f'dependent factors cannot also be regressors: {overlap.tolist()!r}')
        if 'Intercept' in x.columns:
            raise ValueError('factor label Intercept is reserved for the fitted constant')
    nobs = len(y)
    n_parameters = 1 if x is None else 1 + x.shape[1]
    if nobs < 24 or nobs <= n_parameters:
        raise ValueError(f'require at least 24 observations and T > k; got T={nobs}, '
                         f'k={n_parameters}')
    lags = regression.newey_west_lag_rule(nobs) if hac_lags is None else hac_lags
    if (isinstance(lags, (bool, np.bool_)) or not isinstance(lags, Integral)
            or not 0 <= lags < nobs):
        raise ValueError(f'hac_lags must be an integer in [0, {nobs - 1}], got {hac_lags!r}')

    y_scale = np.max(np.abs(y.to_numpy()), axis=0)
    if np.any(y_scale == 0.0):
        raise ValueError(f'degenerate zero responses: {y.columns[y_scale == 0.0].tolist()!r}')
    normalized_y = y / y_scale
    normalized_x = None
    if x is not None:
        x_scale = np.max(np.abs(x.to_numpy()), axis=0)
        if np.any(x_scale == 0.0):
            raise ValueError(f'constant zero factors: {x.columns[x_scale == 0.0].tolist()!r}')
        normalized_x = x / x_scale
    fit = regression.estimate_hac_regression(
        y=normalized_y, x=normalized_x, span=None, hac_lags=int(lags),
        confidence_level=confidence_level,
    )
    design = np.ones((nobs, 1)) if normalized_x is None else np.column_stack(
        (np.ones(nobs), normalized_x.to_numpy()),
    )
    # Evaluate the fitted residuals only; estimation and covariance stay in the shared core.
    residuals = normalized_y.to_numpy() - design @ fit.coefficients.to_numpy()
    residual_rms = np.sqrt(np.mean(np.square(residuals), axis=0))
    coefficients = fit.coefficients.loc['Intercept'].to_numpy()
    standard_errors = fit.hac_se.loc['Intercept'].to_numpy()
    probabilities = fit.pvalues.loc['Intercept'].to_numpy()
    interval_lower = fit.confidence_lower.loc['Intercept'].to_numpy()
    interval_upper = fit.confidence_upper.loc['Intercept'].to_numpy()
    rho = None
    if prewhitened:
        # The normalized responses have unit scale, so these SEs share the fit's units.
        try:
            filtered = estimate_prewhitened_intercepts(
                normalized_y, normalized_x, hac_lags=int(lags), confidence_level=confidence_level,
            )
        except ValueError as exception:
            raise ValueError(f'prewhitened inference for family {family_id!r}: '
                             f'{exception}') from exception
        standard_errors = filtered.table['standard_error'].to_numpy()
        probabilities = filtered.table['pvalue_normal'].to_numpy()
        interval_lower = filtered.table['ci_lower_pointwise'].to_numpy()
        interval_upper = filtered.table['ci_upper_pointwise'].to_numpy()
        rho = filtered.table['rho'].to_numpy()
    invalid = ((residual_rms <= DEGENERACY_RTOL) | (standard_errors <= DEGENERACY_RTOL)
               | ~np.isfinite(residual_rms) | ~np.isfinite(standard_errors))
    if invalid.any():
        raise ValueError(f'degenerate residuals or HAC standard errors relative to response scale: '
                         f'{y.columns[invalid].tolist()!r}')

    adjustment = multiple_testing.adjust_pvalues(
        pd.Series(probabilities, index=y.columns),
        family_id=family_id, method=method, significance_level=significance_level,
    )
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        displayed = np.column_stack((
            coefficients, standard_errors,
            interval_lower, interval_upper,
        )) * y_scale[:, None] * display_factor
    if not np.isfinite(displayed).all() or np.any(displayed[:, 1] <= 0.0):
        raise ValueError('display units produce non-finite values or zero standard errors')
    units = ('annualised fractional simple return' if annualised
             else 'fractional simple return/period')
    table = pd.DataFrame({
        'estimate': displayed[:, 0],
        'hac_se': displayed[:, 1],
        'statistic': coefficients / standard_errors,
        'pvalue_raw': adjustment['pvalue_raw'].to_numpy(),
        'pvalue_adjusted': adjustment['pvalue_adjusted'].to_numpy(),
        'reject': adjustment['reject'].to_numpy(),
        'ci_lower_pointwise': displayed[:, 2],
        'ci_upper_pointwise': displayed[:, 3],
    }, index=y.columns.copy())
    if rho is not None:
        table['rho'] = rho
    table.attrs = dict(adjustment.attrs, confidence_level=confidence_level,
                       confidence_interval='unadjusted/pointwise', output_units=units,
                       intercept_inference=inference.value)
    metadata = dict(
        table.attrs, test='mean' if x is None else 'alpha', null_value=0.0,
        alternative='two-sided', reference_distribution='normal', nobs=fit.nobs,
        sample_start=y.index[0], sample_end=y.index[-1], sampling_frequency=sampling_frequency,
        expected_index=None if expected_index is None else expected_index.copy(),
        n_parameters=n_parameters, factor_columns=() if x is None else tuple(x.columns),
        hac_lags=fit.hac_lags, hac_lags_requested=hac_lags, span=None, weighting='equal',
        filter_nobs=fit.nobs - 1 if prewhitened else None,
        return_convention='simple', return_basis=return_basis, cost_basis=cost_basis,
        af=None if af is None else display_factor, annualised=annualised,
        degeneracy_rtol=DEGENERACY_RTOL,
    )
    return SignificanceResult(table=table, metadata=metadata)


def compute_mean_significance(
        returns: pd.DataFrame,
        family_id: str,
        return_basis: str,
        cost_basis: str,
        sampling_frequency: Optional[str] = None,
        expected_index: Optional[pd.DatetimeIndex] = None,
        hac_lags: Optional[int] = 3,
        confidence_level: float = 0.95,
        method: MultipleTestingMethod = MultipleTestingMethod.HOLM,
        significance_level: float = 0.05,
        af: Optional[float] = None,
        inference: InterceptInference = InterceptInference.BARTLETT_HAC,
) -> SignificanceResult:
    """Test each supplied return mean against zero using one common equal-weight sample.

    Args:
        returns: Complete finite fractional simple returns, one hypothesis per column,
            on ordered, unique dates. At least 24 observations are required.
        family_id: Identifier of the complete predeclared hypothesis family.
        return_basis: Description of funding/RF treatment already applied to the inputs.
        cost_basis: Description of gross/net costs already reflected in the inputs.
        sampling_frequency: Explicit regular calendar, e.g. 'ME'; mutually exclusive
            with expected_index. Attached index.freq alone is insufficient.
        expected_index: Exact custom observation calendar; no missing sessions are inferred.
        hac_lags: Integer in [0, T-1]; None selects newey_west_lag_rule(T). No silent cap.
        confidence_level: Confidence level of unadjusted, pointwise HAC intervals.
        method: Holm FWER control by default, or BHY FDR control.
        significance_level: Family-wise error/FDR target strictly between zero and one.
        af: Optional positive periods per year, explicitly chosen for the sampling grid.
            None displays per-period values; a supplied factor multiplies estimates,
            SEs and interval bounds linearly, without changing statistics or p-values.
        inference: BARTLETT_HAC (default) or PREWHITENED_AR1 standard errors for the
            means. Prewhitening requires hac_lags <= T-2 and adds a rho column.

    Returns:
        A SignificanceResult with mean inference, adjusted decisions and explicit metadata.

    Raises:
        ValueError: If inputs/settings are invalid or a family member has residual RMS
            or intercept SE no larger than 1e-12 of its maximum absolute observation.

    Notes:
        Inference requires stationary, weakly dependent observations, summable
        autocovariances and sufficient finite moments. HAC and multiplicity correction
        do not repair structural breaks, data leakage or omitted research trials.
    """
    return _compute_intercept_significance(
        returns, None, family_id, return_basis, cost_basis, sampling_frequency,
        expected_index, hac_lags, confidence_level, method, significance_level, af, inference,
    )


def compute_alpha_significance(
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        family_id: str,
        return_basis: str,
        cost_basis: str,
        sampling_frequency: Optional[str] = None,
        expected_index: Optional[pd.DatetimeIndex] = None,
        hac_lags: Optional[int] = 3,
        confidence_level: float = 0.95,
        method: MultipleTestingMethod = MultipleTestingMethod.HOLM,
        significance_level: float = 0.05,
        af: Optional[float] = None,
        inference: InterceptInference = InterceptInference.BARTLETT_HAC,
) -> SignificanceResult:
    """Test intercepts in supplied multi-factor regressions with common-sample HAC.

    The model is Y = alpha + X beta + residual, with an intercept and all supplied
    factors in every equation. It tests alphas, not cross-sectional prices of factor risk.
    Regressors must be exogenous and full rank; responses cannot also be regressors.
    No full-sample centering, factor selection, cash subtraction or cost deduction is added.

    Args:
        returns: Complete finite fractional simple returns, one hypothesis per column.
            At least 24 observations and T > 1+number of factors are required.
        factors: Nonempty real-numeric factor-return DataFrame on exactly the same dates.
            No constant, duplicate or linearly dependent columns; Intercept is reserved.
        family_id: Identifier of the complete predeclared alpha-test family.
        return_basis: Description of funding/RF treatment already applied to inputs.
        cost_basis: Description of gross/net costs already reflected in inputs.
        sampling_frequency: Explicit regular observation calendar, e.g. 'ME'.
        expected_index: Exact custom calendar, mutually exclusive with sampling_frequency.
        hac_lags: Integer in [0, T-1]; None uses newey_west_lag_rule(T), without truncation.
        confidence_level: Confidence level of unadjusted, pointwise HAC intervals.
        method: Holm FWER control by default, or BHY FDR control.
        significance_level: Family-wise error/FDR target strictly between zero and one.
        af: Optional explicit periods per year for linear annual display of alpha, SE
            and confidence bounds. None retains per-period units; p-values do not change.
        inference: BARTLETT_HAC (default) or PREWHITENED_AR1 standard errors for the
            alphas. Prewhitening filters each alpha's scalar influence series, not the
            joint score; it requires hac_lags <= T-2 and adds a rho column.

    Returns:
        A SignificanceResult containing intercept inference and family/sample metadata.

    Raises:
        ValueError: If alignment/calendar/settings or the design are invalid, or any
            response has residual RMS or intercept SE at most 1e-12 of its input scale.

    Notes:
        HAC assumes stationary, weakly dependent regression scores with summable
        autocovariances and sufficient finite moments. Endogenous regressors, breaks,
        selection and an unrecorded research search remain outside these tests.
    """
    if factors is None:
        raise ValueError('factors must be a nonempty DataFrame; use the mean test without factors')
    return _compute_intercept_significance(
        returns, factors, family_id, return_basis, cost_basis, sampling_frequency,
        expected_index, hac_lags, confidence_level, method, significance_level, af, inference,
    )


def compute_sharpe_significance(
        returns: pd.DataFrame,
        benchmark: pd.Series,
        family_id: str,
        return_basis: str,
        cost_basis: str,
        sampling_frequency: Optional[str] = None,
        expected_index: Optional[pd.DatetimeIndex] = None,
        hac_lags: Optional[int] = 3,
        confidence_level: float = 0.95,
        method: MultipleTestingMethod = MultipleTestingMethod.HOLM,
        significance_level: float = 0.05,
        af: Optional[float] = None,
) -> SignificanceResult:
    """Test candidate-minus-benchmark arithmetic Sharpe differences as one paired family.

    Each row delegates to sharpe_difference_test on the same complete sample. The null
    is SR_candidate - SR_benchmark = 0, with a two-sided normal reference. Population
    moments (ddof=0), uncorrected Bartlett HAC and the identical/proportional-series
    convention are those of the existing test. No new Sharpe or covariance estimator is used.

    Args:
        returns: Complete finite fractional simple returns, one candidate per uniquely
            labelled column, on ordered unique dates. At least 24 rows are required.
        benchmark: One fixed numeric return Series with exactly the same dates and return
            basis. Its optional name is recorded. It may also appear among the candidates;
            identical or positively proportional series retain z=0, p=1, difference=SE=0.
        family_id: Identifier of the complete predeclared candidate-versus-benchmark family.
        return_basis: Common funding/RF treatment already applied to all supplied returns.
        cost_basis: Description of gross/net costs already reflected in all supplied returns.
        sampling_frequency: Explicit regular observation calendar, e.g. 'ME'; an attached
            index.freq alone is insufficient. Mutually exclusive with expected_index.
        expected_index: Exact custom observation calendar; no sessions are inferred or dropped.
        hac_lags: Integer in [0, T-1]; None selects newey_west_lag_rule(T), without truncation.
        confidence_level: Level of unadjusted pointwise normal intervals for the difference.
        method: Holm FWER control by default, or BHY FDR control over all supplied candidates.
        significance_level: Family-wise error/FDR target strictly between zero and one.
        af: Optional positive periods per year, passed to the existing paired test. None
            uses af=1 for per-period display; otherwise Sharpes, differences, SEs and bounds
            scale by sqrt(af). Statistics, p-values and sample size do not change.

    Returns:
        A SignificanceResult containing paired difference inference, both Sharpes, adjusted
        decisions, and explicit family, fixed-benchmark, sample and display metadata.

    Raises:
        ValueError: If the family/calendar/settings are invalid or any paired test fails.
            The whole family fails without pair-specific deletion or partial results.

    Notes:
        The paired delta method requires stationary, weakly dependent returns with sufficient
        finite moments, including fourth moments. Return-variance degeneracy checks remain
        scale-relative in sharpe_difference_test. Holm/BHY decisions do not turn the intervals
        into simultaneous bounds or correct unrecorded trials, repeated peeking or data leakage.
        No funding subtraction, cost deduction, resampling or all-pairs search occurs here.
    """
    for name, value in (('return_basis', return_basis), ('cost_basis', cost_basis)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonblank description, got {value!r}')
    confidence_level = _real_setting(confidence_level, 'confidence_level')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f'confidence_level must be between zero and one, got {confidence_level}')
    periods_per_year = 1.0 if af is None else _real_setting(af, 'af')
    if periods_per_year <= 0.0:
        raise ValueError(f'af must be positive, got {af!r}')
    y = _finite_return_frame(returns, 'returns')
    if sampling_frequency is None and expected_index is None:
        raise ValueError('declare sampling_frequency or expected_index explicitly')
    if sampling_frequency is not None and not isinstance(sampling_frequency, str):
        raise ValueError('sampling_frequency must be a string or None')
    _validate_calendar(y.index, sampling_frequency, expected_index)
    if not isinstance(benchmark, pd.Series):
        raise ValueError('benchmark must be one pandas Series')
    fixed = _finite_return_frame(benchmark.to_frame(name='benchmark'), 'benchmark').iloc[:, 0]
    fixed = fixed.rename(benchmark.name)
    if not fixed.index.equals(y.index):
        raise ValueError('returns and benchmark must have identical indexes in the same order')
    nobs = len(y)
    if nobs < 24:
        raise ValueError(f'require at least 24 observations, got T={nobs}')
    lags = regression.newey_west_lag_rule(nobs) if hac_lags is None else hac_lags
    if (isinstance(lags, (bool, np.bool_)) or not isinstance(lags, Integral)
            or not 0 <= lags < nobs):
        raise ValueError(f'hac_lags must be an integer in [0, {nobs - 1}], got {hac_lags!r}')

    paired_results = []
    for position, label in enumerate(y.columns):
        try:
            paired = sharpe_inference.sharpe_difference_test(
                returns1=y.iloc[:, position], returns2=fixed, n_lags=int(lags),
                af=periods_per_year,
            )
        except ValueError as exception:
            raise ValueError(f'candidate {label!r} versus benchmark {benchmark.name!r}: '
                             f'{exception}') from exception
        if paired.n_obs != nobs or paired.n_lags != lags:
            raise ValueError(f'paired test changed the common sample or lags for {label!r}')
        paired_results.append(paired)

    adjustment = multiple_testing.adjust_pvalues(
        pd.Series([paired.p_value for paired in paired_results], index=y.columns),
        family_id=family_id, method=method, significance_level=significance_level,
    )
    estimates = np.array([paired.diff_an for paired in paired_results])
    standard_errors = np.array([paired.se_an for paired in paired_results])
    critical = norm.ppf(0.5 + 0.5 * confidence_level)
    with np.errstate(over='ignore', invalid='ignore'):
        lower = estimates - critical * standard_errors
        upper = estimates + critical * standard_errors
    table = pd.DataFrame({
        'estimate': estimates, 'hac_se': standard_errors,
        'statistic': [paired.t_stat for paired in paired_results],
        'pvalue_raw': adjustment['pvalue_raw'].to_numpy(),
        'pvalue_adjusted': adjustment['pvalue_adjusted'].to_numpy(),
        'reject': adjustment['reject'].to_numpy(),
        'ci_lower_pointwise': lower, 'ci_upper_pointwise': upper,
        'sharpe_candidate': [paired.sr1_an for paired in paired_results],
        'sharpe_benchmark': [paired.sr2_an for paired in paired_results],
    }, index=y.columns.copy())
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError('Sharpe display or inference contains non-finite values')
    units = ('per-period arithmetic Sharpe difference' if af is None
             else 'annualised arithmetic Sharpe difference')
    table.attrs = dict(adjustment.attrs, confidence_level=confidence_level,
                       confidence_interval='unadjusted/pointwise', output_units=units)
    metadata = dict(
        table.attrs, test='sharpe_difference', null_value=0.0, alternative='two-sided',
        reference_distribution='normal', comparison='candidate_minus_benchmark',
        benchmark_name=benchmark.name, paired=True, nobs=nobs,
        sample_start=y.index[0], sample_end=y.index[-1], sampling_frequency=sampling_frequency,
        expected_index=None if expected_index is None else expected_index.copy(),
        hac_lags=int(lags), hac_lags_requested=hac_lags, hac_correction='none',
        sharpe_convention='arithmetic mean/population standard deviation (ddof=0)',
        return_convention='simple', return_basis=return_basis, cost_basis=cost_basis,
        af=None if af is None else periods_per_year, annualised=af is not None,
    )
    return SignificanceResult(table=table, metadata=metadata)
