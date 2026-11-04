"""Null-centered, jointly resampled HAC mean and alpha hypothesis families.

Supplied inputs are complete periodic fractional simple returns on a declared calendar.
For means, Y0 = Y - mean(Y); for alphas, Y0 = Y - alpha_hat while factors remain unchanged.
One circular stationary-bootstrap index matrix, shaped (T, B), draws entire paired rows.
Each drawn sequence is refit with the shared equal-weight HAC estimator on the original
analysis grid, never sorted by sampled dates. Studentized two-sided tail counts give
p_boot = (1 + count(|stat_b| >= |stat_observed|)) / (B+1).

Holm/BHY correct these approximate bootstrap p-values. Observed normal HAC p-values and
unadjusted pointwise HAC intervals remain diagnostics; no bootstrap confidence interval is
constructed. Binomial tail-probability intervals describe Monte Carlo uncertainty conditional
on the data and independent draws, not uncertainty about the population effect or exact size.
No sampler, Sharpe test, max-t adjustment, strategy search or invalid-draw replacement is added.
With ``InterceptInference.PREWHITENED_AR1`` the observed sample and every draw use the
AR(1)-prewhitened intercept standard error, refitting rho in each draw.

The procedure is the studentized block bootstrap with HAC standard errors refitted in every
draw of Romano and Wolf (2006, J. Nonparametric Statistics 18(2)) and Ledoit and Wolf (2008,
J. Empirical Finance 15(5)), with a Bartlett rather than a quadratic-spectral kernel. The
family structure follows White (2000) and Romano and Wolf (2005, Econometrica 73(4)); qis
corrects the marginal p-values with Holm/BHY instead of their stepdown maximum statistic.
Specific to qis are the declared-family and provenance conventions and the abort policy.
"""

# packages
import hashlib
import warnings
import numpy as np
import pandas as pd
from numbers import Integral
from typing import Optional, Tuple
from statsmodels.stats.proportion import proportion_confint

# qis
from qis.models.bootstrap import bootstrap_numba
from qis.models.bootstrap.bootstrap_numba import BootstrapType
from qis.perfstats import multiple_testing, significance
from qis.perfstats.multiple_testing import MultipleTestingMethod
from qis.perfstats.significance import InterceptInference, SignificanceResult
from qis.utils import regression
from qis.utils.hac_intercept_prewhitening import estimate_prewhitened_intercepts


def _integer_setting(value: int, name: str, minimum: int = 1,
                     maximum: int = np.iinfo(np.int64).max) -> int:
    """Validate before passing integer settings to the existing Numba sampler."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not minimum <= value <= maximum):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}], got {value!r}')
    return int(value)


def _validate_indices(indices: np.ndarray, nobs: int,
                      num_samples: Optional[int]) -> np.ndarray:
    """Validate orientation, integer dtype and bounds before mapping any source rows."""
    if (not isinstance(indices, np.ndarray) or indices.ndim != 2
            or indices.shape[0] != nobs or indices.shape[1] == 0):
        raise ValueError(f'bootstrapped_indices must have nonempty shape ({nobs}, B)')
    if indices.dtype.kind not in 'iu':
        raise ValueError('bootstrapped_indices must have integer dtype, not bool/float/object')
    if num_samples is not None and indices.shape[1] != num_samples:
        raise ValueError('num_samples conflicts with supplied bootstrapped_indices columns')
    if np.any(indices < 0) or np.any(indices >= nobs):
        raise ValueError(f'bootstrapped_indices must be in [0, {nobs - 1}]')
    return np.array(indices, dtype=np.int64, order='C', copy=True)


def _draw_statistics(y: pd.DataFrame, x: Optional[pd.DataFrame], lags: int,
                     inference: InterceptInference, replicate: int,
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Refit one complete drawn sequence, aborting rather than replacing invalid draws.

    Returns the studentized intercepts and, for prewhitened inference, the AR(1)
    coefficients fitted to this draw; the observed rho is never reused.
    """
    scale = np.max(np.abs(y.to_numpy()), axis=0)
    tolerance = significance.DEGENERACY_RTOL * scale
    rho = None
    try:
        if inference is InterceptInference.PREWHITENED_AR1:
            # Degenerate residuals give zero influence, which the estimator rejects itself.
            filtered = estimate_prewhitened_intercepts(y, x, hac_lags=lags)
            se = filtered.table['standard_error'].to_numpy()
            statistics = filtered.table['statistic'].to_numpy()
            rho = filtered.table['rho'].to_numpy()
            degenerate = se <= tolerance
            nobs_fitted, lags_fitted = filtered.nobs, filtered.hac_lags
        else:
            fitted = regression.estimate_hac_regression(y=y, x=x, span=None, hac_lags=lags)
            design = (np.ones((len(y), 1)) if x is None
                      else np.column_stack((np.ones(len(y)), x)))
            residuals = y.to_numpy() - design @ fitted.coefficients.to_numpy()
            rms = np.sqrt(np.mean(residuals ** 2, axis=0))
            se = fitted.hac_se.loc['Intercept'].to_numpy()
            degenerate = (rms <= tolerance) | (se <= tolerance) | ~np.isfinite(rms)
            nobs_fitted, lags_fitted = fitted.nobs, fitted.hac_lags
    except ValueError as exception:
        raise ValueError(f'bootstrap replicate {replicate}; hypotheses {y.columns.tolist()!r}: '
                         f'{exception}') from exception
    invalid = (scale == 0.0) | degenerate | ~np.isfinite(se)
    if invalid.any():
        raise ValueError(f'bootstrap replicate {replicate}; invalid residuals/SE for hypotheses '
                         f'{y.columns[invalid].tolist()!r}')
    if inference is not InterceptInference.PREWHITENED_AR1:
        statistics = fitted.coefficients.loc['Intercept'].to_numpy() / se
    if not np.isfinite(statistics).all() or nobs_fitted != len(y) or lags_fitted != lags:
        raise ValueError(f'bootstrap replicate {replicate}; invalid statistics/sample for '
                         f'hypotheses {y.columns.tolist()!r}')
    return statistics, rho


def _bootstrap_significance(
        returns: pd.DataFrame, factors: Optional[pd.DataFrame], family_id: str,
        return_basis: str, cost_basis: str, sampling_frequency: Optional[str],
        expected_index: Optional[pd.DatetimeIndex], hac_lags: Optional[int],
        confidence_level: float, method: MultipleTestingMethod, significance_level: float,
        af: Optional[float], num_samples: Optional[int], block_size: int, seed: int,
        bootstrapped_indices: Optional[np.ndarray], include_diagnostics: bool,
        inference: InterceptInference,
) -> SignificanceResult:
    """Validate observed inference once, draw jointly and stream studentized tail counts."""
    if not isinstance(inference, InterceptInference):
        raise ValueError(f'inference must be an InterceptInference member, got {inference!r}')
    prewhitened = inference is InterceptInference.PREWHITENED_AR1
    if num_samples is not None:
        num_samples = _integer_setting(num_samples, 'num_samples')
    # Numba's geometric inverse CDF requires 1 - 1/block_size < 1 in float64.
    block_size = _integer_setting(block_size, 'block_size', maximum=2**53)
    seed = _integer_setting(seed, 'seed', minimum=0, maximum=np.iinfo(np.uint32).max)
    if not isinstance(include_diagnostics, (bool, np.bool_)):
        raise ValueError('include_diagnostics must be boolean')
    display_factor = 1.0 if af is None else significance._real_setting(af, 'af')
    if display_factor <= 0.0:
        raise ValueError('af must be positive')
    settings = dict(family_id=family_id, return_basis=return_basis, cost_basis=cost_basis,
                    sampling_frequency=sampling_frequency, expected_index=expected_index,
                    hac_lags=hac_lags, confidence_level=confidence_level, method=method,
                    significance_level=significance_level, af=None, inference=inference)
    observed = (significance.compute_mean_significance(returns, **settings) if factors is None
                else significance.compute_alpha_significance(returns, factors, **settings))
    nobs, members = observed.metadata['nobs'], observed.table.index
    lags = observed.metadata['hac_lags']
    supplied = bootstrapped_indices is not None
    if supplied:
        indices = _validate_indices(bootstrapped_indices, nobs, num_samples)
    else:
        count = 9999 if num_samples is None else num_samples
        indices = bootstrap_numba.generate_bootstrapped_indices(
            num_data_index=nobs, bootstrap_type=BootstrapType.STATIONARY,
            num_samples=count, index_length=nobs, block_size=block_size,
            min_block_size=1, seed=seed,
        )
        indices = _validate_indices(indices, nobs, count)
    count = indices.shape[1]

    # Work in each input column's positive scale to avoid overflow in centering/moments.
    # These scales cancel from studentized intercepts and do not change the null model.
    values = returns.to_numpy(dtype=float)
    y_scale = np.max(np.abs(values), axis=0)
    values = values / y_scale
    center = (values.mean(axis=0) if factors is None
              else observed.table['estimate'].to_numpy() / y_scale)
    null_values = values - center
    factor_values = None
    if factors is not None:
        factor_values = factors.to_numpy(dtype=float)
        factor_values = factor_values / np.max(np.abs(factor_values), axis=0)
    observed_statistics = np.abs(observed.table['statistic'].to_numpy())
    exceedances = np.zeros(len(members), dtype=np.int64)
    diagnostic_statistics = np.empty((count, len(members))) if include_diagnostics else None
    diagnostic_rho = (np.empty((count, len(members)))
                      if prewhitened and include_diagnostics else None)
    for replicate in range(count):
        rows = indices[:, replicate]
        draw_y = pd.DataFrame(null_values[rows], index=returns.index, columns=members)
        draw_x = None if factor_values is None else pd.DataFrame(
            factor_values[rows], index=returns.index, columns=factors.columns,
        )
        statistics, rho = _draw_statistics(draw_y, draw_x, lags, inference, replicate)
        exceedances += np.abs(statistics) >= observed_statistics
        if diagnostic_statistics is not None:
            diagnostic_statistics[replicate] = statistics
        if diagnostic_rho is not None:
            diagnostic_rho[replicate] = rho

    pvalues = (1.0 + exceedances) / (count + 1.0)
    adjusted = multiple_testing.adjust_pvalues(pd.Series(pvalues, index=members),
                                               family_id=family_id, method=method,
                                               significance_level=significance_level)
    tail_lower, tail_upper = proportion_confint(exceedances, count, alpha=0.05, method='beta')
    table = observed.table.copy(deep=True)
    table['pvalue_hac'] = table['pvalue_raw']
    for column in ('pvalue_raw', 'pvalue_adjusted', 'reject'):
        table[column] = adjusted[column].to_numpy()
    table['bootstrap_exceedances'] = exceedances
    table['bootstrap_tail_mc_lower'] = tail_lower
    table['bootstrap_tail_mc_upper'] = tail_upper
    display_columns = ['estimate', 'hac_se', 'ci_lower_pointwise', 'ci_upper_pointwise']
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        table[display_columns] = table[display_columns] * display_factor
    if not np.isfinite(table.to_numpy(dtype=float)).all() or (table['hac_se'] <= 0).any():
        raise ValueError('bootstrap display or inference contains non-finite values or zero SE')

    harmonic = float(np.sum(1.0 / np.arange(1, len(members) + 1)))
    penalty = 1.0 if method is MultipleTestingMethod.HOLM else harmonic
    first_threshold = adjusted.attrs['significance_level'] / (len(members) * penalty)
    minimum_pvalue = 1.0 / (count + 1.0)
    reachable = minimum_pvalue <= first_threshold
    if not reachable:
        qualifier = (' BHY can still reject at higher ranks.'
                     if method is MultipleTestingMethod.BHY else '')
        warnings.warn(f'B={count}: minimum bootstrap p-value {minimum_pvalue:.6g} exceeds '
                      f'{method.name} first-rank threshold {first_threshold:.6g}.{qualifier}',
                      RuntimeWarning, stacklevel=3)
    units = ('fractional simple return/period' if af is None
             else 'annualised fractional simple return')
    interval_method = 'normal prewhitened HAC' if prewhitened else 'normal HAC'
    table.attrs = dict(adjusted.attrs, confidence_level=observed.metadata['confidence_level'],
                       confidence_interval='unadjusted/pointwise', interval_method=interval_method,
                       output_units=units, pvalue_source='bootstrap',
                       intercept_inference=inference.value)
    metadata = dict(observed.metadata)
    metadata.update(table.attrs)
    metadata.update(
        test='bootstrap_mean' if factors is None else 'bootstrap_alpha',
        reference_distribution='null-centered joint bootstrap',
        inference='approximate studentized bootstrap with finite-resample adjustment',
        null_center='sample_mean' if factors is None else 'fitted_alpha',
        num_samples=count, num_samples_requested=num_samples, bootstrap_index_shape=indices.shape,
        index_source='supplied' if supplied else 'generated',
        bootstrap_type=None if supplied else BootstrapType.STATIONARY,
        block_size=None if supplied else block_size, min_block_size=None if supplied else 1,
        seed=None if supplied else seed,
        indices_sha256=hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest(),
        minimum_pvalue=minimum_pvalue, first_rank_threshold=first_threshold,
        first_rank_reachable=reachable, mc_confidence_level=0.95,
        mc_interval_method='Clopper-Pearson binomial',
        mc_interval_target='conditional bootstrap tail probability before plus-one adjustment',
        mc_assumption='independent replicates from the specified resampling law',
        replicate_numbering='zero-based', include_diagnostics=bool(include_diagnostics),
        af=None if af is None else display_factor, annualised=af is not None,
    )
    if include_diagnostics:
        metadata['bootstrap_indices'] = indices
        metadata['bootstrap_statistics'] = pd.DataFrame(
            diagnostic_statistics, columns=members, index=pd.RangeIndex(count, name='replicate'),
        )
    if diagnostic_rho is not None:
        metadata['bootstrap_prewhitening_rho'] = pd.DataFrame(
            diagnostic_rho, columns=members, index=pd.RangeIndex(count, name='replicate'))
    return SignificanceResult(table=table, metadata=metadata)


def compute_bootstrap_mean_significance(
        returns: pd.DataFrame, family_id: str, return_basis: str, cost_basis: str,
        sampling_frequency: Optional[str] = None,
        expected_index: Optional[pd.DatetimeIndex] = None,
        hac_lags: Optional[int] = 3, confidence_level: float = 0.95,
        method: MultipleTestingMethod = MultipleTestingMethod.HOLM,
        significance_level: float = 0.05, af: Optional[float] = None,
        num_samples: Optional[int] = None, block_size: int = 6, seed: int = 20261101,
        bootstrapped_indices: Optional[np.ndarray] = None, include_diagnostics: bool = False,
        inference: InterceptInference = InterceptInference.BARTLETT_HAC,
) -> SignificanceResult:
    """Test zero return means with a null-centered joint stationary bootstrap.

    Set Y0 = Y - column_mean(Y), resample rows jointly, and recompute mean/HAC SE for
    each draw. The two-sided bootstrap p-value is (1+exceedances)/(B+1), with inclusive
    tail comparisons. This is approximate inference, not an exact randomization test.

    Args:
        returns: Complete finite fractional simple-return DataFrame, one hypothesis per
            column, with ordered unique dates; the compute_mean_significance contract applies.
        family_id: Identifier of the complete predeclared hypothesis family.
        return_basis: Funding/RF treatment already reflected in the supplied inputs.
        cost_basis: Description of gross/net costs already reflected in the inputs.
        sampling_frequency: Explicit regular calendar; mutually exclusive with expected_index.
        expected_index: Exact custom observation calendar. No gaps are filled or dropped.
        hac_lags: Integer in [0, T-1]; None selects the existing lag rule once for the sample.
        confidence_level: Level of retained unadjusted pointwise normal HAC intervals.
        method: Holm (default) or BHY correction of bootstrap marginal p-values.
        significance_level: Family-wise error/FDR target in (0, 1).
        af: Optional positive periods per year, scaling observed estimates/SEs/HAC bounds
            linearly for display only. None displays per-period values.
        num_samples: Positive B. None uses 9999 generated draws, or the supplied column count.
            An explicit value must match supplied indices.
        block_size: Integer expected stationary block length in [1, 2**53], generation only.
            The upper bound protects the geometric sampler's float64 probability calculation.
            min_block_size is fixed at one, and the draw length is T.
        seed: Integer in [0, 2**32-1] for the existing Numba sampler; generation only.
        bootstrapped_indices: Optional authoritative integer ndarray of shape (T, B), with
            values in [0, T-1]. Its origin/law is caller-owned; no stationary claim is inferred.
        include_diagnostics: Retain a copy of indices and labelled (B, M) studentized statistics
            in metadata, plus the (B, M) fitted rho for prewhitened inference. Default False
            streams counts; no B complete panels are retained.
        inference: BARTLETT_HAC (default) or PREWHITENED_AR1 standard errors, used for the
            observed statistics and refitted in every draw. Prewhitening needs hac_lags <= T-2.

    Returns:
        SignificanceResult with corrected bootstrap p-values, pvalue_hac diagnostics,
        bootstrap_exceedances, 95% binomial tail Monte Carlo bounds, and resolution metadata.
        Tail bounds assume independent replicates and are not confidence intervals for means.

    Raises:
        ValueError: If inputs/settings/indices are invalid or any draw has invalid inference.
            The error identifies the zero-based replicate and affected hypotheses; no redraws.

    Warns:
        RuntimeWarning: If 1/(B+1) exceeds the correction's first-rank threshold.
    """
    return _bootstrap_significance(
        returns, None, family_id, return_basis, cost_basis, sampling_frequency, expected_index,
        hac_lags, confidence_level, method, significance_level, af, num_samples, block_size,
        seed, bootstrapped_indices, include_diagnostics, inference,
    )


def compute_bootstrap_alpha_significance(
        returns: pd.DataFrame, factors: pd.DataFrame, family_id: str,
        return_basis: str, cost_basis: str, sampling_frequency: Optional[str] = None,
        expected_index: Optional[pd.DatetimeIndex] = None,
        hac_lags: Optional[int] = 3, confidence_level: float = 0.95,
        method: MultipleTestingMethod = MultipleTestingMethod.HOLM,
        significance_level: float = 0.05, af: Optional[float] = None,
        num_samples: Optional[int] = None, block_size: int = 6, seed: int = 20261101,
        bootstrapped_indices: Optional[np.ndarray] = None, include_diagnostics: bool = False,
        inference: InterceptInference = InterceptInference.BARTLETT_HAC,
) -> SignificanceResult:
    """Test zero multi-factor alphas with paired rows from the recentered regression null.

    Fit observed intercepts, set Y0 = Y - alpha_hat, and draw paired rows of (F, Y0).
    Factors are not demeaned. Refit intercept and all slopes in every draw using the
    shared HAC estimator. Studentized tail counts give (1+exceedances)/(B+1).

    Args:
        returns: Complete finite fractional simple-return responses. The
            compute_alpha_significance contract applies: T >= 24, T > k and a full-rank design.
        factors: Nonempty factor-return DataFrame on exactly the same dates. Self-regressors,
            constants and duplicate/dependent columns are rejected.
        family_id: Identifier of the complete predeclared alpha family.
        return_basis: Funding/RF treatment already applied to the supplied inputs.
        cost_basis: Description of gross/net costs already reflected in inputs.
        sampling_frequency: Explicit regular calendar; mutually exclusive with expected_index.
        expected_index: Exact custom observation calendar, without inferred/fill-in sessions.
        hac_lags: Integer in [0, T-1]; None selects the existing lag rule once.
        confidence_level: Level of retained unadjusted pointwise normal HAC intervals.
        method: Holm (default) or BHY correction of bootstrap marginal p-values.
        significance_level: Family-wise error/FDR target in (0, 1).
        af: Optional positive periods per year for linear display scaling only.
        num_samples: Positive B; None means 9999 generated draws or supplied column count.
            Explicit B must match supplied indices.
        block_size: Integer expected stationary block length in [1, 2**53]; generation only.
            The bound protects the geometric sampler's float64 probability calculation.
        seed: Integer in [0, 2**32-1] for the Numba sampler; generation only.
        bootstrapped_indices: Authoritative integer ndarray (T, B), entries in [0, T-1].
            One matrix is shared by every factor and response; its supplied law is caller-owned.
        include_diagnostics: Keep copied indices and a labelled (B, M) statistic frame in
            metadata, plus the (B, M) fitted rho for prewhitened inference.
        inference: BARTLETT_HAC (default) or PREWHITENED_AR1 standard errors, used for the
            observed statistics and refitted in every draw. Prewhitening needs hac_lags <= T-2.

    Returns:
        SignificanceResult with approximate bootstrap p-values/decisions, observed HAC
        diagnostics and bounds, 95% conditional binomial tail uncertainty and resolution data.
        Monte Carlo bounds assume independent replicates; they are not intervals for alphas.

    Raises:
        ValueError: If observed inputs/settings/indices or any replicate's rank/SE is invalid.
            Draw failures report zero-based replicate and hypotheses; no draws are discarded.

    Warns:
        RuntimeWarning: If 1/(B+1) exceeds the correction's first-rank threshold. BHY can
            still reject at higher ranks; a warning is not a statement that all rejection fails.
    """
    if factors is None:
        raise ValueError('factors must be a nonempty DataFrame; use the mean test without factors')
    return _bootstrap_significance(
        returns, factors, family_id, return_basis, cost_basis, sampling_frequency, expected_index,
        hac_lags, confidence_level, method, significance_level, af, num_samples, block_size,
        seed, bootstrapped_indices, include_diagnostics, inference,
    )
