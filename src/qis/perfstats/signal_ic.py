"""Measure cross-sectional information coefficients (ICs) and summarize them through a cutoff.

Each dated IC correlates formation scores with realised forward returns on a specified
set of eligible assets. The adjusted-minus-reference difference uses the same names.
For normalized date weights w, the horizon-level IC-IR is the weighted mean IC divided
by its sample standard deviation, and the effective count is 1 / sum(w**2). The IC
and its matched difference each use their own finite dates. ICs are dimensionless.
Signal formation and forward-return construction are upstream.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from qis.perfstats.signal_diagnostics import _ic_correlation
from qis.utils.regression import _validate_hac_settings, estimate_hac_regression


class ICStatus(str, Enum):
    """Outcome of one cross-sectional correlation on matched assets.

    Attributes:
        INSUFFICIENT_ASSETS: Fewer eligible, finite names than the specified minimum.
        CONSTANT_SCORE: The selected formation scores are identical.
        CONSTANT_TARGET: The selected target has zero range within numerical tolerance.
        OK: The cross-sectional IC is defined.
    """

    INSUFFICIENT_ASSETS = 'insufficient_assets'
    CONSTANT_SCORE = 'constant_score'
    CONSTANT_TARGET = 'constant_target'
    OK = 'ok'


class ICInferenceStatus(str, Enum):
    """Availability of pointwise inference for a selected series of dated ICs.

    Attributes:
        INSUFFICIENT_HISTORY: The rolling window has yet to fill.
        INSUFFICIENT_DATES: The selected series has too few effective observations.
        ZERO_DISPERSION: All selected values agree within floating-point tolerance.
        OK: A pointwise HAC standard error and confidence interval are available.
    """

    INSUFFICIENT_HISTORY = 'insufficient_history'
    INSUFFICIENT_DATES = 'insufficient_dates'
    ZERO_DISPERSION = 'zero_dispersion'
    OK = 'ok'


@dataclass(frozen=True)
class SignalICReport:
    """Dated weights and endpoint statistics for one historical weighting policy.

    Attributes:
        dated: Input rows through the cutoff, sorted by realisation date. Each selected
            row carries its normalized IC weight and, separately, its paired weight.
            Excluded observations have missing weights.
        summary: One row per group, horizon, sample, signal and target. Includes valid
            and selected date counts, Kish effective counts, mean breadth, mean IC,
            sample standard deviation, IC-IR, positive hit rate, mean paired difference,
            pointwise HAC errors and bounds, inference statuses and estimator settings.
    """

    dated: pd.DataFrame
    summary: pd.DataFrame


_GROUP_KEYS = ('horizon', 'sample', 'signal', 'target')
_SUMMARY_FIELDS = (
    'n_windows', 'n_valid', 'n_selected', 'effective_n', 'n_paired_valid',
    'n_paired_selected', 'paired_effective_n', 'mean_n_assets',
    'mean_ic', 'std_ic', 'ic_ir', 'hit_rate', 'ic_hac_se', 'ic_lower', 'ic_upper',
    'inference', 'effective_hac_lags', 'mean_ic_minus_reference', 'difference_hac_se',
    'difference_lower', 'difference_upper', 'paired_inference', 'paired_effective_hac_lags',
    'as_of', 'weighting', 'ewma_span', 'rolling_span', 'warmup_periods', 'hac_lags',
    'confidence_level',
)


def _matched_ic(score: np.ndarray, outcome: np.ndarray, reference_scale: float,
                min_obs_per_date: int, correlation: Callable) -> tuple[float, ICStatus]:
    if len(score) < min_obs_per_date:
        return np.nan, ICStatus.INSUFFICIENT_ASSETS
    if np.all(score == score[0]):
        return np.nan, ICStatus.CONSTANT_SCORE
    scale = max(reference_scale, np.abs(outcome).max())
    if np.ptp(outcome) <= 8.0 * np.finfo(float).eps * scale:
        return np.nan, ICStatus.CONSTANT_TARGET
    return float(correlation(score, outcome)[0]), ICStatus.OK


def compute_matched_signal_ic(
        observations: pd.DataFrame,
        *,
        signals: Sequence[str],
        targets: Sequence[str],
        reference_target: str,
        group_by: Sequence[str] = (),
        min_obs_per_date: int = 5,
        method: str = 'spearman',
) -> pd.DataFrame:
    """Correlate formation scores with forward returns on matched cross-sections.

    Each observed window emits a common sample across all targets, a reference-versus-target
    sample for each other target, and an available-reference sample. All samples require
    eligibility and finite values of every requested score. Sample labels persist when the
    memberships coincide or the window has zero eligible assets.

    Args:
        observations: One row per group, horizon, formation_date, realisation_date and asset.
            Supply scores and eligibility known at formation, and realised targets for that
            window. Rows with eligible equal to True enter the sample. Scores are used at the
            supplied formation date without resampling or lagging.
        signals: Distinct numeric formation-score columns.
        targets: Distinct numeric forward-outcome columns, including the reference.
        reference_target: Target used for paired comparisons within each sample.
        group_by: Additional identifiers such as fund. Each group has its own cross-section.
        min_obs_per_date: Minimum number of distinct matched assets, at least two.
        method: 'spearman' uses average ranks for ties. 'pearson' uses the supplied values.

    Returns:
        Dated rows with window keys, sample, signal, target, method, reference_target,
        n_assets, ic, status and ic_minus_reference. The difference requires two defined
        ICs on the same names and is missing on reference rows. Pass the table directly
        to compute_signal_ic_summary, using the same group_by to summarize each group.

    Raises:
        ValueError: For invalid columns, duplicate asset-window keys, missing or unordered
            endpoints, invalid breadth or an unsupported method.
        TypeError: If observations is not a DataFrame.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError('observations must be a DataFrame')
    if (isinstance(min_obs_per_date, (bool, np.bool_))
            or not isinstance(min_obs_per_date, (int, np.integer)) or min_obs_per_date < 2):
        raise ValueError('min_obs_per_date must be an integer of at least two')
    correlation = _ic_correlation(method)
    if any(isinstance(names, str) for names in (signals, targets, group_by)):
        raise ValueError('signals, targets and group_by must be sequences of column names')
    signals, targets, group_by = tuple(signals), tuple(targets), tuple(group_by)
    if not signals or not targets or reference_target not in targets:
        raise ValueError('supply signals and targets including the reference_target')
    window_keys = [*group_by, 'horizon', 'formation_date', 'realisation_date']
    required = [*window_keys, 'asset', 'eligible', *signals, *targets]
    fields = ['sample', 'signal', 'target', 'method', 'reference_target',
              'n_assets', 'ic', 'status', 'ic_minus_reference']
    if (not observations.columns.is_unique or len(set(required)) != len(required)
            or set(group_by).intersection([*fields, *_SUMMARY_FIELDS, 'weight', 'paired_weight'])):
        raise ValueError('input column names and group identifiers must be distinct')
    missing = set(required).difference(observations.columns)
    if missing:
        raise ValueError(f'missing required observation columns: {sorted(missing)}')

    panel = observations[required].reset_index(drop=True)
    for date in ('formation_date', 'realisation_date'):
        panel[date] = pd.to_datetime(panel[date], errors='raise')
    if panel[['horizon', 'asset', 'formation_date', 'realisation_date']].isna().any().any():
        raise ValueError('horizon, asset and window dates must be present')
    if (panel['formation_date'] >= panel['realisation_date']).any():
        raise ValueError('formation_date must precede realisation_date')
    if panel.duplicated([*window_keys, 'asset']).any():
        raise ValueError('duplicate asset within a group and decision window')
    for name in (*signals, *targets):
        numeric = pd.to_numeric(panel[name], errors='raise')
        if np.iscomplexobj(numeric):
            raise ValueError(f'{name} must contain real numbers')
        panel[name] = numeric.to_numpy(dtype=float, na_value=np.nan)

    samples = {'common': targets}
    for target in targets:
        if target != reference_target:
            samples[f'{reference_target}_vs_{target}'] = (reference_target, target)
    samples[f'available_{reference_target}'] = (reference_target,)

    records = []
    for key, window in panel.groupby(window_keys, sort=True, observed=True, dropna=False):
        scores = {name: window[name].to_numpy() for name in signals}
        outcomes = {name: window[name].to_numpy() for name in targets}
        eligible = window['eligible'].eq(True).to_numpy(dtype=bool, na_value=False)
        for score in scores.values():
            eligible = eligible & np.isfinite(score)
        finite = {name: np.isfinite(outcome) for name, outcome in outcomes.items()}
        for sample, sample_targets in samples.items():
            mask = eligible.copy()
            for target in sample_targets:
                mask &= finite[target]
            matched = {name: outcomes[name][mask] for name in sample_targets}
            reference_scale = np.abs(matched[reference_target]).max(initial=0.0)
            for signal in signals:
                score = scores[signal][mask]
                estimates = {name: _matched_ic(score, outcome, reference_scale,
                                               min_obs_per_date, correlation)
                             for name, outcome in matched.items()}
                reference_ic = estimates[reference_target][0]
                for target, (ic, status) in estimates.items():
                    difference = np.nan if target == reference_target else ic - reference_ic
                    records.append((*key, sample, signal, target, method, reference_target,
                                    len(score), ic, status.value, difference))
    return pd.DataFrame.from_records(records, columns=[*window_keys, *fields])


@dataclass
class _SeriesEstimate:
    weights: pd.Series
    n_valid: int
    effective_n: float = 0.0
    effective_lags: int = 0
    mean: float = np.nan
    std: float = np.nan
    ic_ir: float = np.nan
    hit_rate: float = np.nan
    se: float = np.nan
    lower: float = np.nan
    upper: float = np.nan
    status: ICInferenceStatus = ICInferenceStatus.INSUFFICIENT_DATES


def _estimate_series(values: pd.Series,
                     ewma_span: Optional[float],
                     rolling_span: Optional[int],
                     warmup_periods: int,
                     hac_lags: int,
                     confidence_level: float) -> _SeriesEstimate:
    finite = values.loc[np.isfinite(values)]
    estimate = _SeriesEstimate(weights=pd.Series(dtype=float), n_valid=len(finite))
    if rolling_span is not None and len(finite) < rolling_span:
        estimate.status = ICInferenceStatus.INSUFFICIENT_HISTORY
        return estimate
    selected = finite.tail(rolling_span) if rolling_span is not None else finite
    n = len(selected)
    if n == 0:
        return estimate

    if ewma_span is None:
        weights = np.full(n, 1.0 / n)
        estimate.effective_n = float(n)
    else:
        # Use the HAC routine's decay so descriptive and inferential weights agree.
        decay = 1.0 - 2.0 / (ewma_span + 1.0)
        if decay == 0.0:
            # At the first float above one, retain the older weight for the sample std.
            decay = (ewma_span - 1.0) / (ewma_span + 1.0)
        ages = np.arange(n - 1, -1, -1)
        objective = np.power(decay, ages)
        estimate.effective_n = float(objective.sum() ** 2 / np.dot(objective, objective))
        weights = objective / objective.sum()

    x = selected.to_numpy()
    estimate.weights = pd.Series(weights, index=selected.index)
    estimate.effective_lags = min(hac_lags, n - 1)
    estimate.mean = float(weights @ x)
    estimate.hit_rate = float(weights @ (x > 0))
    if n == 1:
        return estimate
    if np.ptp(x) <= 8.0 * np.finfo(float).eps:
        estimate.std = 0.0
        estimate.status = ICInferenceStatus.ZERO_DISPERSION
        return estimate

    # This equals 1 - sum(w**2), including when the latest weight rounds to one.
    correction = 2.0 * np.dot(weights[1:], np.cumsum(weights[:-1]))
    estimate.std = float(np.sqrt((weights @ (x - estimate.mean) ** 2) / correction))
    if estimate.std == 0.0:
        estimate.status = ICInferenceStatus.ZERO_DISPERSION
        return estimate
    estimate.ic_ir = estimate.mean / estimate.std

    # Finite geometric histories have effective count strictly below their span.
    if (estimate.effective_n < warmup_periods or estimate.effective_n <= 1.0
            or (ewma_span is not None and ewma_span <= warmup_periods)):
        return estimate

    fit = estimate_hac_regression(y=selected, span=ewma_span, hac_lags=estimate.effective_lags,
                                  confidence_level=confidence_level)
    estimate.se = float(fit.hac_se.iat[0, 0])
    estimate.lower = float(fit.confidence_lower.iat[0, 0])
    estimate.upper = float(fit.confidence_upper.iat[0, 0])
    estimate.status = ICInferenceStatus.OK
    return estimate


def compute_signal_ic_summary(
        dated_ic: pd.DataFrame,
        *,
        group_by: Sequence[str] = (),
        as_of: Optional[pd.Timestamp] = None,
        ewma_span: Optional[float] = None,
        rolling_span: Optional[int] = None,
        warmup_periods: int = 12,
        hac_lags: int = 3,
        confidence_level: float = 0.95,
) -> SignalICReport:
    """Summarize realised ICs and paired IC differences at one endpoint.

    Both spans omitted gives equal weighting over all valid dates. A rolling window
    requires its full number of valid dates before any estimates are available.
    EWMA decays once per valid observation at the group's horizon. The IC and its
    paired difference each have their own finite observations and normalized weights.

    Args:
        dated_ic: Required columns are horizon, sample, signal, target, realisation_date
            and ic. Optional ic_minus_reference contains the difference of the two ICs
            on the same assets and date. Optional n_assets supplies cross-sectional breadth.
            Optional method and reference_target must each be constant within a group.
        group_by: Additional identifier columns, such as fund, to summarize independently.
        as_of: Inclusive cutoff on realisation_date. Defaults to the latest supplied date.
        ewma_span: Finite span greater than one, in valid observations.
        rolling_span: Number of recent valid observations to weight equally, at least two.
            Supply at most one of rolling_span and ewma_span.
        warmup_periods: Minimum Kish effective observations for a HAC interval. Zero
            disables this threshold. Inference requires at least two observations,
            numerically distinct effective count above one, and nonconstant values.
            EWMA spans at or below a positive threshold cannot qualify in finite time.
        hac_lags: Nonnegative Bartlett lag count, capped at the selected count minus one.
        confidence_level: Two-sided pointwise confidence level strictly between zero and one.

    Returns:
        Dated rows with weight and paired_weight and a summary with estimates in the
        observed horizon's units. mean_n_assets averages recorded breadth over selected
        IC dates. Each series reports its own HAC status, standard error and interval.

    Raises:
        ValueError: For invalid settings, ambiguous columns, dates, metadata or groups.
        TypeError: For unsupported input types or noninteger HAC lags.
    """
    if not isinstance(dated_ic, pd.DataFrame):
        raise TypeError('dated_ic must be a DataFrame')
    if ewma_span is not None and rolling_span is not None:
        raise ValueError('supply only one of ewma_span and rolling_span')
    if rolling_span is not None and (
            isinstance(rolling_span, (bool, np.bool_))
            or not isinstance(rolling_span, (int, np.integer)) or rolling_span < 2):
        raise ValueError('rolling_span must be an integer of at least two')
    if (isinstance(warmup_periods, (bool, np.bool_))
            or not isinstance(warmup_periods, (int, np.integer)) or warmup_periods < 0):
        raise ValueError('warmup_periods must be a nonnegative integer')
    ewma_span = _validate_hac_settings(ewma_span, hac_lags, confidence_level)

    if isinstance(group_by, str):
        raise ValueError('group_by must be a sequence of column names')
    keys = [*group_by, *_GROUP_KEYS]
    metadata = [name for name in ('method', 'reference_target') if name in dated_ic]
    reserved = {'weight', 'paired_weight', 'ic', 'ic_minus_reference', 'n_assets',
                'realisation_date', *_SUMMARY_FIELDS}
    if (len(set(keys)) != len(keys) or reserved.intersection(keys)
            or not dated_ic.columns.is_unique):
        raise ValueError('expected distinct input columns and distinct group identifiers')
    missing = {*keys, 'realisation_date', 'ic'}.difference(dated_ic.columns)
    if missing:
        raise ValueError(f'missing required dated IC columns: {sorted(missing)}')

    dated = dated_ic.copy()
    dates = pd.to_datetime(dated['realisation_date'], errors='raise')
    if dates.isna().any():
        raise ValueError('missing realisation_date')
    dated['realisation_date'] = dates
    cutoff = dates.max() if as_of is None else pd.Timestamp(as_of)
    if as_of is not None and pd.isna(cutoff):
        raise ValueError('as_of must be a valid timestamp')
    dated = dated.loc[dates <= cutoff].sort_values('realisation_date', kind='stable')
    dated = dated.reset_index(drop=True)
    if 'formation_date' in dated:
        dated['formation_date'] = pd.to_datetime(dated['formation_date'], errors='raise')
    if dated.duplicated([*keys, 'realisation_date']).any():
        raise ValueError('duplicate realisation_date within a dated IC group')
    for name in ('ic', 'ic_minus_reference', 'n_assets'):
        if name in dated:
            values = pd.to_numeric(dated[name], errors='raise')
            if np.iscomplexobj(values):
                raise ValueError(f'{name} must contain real numbers')
            dated[name] = values.to_numpy(dtype=float, na_value=np.nan)
    dated['weight'] = np.nan
    dated['paired_weight'] = np.nan
    weighting = 'equal'
    if rolling_span is not None:
        weighting = 'rolling'
    elif ewma_span is not None:
        weighting = 'ewma'

    records = []
    for group_key, group in dated.groupby(keys, sort=False, observed=True, dropna=False):
        row = dict(zip(keys, group_key))
        for name in metadata:
            if group[name].nunique(dropna=False) != 1:
                raise ValueError(f'{name} must be constant within group {group_key}')
            row[name] = group[name].iloc[0]
        ic = _estimate_series(group['ic'], ewma_span, rolling_span, warmup_periods,
                              hac_lags, confidence_level)
        differences = group.get('ic_minus_reference', pd.Series(dtype=float))
        paired = _estimate_series(differences, ewma_span, rolling_span, warmup_periods,
                                  hac_lags, confidence_level)
        dated.loc[ic.weights.index, 'weight'] = ic.weights
        dated.loc[paired.weights.index, 'paired_weight'] = paired.weights
        breadth = np.nan
        if 'n_assets' in group:
            counts = group.loc[ic.weights.index, 'n_assets']
            breadth = counts.loc[np.isfinite(counts)].mean()

        row.update(n_windows=len(group), n_valid=ic.n_valid, n_selected=len(ic.weights),
                   effective_n=ic.effective_n, n_paired_valid=paired.n_valid,
                   n_paired_selected=len(paired.weights), paired_effective_n=paired.effective_n,
                   mean_n_assets=breadth, mean_ic=ic.mean, std_ic=ic.std, ic_ir=ic.ic_ir,
                   hit_rate=ic.hit_rate, ic_hac_se=ic.se, ic_lower=ic.lower, ic_upper=ic.upper,
                   inference=ic.status.value, effective_hac_lags=ic.effective_lags,
                   mean_ic_minus_reference=paired.mean, difference_hac_se=paired.se,
                   difference_lower=paired.lower, difference_upper=paired.upper,
                   paired_inference=paired.status.value,
                   paired_effective_hac_lags=paired.effective_lags,
                   as_of=cutoff, weighting=weighting, ewma_span=ewma_span,
                   rolling_span=rolling_span, warmup_periods=warmup_periods,
                   hac_lags=hac_lags, confidence_level=confidence_level)
        records.append(row)

    columns = keys + list(_SUMMARY_FIELDS) + [name for name in metadata if name not in keys]
    return SignalICReport(dated=dated, summary=pd.DataFrame.from_records(records, columns=columns))
