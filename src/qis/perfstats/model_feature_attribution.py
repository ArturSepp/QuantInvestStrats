"""Factorial and Shapley attribution for features of a layered portfolio model.

The caller supplies one :class:`ModelLayerNavs` bundle for every coalition of model features,
including the empty production coalition. QIS aligns all scenarios on one common sample and
requires their benchmark paths to be identical. For every risk, signal, full-model and optional
net-model layer, factorial effects are constructed as Harsanyi dividends and order-independent
feature effects are constructed with the exact Shapley weights.

All effect paths are positive NAV ratios. Their log returns are therefore the corresponding
linear combinations of scenario log returns. Each effect bundle is passed directly to
``compute_model_layer_alpha_beta_attribution`` so alpha differences and their Bartlett-HAC
intervals come from one regression on the effect path; regression tables are never subtracted.
The total-return rows of the summary have no regressor, so their intervals come from the
constant-only HAC mean in ``qis.utils.regression``.
The Shapley paths and the complete set of factorial effects are both checked to reconstruct the
joint all-features-versus-production return path at every observation.

This is full-sample descriptive analytics. It does not estimate point-in-time feature values and
must not be used as a portfolio construction input. A complete experiment with ``n`` features
requires all ``2**n`` coalitions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from qis.perfstats.config import PerfStat
from qis.perfstats.model_layer_attribution import (
    ALPHA_AN_CI_HIGH_COLUMN,
    ALPHA_AN_CI_LOW_COLUMN,
    ALPHA_CONFIDENCE_LEVEL,
    ALPHA_HAC_LAGS,
    ModelLayerAlphaBetaAttribution,
    compute_model_layer_alpha_beta_attribution,
)
from qis.utils.annualisation import get_annualization_factor
from qis.utils.regression import estimate_hac_mean


MODEL_FEATURE_IDENTITY_TOLERANCE: float = 1.0e-10
_LAYER_FIELDS: tuple[str, ...] = (
    'risk_layer_nav',
    'signal_layer_nav',
    'full_model_nav',
)


@dataclass(frozen=True)
class ModelLayerNavs:
    """Benchmark and model-layer NAVs for one feature coalition or effect.

    Attributes:
        benchmark_nav: Benchmark NAV or price index, identical across feature coalitions.
        risk_layer_nav: NAV produced by the standalone risk layer.
        signal_layer_nav: NAV produced by the signal layer.
        full_model_nav: NAV produced by the fully integrated model.
        full_model_net_nav: Optional fully integrated model NAV after trading costs.
    """

    benchmark_nav: pd.Series
    risk_layer_nav: pd.Series
    signal_layer_nav: pd.Series
    full_model_nav: pd.Series
    full_model_net_nav: pd.Series | None = None


@dataclass(frozen=True)
class ModelFeatureAlphaBetaAttribution:
    """Factorial and Shapley attribution of layered-model feature effects.

    Attributes:
        scenario_layer_navs: Common-sample layer NAVs for every supplied feature coalition.
        factorial_effect_navs: Harsanyi dividend NAVs keyed by non-empty feature coalition.
        shapley_feature_navs: Order-independent Shapley effect NAVs keyed by feature name.
        joint_effect_navs: All-features-versus-production layer NAV ratios.
        factorial_effect_attributions: Model-layer attributions for every factorial effect.
        feature_attributions: Model-layer attributions for every Shapley feature effect.
        joint_attribution: Model-layer attribution of the joint feature effect.
        summary: Factorial, Shapley and joint return, alpha, beta and interval estimates.
        identity_errors: Maximum log-return reconstruction errors for audited identities.
        freq: Regression and return frequency.
        hac_lags: Bartlett-kernel lag count used for inference.
        confidence_level: Two-sided confidence level used for intervals.
    """

    scenario_layer_navs: Mapping[frozenset[str], ModelLayerNavs]
    factorial_effect_navs: Mapping[frozenset[str], ModelLayerNavs]
    shapley_feature_navs: Mapping[str, ModelLayerNavs]
    joint_effect_navs: ModelLayerNavs
    factorial_effect_attributions: Mapping[
        frozenset[str], ModelLayerAlphaBetaAttribution
    ]
    feature_attributions: Mapping[str, ModelLayerAlphaBetaAttribution]
    joint_attribution: ModelLayerAlphaBetaAttribution
    summary: pd.DataFrame
    identity_errors: pd.Series
    freq: str
    hac_lags: int
    confidence_level: float


def _coalition_sort_key(coalition: frozenset[str]) -> tuple[int, tuple[str, ...]]:
    """Return a deterministic size-then-name ordering for a feature coalition."""
    return len(coalition), tuple(sorted(coalition))


def _all_coalitions(features: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    """Return the complete power set in deterministic order."""
    return tuple(
        frozenset(coalition)
        for size in range(len(features) + 1)
        for coalition in combinations(features, size)
    )


def _validate_scenario_keys(
        scenario_layer_navs: Mapping[frozenset[str], ModelLayerNavs],
) -> tuple[tuple[str, ...], tuple[frozenset[str], ...]]:
    """Validate coalition keys and return features plus the expected power set."""
    if not scenario_layer_navs:
        raise ValueError('feature attribution requires at least two feature coalitions')
    for coalition in scenario_layer_navs:
        if not isinstance(coalition, frozenset):
            raise TypeError('feature-coalition keys must be frozenset[str] instances')
        if any(not isinstance(feature, str) or not feature for feature in coalition):
            raise ValueError('feature names must be non-empty strings')
    features = tuple(sorted(set().union(*scenario_layer_navs)))
    if not features:
        raise ValueError('feature attribution requires at least one feature')
    expected = _all_coalitions(features=features)
    supplied = set(scenario_layer_navs)
    missing = [coalition for coalition in expected if coalition not in supplied]
    extra = sorted(supplied.difference(expected), key=_coalition_sort_key)
    if missing or extra:
        raise ValueError(
            'feature attribution requires the complete factorial experiment: '
            f'{missing=}, {extra=}'
        )
    return features, expected


def _align_scenario_layer_navs(
        scenario_layer_navs: Mapping[frozenset[str], ModelLayerNavs],
        coalitions: tuple[frozenset[str], ...],
) -> dict[frozenset[str], ModelLayerNavs]:
    """Align, validate and unit-normalise all scenario layer NAVs."""
    net_flags = [
        scenario_layer_navs[coalition].full_model_net_nav is not None
        for coalition in coalitions
    ]
    if any(net_flags) and not all(net_flags):
        raise ValueError('full-model net NAV must be supplied for every coalition or none')
    layer_fields = ('benchmark_nav', *_LAYER_FIELDS)
    if all(net_flags):
        layer_fields = (*layer_fields, 'full_model_net_nav')

    columns: dict[tuple[frozenset[str], str], pd.Series] = {}
    for coalition in coalitions:
        layer_navs = scenario_layer_navs[coalition]
        if not isinstance(layer_navs, ModelLayerNavs):
            raise TypeError('every feature coalition must contain ModelLayerNavs')
        for layer_field in layer_fields:
            nav = getattr(layer_navs, layer_field)
            if nav is None:
                raise ValueError(f'{coalition} is missing {layer_field}')
            if not isinstance(nav, pd.Series):
                raise TypeError(f'{coalition} {layer_field} must be a pandas Series')
            if not isinstance(nav.index, pd.DatetimeIndex):
                raise TypeError(f'{coalition} {layer_field} must have a DatetimeIndex')
            if nav.index.has_duplicates:
                raise ValueError(f'{coalition} {layer_field} index contains duplicate dates')
            columns[(coalition, layer_field)] = nav.astype(float)

    common = pd.concat(columns, axis=1, join='inner', sort=True).sort_index().dropna(how='any')
    if len(common.index) < 3:
        raise ValueError('factorial layer NAVs require at least three common observations')
    if not np.isfinite(common.to_numpy(dtype=float)).all() or (common <= 0.0).any().any():
        raise ValueError('factorial layer NAVs must be finite and strictly positive')
    common = common.divide(common.iloc[0], axis=1)

    empty_coalition = frozenset()
    benchmark = common[(empty_coalition, 'benchmark_nav')]
    for coalition in coalitions[1:]:
        difference = (common[(coalition, 'benchmark_nav')] - benchmark).abs().max()
        if difference > MODEL_FEATURE_IDENTITY_TOLERANCE:
            raise ValueError(
                f'benchmark changed in coalition {sorted(coalition)}: '
                f'max difference {difference:.3e}'
            )

    aligned: dict[frozenset[str], ModelLayerNavs] = {}
    for coalition in coalitions:
        label = '+'.join(sorted(coalition)) or 'production'
        aligned[coalition] = ModelLayerNavs(
            benchmark_nav=benchmark.rename('Benchmark'),
            risk_layer_nav=common[(coalition, 'risk_layer_nav')].rename(f'{label} Risk'),
            signal_layer_nav=common[(coalition, 'signal_layer_nav')].rename(f'{label} Signal'),
            full_model_nav=common[(coalition, 'full_model_nav')].rename(f'{label} Full'),
            full_model_net_nav=(
                common[(coalition, 'full_model_net_nav')].rename(f'{label} Full Net')
                if all(net_flags) else None
            ),
        )
    return aligned


def _combine_layer_navs(
        scenarios: Mapping[frozenset[str], ModelLayerNavs],
        name: str,
        scenario_powers: Mapping[frozenset[str], float],
) -> ModelLayerNavs:
    """Construct one effect bundle from products of scenario NAV powers."""
    benchmark = scenarios[frozenset()].benchmark_nav

    def combine(layer_field: str) -> pd.Series | None:
        """Combine one layer, preserving an absent optional net layer."""
        values: pd.Series | None = None
        for coalition, power in scenario_powers.items():
            nav = getattr(scenarios[coalition], layer_field)
            if nav is None:
                return None
            term = nav.pow(power)
            values = term if values is None else values * term
        assert values is not None
        return (values / values.iloc[0]).rename(f'{name} {layer_field}')

    return ModelLayerNavs(
        benchmark_nav=benchmark,
        risk_layer_nav=combine('risk_layer_nav'),
        signal_layer_nav=combine('signal_layer_nav'),
        full_model_nav=combine('full_model_nav'),
        full_model_net_nav=combine('full_model_net_nav'),
    )


def _harsanyi_powers(coalition: frozenset[str]) -> dict[frozenset[str], float]:
    """Return inclusion-exclusion powers for one Harsanyi dividend."""
    ordered = tuple(sorted(coalition))
    all_subsets = _all_coalitions(features=ordered)
    multiplication_order = (
        coalition,
        frozenset(),
        *(subset for subset in all_subsets if subset not in {coalition, frozenset()}),
    )
    return {
        subset: float((-1) ** (len(coalition) - len(subset)))
        for subset in multiplication_order
    }


def _shapley_powers(
        feature: str,
        features: tuple[str, ...],
) -> dict[frozenset[str], float]:
    """Return scenario powers producing one exact Shapley feature path."""
    other_features = tuple(item for item in features if item != feature)
    denominator = factorial(len(features))
    powers: dict[frozenset[str], float] = {}
    for coalition in _all_coalitions(features=other_features):
        weight = (
            factorial(len(coalition))
            * factorial(len(features) - len(coalition) - 1)
            / denominator
        )
        with_feature = coalition.union({feature})
        powers[with_feature] = powers.get(with_feature, 0.0) + weight
        powers[coalition] = powers.get(coalition, 0.0) - weight
    return powers


def _compute_layer_attribution(
        layer_navs: ModelLayerNavs,
        freq: str,
        hac_lags: int,
        confidence_level: float,
) -> ModelLayerAlphaBetaAttribution:
    """Run the existing QIS layer attribution for one effect bundle."""
    return compute_model_layer_alpha_beta_attribution(
        benchmark_nav=layer_navs.benchmark_nav,
        risk_layer_nav=layer_navs.risk_layer_nav,
        signal_layer_nav=layer_navs.signal_layer_nav,
        full_model_nav=layer_navs.full_model_nav,
        full_model_net_nav=layer_navs.full_model_net_nav,
        freq=freq,
        hac_lags=hac_lags,
        confidence_level=confidence_level,
    )


def _annualised_mean_hac_interval(
        attribution: ModelLayerAlphaBetaAttribution,
        component: str,
) -> tuple[float, float, float]:
    """Return the annualised mean and Bartlett-HAC interval of one return component.

    The component has no regressor, so the interval comes from the constant-only HAC estimator
    with the one-parameter small-sample correction, not from a regression on the benchmark.
    """
    component_returns = attribution.component_returns[component].dropna()
    regression = estimate_hac_mean(
        y=component_returns,
        hac_lags=attribution.hac_lags,
        confidence_level=attribution.confidence_level,
    )
    annualisation = get_annualization_factor(freq=attribution.freq)
    estimate = annualisation * regression.mean
    expected = float(attribution.annualised_components[component])
    if not np.isclose(
            estimate,
            expected,
            atol=MODEL_FEATURE_IDENTITY_TOLERANCE,
            rtol=0.0,
    ):
        raise RuntimeError(f'{component} HAC mean does not match its annualised component')
    return (
        estimate,
        annualisation * regression.confidence_interval[0],
        annualisation * regression.confidence_interval[1],
    )


def _attribution_summary(
        named_attributions: Mapping[
            tuple[str, str], ModelLayerAlphaBetaAttribution
        ],
) -> pd.DataFrame:
    """Build the auditable return, alpha, beta and interval table for every effect."""
    annualised_alpha = PerfStat.ALPHA_AN.to_str()
    beta = PerfStat.BETA.to_str()
    pvalue = PerfStat.ALPHA_PVALUE.to_str()
    rows: dict[str, dict[str, float]] = {}
    for effect_key, attribution in named_attributions.items():
        table = attribution.regression_table
        row: dict[str, float] = {}
        for component in ('Full Model Return', 'Full Model Net Return'):
            if component not in attribution.component_returns:
                continue
            estimate, ci_low, ci_high = _annualised_mean_hac_interval(
                attribution=attribution,
                component=component,
            )
            prefix = f'Annualised {component}'
            row[prefix] = estimate
            row[f'{prefix} CI Low'] = ci_low
            row[f'{prefix} CI High'] = ci_high
        for layer in (
                'Risk Layer',
                'Signal Layer',
                'Integration',
                'Full Model',
                'Full Model Net',
        ):
            if layer not in table.index:
                continue
            row[f'{layer} Alpha'] = float(table.loc[layer, annualised_alpha])
            row[f'{layer} Alpha CI Low'] = float(table.loc[layer, ALPHA_AN_CI_LOW_COLUMN])
            row[f'{layer} Alpha CI High'] = float(
                table.loc[layer, ALPHA_AN_CI_HIGH_COLUMN]
            )
            row[f'{layer} Beta'] = float(table.loc[layer, beta])
            row[f'{layer} Alpha p-value'] = float(table.loc[layer, pvalue])
        rows[effect_key] = row
    summary = pd.DataFrame.from_dict(rows, orient='index')
    summary.index = pd.MultiIndex.from_tuples(
        summary.index,
        names=['Effect Type', 'Feature'],
    )
    return summary


def _max_log_return_identity_error(
        left: pd.Series,
        right_parts: list[pd.Series],
) -> float:
    """Return the maximum pointwise log-return reconstruction error."""
    left_returns = np.log(left).diff().dropna()
    right_returns = sum(np.log(part).diff().dropna() for part in right_parts)
    return float((left_returns - right_returns).abs().max())


def _identity_errors(
        factorial_effect_navs: Mapping[frozenset[str], ModelLayerNavs],
        shapley_feature_navs: Mapping[str, ModelLayerNavs],
        joint_effect_navs: ModelLayerNavs,
        factorial_effect_attributions: Mapping[
            frozenset[str], ModelLayerAlphaBetaAttribution
        ],
        feature_attributions: Mapping[str, ModelLayerAlphaBetaAttribution],
        joint_attribution: ModelLayerAlphaBetaAttribution,
) -> pd.Series:
    """Compute and enforce pathwise reconstruction and QIS alpha-bridge identities."""
    layer_fields = list(_LAYER_FIELDS)
    if joint_effect_navs.full_model_net_nav is not None:
        layer_fields.append('full_model_net_nav')
    checks: dict[str, float] = {}
    for layer_field in layer_fields:
        joint = getattr(joint_effect_navs, layer_field)
        assert joint is not None
        factorial_parts = [
            getattr(layer_navs, layer_field)
            for layer_navs in factorial_effect_navs.values()
        ]
        shapley_parts = [
            getattr(layer_navs, layer_field)
            for layer_navs in shapley_feature_navs.values()
        ]
        checks[f'{layer_field} factorial sum = joint'] = _max_log_return_identity_error(
            joint,
            [part for part in factorial_parts if part is not None],
        )
        checks[f'{layer_field} Shapley sum = joint'] = _max_log_return_identity_error(
            joint,
            [part for part in shapley_parts if part is not None],
        )

    annualised_alpha = PerfStat.ALPHA_AN.to_str()
    named_attributions: dict[str, ModelLayerAlphaBetaAttribution] = {
        f'factorial:{"+".join(sorted(coalition))}': attribution
        for coalition, attribution in factorial_effect_attributions.items()
    }
    named_attributions.update({
        f'Shapley:{feature}': attribution
        for feature, attribution in feature_attributions.items()
    })
    named_attributions['joint'] = joint_attribution
    for name, attribution in named_attributions.items():
        table = attribution.regression_table
        checks[f'{name} QIS alpha bridge'] = abs(
            float(table.loc['Full Model', annualised_alpha])
            - float(table.loc[
                ['Risk Layer', 'Signal Layer', 'Integration'], annualised_alpha
            ].sum())
        )

    errors = pd.Series(checks, name='Maximum Absolute Error')
    failed = errors.loc[errors > MODEL_FEATURE_IDENTITY_TOLERANCE]
    if not failed.empty:
        raise RuntimeError(f'model-feature attribution identity checks failed:\n{failed}')
    return errors


def compute_model_feature_alpha_beta_attribution(
        scenario_layer_navs: Mapping[frozenset[str], ModelLayerNavs],
        freq: str = 'ME',
        hac_lags: int = ALPHA_HAC_LAGS,
        confidence_level: float = ALPHA_CONFIDENCE_LEVEL,
) -> ModelFeatureAlphaBetaAttribution:
    """Compute factorial and Shapley attribution for layered-model feature changes.

    Coalition keys are ``frozenset`` instances containing the features enabled in that scenario;
    the empty key is the production baseline. Every subset of the feature set must be supplied.
    QIS constructs Harsanyi factorial effects, order-independent Shapley feature effects and the
    joint all-features effect as NAV ratios. Each effect is then analysed by
    :func:`compute_model_layer_alpha_beta_attribution` on the same common sample.

    Args:
        scenario_layer_navs: Complete mapping from feature coalitions to model-layer NAV bundles.
        freq: Regression and return frequency. Defaults to month-end.
        hac_lags: Bartlett-kernel lag count for HAC inference. Defaults to three periods.
        confidence_level: Two-sided confidence-interval level. Defaults to 0.95.

    Returns:
        Factorial paths, Shapley paths, layer attributions, estimates and identity checks.

    Raises:
        TypeError: If coalition keys, bundles, NAVs or their indices have invalid types.
        ValueError: If the factorial experiment is incomplete, NAVs cannot be aligned, benchmark
            paths differ, optional net NAVs are inconsistent, or inference settings are invalid.
        RuntimeError: If a factorial, Shapley or model-layer identity fails.
    """
    if hac_lags < 0:
        raise ValueError(f'hac_lags must be non-negative, got {hac_lags}')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f'confidence_level must be between zero and one, got {confidence_level}'
        )
    features, coalitions = _validate_scenario_keys(scenario_layer_navs=scenario_layer_navs)
    scenarios = _align_scenario_layer_navs(
        scenario_layer_navs=scenario_layer_navs,
        coalitions=coalitions,
    )

    factorial_effect_navs = {
        coalition: _combine_layer_navs(
            scenarios=scenarios,
            name=f'factorial:{"+".join(sorted(coalition))}',
            scenario_powers=_harsanyi_powers(coalition=coalition),
        )
        for coalition in coalitions
        if coalition
    }
    shapley_feature_navs = {
        feature: _combine_layer_navs(
            scenarios=scenarios,
            name=f'Shapley:{feature}',
            scenario_powers=_shapley_powers(feature=feature, features=features),
        )
        for feature in features
    }
    all_features = frozenset(features)
    joint_effect_navs = _combine_layer_navs(
        scenarios=scenarios,
        name='joint',
        scenario_powers={all_features: 1.0, frozenset(): -1.0},
    )

    factorial_effect_attributions = {
        coalition: _compute_layer_attribution(
            layer_navs=layer_navs,
            freq=freq,
            hac_lags=hac_lags,
            confidence_level=confidence_level,
        )
        for coalition, layer_navs in factorial_effect_navs.items()
    }
    feature_attributions = {
        feature: _compute_layer_attribution(
            layer_navs=layer_navs,
            freq=freq,
            hac_lags=hac_lags,
            confidence_level=confidence_level,
        )
        for feature, layer_navs in shapley_feature_navs.items()
    }
    joint_attribution = _compute_layer_attribution(
        layer_navs=joint_effect_navs,
        freq=freq,
        hac_lags=hac_lags,
        confidence_level=confidence_level,
    )
    named_attributions: dict[tuple[str, str], ModelLayerAlphaBetaAttribution] = {
        ('Factorial', ' + '.join(sorted(coalition))): attribution
        for coalition, attribution in factorial_effect_attributions.items()
    }
    named_attributions.update({
        ('Shapley', feature): attribution
        for feature, attribution in feature_attributions.items()
    })
    named_attributions[('Joint', ' + '.join(features))] = joint_attribution
    identity_errors = _identity_errors(
        factorial_effect_navs=factorial_effect_navs,
        shapley_feature_navs=shapley_feature_navs,
        joint_effect_navs=joint_effect_navs,
        factorial_effect_attributions=factorial_effect_attributions,
        feature_attributions=feature_attributions,
        joint_attribution=joint_attribution,
    )
    return ModelFeatureAlphaBetaAttribution(
        scenario_layer_navs=scenarios,
        factorial_effect_navs=factorial_effect_navs,
        shapley_feature_navs=shapley_feature_navs,
        joint_effect_navs=joint_effect_navs,
        factorial_effect_attributions=factorial_effect_attributions,
        feature_attributions=feature_attributions,
        joint_attribution=joint_attribution,
        summary=_attribution_summary(named_attributions=named_attributions),
        identity_errors=identity_errors,
        freq=freq,
        hac_lags=hac_lags,
        confidence_level=confidence_level,
    )
