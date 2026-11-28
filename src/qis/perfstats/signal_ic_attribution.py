"""Cross-fitted hedge diagnostics and persistent/within-asset attribution of mean rank IC."""
from collections.abc import Sequence

import numpy as np
import pandas as pd

from qis.perfstats.forward_return_targets import (
    ForwardReturnAdjustment, compute_forward_return_targets,
)
from qis.perfstats.signal_ic import (
    SignalICReport, compute_matched_signal_ic, compute_signal_ic_summary,
)
from qis.portfolio.risk.ewm_factor_model import EwmLinearModel


def compute_beta_reuse_ic(
        asset_returns: pd.DataFrame, factor_returns: pd.DataFrame, decision_dates: pd.DatetimeIndex,
        *, score_factor: str, horizons: Sequence[int], freq: str = 'ME',
        beta_ewma_span: float = 36, ewma_span: float | None = None,
        rolling_span: int | None = None, warmup_periods: int = 12,
        min_obs_per_date: int = 5,
) -> tuple[SignalICReport, pd.DataFrame]:
    """Compare same-fold and disjoint-fold hedges for a negative factor-loading score.

    Args:
        asset_returns: Finite additive increments, dates by assets on a complete native grid.
        factor_returns: Joint benchmark increments on the same grid, dates by named factors.
        decision_dates: Ordered native closes defining the non-overlapping holding windows.
        score_factor: Factor whose negative signed loading ranks the assets.
        horizons: Holding periods in decision steps.
        freq: Native pandas period-end frequency.
        beta_ewma_span: EWMA span in native periods for the through-origin loading fits.
        ewma_span: Decay span in valid dated ICs for the summary.
        rolling_span: Optional equal-weight window of valid dated ICs.
        warmup_periods: Minimum effective dates for paired HAC inference.
        min_obs_per_date: Minimum matched assets for each Spearman IC.

    Returns:
        Summary and dated weights in a SignalICReport, plus the two directional IC pairs.
        In the score-factor test other loadings come from the full fit. In the joint test
        every hedge loading comes from the chosen fold. A date needs both orientations and
        both target ICs for that test. Each date receives one weight in the HAC calculation.
    """
    if not np.isfinite(beta_ewma_span) or beta_ewma_span <= 1:
        raise ValueError('beta_ewma_span must be finite and greater than one')
    dates = asset_returns.index
    if (not isinstance(dates, pd.DatetimeIndex) or dates.empty
            or not dates.equals(factor_returns.index)
            or not dates.equals(pd.date_range(dates[0], dates[-1], freq=freq))
            or not decision_dates.isin(dates).all()):
        raise ValueError('returns must share a complete native grid containing the decision dates')
    if (not asset_returns.columns.is_unique or not factor_returns.columns.is_unique
            or score_factor not in factor_returns or not np.isfinite(asset_returns).all().all()
            or not np.isfinite(factor_returns).all().all()):
        raise ValueError('expected finite labelled asset and factor returns including score_factor')
    if len(decision_dates) < 2 or len(dates) < 4:
        raise ValueError('require at least two decisions and four native returns')
    decay = 1 - 2 / (beta_ewma_span + 1)
    models = {}
    for fold, selection, step in [('full', slice(None), 1), ('A', slice(0, None, 2), 2),
                                  ('B', slice(1, None, 2), 2)]:
        model = EwmLinearModel(x=factor_returns.iloc[selection], y=asset_returns.iloc[selection])
        model.fit(span=None, ewm_lambda=decay ** step, is_x_correlated=True, warmup_period=0)
        models[fold] = {t: model.get_loadings_at_date(t).T for t in decision_dates[:-1]
                        if t >= model.x.index[0]}

    adjustments, scores = {}, {}
    for fold in ('A', 'B'):
        snapshots = models[fold]
        mixed = {t: models['full'][t].assign(**{score_factor: b[score_factor]})
                 for t, b in snapshots.items()}
        for leg, matrices in [('score_factor', mixed), ('joint', snapshots)]:
            adjustments[f'{leg}_{fold}'] = ForwardReturnAdjustment({freq: factor_returns}, matrices)
        scores[fold] = pd.DataFrame({t: -b[score_factor] for t, b in snapshots.items()}).T
    panel = compute_forward_return_targets(decision_dates, {freq: asset_returns},
                                           horizons=horizons, adjustments=adjustments)
    for fold, frame in scores.items():
        series = frame.rename_axis(index='formation_date', columns='asset').stack()
        panel = panel.join(series.rename(fold), on=['formation_date', 'asset'], validate='m:1')
    panel['eligible'] = True
    dated = compute_matched_signal_ic(panel, signals=['A', 'B'], targets=['raw', *adjustments],
                                      reference_target='raw', min_obs_per_date=min_obs_per_date)
    keys = ['horizon', 'formation_date', 'realisation_date', 'n_assets']
    wide = dated.query("sample == 'common'").pivot(index=keys, columns=['signal', 'target'],
                                                   values='ic')
    parts = []
    for leg in ('score_factor', 'joint'):
        for fold, other in [('A', 'B'), ('B', 'A')]:
            pair = wide[[(fold, f'{leg}_{fold}'), (fold, f'{leg}_{other}')]].copy()
            pair.columns = ['same', 'disjoint']
            parts.append(pair.assign(hedge=leg, score_fold=fold).reset_index())
    folds = pd.concat(parts, ignore_index=True)
    groups = folds.groupby([*keys, 'hedge'])[['same', 'disjoint']]
    means = groups.mean().where(groups.count().eq(2).all(axis=1), axis=0)
    delta = (means.same - means.disjoint).rename('ic_minus_reference')
    observations = means.reset_index().melt(id_vars=[*keys, 'hedge'], var_name='target',
                                            value_name='ic')
    observations = observations.join(delta, on=[*keys, 'hedge'])
    observations.loc[observations.target.eq('disjoint'), 'ic_minus_reference'] = np.nan
    observations = observations.assign(sample='common', signal='low_beta', method='spearman',
                                       reference_target='disjoint')
    report = compute_signal_ic_summary(observations, group_by=['hedge'], ewma_span=ewma_span,
                                       rolling_span=rolling_span, warmup_periods=warmup_periods)
    return report, folds


def decompose_rank_ic(scores: pd.DataFrame, outcomes: pd.DataFrame,
                      date_weights: pd.Series, *, varying_membership: bool = False) -> pd.DataFrame:
    """Attribute a weighted mean dated Spearman IC to persistent and within-asset ranks.

    Args:
        scores: Formation scores by asset, indexed by the realisation dates of their targets.
        outcomes: Corresponding forward returns with the same asset labels.
        date_weights: Weights by realisation date, e.g. the selected rows of SignalICReport.dated.
            Positive finite weights are normalised. Zero and missing weights exclude dates.
        varying_membership: Rank finite score/outcome pairs on each selected date. Each asset's
            persistent contribution uses its conditional mean unit ranks multiplied by its
            share of date weight. Within-asset deviations use those same conditional means.

    Returns:
        One row per asset with its weighted mean ranks and its two IC contributions.
        Summing both contributions over assets reproduces the weighted mean rank IC.
        Ties receive average ranks. By default selected dates require the same finite basket and
        nonconstant ranks. With no selected dates the contributions are missing.
    """
    if (not scores.index.is_unique or not outcomes.index.is_unique
            or not date_weights.index.is_unique
            or not scores.columns.is_unique or not outcomes.columns.is_unique
            or set(scores.columns) != set(outcomes.columns)):
        raise ValueError('expected unique dates and matching unique asset labels')
    if (date_weights.dropna() < 0).any() or np.isinf(date_weights).any():
        raise ValueError('date weights must be nonnegative and finite when supplied')
    w = date_weights[date_weights.gt(0)].astype(float)
    result = pd.DataFrame(index=scores.columns.rename('asset'),
                          columns=['mean_score_rank', 'mean_outcome_rank', 'persistent', 'within'],
                          dtype=float)
    if w.empty:
        return result
    w = w / w.sum()
    aligned = [p.reindex(index=w.index, columns=result.index) for p in (scores, outcomes)]
    matched = np.isfinite(aligned[0]) & np.isfinite(aligned[1])
    mass = matched.mul(w, axis=0).sum()
    if not varying_membership and not matched.all().all():
        raise ValueError('rank attribution requires a complete fixed basket on selected dates')
    units = []
    for panel, name in zip(aligned, ['mean_score_rank', 'mean_outcome_rank']):
        ranks = panel.where(matched).rank(axis=1, method='average')
        result[name] = ranks.mul(w, axis=0).sum() / mass
        centered = ranks.sub(ranks.mean(axis=1), axis=0)
        norm = np.sqrt(centered.pow(2).sum(axis=1))
        if norm.eq(0).any():
            raise ValueError('rank attribution requires nonconstant ranks on selected dates')
        units.append(centered.div(norm, axis=0))
    a, b = units
    mean_a = a.mul(w, axis=0).sum() / mass
    mean_b = b.mul(w, axis=0).sum() / mass
    result['persistent'] = mass * mean_a * mean_b
    result['within'] = ((a - mean_a) * (b - mean_b)).mul(w, axis=0).sum(min_count=1)
    return result
