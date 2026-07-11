"""
Cross-sectional predictive regression diagnostics for trading signals,
with per-asset native-cadence handling.

For an N-asset universe with a panel of signal scores and per-frequency
return panels, this module quantifies the cross-sectional predictive
content of the signal at one or more forward-return horizons via the
regression

        ỹ_{i,t,t+h} = β · z_{i,t-1} + ε_{i,t}      (default: no intercept)

where the forward window length h is expressed in **the asset's native
rebalancing cadence** — h=1 means one month for a monthly asset and one
quarter for a quarterly asset. This is the key difference from a naive
price-based diagnostic: assets that print quarterly are not forced onto
a monthly grid (which produces zero-then-jump return artefacts), but
instead are evaluated at horizon h in units of their native cadence.

The cross-sectional normalisation at each "regression date" uses
**whichever assets are active on that date** (i.e. have a non-NaN
forward return at that horizon). This is universe-wide and includes
mixed-cadence assets simultaneously when they happen to print on the
same date.

Two views are produced for each horizon:

    1. Pooled regression on universe-normalised pairs — a single
       (β, t-stat, IC) statement of overall signal quality across all
       assets and all rebalance dates.

    2. Per-group regression with within-group normalisation — one β per
       group label, useful for attributing where the signal works in
       segmented universes.

String horizons (e.g. 'YE') override per-asset cadence: all assets are
resampled to that frequency directly. Use for headline 12-month tests.

This module's ``estimate_signal_diagnostics`` is the compute entry
point; see ``qis.plots.derived.signal_diagnostics_plot`` for plots.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Sequence, Tuple, Union


class SignalDiagnosticsColumns(str, Enum):
    """Column-name constants for the regression result DataFrames."""
    N = 'n'
    BETA = 'beta'
    SE = 'se'
    T_STAT = 't_stat'
    IC_PEARSON = 'IC_pearson'
    IC_SPEARMAN = 'IC_spearman'


_STAT_COLS = [
    SignalDiagnosticsColumns.N.value,
    SignalDiagnosticsColumns.BETA.value,
    SignalDiagnosticsColumns.SE.value,
    SignalDiagnosticsColumns.T_STAT.value,
    SignalDiagnosticsColumns.IC_PEARSON.value,
    SignalDiagnosticsColumns.IC_SPEARMAN.value,
]


# IC-IR table columns (the time-series IC summary in ``estimate_ic_ir``).
# Kept separate from ``_STAT_COLS`` — those are the per-regression stats,
# these summarise the per-date IC series.
_IC_IR_COLS = [
    'n_dates', 'mean_IC', 'std_IC', 'IC_IR', 'IC_IR_an', 't_stat', 'hit_rate',
]


@dataclass
class SignalDiagnosticsResult:
    """Container for cross-sectional signal diagnostic outputs.

    Attributes:
        pooled_universe: DataFrame indexed by horizon label (e.g. '1', '3',
            '6' or 'YE'); columns are SignalDiagnosticsColumns. Each row
            is the universe-pooled regression for that horizon.

        per_group: DataFrame indexed by (horizon, group) MultiIndex with the
            same columns. Each row is the within-group regression. Empty
            DataFrame when no group_data was passed.

        pairs: dict keyed by horizon label, each value a long-format
            DataFrame with columns ['date', 'asset', 'group', 'asset_freq',
            'z', 'r_norm_univ', 'r_norm_group']. The underlying (signal,
            return) panel consumed by both regressions and by the plotting
            layer.

        horizon_labels: ordered list of horizon labels (matches index
            order of pooled_universe).

        group_order: ordered list of group labels present in per_group.

        start_date, end_date: span of the pairs sample.
    """
    pooled_universe: pd.DataFrame
    per_group: pd.DataFrame
    pairs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    horizon_labels: List[str] = field(default_factory=list)
    group_order: List[str] = field(default_factory=list)
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────


def _horizon_label(horizon: Union[int, str]) -> str:
    """Pretty label for a horizon. Integers become 'Nx' marker-free strings.

    Horizon is in *native cadence units* of each asset, so the same
    integer label e.g. '3' means 3m for monthly assets and 3q for
    quarterly assets. The label is left frequency-agnostic.
    """
    if isinstance(horizon, str):
        return horizon
    return f"{int(horizon)}"


def _asset_to_freq_map(
        asset_returns_dict: Dict[str, pd.DataFrame],
) -> Dict[str, str]:
    """Map each asset → its native frequency (the dict key it appears under).

    An asset that appears in multiple frequency panels (it should not)
    is mapped to the first frequency in dict-insertion order; a warning
    is emitted via the result of the call when this happens.
    """
    mapping: Dict[str, str] = {}
    for freq, df in asset_returns_dict.items():
        if df is None or df.empty:
            continue
        for col in df.columns:
            if col not in mapping:
                mapping[col] = freq
    return mapping


def _fit_through_origin(z: np.ndarray, r: np.ndarray) -> Optional[Dict[str, float]]:
    """No-intercept OLS: β = Σ(zr) / Σ(z²)."""
    mask = np.isfinite(z) & np.isfinite(r)
    z, r = z[mask], r[mask]
    n = len(z)
    if n < 5:
        return None
    zz = float((z * z).sum())
    if zz <= 0.0:
        return None
    beta = float((z * r).sum() / zz)
    e = r - beta * z
    sigma2 = float((e ** 2).sum() / (n - 1)) if n > 1 else np.nan
    se = float(np.sqrt(sigma2 / zz)) if np.isfinite(sigma2) and sigma2 > 0 else np.nan
    t_stat = beta / se if se > 0 else np.nan
    try:
        ic_p = float(scipy_stats.pearsonr(z, r)[0])
    except Exception:
        ic_p = np.nan
    try:
        ic_s = float(scipy_stats.spearmanr(z, r)[0])
    except Exception:
        ic_s = np.nan
    return {
        SignalDiagnosticsColumns.N.value: int(n),
        SignalDiagnosticsColumns.BETA.value: beta,
        SignalDiagnosticsColumns.SE.value: se,
        SignalDiagnosticsColumns.T_STAT.value: t_stat,
        SignalDiagnosticsColumns.IC_PEARSON.value: ic_p,
        SignalDiagnosticsColumns.IC_SPEARMAN.value: ic_s,
    }


def _fit_with_intercept(z: np.ndarray, r: np.ndarray) -> Optional[Dict[str, float]]:
    """OLS with intercept: returns slope coefficient stats only."""
    mask = np.isfinite(z) & np.isfinite(r)
    z, r = z[mask], r[mask]
    n = len(z)
    if n < 5:
        return None
    try:
        slope, _intercept, rp, _p, slope_se = scipy_stats.linregress(z, r)
    except ValueError:
        return None
    t_stat = slope / slope_se if slope_se > 0 else np.nan
    try:
        ic_s = float(scipy_stats.spearmanr(z, r)[0])
    except Exception:
        ic_s = np.nan
    return {
        SignalDiagnosticsColumns.N.value: int(n),
        SignalDiagnosticsColumns.BETA.value: float(slope),
        SignalDiagnosticsColumns.SE.value: float(slope_se),
        SignalDiagnosticsColumns.T_STAT.value: float(t_stat),
        SignalDiagnosticsColumns.IC_PEARSON.value: float(rp),
        SignalDiagnosticsColumns.IC_SPEARMAN.value: ic_s,
    }


def _build_pairs_int_horizon(
        asset_returns_dict: Dict[str, pd.DataFrame],
        asset_freq: Dict[str, str],
        signal_rs_by_freq: Dict[str, pd.DataFrame],
        horizon: int,
        group_data: Optional[pd.Series],
        is_log_returns: bool,
) -> pd.DataFrame:
    """Build pairs for an integer horizon in **per-asset native cadence**.

    For each frequency frame in the dict:
        - take that frame's returns
        - cumulate over a window of `horizon` periods of that frame
        - sample every `horizon`-th period for non-overlap
        - lag the signal one period of the same frame
        - tag each row with the asset's native freq

    Returns a long DataFrame with one row per (date, asset). The
    'date' column is the asset's regression date in its native cadence.
    Cross-sectional normalisation across the universe happens
    downstream after pooling.
    """
    rows: List[Dict] = []
    for freq, returns_df in asset_returns_dict.items():
        if returns_df is None or returns_df.empty:
            continue
        # signal_rs is the signal panel resampled to this frequency
        signal_rs = signal_rs_by_freq[freq]
        # Only consider assets whose native freq is THIS freq
        assets_here = [c for c in returns_df.columns if asset_freq.get(c) == freq]
        if not assets_here:
            continue
        # Forward cumulative return over h periods of this freq:
        # at date t, cum_fwd[t] = return over [t, t + h-1] (h periods)
        ret = returns_df[assets_here]
        if is_log_returns:
            cum = ret.rolling(horizon).sum()
        else:
            cum = (1.0 + ret).rolling(horizon).apply(np.prod, raw=True) - 1.0
        cum_fwd = cum.shift(-horizon + 1)
        # Lagged signal, sampled at same frequency
        sig_lag = signal_rs[assets_here].shift(1)
        # Align indices: keep dates present in both
        common = cum_fwd.index.intersection(sig_lag.index)
        # Non-overlapping: every h-th date in this asset's native cadence
        sampled = common[::horizon]
        for d in sampled:
            for asset in assets_here:
                z = sig_lag.loc[d, asset]
                r = cum_fwd.loc[d, asset]
                if not (np.isfinite(z) and np.isfinite(r)):
                    continue
                rows.append({
                    'date': d, 'asset': asset, 'asset_freq': freq,
                    'group': group_data.get(asset) if group_data is not None else None,
                    'z': float(z), 'r': float(r),
                })
    return pd.DataFrame(rows, columns=['date', 'asset', 'asset_freq',
                                       'group', 'z', 'r'])


def _build_pairs_string_horizon(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: pd.DataFrame,
        horizon_freq: str,
        group_data: Optional[pd.Series],
        is_log_returns: bool,
) -> pd.DataFrame:
    """String-horizon override: resample every asset's NAV to this frequency.

    Reconstructs NAVs per native-cadence frame via cumulation, concatenates
    column-wise, then re-resamples to ``horizon_freq``. One observation
    per asset per horizon_freq period.
    """
    # Reconstruct NAVs at each asset's native cadence, then merge column-wise
    per_freq_nav = []
    asset_freq = _asset_to_freq_map(asset_returns_dict)
    for freq, returns_df in asset_returns_dict.items():
        if returns_df is None or returns_df.empty:
            continue
        # Build NAV from returns at native cadence; first row → 1.0
        if is_log_returns:
            nav = np.exp(returns_df.cumsum())
        else:
            nav = (1.0 + returns_df).cumprod()
        # First observation set to 1.0 (preserves the diff)
        nav.iloc[0] = nav.iloc[0].where(nav.iloc[0].notna(), 1.0)
        per_freq_nav.append(nav)
    if not per_freq_nav:
        return pd.DataFrame(columns=['date', 'asset', 'asset_freq', 'group', 'z', 'r'])
    full_nav = pd.concat(per_freq_nav, axis=1).sort_index().ffill()
    # Resample to horizon_freq
    nav_rs = full_nav.resample(horizon_freq).last().ffill()
    # Returns at this freq
    if is_log_returns:
        ret_rs = np.log(nav_rs).diff()
    else:
        ret_rs = nav_rs.pct_change()
    sig_rs = signal.resample(horizon_freq).last().shift(1)
    common = ret_rs.index.intersection(sig_rs.index)
    rows: List[Dict] = []
    for d in common:
        for asset in ret_rs.columns:
            if asset not in sig_rs.columns:
                continue
            z = sig_rs.loc[d, asset]
            r = ret_rs.loc[d, asset]
            if not (np.isfinite(z) and np.isfinite(r)):
                continue
            rows.append({
                'date': d, 'asset': asset, 'asset_freq': asset_freq.get(asset),
                'group': group_data.get(asset) if group_data is not None else None,
                'z': float(z), 'r': float(r),
            })
    return pd.DataFrame(rows, columns=['date', 'asset', 'asset_freq',
                                       'group', 'z', 'r'])


def _apply_cross_sectional_normalisation(
        df: pd.DataFrame, is_vol_normalised: bool,
        min_obs_per_date: int,
) -> pd.DataFrame:
    """Add r_norm_univ and r_norm_group columns to a pairs DataFrame.

    Cross-sectional normalisation is universe-wide at each regression
    date, using whichever assets are active on that date. Within-group
    normalisation uses the same date filter, restricted to the group.

    Dates with fewer than ``min_obs_per_date`` active assets are dropped.
    Group cells with fewer than 2 active members are left as NaN in
    r_norm_group.
    """
    df = df.copy()
    df['r_norm_univ'] = np.nan
    df['r_norm_group'] = np.nan

    keep_rows: List[int] = []
    for d, grp_df in df.groupby('date'):
        r_vals = grp_df['r'].to_numpy()
        if len(r_vals) < min_obs_per_date:
            continue
        u_mean = float(r_vals.mean())
        if is_vol_normalised:
            u_std = float(r_vals.std(ddof=1))
            if u_std <= 0.0:
                continue
        else:
            u_std = 1.0
        idx = grp_df.index
        df.loc[idx, 'r_norm_univ'] = (grp_df['r'].to_numpy() - u_mean) / u_std
        keep_rows.extend(idx.tolist())

        # Per-group, this same date
        if 'group' in grp_df.columns:
            for g, sub in grp_df.groupby('group', dropna=False):
                if g is None or pd.isna(g):
                    continue
                rg = sub['r'].to_numpy()
                if len(rg) < 2:
                    continue
                g_mean = float(rg.mean())
                if is_vol_normalised:
                    g_std = float(rg.std(ddof=1))
                    if g_std <= 0.0:
                        continue
                else:
                    g_std = 1.0
                df.loc[sub.index, 'r_norm_group'] = (rg - g_mean) / g_std
    return df.loc[keep_rows].reset_index(drop=True)


# ───────────────────────────────────────────────────────────────────────────────
# Public entry point
# ───────────────────────────────────────────────────────────────────────────────


def estimate_signal_diagnostics(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: pd.DataFrame,
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 3, 6),
        fit_intercept: bool = False,
        is_log_returns: bool = True,
        is_vol_normalised: bool = True,
        min_obs_per_date: int = 5,
        min_obs_per_group: int = 10,
        group_order: Optional[Sequence[str]] = None,
) -> SignalDiagnosticsResult:
    """Cross-sectional predictive regression of forward returns on lagged signal.

    For each horizon h, builds non-overlapping (signal, return) pairs
    ``(z_{i,t-1}, ỹ_{i,t,t+h})`` and fits

        ỹ_{i,t,t+h} = β · z_{i,t-1} + ε   (default: no intercept)

    The forward window length h is in **native cadence units of each
    asset** — h=1 means 1 month for a monthly asset and 1 quarter for a
    quarterly asset. This avoids the zero-then-jump return artefacts
    that arise when quarterly NAVs are forced onto a monthly grid.

    String horizons (e.g. 'YE') override per-asset cadence — all assets
    are resampled to that frequency. Use for headline annual tests.

    Args:
        asset_returns_dict: Per-frequency returns dict from a pipeline
            that already handles FX adjustment and unsmoothing. Keys are
            pandas frequency strings (e.g. 'ME', 'QE', 'YE'); values are
            return DataFrames indexed at that frequency's period-ends
            with asset tickers as columns. Each asset must appear in
            exactly one frame (its native cadence).

        signal: T x N signal panel (e.g. ``AlphasData.alpha_scores``).
            Columns must include every asset in ``asset_returns_dict``.
            Signal panel is typically at the finest frequency present in
            the dict (e.g. monthly); the function resamples it to each
            asset's native cadence.

        group_data: Optional Series mapping asset name → group label.
            When None, only the pooled regression is run.

        horizons: Forward-return horizons. Integers are in native-cadence
            units (1, 3, 6 → 1, 3, 6 native periods per asset). Strings
            like 'YE' override per-asset cadence and resample uniformly.

        fit_intercept: Include α in the regression. Default False —
            cross-sectional demeaning of the LHS makes α = 0 by
            construction.

        is_log_returns: Set True when ``asset_returns_dict`` contains log
            returns (default), False for arithmetic. Affects cumulation
            across horizons.

        is_vol_normalised: Divide cross-sectional return by cross-
            sectional std at each date (default True).

        min_obs_per_date: Minimum cross-sectional sample at a date.

        min_obs_per_group: Minimum sample size for a group's regression
            to be reported.

        group_order: Explicit ordering for the groups in per_group.

    Returns:
        ``SignalDiagnosticsResult`` with ``.pooled_universe``,
        ``.per_group``, and ``.pairs`` populated.
    """
    if group_data is not None and not isinstance(group_data, pd.Series):
        raise TypeError("group_data must be a pandas Series mapping asset -> group label")
    if not isinstance(asset_returns_dict, dict) or not asset_returns_dict:
        raise ValueError("asset_returns_dict must be a non-empty dict")

    # Restrict each frequency frame to assets the signal panel actually
    # covers. Per-component diagnostics (e.g. running against
    # momentum_score on a universe that includes PE/HF funds) naturally
    # produce a signal panel narrower than the full universe — the
    # MANAGERS_ALPHA signal covers PE/HF, but MOMENTUM does not, so the
    # MOMENTUM score panel has fewer columns. Project the returns dict
    # accordingly rather than raising, but warn if every column drops.
    signal_cols = set(signal.columns)
    projected_returns: Dict[str, pd.DataFrame] = {}
    dropped_assets: List[str] = []
    for freq, df in asset_returns_dict.items():
        if df is None or df.empty:
            continue
        keep_cols = [c for c in df.columns if c in signal_cols]
        if keep_cols:
            projected_returns[freq] = df[keep_cols]
        dropped_assets.extend([c for c in df.columns if c not in signal_cols])
    if not projected_returns:
        raise ValueError(
            "No overlap between asset_returns_dict columns and signal "
            "panel columns — signal does not cover any asset in the "
            f"returns dict (first 5 dropped: {sorted(set(dropped_assets))[:5]})."
        )
    asset_returns_dict = projected_returns

    # Group ordering
    if group_data is not None:
        if group_order is None:
            seen: List[str] = []
            for _, g in group_data.items():
                if g is not None and not pd.isna(g) and g not in seen:
                    seen.append(g)
            group_order_list = seen
        else:
            group_order_list = list(group_order)
    else:
        group_order_list = []

    # Asset → native frequency map
    asset_freq = _asset_to_freq_map(asset_returns_dict)

    # Pre-resample signal panel to each frequency
    signal_rs_by_freq: Dict[str, pd.DataFrame] = {}
    for freq in asset_returns_dict.keys():
        signal_rs_by_freq[freq] = signal.resample(freq).last()

    pooled_rows: Dict[str, Dict[str, float]] = {}
    group_rows: Dict[Tuple[str, str], Dict[str, float]] = {}
    pairs_by_horizon: Dict[str, pd.DataFrame] = {}
    horizon_labels: List[str] = []
    fitter = _fit_with_intercept if fit_intercept else _fit_through_origin

    overall_start: Optional[pd.Timestamp] = None
    overall_end: Optional[pd.Timestamp] = None

    for horizon in horizons:
        if isinstance(horizon, str):
            raw_pairs = _build_pairs_string_horizon(
                asset_returns_dict=asset_returns_dict, signal=signal,
                horizon_freq=horizon, group_data=group_data,
                is_log_returns=is_log_returns,
            )
            label = horizon
        elif isinstance(horizon, (int, np.integer)) and horizon >= 1:
            raw_pairs = _build_pairs_int_horizon(
                asset_returns_dict=asset_returns_dict,
                asset_freq=asset_freq,
                signal_rs_by_freq=signal_rs_by_freq,
                horizon=int(horizon),
                group_data=group_data,
                is_log_returns=is_log_returns,
            )
            label = _horizon_label(horizon)
        else:
            raise ValueError(f"horizon {horizon!r} must be a positive int or a "
                             f"pandas frequency string")
        horizon_labels.append(label)

        # Cross-sectional normalisation across the active universe at each date
        normed = _apply_cross_sectional_normalisation(
            raw_pairs, is_vol_normalised=is_vol_normalised,
            min_obs_per_date=min_obs_per_date,
        )
        pairs_by_horizon[label] = normed

        if len(normed) > 0:
            d_min = normed['date'].min()
            d_max = normed['date'].max()
            overall_start = d_min if overall_start is None else min(overall_start, d_min)
            overall_end = d_max if overall_end is None else max(overall_end, d_max)

        # Pooled regression (universe-normalised)
        if len(normed) > 0:
            fit = fitter(normed['z'].to_numpy(), normed['r_norm_univ'].to_numpy())
            pooled_rows[label] = fit if fit is not None else {c: np.nan for c in _STAT_COLS}
        else:
            pooled_rows[label] = {c: np.nan for c in _STAT_COLS}

        # Per-group (within-group normalised)
        if group_data is not None:
            for g in group_order_list:
                sub = normed[normed['group'] == g].dropna(
                    subset=['z', 'r_norm_group'])
                if len(sub) < min_obs_per_group:
                    continue
                fit_g = fitter(sub['z'].to_numpy(), sub['r_norm_group'].to_numpy())
                if fit_g is not None:
                    group_rows[(label, g)] = fit_g

    pooled_df = pd.DataFrame.from_dict(pooled_rows, orient='index')[_STAT_COLS]
    pooled_df.index.name = 'horizon'

    if group_rows:
        per_group_df = pd.DataFrame.from_dict(group_rows, orient='index')[_STAT_COLS]
        per_group_df.index = pd.MultiIndex.from_tuples(
            per_group_df.index, names=['horizon', 'group'],
        )
    else:
        per_group_df = pd.DataFrame(
            columns=_STAT_COLS,
            index=pd.MultiIndex.from_tuples([], names=['horizon', 'group']),
        )

    return SignalDiagnosticsResult(
        pooled_universe=pooled_df,
        per_group=per_group_df,
        pairs=pairs_by_horizon,
        horizon_labels=horizon_labels,
        group_order=group_order_list,
        start_date=overall_start,
        end_date=overall_end,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Per-asset β extraction (for cross-asset dispersion boxplots)
# ───────────────────────────────────────────────────────────────────────────────


def compute_per_asset_betas(
        result: SignalDiagnosticsResult,
        min_obs_per_asset: int = 12,
        fit_intercept: bool = False,
) -> pd.DataFrame:
    """Estimate one β per (asset, horizon) from the diagnostic pairs.

    For each asset and each horizon, runs the same no-intercept (default)
    regression as the pooled diagnostic but restricted to that asset's
    time-series of (z, r_norm_univ) pairs:

        ỹ_{i,t,t+h} = β_i · z_{i,t-1} + ε

    The LHS is the universe-normalised forward return (``r_norm_univ``)
    — same convention as the pooled regression, so per-asset β values
    are directly comparable to the pooled β.

    Useful for cross-asset dispersion visualisations (e.g. boxplot of β
    across assets at each horizon) — a complement to the pooled and
    per-group regressions.

    Args:
        result: ``SignalDiagnosticsResult`` from
            ``estimate_signal_diagnostics``.
        min_obs_per_asset: Minimum (z, r) pair count per asset per
            horizon required to report a β. Assets with fewer
            observations are dropped from that horizon's row set.
        fit_intercept: Match the corresponding flag in the pooled fit.
            Default ``False`` for symmetry with the pooled regression.

    Returns:
        Long-format DataFrame with columns
        ``[horizon, asset, asset_freq, group, beta, t_stat, n]``. One
        row per (asset, horizon) cell that passed the
        ``min_obs_per_asset`` filter.
    """
    fitter = _fit_with_intercept if fit_intercept else _fit_through_origin

    rows: List[Dict] = []
    for horizon_label in result.horizon_labels:
        pairs = result.pairs.get(horizon_label)
        if pairs is None or pairs.empty:
            continue
        for asset, sub in pairs.groupby('asset'):
            if len(sub) < min_obs_per_asset:
                continue
            fit = fitter(sub['z'].to_numpy(),
                         sub['r_norm_univ'].to_numpy())
            if fit is None:
                continue
            # Use the asset's freq and group from the first row of this
            # asset's pair frame — they're constant per asset.
            asset_freq = sub['asset_freq'].iloc[0]
            group = sub['group'].iloc[0]
            rows.append({
                'horizon': horizon_label,
                'asset': asset,
                'asset_freq': asset_freq,
                'group': group,
                SignalDiagnosticsColumns.BETA.value:
                    fit[SignalDiagnosticsColumns.BETA.value],
                SignalDiagnosticsColumns.T_STAT.value:
                    fit[SignalDiagnosticsColumns.T_STAT.value],
                SignalDiagnosticsColumns.N.value:
                    fit[SignalDiagnosticsColumns.N.value],
            })

    if not rows:
        return pd.DataFrame(columns=['horizon', 'asset', 'asset_freq',
                                     'group',
                                     SignalDiagnosticsColumns.BETA.value,
                                     SignalDiagnosticsColumns.T_STAT.value,
                                     SignalDiagnosticsColumns.N.value])
    out = pd.DataFrame(rows)
    # Preserve horizon ordering as in result.horizon_labels.
    out['horizon'] = pd.Categorical(out['horizon'],
                                    categories=result.horizon_labels,
                                    ordered=True)
    out = out.sort_values(['horizon', 'asset']).reset_index(drop=True)
    return out
# ───────────────────────────────────────────────────────────────────────────────
# IC information ratio (IC-IR): time-series stability of the per-date IC
# ───────────────────────────────────────────────────────────────────────────────


def _per_date_ic(
        pairs: pd.DataFrame,
        method: str = 'spearman',
        return_col: str = 'r_norm_univ',
        min_obs_per_date: int = 5,
) -> pd.DataFrame:
    """Per-date cross-sectional IC of (z, forward return) from a pairs frame.

    ``estimate_signal_diagnostics`` pools the whole panel into a single IC;
    this instead computes ONE IC per rebalance date, giving the time
    series whose mean/std define the IC-IR.

    Args:
        pairs: One horizon's normalised pairs frame (a value of
            ``SignalDiagnosticsResult.pairs``). Needs ``date``, ``z`` and
            ``return_col``.
        method: 'spearman' (rank IC, default) or 'pearson'.
        return_col: Forward-return column — ``r_norm_univ`` (universe
            cross-section, default) or ``r_norm_group`` (within group),
            matching the pooled vs per-group views.
        min_obs_per_date: Dates with fewer active names are dropped; a
            rank IC on 2-3 names is meaningless.

    Returns:
        DataFrame indexed by date with columns ``['n', 'IC']``, sorted by
        date. Empty when no date clears ``min_obs_per_date``.
    """
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame(columns=['n', 'IC'])
    corr = scipy_stats.spearmanr if method == 'spearman' else scipy_stats.pearsonr
    recs: List[Tuple] = []
    for d, sub in pairs.groupby('date'):
        z = sub['z'].to_numpy()
        r = sub[return_col].to_numpy()
        mask = np.isfinite(z) & np.isfinite(r)
        z, r = z[mask], r[mask]
        if len(z) < min_obs_per_date:
            continue
        if np.std(z) == 0.0 or np.std(r) == 0.0:  # corr undefined on a constant
            continue
        try:
            ic = float(corr(z, r)[0])
        except Exception:
            continue
        if np.isfinite(ic):
            recs.append((d, len(z), ic))
    if not recs:
        return pd.DataFrame(columns=['n', 'IC'])
    return (pd.DataFrame(recs, columns=['date', 'n', 'IC'])
            .set_index('date').sort_index())


def _infer_periods_per_year(index: pd.Index) -> float:
    """Annualisation factor from the median spacing of the IC-series dates.

    Handles native-cadence / non-overlapping spacing automatically (a
    horizon-3 monthly series is spaced ~3m → ~4/yr), so the caller need
    not know the rebalance frequency.
    """
    if len(index) < 2:
        return float('nan')
    days = pd.Series(pd.to_datetime(index)).sort_values().diff().dropna().dt.days
    med = float(days.median()) if len(days) else float('nan')
    return 365.25 / med if (med and med > 0) else float('nan')


def compute_ic_timeseries(
        result: SignalDiagnosticsResult,
        method: str = 'spearman',
        return_col: str = 'r_norm_univ',
        min_obs_per_date: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Per-date IC series for every horizon in a diagnostic result.

    Mirrors ``compute_per_asset_betas`` in shape — consumes a
    ``SignalDiagnosticsResult`` and loops ``result.horizon_labels``.
    Useful for plotting the IC time series / cumulative IC and inspecting
    IC decay across horizons.

    Args:
        result: ``SignalDiagnosticsResult`` from
            ``estimate_signal_diagnostics``.
        method: 'spearman' (default) or 'pearson' for the per-date IC.
        return_col: 'r_norm_univ' (default) or 'r_norm_group'.
        min_obs_per_date: Minimum cross-section per date.

    Returns:
        ``{horizon_label: DataFrame[date -> (n, IC)]}`` in horizon order.
    """
    return {
        label: _per_date_ic(
            result.pairs.get(label), method=method,
            return_col=return_col, min_obs_per_date=min_obs_per_date,
        )
        for label in result.horizon_labels
    }


def estimate_ic_ir(
        result: SignalDiagnosticsResult,
        method: str = 'spearman',
        return_col: str = 'r_norm_univ',
        periods_per_year: Optional[float] = None,
        min_obs_per_date: int = 5,
) -> pd.DataFrame:
    """IC information ratio per horizon — the time-series counterpart to the
    pooled IC in ``pooled_universe``.

    For each horizon the per-date cross-sectional IC series is summarised::

        IC_IR     = mean(IC) / std(IC)               (per period)
        IC_IR_an  = IC_IR * sqrt(periods_per_year)   (annualised)
        t_stat    = IC_IR * sqrt(n_dates)            (significance of mean IC)
        hit_rate  = mean(IC > 0)

    ``IC_IR`` is the breadth-adjusted quality of the signal: it rewards an
    IC that is consistently the right sign, not merely large on average.
    It is the honest stability number the pooled ``t_stat`` is not — the
    pooled regression in ``estimate_signal_diagnostics`` treats every
    (asset, date) pair as iid and so overstates significance when the
    cross-section is correlated within a date; this collapses each date to
    one observation.

    Args:
        result: ``SignalDiagnosticsResult`` from
            ``estimate_signal_diagnostics``.
        method: 'spearman' (default) or 'pearson' for the per-date IC.
        return_col: 'r_norm_univ' (universe cross-section, default) or
            'r_norm_group' (within-group — pass ``group_data`` to
            ``estimate_signal_diagnostics`` to populate it).
        periods_per_year: Annualisation factor for ``IC_IR_an``; inferred
            from the median IC-date spacing when None.
        min_obs_per_date: Minimum cross-section per date.

    Returns:
        DataFrame indexed by horizon label, columns ``_IC_IR_COLS``.
    """
    ts = compute_ic_timeseries(
        result, method=method, return_col=return_col,
        min_obs_per_date=min_obs_per_date,
    )
    rows: Dict[str, Dict[str, float]] = {}
    for label in result.horizon_labels:
        ic = ts.get(label)
        if ic is None or ic.empty:
            rows[label] = {c: np.nan for c in _IC_IR_COLS}
            continue
        s = ic['IC']
        n_dates = int(s.shape[0])
        mean_ic = float(s.mean())
        std_ic = float(s.std(ddof=1)) if n_dates > 1 else np.nan
        ppy = (periods_per_year if periods_per_year is not None
               else _infer_periods_per_year(ic.index))
        ic_ir = mean_ic / std_ic if (std_ic and std_ic > 0) else np.nan
        rows[label] = {
            'n_dates': n_dates,
            'mean_IC': mean_ic,
            'std_IC': std_ic,
            'IC_IR': ic_ir,
            'IC_IR_an': ic_ir * np.sqrt(ppy) if (np.isfinite(ic_ir) and np.isfinite(ppy)) else np.nan,
            't_stat': ic_ir * np.sqrt(n_dates) if np.isfinite(ic_ir) else np.nan,
            'hit_rate': float((s > 0).mean()),
        }
    out = pd.DataFrame.from_dict(rows, orient='index')[_IC_IR_COLS]
    out.index.name = 'horizon'
    return out