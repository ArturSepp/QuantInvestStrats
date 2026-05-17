"""
Cross-sectional predictive regression diagnostics for trading signals.

For an N-asset universe with a panel of signal scores and prices, this module
quantifies the cross-sectional predictive content of the signal at one or more
forward-return horizons via the regression

        ỹ_{i,t,t+h} = β · z_{i,t-1} + ε_{i,t}      (default: no intercept)

where
    - z_{i,t-1}    is the signal value for asset i at the prior rebalance date
    - ỹ_{i,t,t+h} is the cross-sectionally vol-normalised cumulative return of
                  asset i over the forward window [t, t+h-1]:

                      ỹ_{i,t,t+h} = (r_{i,t,t+h} - mean_j r_{j,t,t+h})
                                    / std_j r_{j,t,t+h}

The cross-sectional normalisation strips out the universe-wide directional move
at each date, isolating the relative ranking information. The regression
intercept is dropped by default because, with cross-sectionally demeaned LHS,
α = 0 by construction.

Two views are produced for each horizon:

    1. Pooled regression on universe-normalised pairs across all assets / all
       dates --- a single (β, t-stat, IC) statement of overall signal quality.

    2. Per-group regression with within-group normalisation, where each group's
       β is estimated against returns demeaned within the group. This tests
       whether the signal discriminates inside each segment of the universe.

The non-overlapping rebalancing scheme samples one observation per asset per
forward-window length (e.g. for h = 6 with monthly rebalancing, every 6th
month-end). This keeps standard errors interpretable without requiring
heteroskedasticity-and-autocorrelation corrections.

This module's `estimate_signal_diagnostics` function is the compute entry
point; see `qis.plots.derived.signal_diagnostics_plot` for visualisation.
"""
# built-in
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Sequence, Union


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


@dataclass
class SignalDiagnosticsResult:
    """Container for cross-sectional signal diagnostic outputs.

    Attributes:
        pooled_universe: DataFrame indexed by horizon label (e.g. '1m', '3m',
            '6m' or 'YE'); columns are SignalDiagnosticsColumns. Each row is
            the universe-pooled regression for that horizon.

        per_group: DataFrame indexed by (horizon, group) MultiIndex; same
            columns. Each row is the within-group regression for that
            (horizon, group) cell. Empty DataFrame when no group_data passed.

        pairs: dict keyed by horizon label, each value a long-format
            DataFrame with columns ['date', 'asset', 'group', 'z',
            'r_norm_univ', 'r_norm_group']. The underlying (signal, return)
            panel consumed by both regressions and by the plotting layer.

        horizon_labels: ordered list of horizon labels (matches index order of
            pooled_universe).

        group_order: ordered list of group labels present in per_group;
            empty if group_data not provided.

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


def _horizon_label(horizon: Union[int, str], rebalance_freq: str) -> str:
    """Pretty label for a horizon spec.

    Integer horizons are labelled as '{n}m' when rebalance_freq starts with 'M'
    (monthly), '{n}q' for quarterly, etc. String horizons (e.g. 'YE') are passed
    through verbatim.
    """
    if isinstance(horizon, str):
        return horizon
    suffix_map = {'M': 'm', 'Q': 'q', 'W': 'w', 'D': 'd', 'A': 'y', 'Y': 'y'}
    suffix = suffix_map.get(rebalance_freq[0].upper(), '')
    return f"{horizon}{suffix}"


def _fit_through_origin(z: np.ndarray, r: np.ndarray) -> Optional[Dict[str, float]]:
    """No-intercept OLS: β = Σ(zr) / Σ(z²).

    Returns dict with n, beta, se, t_stat, IC_pearson, IC_spearman.
    Returns None when the sample is too small or singular.
    """
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
    # σ̂² = e'e / (n - 1)  (one estimated parameter)
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


def _build_pairs_for_horizon(
        prices_rs: pd.DataFrame,
        signal_rs: pd.DataFrame,
        horizon_periods: int,
        group_data: Optional[pd.Series],
        group_order: List[str],
        is_log_returns: bool,
        is_vol_normalised: bool,
        min_obs_per_date: int,
) -> pd.DataFrame:
    """Non-overlapping pair builder for a single horizon.

    Returns long-format DataFrame with columns
    ['date', 'asset', 'group', 'z', 'r_norm_univ', 'r_norm_group'].
    """
    # Forward cumulative returns: cum_fwd[t] = log return over [t, t+horizon-1]
    if is_log_returns:
        per_period = np.log(prices_rs).diff()
        cum = per_period.rolling(horizon_periods).sum()
    else:
        per_period = prices_rs.pct_change()
        cum = (1.0 + per_period).rolling(horizon_periods).apply(np.prod, raw=True) - 1.0
    cum_fwd = cum.shift(-horizon_periods + 1)

    # Signal at t-1 (one rebalance lag, by convention)
    signal_lag = signal_rs.shift(1)

    common_index = cum_fwd.index.intersection(signal_lag.index)
    sampled_dates = common_index[::horizon_periods]   # non-overlapping along time

    asset_cols = list(signal_rs.columns)
    rows: List[Dict] = []
    for d in sampled_dates:
        z_row = signal_lag.loc[d]
        r_row = cum_fwd.loc[d]
        valid = z_row.notna() & r_row.notna()
        cols_ok = valid[valid].index.tolist()
        if len(cols_ok) < min_obs_per_date:
            continue

        r_vals = r_row[cols_ok].to_numpy()
        u_mean = float(r_vals.mean())
        if is_vol_normalised:
            u_std = float(r_vals.std(ddof=1))
            if u_std <= 0.0:
                continue
        else:
            u_std = 1.0

        group_stats: Dict[str, Optional[tuple]] = {}
        if group_data is not None:
            for g in group_order:
                cols_g = [t for t in cols_ok if group_data.get(t) == g]
                if len(cols_g) < 2:
                    group_stats[g] = None
                    continue
                rg = r_row[cols_g].to_numpy()
                g_mean = float(rg.mean())
                if is_vol_normalised:
                    g_std = float(rg.std(ddof=1))
                    group_stats[g] = (g_mean, g_std if g_std > 0.0 else None)
                else:
                    group_stats[g] = (g_mean, 1.0)

        for asset in cols_ok:
            grp = group_data.get(asset) if group_data is not None else None
            r_value = float(r_row[asset])
            r_norm_univ = (r_value - u_mean) / u_std
            r_norm_group = np.nan
            if grp is not None and group_stats.get(grp) is not None:
                g_mean, g_std = group_stats[grp]
                if g_std is not None:
                    r_norm_group = (r_value - g_mean) / g_std
            rows.append({
                'date': d, 'asset': asset, 'group': grp,
                'z': float(z_row[asset]),
                'r_norm_univ': r_norm_univ,
                'r_norm_group': r_norm_group,
            })

    return pd.DataFrame(rows, columns=['date', 'asset', 'group', 'z',
                                       'r_norm_univ', 'r_norm_group'])


# ───────────────────────────────────────────────────────────────────────────────
# Public entry point
# ───────────────────────────────────────────────────────────────────────────────


def estimate_signal_diagnostics(
        prices: pd.DataFrame,
        signal: pd.DataFrame,
        group_data: Optional[pd.Series] = None,
        horizons: Sequence[Union[int, str]] = (1, 3, 6),
        rebalance_freq: str = 'ME',
        fit_intercept: bool = False,
        is_log_returns: bool = True,
        is_vol_normalised: bool = True,
        min_obs_per_date: int = 5,
        min_obs_per_group: int = 10,
        group_order: Optional[Sequence[str]] = None,
) -> SignalDiagnosticsResult:
    """Cross-sectional predictive regression of forward returns on lagged signal.

    For each horizon h in `horizons`, builds non-overlapping (signal, return)
    pairs (z_{i,t-1}, ỹ_{i,t,t+h}) and fits

        ỹ_{i,t,t+h} = β · z_{i,t-1} + ε   (default: no intercept)

    where ỹ is the cross-sectionally (and optionally vol-) normalised forward
    return. Two regressions are run per horizon:

        - pooled with universe-wide normalisation: a single β across all assets
        - per-group with within-group normalisation: one β per group label

    The signal should already be on the scale you want to evaluate (e.g. mapped
    via `qis.map_signal_to_weight`). No transformation is applied here.

    Args:
        prices: T x N price panel. Index is date; columns are asset identifiers.
            Stale prices (e.g. quarterly NAVs) should be forward-filled before
            passing in.

        signal: T x N signal panel with the same column names as prices.
            Index need not match prices exactly; both are resampled to
            `rebalance_freq` internally.

        group_data: optional Series mapping asset name -> group label. When
            None, only the pooled regression is run and per_group in the
            output is empty.

        horizons: sequence of forward-return horizons. Integers are
            interpreted as units of `rebalance_freq` (1, 3, 6 -> 1m, 3m, 6m
            when rebalance_freq='ME'). Strings like 'YE' or '1Q' are
            interpreted as standalone resampling frequencies for that
            horizon (one observation per period).

        rebalance_freq: pandas frequency string for resampling prices and
            signal before pair construction (default 'ME'). Use this when
            horizons are integers; ignored for string horizons.

        fit_intercept: include α in the regression. Default False because the
            cross-sectional demeaning of the LHS makes α = 0 by construction.
            Set True if you want to allow a non-zero intercept.

        is_log_returns: cumulate forward returns in log space (default True).

        is_vol_normalised: divide cross-sectional (mean-demeaned) return by
            the cross-sectional std at each date (default True). When False,
            ỹ is the raw cross-sectional return.

        min_obs_per_date: minimum number of (z, r) pairs required at a
            rebalance date for the date to enter the sample.

        min_obs_per_group: minimum sample size for a group's regression to
            be reported. Groups with fewer observations are dropped from
            per_group.

        group_order: optional explicit ordering for the groups in per_group.
            Defaults to the order of first appearance in group_data.

    Returns:
        SignalDiagnosticsResult with pooled_universe, per_group, and
        pairs (per horizon) populated.

    Notes:
        - Non-overlapping sampling at horizon h takes every h-th rebalance
          date along time, producing independent (uncorrelated-residual)
          observations per asset across time.
        - Cross-sectional vol normalisation makes β comparable across
          horizons: without it, longer horizons mechanically have larger
          coefficients due to accumulating dispersion.
        - For string horizons like 'YE', prices and signal are resampled to
          that frequency directly; the per-period return is the full annual
          return.
    """
    if group_data is not None and not isinstance(group_data, pd.Series):
        raise TypeError("group_data must be a pandas Series mapping asset -> group label")
    if not set(signal.columns).issubset(set(prices.columns)):
        raise ValueError("signal columns must be a subset of prices columns")

    # Group ordering
    if group_data is not None:
        if group_order is None:
            seen = []
            for _, g in group_data.items():
                if g is not None and not pd.isna(g) and g not in seen:
                    seen.append(g)
            group_order = seen
        else:
            group_order = list(group_order)
    else:
        group_order = []

    # Resampled inputs (monthly by default) — forward-fill prices to handle
    # stale NAVs gracefully; this only affects per-date denominators for
    # cross-sectional normalisation and the cumulative return when the
    # underlying NAV has not updated.
    prices_rs_monthly = prices.resample(rebalance_freq).last().ffill()
    signal_rs_monthly = signal.resample(rebalance_freq).last()

    pooled_rows: Dict[str, Dict[str, float]] = {}
    group_rows: Dict[tuple, Dict[str, float]] = {}
    pairs_by_horizon: Dict[str, pd.DataFrame] = {}
    horizon_labels: List[str] = []
    fitter = _fit_with_intercept if fit_intercept else _fit_through_origin

    overall_start: Optional[pd.Timestamp] = None
    overall_end: Optional[pd.Timestamp] = None

    for horizon in horizons:
        if isinstance(horizon, str):
            # standalone-frequency horizon: resample to that frequency, h = 1 period
            prices_rs = prices.resample(horizon).last().ffill()
            signal_rs = signal.resample(horizon).last()
            horizon_periods = 1
            label = horizon
        elif isinstance(horizon, (int, np.integer)) and horizon >= 1:
            prices_rs = prices_rs_monthly
            signal_rs = signal_rs_monthly
            horizon_periods = int(horizon)
            label = _horizon_label(horizon, rebalance_freq)
        else:
            raise ValueError(f"horizon {horizon!r} must be a positive int or a "
                             f"pandas frequency string")
        horizon_labels.append(label)

        pairs = _build_pairs_for_horizon(
            prices_rs=prices_rs,
            signal_rs=signal_rs,
            horizon_periods=horizon_periods,
            group_data=group_data,
            group_order=group_order,
            is_log_returns=is_log_returns,
            is_vol_normalised=is_vol_normalised,
            min_obs_per_date=min_obs_per_date,
        )
        pairs_by_horizon[label] = pairs
        if len(pairs) > 0:
            overall_start = (pairs['date'].min() if overall_start is None
                             else min(overall_start, pairs['date'].min()))
            overall_end = (pairs['date'].max() if overall_end is None
                           else max(overall_end, pairs['date'].max()))

        # Pooled (universe-normalised)
        if len(pairs) > 0:
            fit = fitter(pairs['z'].to_numpy(), pairs['r_norm_univ'].to_numpy())
            pooled_rows[label] = fit if fit is not None else {c: np.nan for c in _STAT_COLS}
        else:
            pooled_rows[label] = {c: np.nan for c in _STAT_COLS}

        # Per-group (within-group normalised)
        if group_data is not None:
            for g in group_order:
                sub = pairs[pairs['group'] == g].dropna(subset=['z', 'r_norm_group'])
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
        group_order=group_order,
        start_date=overall_start,
        end_date=overall_end,
    )
