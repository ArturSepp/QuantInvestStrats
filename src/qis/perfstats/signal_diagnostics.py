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

String horizons (e.g. 'YE') override per-asset cadence: each asset's
native returns are compounded within the periods of that frequency, and
only periods fully covered by the asset's native returns are kept. Use for
headline 12-month tests.

The pooled t-statistic charges one residual degree of freedom per
regression date for the cross-sectional demeaning, and the annualised IC
ratio uses qis's annualisation factor of the IC grid
(``qis.get_annualization_factor``), divided by the horizon.

This module's ``estimate_signal_diagnostics`` is the compute entry
point; see ``qis.plots.derived.signal_diagnostics_plot`` for plots.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Sequence, Tuple, Union

from qis.utils.annualisation import get_annualization_factor


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

_PAIR_COLUMNS = ['date', 'asset', 'asset_freq', 'group', 'z', 'r']

# per-date IC methods of compute_ic_timeseries / estimate_ic_ir
_IC_METHODS = {'spearman': scipy_stats.spearmanr, 'pearson': scipy_stats.pearsonr}


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
            DataFrame with columns ['date', 'asset', 'asset_freq', 'group',
            'z', 'r', 'r_norm_univ', 'r_norm_group'], ordered by date and
            then asset: ``z`` is the lagged signal, ``r`` the raw forward
            return and the last two the forward return normalised across
            the universe and within the asset's group. The underlying
            (signal, return) panel consumed by both regressions and by the
            plotting layer.

        horizon_labels: ordered list of horizon labels (matches index
            order of pooled_universe).

        group_order: ordered list of group labels present in per_group.

        start_date, end_date: span of the pairs sample.

        fit_intercept: whether the pooled and per-group regressions were
            fitted with an intercept.
    """
    pooled_universe: pd.DataFrame
    per_group: pd.DataFrame
    pairs: Dict[str, pd.DataFrame] = field(default_factory=dict)
    horizon_labels: List[str] = field(default_factory=list)
    group_order: List[str] = field(default_factory=list)
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    fit_intercept: bool = False


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
    is mapped to the first frequency in dict-insertion order, and a
    UserWarning names the duplicated assets; their other frames are
    ignored by every horizon.
    """
    mapping: Dict[str, str] = {}
    duplicated: Dict[str, List[str]] = {}
    for freq, df in asset_returns_dict.items():
        if df is None or df.empty:
            continue
        for col in df.columns:
            if col not in mapping:
                mapping[col] = freq
            elif mapping[col] != freq:
                duplicated.setdefault(col, [mapping[col]]).append(freq)
    if duplicated:
        listed = ', '.join(f"{asset} {freqs}" for asset, freqs in list(duplicated.items())[:5])
        warnings.warn(f"estimate_signal_diagnostics: {len(duplicated)} asset(s) appear in "
                      f"several frequency frames and are assigned to the first one in dict "
                      f"order; the other frames are ignored for them: {listed}",
                      UserWarning, stacklevel=3)
    return mapping


def _residual_dof(n: int, n_params: int, dates: Optional[np.ndarray],
                  mask: np.ndarray) -> int:
    """Residual degrees of freedom of a slope fit on per-date demeaned returns.

    Without ``dates`` the classical count ``n - n_params``. With ``dates``
    the returns were demeaned across names at each date, which is a date
    fixed effect: each of the ``T`` distinct dates costs one degree of
    freedom and absorbs any intercept, leaving ``n - T - 1``.
    """
    if dates is None:
        return n - n_params
    n_dates = int(pd.Series(np.asarray(dates)[mask]).nunique())
    return n - n_dates - 1


def _fit_through_origin(z: np.ndarray, r: np.ndarray,
                        dates: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """No-intercept OLS: β = Σ(zr) / Σ(z²).

    ``dates`` (one per pair) marks returns demeaned across names at each
    date; the residual variance then uses ``n - T - 1`` degrees of freedom
    instead of ``n - 1``.
    """
    mask = np.isfinite(z) & np.isfinite(r)
    dof = _residual_dof(n=int(mask.sum()), n_params=1, dates=dates, mask=mask)
    z, r = z[mask], r[mask]
    n = len(z)
    if n < 5:
        return None
    zz = float((z * z).sum())
    if zz <= 0.0:
        return None
    beta = float((z * r).sum() / zz)
    e = r - beta * z
    sigma2 = float((e ** 2).sum() / dof) if dof > 0 else np.nan
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


def _fit_with_intercept(z: np.ndarray, r: np.ndarray,
                        dates: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """OLS with intercept: returns slope coefficient stats only.

    ``dates`` (one per pair) marks returns demeaned across names at each
    date; the intercept is then absorbed by the date effects and the
    residual variance uses ``n - T - 1`` degrees of freedom instead of
    ``n - 2``.
    """
    mask = np.isfinite(z) & np.isfinite(r)
    dof = _residual_dof(n=int(mask.sum()), n_params=2, dates=dates, mask=mask)
    z, r = z[mask], r[mask]
    n = len(z)
    if n < 5:
        return None
    try:
        slope, _intercept, rp, _p, slope_se = scipy_stats.linregress(z, r)
    except ValueError:
        return None
    # linregress divides the residual sum of squares by n - 2
    slope_se = float(slope_se * np.sqrt((n - 2) / dof)) if dof > 0 else np.nan
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


def _pair_frame_from_aligned_values(
        signal_frame: pd.DataFrame,
        return_frame: pd.DataFrame,
        asset_freq: Dict[str, str],
        group_data: Optional[pd.Series],
) -> pd.DataFrame:
    """Stack jointly finite values from identically aligned pair frames.

    Rolling, resampling, and signal-lag rules stay in the two callers. This
    helper changes only the expensive final projection: ordinary numerical
    frames are converted to arrays once rather than read through pandas one
    cell at a time.

    The joint finite mask selects the same signal/return pairs as the former
    nested loops. Flattening in C order then retains their date-major,
    asset-minor output order.

    Args:
        signal_frame: Lagged signal values in regression-date and asset order.
        return_frame: Forward returns with the same index and columns.
        asset_freq: Native-frequency lookup by asset.
        group_data: Optional group-label lookup by asset.

    Returns:
        Long pair frame retaining date-major, asset-minor input order.
    """
    if return_frame.empty or len(return_frame.columns) == 0:
        return pd.DataFrame(columns=_PAIR_COLUMNS)

    dtypes = [*signal_frame.dtypes, *return_frame.dtypes]
    if any(isinstance(dtype, pd.api.extensions.ExtensionDtype) for dtype in dtypes):
        # Nullable scalars have distinct success/error behavior, so keep that established path
        # rather than letting this performance refactor silently broaden the public contract.
        rows: List[Dict] = []
        for date in return_frame.index:
            for asset in return_frame.columns:
                z = signal_frame.loc[date, asset]
                r = return_frame.loc[date, asset]
                if not (np.isfinite(z) and np.isfinite(r)):
                    continue
                rows.append({
                    'date': date,
                    'asset': asset,
                    'asset_freq': asset_freq.get(asset),
                    'group': group_data.get(asset) if group_data is not None else None,
                    'z': float(z),
                    'r': float(r),
                })
        return pd.DataFrame(rows, columns=_PAIR_COLUMNS)

    # Convert each aligned frame once; the old nested loop paid pandas indexing cost per pair.
    signal_values = signal_frame.to_numpy()
    return_values = return_frame.to_numpy()
    finite = np.isfinite(signal_values) & np.isfinite(return_values)
    if not finite.any():
        return pd.DataFrame(columns=_PAIR_COLUMNS)

    n_dates, n_assets = finite.shape
    assets = return_frame.columns.to_numpy()
    # C-order flattening preserves the established date-major, asset-minor row order.
    selected = finite.ravel()
    dates = np.repeat(return_frame.index.to_numpy(), n_assets)[selected]
    asset_values = np.tile(assets, n_dates)[selected]
    freq_values = np.tile(
        np.asarray([asset_freq.get(asset) for asset in assets], dtype=object),
        n_dates,
    )[selected]
    if group_data is None:
        group_values = np.full(int(selected.sum()), None, dtype=object)
    else:
        selected_group_values = np.tile(
            np.asarray([group_data.get(asset) for asset in assets], dtype=object),
            n_dates,
        )[selected]
        # Infer from surviving labels just as the former scalar row construction did.
        group_values = pd.Series(selected_group_values.tolist())
    return pd.DataFrame({
        'date': dates,
        'asset': asset_values,
        'asset_freq': freq_values,
        'group': group_values,
        'z': signal_values.ravel()[selected].astype(float, copy=False),
        'r': return_values.ravel()[selected].astype(float, copy=False),
    })


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
    pair_frames: List[pd.DataFrame] = []
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
        # Batch only final extraction; cadence, compounding, and lag rules above are unchanged.
        pairs = _pair_frame_from_aligned_values(
            signal_frame=sig_lag.loc[sampled, assets_here],
            return_frame=cum_fwd.loc[sampled, assets_here],
            asset_freq=asset_freq,
            group_data=group_data,
        )
        if not pairs.empty:
            pair_frames.append(pairs)
    if not pair_frames:
        return pd.DataFrame(columns=_PAIR_COLUMNS)
    return pd.concat(pair_frames, ignore_index=True)


def _native_label(date: pd.Timestamp, native_freq: str) -> pd.Timestamp:
    """Label of the ``native_freq`` period that holds ``date``, as ``resample`` labels it."""
    return pd.Series(1, index=pd.DatetimeIndex([date])).resample(native_freq).sum().index[0]


def _rows_per_complete_period(
        index: pd.DatetimeIndex, native_freq: str, horizon_freq: str,
) -> pd.Series:
    """Rows of a native frame in each ``horizon_freq`` period; zero where not fully covered.

    Interior periods are counted on the frame's own calendar, so a row that
    is absent for every asset (an exchange holiday) does not make a period
    incomplete. The first and last periods of the frame are covered only
    when its first (last) row falls in the first (last) native period of
    the key inside them; otherwise the frame starts or ends mid-period and
    their count is set to zero.
    """
    rows = pd.Series(1, index=index).resample(horizon_freq).sum()
    offset = pd.tseries.frequencies.to_offset(horizon_freq)
    grid = pd.date_range(start=index.min() - 2 * offset, end=index.max() + 2 * offset,
                         freq=native_freq)
    grid_by_period = pd.Series(grid, index=grid).resample(horizon_freq)
    first_native, last_native = grid_by_period.min(), grid_by_period.max()
    first_period, last_period = rows.index[0], rows.index[-1]
    if _native_label(index.min(), native_freq) != first_native.get(first_period):
        rows.loc[first_period] = 0
    if _native_label(index.max(), native_freq) != last_native.get(last_period):
        rows.loc[last_period] = 0
    return rows


def _compound_complete_periods(
        returns_df: pd.DataFrame, native_freq: str, horizon_freq: str,
        is_log_returns: bool,
) -> pd.DataFrame:
    """Compound native returns within ``horizon_freq`` periods, complete periods only.

    A period is kept for an asset only when the asset has a finite return
    at every row of the frame inside it and the frame covers the period
    from its first to its last native period: periods before the asset's
    first return, after its last one, partially covered at the sample
    edges, or containing a missing return are missing, the same rule as an
    integer-horizon window.
    """
    ret = returns_df.astype(float)
    finite = pd.DataFrame(np.isfinite(ret.to_numpy()), index=ret.index, columns=ret.columns)
    ret = ret.where(finite)
    if is_log_returns:
        total = ret.resample(horizon_freq).sum(min_count=1)
    else:
        total = (1.0 + ret).resample(horizon_freq).prod(min_count=1) - 1.0
    observed = finite.resample(horizon_freq).sum()
    rows = _rows_per_complete_period(ret.index, native_freq, horizon_freq).reindex(
        total.index, fill_value=0)
    complete = observed.eq(rows, axis=0) & (rows.to_numpy() > 0)[:, None]
    return total.where(complete)


def _build_pairs_string_horizon(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: pd.DataFrame,
        horizon_freq: str,
        group_data: Optional[pd.Series],
        is_log_returns: bool,
        asset_freq: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """String-horizon override: compound every asset's returns to this frequency.

    Each asset's native returns are compounded (log returns summed) within
    the periods of ``horizon_freq``. A period enters only when the frame
    covers it and the asset has a finite return at every row of the frame
    inside it, so there are no zero returns before an asset's first return
    or after its last, and no partial periods at the sample edges. One observation per asset per
    complete ``horizon_freq`` period, paired with the last signal of the
    previous period.
    """
    if asset_freq is None:
        asset_freq = _asset_to_freq_map(asset_returns_dict)
    per_freq_returns = []
    for freq, returns_df in asset_returns_dict.items():
        if returns_df is None or returns_df.empty:
            continue
        # an asset listed in several frames is taken from its mapped frame only
        assets_here = [c for c in returns_df.columns if asset_freq.get(c) == freq]
        if not assets_here:
            continue
        per_freq_returns.append(_compound_complete_periods(
            returns_df=returns_df[assets_here], native_freq=freq, horizon_freq=horizon_freq,
            is_log_returns=is_log_returns))
    if not per_freq_returns:
        return pd.DataFrame(columns=_PAIR_COLUMNS)
    ret_rs = pd.concat(per_freq_returns, axis=1, sort=True).sort_index()
    sig_rs = signal.resample(horizon_freq).last().shift(1)
    common = ret_rs.index.intersection(sig_rs.index)
    assets = [asset for asset in ret_rs.columns if asset in sig_rs.columns]
    # Share the final projection so string and integer horizons keep one ordering contract.
    return _pair_frame_from_aligned_values(
        signal_frame=sig_rs.loc[common, assets],
        return_frame=ret_rs.loc[common, assets],
        asset_freq=asset_freq,
        group_data=group_data,
    )


def _align_to_period_labels(
        returns_df: pd.DataFrame, freq: str, signal: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Put a native returns frame on the period-end labels of its key, point in time.

    The integer-horizon pairs join returns and the signal resampled with
    ``signal.resample(freq).last()`` on the resample labels. A frame whose
    dates sit inside their periods but off the labels (business month-ends
    under ``'ME'``, say) is relabelled to the label of the period holding
    each date. A signal value dated after such a return date but inside the
    same period was not known at that date, so it is removed before the
    signal is resampled: the signal paired with the next return is the last
    value observed at or before the return date that ends the previous
    native period.

    Args:
        returns_df: returns of one frequency frame.
        freq: the frame's key.
        signal: the signal panel.

    Returns:
        The relabelled frame and the signal to resample for this frame. Both
        are returned unchanged when the dates are already on the labels, and,
        with a UserWarning, when the frame holds more than one date in some
        period (an index finer than its key) or its key labels periods by
        their start.
    """
    index = returns_df.index
    if len(index) == 0 or not (index.is_monotonic_increasing and index.is_unique):
        return returns_df, signal
    counts = pd.Series(1, index=index).resample(freq).sum()
    if int(counts.max()) > 1:
        warnings.warn(f"estimate_signal_diagnostics: the returns frame under key '{freq}' has "
                      f"more than one date in some '{freq}' periods, so its index is finer than "
                      f"its key; only dates on the '{freq}' period labels are paired. Store the "
                      f"frame under the key of its own sampling frequency.",
                      UserWarning, stacklevel=3)
        return returns_df, signal
    labels = counts.index[counts.to_numpy() > 0]
    if labels.equals(index):
        return returns_df, signal
    off_label = labels.to_numpy() != index.to_numpy()
    off_dates = index.to_numpy()[off_label]
    off_labels = labels.to_numpy()[off_label]
    if np.any(off_labels < off_dates):
        warnings.warn(f"estimate_signal_diagnostics: the key '{freq}' labels periods by their "
                      f"start, so returns dated inside a period cannot be aligned point in time; "
                      f"only dates on the '{freq}' period labels are paired. Use a period-end "
                      f"key.", UserWarning, stacklevel=3)
        return returns_df, signal
    # drop signal values dated in (d, label(d)] for every off-label return date d
    signal_dates = signal.index.to_numpy()
    position = np.searchsorted(off_dates, signal_dates, side='left') - 1
    after_return = (position >= 0) & (
        signal_dates <= off_labels[np.clip(position, 0, None)])
    return returns_df.set_axis(labels, axis=0), signal.loc[~after_return]


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

    String horizons (e.g. 'YE') override per-asset cadence — each asset's
    native returns are compounded within the periods of that frequency,
    keeping only periods that the asset's frame covers and in which the
    asset has a finite return at every row of its frame, the same rule as
    an integer-horizon window. Use for headline annual tests.

    The standard error of β treats the pairs as independent but charges
    one residual degree of freedom per regression date for the
    cross-sectional demeaning: ``n - T - 1`` with T dates, with or without
    an intercept, since the date effects absorb it.

    Args:
        asset_returns_dict: Per-frequency returns dict from a pipeline
            that already handles FX adjustment and unsmoothing. Keys are
            pandas period-end frequency strings (e.g. 'ME', 'QE', 'YE');
            values are return DataFrames indexed at that frequency's
            period-ends with asset tickers as columns. A frame whose dates
            fall inside their periods but off the labels of
            ``resample(key)`` (business month-ends under 'ME', say) is
            relabelled to the period-end labels, and pairs are dated at
            those labels. Each asset should appear in exactly one frame
            (its native cadence); an asset in several frames is assigned
            to the first one, with a UserWarning.

        signal: T x N signal panel (e.g. ``AlphasData.alpha_scores``),
            dated when each value becomes known. Assets in
            ``asset_returns_dict`` without a signal column are dropped
            with a UserWarning; if none remains, ValueError. Signal panel
            is typically at the finest frequency present in the dict
            (e.g. monthly); the function resamples it to each asset's
            native cadence and lags it one native period.

        group_data: Optional Series mapping asset name → group label.
            When None, only the pooled regression is run.

        horizons: Forward-return horizons. Integers are in native-cadence
            units (1, 3, 6 → 1, 3, 6 native periods per asset). Strings
            like 'YE' override per-asset cadence and compound uniformly.

        fit_intercept: Include α in the regression. Default False. The
            forward returns are demeaned across names at each date, so
            their pooled mean is zero and the fitted intercept is
            ``α̂ = -β̂ · mean(z)`` over the pooled pairs: zero only when the
            pooled signal mean is zero. Otherwise the two fits give
            different slopes: a constant level in the signal attenuates the
            no-intercept β and is removed by the intercept, while a level
            that varies by date affects both.

        is_log_returns: Set True when ``asset_returns_dict`` contains log
            returns (default), False for simple returns. Affects
            cumulation across horizons (log returns are summed, simple
            returns compounded). The default is the opposite of the
            ``qis.to_returns`` default, which produces simple returns;
            pass the flag that matches the data.

        is_vol_normalised: Divide cross-sectional return by cross-
            sectional std at each date (default True).

        min_obs_per_date: Minimum cross-sectional sample at a date.

        min_obs_per_group: Minimum sample size for a group's regression
            to be reported.

        group_order: Explicit ordering for the groups in per_group.

    Returns:
        ``SignalDiagnosticsResult`` with ``.pooled_universe``,
        ``.per_group``, ``.pairs`` and ``.fit_intercept`` populated.

    Raises:
        TypeError: if ``group_data`` is not a pandas Series.
        ValueError: if ``asset_returns_dict`` is empty, if the signal
            covers none of its assets, or if a horizon is neither a
            positive integer nor a string.
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
    # accordingly with a warning naming the dropped assets, and raise only
    # if every column drops.
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
    if dropped_assets:
        dropped = sorted(set(dropped_assets), key=str)
        warnings.warn(f"estimate_signal_diagnostics: {len(dropped)} asset(s) of "
                      f"asset_returns_dict have no signal column and are dropped "
                      f"(first 5: {dropped[:5]})", UserWarning, stacklevel=2)
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

    # Put each frame on its key's period labels and pre-resample the signal
    # panel to each frequency with only values known at the return dates
    aligned_returns: Dict[str, pd.DataFrame] = {}
    signal_rs_by_freq: Dict[str, pd.DataFrame] = {}
    for freq, df in asset_returns_dict.items():
        aligned_returns[freq], signal_for_freq = _align_to_period_labels(
            returns_df=df, freq=freq, signal=signal)
        signal_rs_by_freq[freq] = signal_for_freq.resample(freq).last()

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
                is_log_returns=is_log_returns, asset_freq=asset_freq,
            )
            label = horizon
        elif isinstance(horizon, (int, np.integer)) and horizon >= 1:
            raw_pairs = _build_pairs_int_horizon(
                asset_returns_dict=aligned_returns,
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

        # Pooled regression (universe-normalised); the per-date demeaning
        # costs one degree of freedom per date
        if len(normed) > 0:
            fit = fitter(normed['z'].to_numpy(), normed['r_norm_univ'].to_numpy(),
                         dates=normed['date'].to_numpy())
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
                fit_g = fitter(sub['z'].to_numpy(), sub['r_norm_group'].to_numpy(),
                               dates=sub['date'].to_numpy())
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
        fit_intercept=bool(fit_intercept),
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

    Raises:
        ValueError: if ``method`` is neither 'spearman' nor 'pearson'.
    """
    corr = _ic_correlation(method)
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame(columns=['n', 'IC'])
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


def _ic_correlation(method: str):
    """The per-date correlation function of an IC ``method``."""
    if method not in _IC_METHODS:
        raise ValueError(f"IC method must be one of {sorted(_IC_METHODS)}, got method={method!r}")
    return _IC_METHODS[method]


# median day gap of a grid -> its qis frequency, for the fallback inference below
_GAP_TIERS = ((3.0, 'B'), (10.0, 'W'), (20.0, '2W'), (45.0, 'ME'), (75.0, '2ME'),
              (135.0, 'QE'), (270.0, '2QE'))


def _infer_periods_per_year(index: pd.Index) -> float:
    """Periods per year of an IC-date grid classified by its median spacing.

    Fallback for pairs without native-frequency information: the median day
    gap is classified into a qis frequency (business days, weeks, months,
    quarters, half-years or years) and mapped to its
    ``get_annualization_factor``, so business days give 252 and month-ends
    12, not the calendar ratios 365.25 and 11.78.
    """
    if len(index) < 2:
        return float('nan')
    days = pd.Series(pd.to_datetime(index)).sort_values().diff().dropna().dt.days
    med = float(days.median()) if len(days) else float('nan')
    if not (np.isfinite(med) and med > 0):
        return float('nan')
    freq = next((f for gap, f in _GAP_TIERS if med <= gap), 'YE')
    return get_annualization_factor(freq)


def _ic_periods_per_year(
        label: str, pairs: Optional[pd.DataFrame], periods_per_year: Optional[float],
        ic_index: pd.Index,
) -> float:
    """Periods per year of one horizon's IC series.

    A string horizon ('YE', 'QE', ...) is its own grid:
    ``get_annualization_factor(label)``. An integer horizon h samples every
    h-th native period, so its IC series has ``AN / h`` periods per year,
    where AN is ``periods_per_year`` if given, and otherwise the qis
    annualisation factor of the finest native frequency in the pairs
    (``asset_freq``), which sets the IC dates.
    """
    label = str(label)
    if not label.isdigit():
        return get_annualization_factor(label)
    horizon = int(label)
    if periods_per_year is not None:
        return float(periods_per_year) / horizon
    if pairs is not None and 'asset_freq' in pairs.columns:
        freqs = [str(f) for f in pd.unique(pairs['asset_freq'].dropna())]
        if freqs:
            return max(get_annualization_factor(f) for f in freqs) / horizon
    return _infer_periods_per_year(ic_index)


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

    Raises:
        ValueError: if ``method`` is neither 'spearman' nor 'pearson'.
    """
    _ic_correlation(method)
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

        IC_IR     = mean(IC) / std(IC)               (per IC period)
        IC_IR_an  = IC_IR * sqrt(AN_h)               (annualised)
        t_stat    = IC_IR * sqrt(n_dates)            (significance of mean IC)
        hit_rate  = mean(IC > 0)

    ``IC_IR`` measures the stability of the IC over time: it rewards an IC
    that is consistently the right sign, not merely large on average. It
    does not adjust for breadth; the number of names enters only through
    the sampling noise of each per-date IC. It is the honest stability
    number the pooled ``t_stat`` is not — the pooled regression in
    ``estimate_signal_diagnostics`` treats every (asset, date) pair as
    independent and so overstates significance when the cross-section is
    correlated within a date; this collapses each date to one observation.

    ``AN_h`` is the number of IC periods per year. A string horizon uses
    ``qis.get_annualization_factor(label)`` (1 for 'YE'). An integer
    horizon h samples every h-th native period, so ``AN_h = AN / h`` with
    AN the qis annualisation factor of the finest native frequency in the
    pairs (12 for 'ME', 4 for 'QE', 252 for 'B'), or ``periods_per_year``
    when given.

    Args:
        result: ``SignalDiagnosticsResult`` from
            ``estimate_signal_diagnostics``.
        method: 'spearman' (default) or 'pearson' for the per-date IC.
        return_col: 'r_norm_univ' (universe cross-section, default) or
            'r_norm_group' (within-group — pass ``group_data`` to
            ``estimate_signal_diagnostics`` to populate it).
        periods_per_year: Periods per year of the native grid (the h = 1
            grid), divided by h for an integer horizon h; not used for
            string horizons. Inferred from the pairs' native frequencies
            when None.
        min_obs_per_date: Minimum cross-section per date.

    Returns:
        DataFrame indexed by horizon label, columns ``_IC_IR_COLS``.

    Raises:
        ValueError: if ``method`` is neither 'spearman' nor 'pearson'.
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
        ppy = _ic_periods_per_year(label=label, pairs=result.pairs.get(label),
                                   periods_per_year=periods_per_year, ic_index=ic.index)
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
