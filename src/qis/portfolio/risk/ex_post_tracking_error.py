"""
Ex-post tracking error from NAVs and return differences.

``compute_ewma_realised_tracking_error`` estimates a conditional EWMA series for
point-in-time monitoring. ``compute_te_ir_errors`` and ``compute_info_ratio_table``
estimate unconditional whole-sample tracking error and information-ratio scalars.
Ex-ante tracking error from weights and covariance lives in ``risk_model.py``;
nothing here takes weights.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from qis.models.linear.ewm import compute_ewm_vol
from qis.perfstats.returns import to_returns
from qis.utils.annualisation import infer_annualisation_factor_from_df


def compute_ewma_realised_tracking_error(
        portfolio_nav: pd.Series,
        benchmark_nav: pd.Series,
        ewma_span: int = 36,
        freq: str = 'ME',
        is_log_returns: bool = False,
) -> pd.Series:
    """Compute annualised EWMA realised (ex-post) tracking error from NAVs.

    Both NAVs are resampled to ``freq`` before their period returns are differenced.
    The EWMA variance recursion, its initialisation, and the annualisation factor implied
    by ``freq`` all come from ``compute_ewm_vol``. The first ``ewma_span`` estimates are
    masked while the EWMA state warms up. For unconditional whole-sample TE and IR scalars,
    use ``compute_te_ir_errors`` or ``compute_info_ratio_table``.

    Args:
        portfolio_nav: Portfolio NAV time series.
        benchmark_nav: Benchmark NAV time series.
        ewma_span: EWMA span in periods of ``freq``.
        freq: Resampling frequency for the return differences.
        is_log_returns: If True, use log returns; otherwise use simple returns.

    Returns:
        Annualised realised tracking error named ``'Tracking error'``, indexed by the
        resampled period and NaN during the EWMA warm-up.

    Note:
        The join is ``pd.concat([...], axis=1, sort=True).dropna(how='all').ffill()``.
        A benchmark NAV starting before the portfolio NAV should be clipped by the caller,
        for example with ``TimePeriod.locate``, so the two series share the intended sample.
    """
    navs = pd.concat(
        [portfolio_nav.rename('portfolio'), benchmark_nav.rename('benchmark')],
        axis=1,
        sort=True,
    ).dropna(how='all').ffill()
    returns = to_returns(
        prices=navs,
        freq=freq,
        is_log_returns=is_log_returns,
        drop_first=True,
    )
    return_diff = (returns['portfolio'] - returns['benchmark']).dropna()
    tracking_error = compute_ewm_vol(
        data=return_diff,
        span=ewma_span,
        annualize=True,
        warmup_period=ewma_span,
    )
    return tracking_error.rename('Tracking error')


def compute_te_ir_errors(return_diffs: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute whole-sample tracking error and information ratio from return differentials.

    Use ``compute_ewma_realised_tracking_error`` instead when point-in-time monitoring or a
    conditional tracking-error series is required.

    Args:
        return_diffs: DataFrame of (strategy_return - benchmark_return) per period.

    Returns:
        Tuple of (tracking_error_series, information_ratio_series), both indexed by
        the columns of return_diffs and annualised.
    """
    vol_dt = np.sqrt(infer_annualisation_factor_from_df(return_diffs))
    avg = np.nanmean(return_diffs, axis=0)
    vol = np.nanstd(return_diffs, axis=0, ddof=1)
    # NumPy 2.x: explicit out= buffer so masked positions (vol==0) are deterministic nan.
    ir = vol_dt * np.divide(
        avg, vol,
        out=np.full_like(avg, np.nan, dtype=float),
        where=np.greater(vol, 0.0),
    )
    te = pd.Series(vol_dt * vol, index=return_diffs.columns, name='TE')
    ir = pd.Series(ir, index=return_diffs.columns, name='IR')
    return te, ir


def compute_info_ratio_table(
        return_diffs_dict: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute whole-sample TE and IR tables across return-differential panels.

    Use ``compute_ewma_realised_tracking_error`` instead when point-in-time monitoring or a
    conditional tracking-error series is required.

    Args:
        return_diffs_dict: Mapping of asset class label to return-diffs DataFrame.

    Returns:
        Tuple of (te_table, ir_table) DataFrames, columns = asset classes,
        rows = strategies.
    """
    te_ac_datas = []
    ir_ac_datas = []
    for ac, data in return_diffs_dict.items():
        te, ir = compute_te_ir_errors(return_diffs=data)
        te_ac_datas.append(te.rename(ac))
        ir_ac_datas.append(ir.rename(ac))
    te_table = pd.concat(te_ac_datas, axis=1, sort=False)
    ir_table = pd.concat(ir_ac_datas, axis=1, sort=False)
    return te_table, ir_table
