"""
Dimson (1979) aggregated-coefficient beta for non-synchronous / smoothed returns.

Stand-alone qis function. Suggested home: qis/perfstats/returns.py (alongside
the leverage helpers) or a qis/models regression module. Pure numpy/pandas, no
qis dependency, so it can also be imported directly.
"""
# packages
import numpy as np
import pandas as pd
from typing import Union


def estimate_dimson_beta(asset_returns: Union[pd.Series, pd.DataFrame],
                         market_returns: pd.Series,
                         num_lags: int = 3,
                         min_obs: int = 36,
                         ) -> pd.DataFrame:
    """estimate the Dimson aggregated-coefficient beta to detect return smoothing.

    Fits, per asset, the time-series regression of asset returns on the
    contemporaneous and lagged market return

        r_i_t = a_i + sum_{k=0}^{L} b_{i,k} r_m_{t-k} + e_i_t

    and reports the Dimson beta beta_dimson = sum_{k=0}^{L} b_{i,k}. When an asset
    prices with a lag (stale marks, return smoothing, illiquid holdings), the
    contemporaneous slope b_0 understates the true market exposure and the lagged
    slopes recover it. The ratio beta_dimson / b_0 measures the understatement and
    the t-stat on the summed lagged slopes tests whether the lag effect is real.

    Beta is invariant to using total or excess returns provided the same
    convention is used for asset and market, so total returns are the natural
    input when the goal is to detect smoothing in the raw reported series.

    Args:
        asset_returns: asset return panel, one column per asset, at the same frequency as
            ``market_returns``. Monthly for a quarter-smoothing test with L=3
        market_returns: market or factor return series, the timing reference. Use a liquid,
            frequently priced index
        num_lags: number of lagged market terms L. L=3 on monthly data tests whether
            aggregating to quarterly recovers exposure the monthly regression misses
        min_obs: minimum overlapping observations required to fit an asset

    Returns:
        one row per asset, with columns ``beta_0`` the contemporaneous slope, ``beta_dimson``
        the aggregated slope, ``smoothing_ratio`` their ratio (NaN when ``b_0`` is near zero),
        ``t_beta_0``, ``sum_lag_beta`` the sum of the lagged slopes, ``t_sum_lag`` its t-stat,
        ``ar1`` the first-order autocorrelation of the asset return, ``r2``, and ``n_obs``

    Raises:
        ValueError: if ``market_returns`` is not a pd.Series, or ``num_lags`` is negative
    """
    if not isinstance(market_returns, pd.Series):
        raise ValueError(f"market_returns must be a pd.Series, got {type(market_returns)!r}")
    if num_lags < 0:
        raise ValueError(f"num_lags must be >= 0, got {num_lags}")
    if isinstance(asset_returns, pd.Series):
        asset_returns = asset_returns.to_frame()

    # contemporaneous + lagged market design columns
    mkt = pd.DataFrame({f'mkt_l{k}': market_returns.shift(k) for k in range(num_lags + 1)})
    mkt_cols = list(mkt.columns)

    out = {}
    for col in asset_returns.columns:
        df = pd.concat([asset_returns[col].rename('y'), mkt], axis=1, sort=True).dropna()
        n = len(df)
        if n < max(min_obs, num_lags + 3):
            out[col] = dict(beta_0=np.nan, beta_dimson=np.nan, smoothing_ratio=np.nan,
                            t_beta_0=np.nan, sum_lag_beta=np.nan, t_sum_lag=np.nan,
                            ar1=np.nan, r2=np.nan, n_obs=n)
            continue

        y = df['y'].to_numpy(dtype=float)
        x_mkt = df[mkt_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(n), x_mkt])  # [1, m_t, m_{t-1}, ..., m_{t-L}]

        xtx_inv = np.linalg.inv(x.T @ x)
        b = xtx_inv @ (x.T @ y)
        resid = y - x @ b
        dof = n - x.shape[1]
        sigma2 = float(resid @ resid) / dof if dof > 0 else np.nan
        cov_b = sigma2 * xtx_inv

        # selectors over the market coefficients (indices 1 .. num_lags+1)
        c_all = np.zeros(x.shape[1]); c_all[1:] = 1.0          # contemporaneous + all lags
        c_lag = np.zeros(x.shape[1]); c_lag[2:] = 1.0          # lagged terms only

        beta_0 = float(b[1])
        beta_dimson = float(c_all @ b)
        sum_lag = float(c_lag @ b)
        se_beta_0 = float(np.sqrt(cov_b[1, 1]))
        se_sum_lag = float(np.sqrt(c_lag @ cov_b @ c_lag)) if num_lags >= 1 else np.nan

        out[col] = dict(
            beta_0=beta_0,
            beta_dimson=beta_dimson,
            smoothing_ratio=(beta_dimson / beta_0 if abs(beta_0) > 1e-8 else np.nan),
            t_beta_0=(beta_0 / se_beta_0 if se_beta_0 > 0 else np.nan),
            sum_lag_beta=sum_lag,
            t_sum_lag=(sum_lag / se_sum_lag if (num_lags >= 1 and se_sum_lag > 0) else np.nan),
            ar1=float(pd.Series(y).autocorr(lag=1)),
            r2=(1.0 - float(resid @ resid) / float(np.sum((y - y.mean()) ** 2))
                if np.sum((y - y.mean()) ** 2) > 0 else np.nan),
            n_obs=n,
        )

    return pd.DataFrame.from_dict(out, orient='index')