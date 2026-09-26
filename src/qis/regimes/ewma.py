"""
regime-time EWMA moments: conditional means and betas that decay within each regime's own stream.

A calendar-time EWMA discounts a crisis by its calendar age, so the Bear-regime moments of a long
sample are set by whichever Bear periods happen to be recent. Running the recursion over each
regime's own return stream discounts in regime time instead: the last Bear period carries the
most weight among Bear periods, however long ago it occurred, and past crises keep their weight
in the Bear moments. These are the regime-time counterparts of ``compute_regime_avg``, whose
conditional means are equal-weighted.

Both functions seed the recursion with the stream's full-sample mean (``InitType.MEAN``, the
look-ahead seed), so at a span far longer than the stream they return the equal-weighted
estimates. Inside a backtest they must be evaluated on the data available at each decision date,
as an expanding window does.
"""
# packages
import numpy as np
import pandas as pd
from typing import Tuple
# qis
from qis.models.linear.ewm import InitType, compute_ewm, compute_ewm_covar
from qis.regimes.partition import REGIME_COLUMN, get_ordered_regimes


def compute_regime_ewm_avg(sampled_returns_with_regime_id: pd.DataFrame,
                           span: float = 40.0,
                           regime_column: str = REGIME_COLUMN
                           ) -> pd.DataFrame:
    """EWMA conditional means with the span applied within each regime's own return stream.

    Args:
        sampled_returns_with_regime_id: periodic returns with a regime column
        span: EWMA span in regime-time periods
        regime_column: name of the regime column

    Returns:
        regimes in rows, in bucket order, and assets in columns: the last EWMA value of each
        regime's stream, seeded at the stream mean
    """
    data = sampled_returns_with_regime_id.dropna(subset=[regime_column])
    out = {}
    for regime, block in data.groupby(regime_column, sort=False, observed=True):
        rets = block.drop(columns=regime_column)
        out[regime] = compute_ewm(data=rets, span=span, init_type=InitType.MEAN).iloc[-1]
    return pd.DataFrame(out).T.reindex(get_ordered_regimes(data[regime_column]))


def compute_regime_ewm_betas(sampled_returns_with_regime_id: pd.DataFrame,
                             benchmark: str,
                             span: float = 40.0,
                             regime_column: str = REGIME_COLUMN
                             ) -> Tuple[pd.DataFrame, pd.Series]:
    """EWMA-weighted per-regime betas on the benchmark and the calendar-time residual variances.

    Within each regime the betas use the regime-time EWMA covariance of the demeaned pair, seeded
    at the sample covariance, and an intercept at the EWMA means, which is discarded. The
    idiosyncratic variance is the calendar-time EWMA of the squared pooled residuals.

    Args:
        sampled_returns_with_regime_id: periodic returns with a regime column
        benchmark: name of the benchmark column
        span: EWMA span in periods
        regime_column: name of the regime column

    Returns:
        the betas, assets in rows and regimes in columns in bucket order, and the per-period
        residual variance of each asset, to be annualised by the caller
    """
    data = sampled_returns_with_regime_id.dropna(subset=[regime_column])
    regimes = get_ordered_regimes(data[regime_column])
    assets = [c for c in data.columns if c not in (regime_column, benchmark)]
    betas = {}
    residuals = pd.DataFrame(index=data.index, columns=assets, dtype=float)
    for regime, block in data.groupby(regime_column, sort=False, observed=True):
        x = block[benchmark].to_numpy()
        xm = float(compute_ewm(data=block[benchmark], span=span, init_type=InitType.MEAN).iloc[-1])
        for asset in assets:
            y = block[asset].to_numpy()
            ym = float(compute_ewm(data=block[asset], span=span, init_type=InitType.MEAN).iloc[-1])
            xy = np.stack([x - xm, y - ym], axis=1)
            covar = compute_ewm_covar(a=xy, span=span, covar0=(xy.T @ xy) / len(xy))
            beta = float(covar[0, 1] / covar[0, 0])
            betas.setdefault(asset, {})[regime] = beta
            residuals.loc[block.index, asset] = y - (ym - beta * xm) - beta * x
    betas = pd.DataFrame(betas).T[regimes]
    idio_vars = residuals.apply(
        lambda r: float(compute_ewm(data=r.dropna() ** 2, span=span,
                                    init_type=InitType.MEAN).iloc[-1]))
    return betas, idio_vars
