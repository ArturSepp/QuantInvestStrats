"""
EWMA realised tracking error from portfolio and benchmark NAVs.

The entry point ``compute_ewma_realised_tracking_error`` estimates ex-post tracking
error from return differences:

    TE_t = sqrt(EWMA(d_t²; span)) · sqrt(a_freq)

The output is an annualised return volatility. Ex-ante tracking error from weights and
a covariance lives in ``qis.RiskModel``; nothing here takes weights.
"""
import pandas as pd

from qis.models.linear.ewm import compute_ewm_vol
from qis.perfstats.returns import to_returns


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
    masked while the EWMA state warms up.

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
