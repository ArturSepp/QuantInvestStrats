"""
the principal-overlay mixes behind a smart-diversification curve.

``create_overlay_portfolio_curve`` mixes a principal portfolio with one overlay at eleven weights
from zero to ``max_overlay_weight``, backtests each mix with ``backtest_model_portfolio`` on the
first price date and at ``rebalancing_freq``, and returns the navs, one column per mix.
``SmartDiversificationReport`` in ``report.py`` reads its statistics off these navs.
"""
# packages
import numpy as np
import pandas as pd
# qis
from qis.portfolio.backtester import backtest_model_portfolio


def create_overlay_portfolio_curve(principal_nav: pd.Series,
                                   overlay_nav: pd.Series,
                                   principal_weight: float = 1.0,
                                   max_overlay_weight: float = 1.0,
                                   rebalancing_freq: str = 'QE',
                                   is_principal_weight_fixed: bool = True
                                   ) -> pd.DataFrame:
    """Navs of the principal portfolio mixed with one overlay at eleven overlay weights.

    The overlay weights run from zero to ``max_overlay_weight`` in equal steps, so the first
    column holds the principal portfolio alone. Each mix is a ``backtest_model_portfolio`` with
    constant weights, invested on the first price date and rebalanced at ``rebalancing_freq``, so
    a history that starts between two rebalancing dates is invested in its first period too.
    Up to qis 5.31.0 the mixes held no position until the first scheduled rebalancing date.

    With ``is_principal_weight_fixed`` the mixes are levered, and the backtest finances the
    leverage at a zero rate. On excess-of-cash navs, with ``principal_weight=1``, a mix is then the
    stacked portfolio ``r_B + w r_A`` of Sepp and Kastenholz (2026, Definition 3), up to the drift
    between rebalancing dates; on total-return navs it overstates the mix by ``w`` times the cash
    return.

    Args:
        principal_nav: nav of the principal portfolio
        overlay_nav: nav of the overlay; its name labels the columns
        principal_weight: weight of the principal portfolio when ``is_principal_weight_fixed``
        max_overlay_weight: overlay weight of the last mix
        rebalancing_freq: rebalancing frequency of the mixes
        is_principal_weight_fixed: keep ``principal_weight`` fixed and add the overlay on top;
            False funds the overlay from the principal, with weights ``1 - w`` and ``w``

    Returns:
        one nav column per mix, named ``"<overlay> <weight>"``, on the union of the two indices
    """
    prices = pd.concat([principal_nav, overlay_nav], axis=1, sort=True)

    overlay_weights = np.linspace(0, max_overlay_weight, 11)
    portfolio_navs = []
    for overlay_weight in overlay_weights:
        if is_principal_weight_fixed:
            weights = np.array([principal_weight, overlay_weight])
        else:
            weights = np.array([1.0-overlay_weight, overlay_weight])

        portfolio_nav = backtest_model_portfolio(prices=prices,
                                                 weights=np.array(weights),
                                                 rebalancing_freq=rebalancing_freq,
                                                 is_rebalanced_at_first_date=True
                                                 ).get_portfolio_nav()
        portfolio_nav.name = f"{overlay_nav.name} {'{:.2%}'.format(overlay_weight)}"
        portfolio_navs.append(portfolio_nav)
    portfolio_navs = pd.concat(portfolio_navs, axis=1, sort=True)
    return portfolio_navs
