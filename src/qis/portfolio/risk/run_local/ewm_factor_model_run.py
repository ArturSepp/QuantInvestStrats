"""Development runner extracted from ``qis.portfolio.risk.ewm_factor_model``."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import qis.perfstats.returns as ret
import qis.plots.time_series as pts

from qis.portfolio.risk.ewm_factor_model import (
    EwmLinearModel,
    compute_portfolio_benchmark_ewm_beta_alpha_attribution,
)

class Locals(Enum):
    MODEL = 1
    ATTRIBUTION = 2

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()

    if local == Locals.MODEL:
        returns = np.log(prices.divide(prices.shift(1)))

        # factors
        factors = ['SPY', 'TLT', 'GLD']
        # factors = ['SPY']
        factor_returns = returns[factors]

        # assets
        is_check = False
        if is_check:
            asset_returns = returns[factors]
            asset_returns.columns = [f"{x.split('_')[0]}_asset" for x in factors]
        else:
            assets = ['QQQ', 'HYG']
            asset_returns = returns[assets]
        ewm_linear_model = EwmLinearModel(x=factor_returns, y=asset_returns)
        ewm_linear_model.fit(ewm_lambda=0.94, is_x_correlated=True)

        ewm_linear_model.print()
        ewm_linear_model.plot_factor_loadings(factor='SPY')

        factor_alpha, explained_returns = ewm_linear_model.get_factor_alpha()
        pts.plot_time_series(df=factor_alpha.cumsum(axis=0), title='Cumulative alpha')
        pts.plot_time_series(df=explained_returns.cumsum(axis=0), title='Cumulative explained return')

    elif local == Locals.ATTRIBUTION:
        benchmark_prices = prices[['SPY', 'TLT']]
        instrument_prices = prices[['QQQ', 'HYG', 'GLD']]
        exposures = pd.DataFrame(1.0/3.0, index=instrument_prices.index, columns=instrument_prices.columns)
        portfolio_nav = ret.returns_to_nav(returns=(exposures.shift(1)).multiply(instrument_prices.pct_change()).sum(axis=1))
        print(portfolio_nav)

        attribution = compute_portfolio_benchmark_ewm_beta_alpha_attribution(instrument_prices=instrument_prices,
                                                                             weights=exposures,
                                                                             benchmark_prices=benchmark_prices,
                                                                             portfolio_nav=portfolio_nav,
                                                                             time_period=None,
                                                                             freq_beta='W-WED',
                                                                             factor_beta_span=52,  # quarter
                                                                             residual_name='Alpha')
        pts.plot_time_series(df=attribution.cumsum(axis=0))

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.MODEL)
