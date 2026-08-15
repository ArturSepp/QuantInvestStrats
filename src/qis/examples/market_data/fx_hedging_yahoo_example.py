"""
Illustrate the FxRatesData FX-hedging analytics on real Yahoo data.

Data-driven companion to the hedging cases in fx_rates_data_test (CHECK_HEDGED_RETURN,
PLOT_HEDGE_REPORT, MULTI_ASSET_HEDGE, MULTI_ASSET_HEDGE_REPORT): a panel of USD-denominated assets
is viewed by a CHF investor, and the example exercises the optimal / max-carry / beta hedge ratios,
the hedged NAVs across hedge ratios, the EWMA FX vol and beta, the single-asset hedging report, and
the multi-asset hedging metrics and heatmap report.

FX spots are real Yahoo data; domestic rates are USD-real plus stylised differentials (see
fx_rates_data_yahoo_example), so the carry component is illustrative — production rates come from
Bloomberg, per the FxRatesData README.
"""
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis
from enum import Enum

from qis.examples.market_data.fx_rates_data_yahoo_example import fetch_fx_rates_data_from_yahoo
from qis.market_data.fx_hedging import (compute_fx_optimal_hedge,
                                        compute_fx_vol_beta,
                                        compute_performance_of_local_ccy_asset_in_reference_ccy)
from qis.market_data.reports.fx_hedging_report import (run_asset_fx_hedging_report,
                                                       compute_multi_asset_fx_hedging,
                                                       plot_multi_asset_fx_hedging_report)

USD_ASSETS = ['SPY', 'TLT', 'LQD', 'HYG', 'GLD']   # USD-denominated; viewed by a CHF investor
ASSET = 'SPY'
LOCAL_CCY = 'USD'
REFERENCE_CCY = 'CHF'
FREQ = 'ME'


def _load_assets(start_date: str = '2005-12-31') -> pd.DataFrame:
    return yf.download(USD_ASSETS, start=start_date, auto_adjust=True, progress=False)['Close'][USD_ASSETS].dropna()


class LocalTests(Enum):
    HEDGE_RATIOS = 1        # optimal / max-carry / beta hedge ratios over time (single asset)
    HEDGED_NAVS = 2         # asset NAV in CHF across hedge ratios h in {0, 0.5, 1, optimal}
    FX_VOL_BETA = 3         # EWMA FX vol and asset-to-FX beta
    ASSET_HEDGE_REPORT = 4  # single-asset one-call hedging report
    MULTI_ASSET_METRICS = 5  # compute_multi_asset_fx_hedging: p.a. / Sharpe / hedge-ratio tables
    MULTI_ASSET_REPORT = 6   # plot_multi_asset_fx_hedging_report: heatmap report across the panel


def run_local_test(local_test: LocalTests):
    pd.set_option('display.width', 180)
    pd.set_option('display.max_columns', 50)

    fx_rates_data = fetch_fx_rates_data_from_yahoo(start_date='2005-12-31')
    assets = _load_assets()
    time_period = qis.TimePeriod('31Dec2010', assets.index[-1])

    if local_test in (LocalTests.HEDGE_RATIOS, LocalTests.HEDGED_NAVS, LocalTests.FX_VOL_BETA):
        # FX rate and CIP forward premium for the USD -> CHF leg, shared by the single-asset cases
        l2r = fx_rates_data.get_local_to_reference_fx_rate(local_ccy=LOCAL_CCY, reference_ccy=REFERENCE_CCY)
        fwd = fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=LOCAL_CCY, reference_ccy=REFERENCE_CCY, freq=FREQ)

    if local_test == LocalTests.HEDGE_RATIOS:
        optimal, max_carry, beta_hedged = compute_fx_optimal_hedge(
            asset_price_local_ccy=assets[ASSET], local_to_reference_fx_rate=l2r,
            forward_rate_for_local_ccy=fwd, freq=FREQ)
        hedges = pd.concat([optimal.rename('Optimal'), max_carry.rename('Max carry'),
                            beta_hedged.rename('Beta-hedged')], axis=1)
        qis.plot_time_series(hedges, title=f'{ASSET} USD->CHF hedge ratios')

    elif local_test == LocalTests.HEDGED_NAVS:
        optimal, _, _ = compute_fx_optimal_hedge(asset_price_local_ccy=assets[ASSET], local_to_reference_fx_rate=l2r,
                                                 forward_rate_for_local_ccy=fwd, freq=FREQ)
        kwargs = dict(asset_price_local_ccy=assets[ASSET], local_to_reference_fx_rate=l2r,
                      forward_rate_for_local_ccy=fwd, freq=FREQ)
        nav0, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.0, **kwargs)
        nav05, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.5, **kwargs)
        nav1, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=1.0, **kwargs)
        nav_opt, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=optimal, **kwargs)
        navs = pd.concat([nav0.rename('h=0.0'), nav05.rename('h=0.5'), nav1.rename('h=1.0'),
                          nav_opt.rename('Optimal')], axis=1)
        qis.plot_prices_with_dd(prices=navs, perf_params=qis.PerfParams(freq=FREQ),
                                title=f'{ASSET} in {REFERENCE_CCY}: NAV by hedge ratio')

    elif local_test == LocalTests.FX_VOL_BETA:
        fx_vol, fx_beta = compute_fx_vol_beta(asset_price_local_ccy=assets[ASSET], local_to_reference_fx_rate=l2r,
                                              freq=FREQ, span=3 * 12)
        panel = pd.concat([fx_vol.rename(f'{ASSET} FX vol'), fx_beta.rename(f'{ASSET} FX beta')], axis=1)
        qis.plot_time_series(panel, title=f'{ASSET} EWMA FX vol and beta (USD->CHF)')

    elif local_test == LocalTests.ASSET_HEDGE_REPORT:
        run_asset_fx_hedging_report(asset_price_local_ccy=assets[ASSET], fx_rates_data=fx_rates_data,
                                    local_ccy=LOCAL_CCY, reference_ccy=REFERENCE_CCY, time_period=time_period)

    elif local_test == LocalTests.MULTI_ASSET_METRICS:
        out = compute_multi_asset_fx_hedging(asset_prices=assets, fx_rates_data=fx_rates_data,
                                             local_ccys=LOCAL_CCY, reference_ccy=REFERENCE_CCY,
                                             time_period=time_period)
        print(f"USD assets viewed in {REFERENCE_CCY} — multi-asset hedging metrics (asset x strategy):")
        print("\nP.a. return:\n", out['pas'])
        print("\nSharpe (rf=0):\n", out['sharpes'])
        print("\nLast hedge ratios:\n", out['last_hedges'])

    elif local_test == LocalTests.MULTI_ASSET_REPORT:
        plot_multi_asset_fx_hedging_report(asset_prices=assets, fx_rates_data=fx_rates_data,
                                           local_ccy=LOCAL_CCY, reference_ccy=REFERENCE_CCY,
                                           time_period=time_period)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MULTI_ASSET_REPORT)

    plt.show()