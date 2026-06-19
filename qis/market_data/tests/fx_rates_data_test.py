"""
implementation tests for fx_rates_data object
"""
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum

from qis.market_data import FxRatesData, load_fx_rates_data
from qis.market_data.fx_hedging import (compute_fx_optimal_hedge,
                                        compute_fx_vol_beta,
                                        compute_performance_of_local_ccy_asset_in_reference_ccy)
from qis.market_data.reports.fx_hedging_report import (run_asset_fx_hedging_report,
                                                       compute_multi_asset_fx_hedging,
                                                       plot_multi_asset_fx_hedging_report)
# NOTE: create_fx_rates_data (Bloomberg build) stays in the production layer (rosaa); the
# CREATE_DATA case is exercised there, not here.



class LocalTests(Enum):
    CREATE_DATA = 1
    LOAD_DATA = 2
    CHECK_HEDGED_RETURN = 3
    CHECK_CHF = 4
    PLOT_HEDGE_REPORT = 5
    MULTI_ASSET_HEDGE = 6
    MULTI_ASSET_HEDGE_REPORT = 7
    LOCAL_RATE_ADJUSTMENT = 8


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from rosaa import local_path as lp

    if local_test == LocalTests.CREATE_DATA:
        create_fx_rates_data(local_path=lp.get_resource_path())

    elif local_test == LocalTests.LOAD_DATA:
        usd_assets, fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        print(usd_assets)
        print(fx_spots)
        print(domestic_rates)

    elif local_test == LocalTests.CHECK_HEDGED_RETURN:
        usd_assets, fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        asset_price_local_ccy = usd_assets['Equities']

        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        local_ccy = 'USD'
        reference_ccy = 'CHF'
        freq = 'ME'
        dt = 1.0 / 12.0
        # get rates universe
        local_to_reference_fx_rate = fx_rates_data.get_local_to_reference_fx_rate(local_ccy=local_ccy,
                                                                                  reference_ccy=reference_ccy)
        forward_rate_for_local_ccy = fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=local_ccy,
                                                                                  reference_ccy=reference_ccy, dt=dt)
        carry_fx_nav = fx_rates_data.get_carry_fx_return_nav(local_ccy=local_ccy, reference_ccy=reference_ccy,
                                                             freq=freq)

        kwargs = dict(asset_price_local_ccy=asset_price_local_ccy,
                      local_to_reference_fx_rate=local_to_reference_fx_rate,
                      forward_rate_for_local_ccy=forward_rate_for_local_ccy, freq=freq)

        optimal_hedge, max_carry, beta_hedged = compute_fx_optimal_hedge(**kwargs, dt=dt)

        hedges = pd.concat([optimal_hedge, max_carry, beta_hedged], axis=1)
        qis.plot_time_series(hedges, title='hedges')

        nav0, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.0, **kwargs)
        nav05, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.5, **kwargs)
        nav1, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=1.0, **kwargs)
        nav_optimal, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=optimal_hedge, **kwargs)

        navs = pd.concat([asset_price_local_ccy,
                          local_to_reference_fx_rate.rename(f"{carry_fx_nav.name} spot return"),
                          carry_fx_nav.rename(f"{carry_fx_nav.name} carry return"),
                          nav0.rename('h=0.0'), nav05.rename('h=0.5'), nav1.rename('h=1.0'),
                          nav_optimal.rename('Optimal')], axis=1)
        qis.plot_prices_with_dd(prices=navs, perf_params=qis.PerfParams(freq='ME'))

        fx_vol, fx_beta = compute_fx_vol_beta(asset_price_local_ccy=asset_price_local_ccy,
                                              local_to_reference_fx_rate=local_to_reference_fx_rate, freq=freq,
                                              span=3 * 12)
        qis.plot_time_series(fx_beta)
        qis.plot_time_series(fx_vol)

    elif local_test == LocalTests.CHECK_CHF:

        from bbg_fetch import fetch_field_timeseries_per_tickers

        _, fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)

        assets = {'LGCPTRUH Index': 'IG USD', 'LGCPTRCH Index': 'IG CHF Hedged'}
        prices = fetch_field_timeseries_per_tickers(tickers=assets, freq='B', field='PX_LAST', start_date=pd.Timestamp('31Dec2001')).ffill()

        nav_hedged, _ = fx_rates_data._compute_performance_of_local_ccy_asset_in_reference_ccy(
            local_ccy='USD',
            reference_ccy='CHF',
            asset_price_local_ccy=prices.iloc[:, 0],
            hedge_ratio=1.0)

        prices: pd.DataFrame = pd.concat([prices, nav_hedged.rename('CHF-hedge replica')], axis=1)
        prices = prices.asfreq('QE').ffill()
        qis.plot_prices_with_dd(prices=prices, perf_params=qis.PerfParams(freq='ME'), title='Hedged')

        assets = {'NDUEACWF Index': 'ACWI USD', 'MEWD Index': 'ACWI CHF'}
        prices = fetch_field_timeseries_per_tickers(tickers=assets, freq='B', field='PX_LAST', start_date=pd.Timestamp('31Dec2013')).ffill()
        nav_hedged, _ = fx_rates_data._compute_performance_of_local_ccy_asset_in_reference_ccy(
            local_ccy='USD',
            reference_ccy='CHF',
            asset_price_local_ccy=prices.iloc[:, 0],
            hedge_ratio=0.0)

        prices = pd.concat([prices, nav_hedged.rename('CHF-local replica')], axis=1).asfreq('QE').ffill().loc['31Dec2013':, :]
        qis.plot_prices_with_dd(prices=prices, perf_params=qis.PerfParams(freq='ME'), title='Unhedged')

    elif local_test == LocalTests.PLOT_HEDGE_REPORT:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        usd_assets, fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        asset_price_local_ccy = usd_assets['IG Bonds']
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)

        run_asset_fx_hedging_report(asset_price_local_ccy=asset_price_local_ccy,
                                    fx_rates_data=fx_rates_data,
                                    local_ccy='USD',
                                    reference_ccy='CHF',
                                    time_period=time_period)

    elif local_test == LocalTests.MULTI_ASSET_HEDGE:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        usd_assets, fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        out = compute_multi_asset_fx_hedging(asset_prices=usd_assets,
                                             fx_rates_data=fx_rates_data,
                                             time_period=time_period,
                                             local_ccys='USD',
                                             reference_ccy='CHF')
        print(out)

    elif local_test == LocalTests.MULTI_ASSET_HEDGE_REPORT:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        usd_assets, fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        plot_multi_asset_fx_hedging_report(asset_prices=usd_assets,
                                          fx_rates_data=fx_rates_data,
                                          time_period=time_period,
                                          local_ccy='USD',
                                          reference_ccy='CHF')

    elif local_test == LocalTests.LOCAL_RATE_ADJUSTMENT:
        usd_assets, fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        local_ccys = pd.Series('USD', index=usd_assets.columns)
        usd_assets.loc[:'31Dec2004', 'HY Bonds'] = pd.NA
        local_returns, excess_returns, local_rates_df = fx_rates_data.compute_returns_adjusted_by_local_rate(asset_prices=usd_assets,
                                                                             local_ccys=local_ccys)
        print(local_returns)
        local_returns, excess_returns, local_rates_df = fx_rates_data.compute_returns_adjusted_by_local_rate(asset_prices=usd_assets,
                                                                             local_ccys=pd.Series('USD', index=usd_assets.columns))
        print(local_returns)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.LOCAL_RATE_ADJUSTMENT)
