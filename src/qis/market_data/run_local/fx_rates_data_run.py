"""
implementation tests for fx_rates_data object
"""

from qis.market_data import fx_returns as qfx
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
import qis.local_path as lp
from enum import Enum

from qis.market_data import FxRatesData, load_fx_rates_data
from qis.market_data.fx_hedging import (compute_fx_optimal_hedge,
                                        compute_fx_vol_beta,
                                        compute_performance_of_local_ccy_asset_in_reference_ccy)
from qis.market_data.reports.fx_hedging_report import (run_asset_fx_hedging_report,
                                                       compute_multi_asset_fx_hedging,
                                                       plot_multi_asset_fx_hedging_report)


def _load_runner_data(local_path: str,
                      file_name: str = 'fx_hedging_data'
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the example-only USD assets alongside the provider-neutral FX panels."""
    fx_spots, domestic_rates = load_fx_rates_data(local_path=local_path, file_name=file_name)
    data = qis.load_df_dict_from_csv(dataset_keys=['usd_assets'],
                                     file_name=file_name, local_path=local_path)
    return data['usd_assets'], fx_spots, domestic_rates


class Locals(Enum):
    LOAD_DATA = 2
    CHECK_HEDGED_RETURN = 3
    CHECK_CHF = 4
    PLOT_HEDGE_REPORT = 5
    MULTI_ASSET_HEDGE = 6
    MULTI_ASSET_HEDGE_REPORT = 7
    LOCAL_RATE_ADJUSTMENT = 8


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    if local == Locals.LOAD_DATA:
        usd_assets, fx_spots, domestic_rates = _load_runner_data(local_path=lp.get_resource_path())
        print(usd_assets)
        print(fx_spots)
        print(domestic_rates)

    elif local == Locals.CHECK_HEDGED_RETURN:
        usd_assets, fx_spots, domestic_rates = _load_runner_data(local_path=lp.get_resource_path())
        asset_price_local_ccy = usd_assets['Equities']

        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        local_ccy = 'USD'
        reference_ccy = 'CHF'
        freq = 'ME'
        # get rates universe
        local_to_reference_fx_rate = fx_rates_data.get_local_to_reference_fx_rate(
            local_ccy=local_ccy, reference_ccy=reference_ccy)
        forward_rate_for_local_ccy = fx_rates_data.get_forward_rate_for_local_ccy(
            local_ccy=local_ccy, reference_ccy=reference_ccy, freq=freq)
        forward_result = qis.compute_fx_returns(fx_rates_data, qis.FxCurrencyReturnsRequest(
            (local_ccy,), reference_ccy, qis.FxReturnType.FX_FORWARD, frequency='B'))['B']
        forward_fx_nav = forward_result.to_index(qis.FxIndexNormalization(
            initial_value=1.0, notional_to_capital=1.0, notional_rule='capital_proportional',
            settlement_frequency='B', reinvestment='each_interval',
            cash_treatment='as_returned', missing_returns='skip')).iloc[:, 0]
        forward_fx_nav = qis.df_asfreq(forward_fx_nav, freq=freq).rename(f'{local_ccy}-{reference_ccy}')

        kwargs = dict(asset_price_local_ccy=asset_price_local_ccy,
                      local_to_reference_fx_rate=local_to_reference_fx_rate,
                      forward_rate_for_local_ccy=forward_rate_for_local_ccy, freq=freq)

        optimal_hedge, max_carry, beta_hedged = compute_fx_optimal_hedge(**kwargs)

        hedges = pd.concat([optimal_hedge, max_carry, beta_hedged], axis=1)
        qis.plot_time_series(hedges, title='hedges')

        nav0, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.0, **kwargs)
        nav05, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(
            hedge_ratio=0.5, **kwargs)
        nav1, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=1.0, **kwargs)
        nav_optimal, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(
            hedge_ratio=optimal_hedge, **kwargs)

        navs = pd.concat([asset_price_local_ccy,
                          local_to_reference_fx_rate.rename(f"{forward_fx_nav.name} spot return"),
                          forward_fx_nav.rename(f"{forward_fx_nav.name} forward index (normalized)"),
                          nav0.rename('h=0.0'), nav05.rename('h=0.5'), nav1.rename('h=1.0'),
                          nav_optimal.rename('Optimal')], axis=1)
        qis.plot_prices_with_dd(prices=navs, perf_params=qis.PerfParams(freq='ME'))

        fx_vol, fx_beta = compute_fx_vol_beta(
            asset_price_local_ccy=asset_price_local_ccy,
            local_to_reference_fx_rate=local_to_reference_fx_rate,
            freq=freq,
            span=3 * 12)
        qis.plot_time_series(fx_beta)
        qis.plot_time_series(fx_vol)

    elif local == Locals.CHECK_CHF:

        from bbg_fetch import fetch_field_timeseries_per_tickers

        fx_spots, domestic_rates = load_fx_rates_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)

        assets = {'LGCPTRUH Index': 'IG USD', 'LGCPTRCH Index': 'IG CHF Hedged'}
        prices = fetch_field_timeseries_per_tickers(
            tickers=assets, freq='B', field='PX_LAST',
            start_date=pd.Timestamp('31Dec2001')).ffill()

        fx_result_116 = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=prices.iloc[:, 0].to_frame(),
                local_ccys='USD',
                reference_ccy='CHF',
                return_type=qfx.FxReturnType.REFERENCE_TOTAL,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                hedge_ratios=1.0,
                frequency='ME',
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
            ),
        )['ME']
        nav_hedged = fx_result_116.to_index(qfx.FxIndexNormalization(
            1., 1., 'capital_proportional', 'ME', 'each_interval', 'as_returned', 'skip')).iloc[:, 0]

        prices: pd.DataFrame = pd.concat([prices, nav_hedged.rename('CHF-hedge replica')], axis=1)
        prices = prices.asfreq('QE').ffill()
        qis.plot_prices_with_dd(
            prices=prices, perf_params=qis.PerfParams(freq='ME'), title='Hedged')

        assets = {'NDUEACWF Index': 'ACWI USD', 'MEWD Index': 'ACWI CHF'}
        prices = fetch_field_timeseries_per_tickers(
            tickers=assets, freq='B', field='PX_LAST',
            start_date=pd.Timestamp('31Dec2013')).ffill()
        fx_result_131 = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=prices.iloc[:, 0].to_frame(),
                local_ccys='USD',
                reference_ccy='CHF',
                return_type=qfx.FxReturnType.REFERENCE_TOTAL,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                hedge_ratios=0.0,
                frequency='ME',
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
            ),
        )['ME']
        nav_hedged = fx_result_131.to_index(qfx.FxIndexNormalization(
            1., 1., 'capital_proportional', 'ME', 'each_interval', 'as_returned', 'skip')).iloc[:, 0]

        prices = (pd.concat([prices, nav_hedged.rename('CHF-local replica')], axis=1)
                  .asfreq('QE').ffill().loc['31Dec2013':, :])
        qis.plot_prices_with_dd(
            prices=prices, perf_params=qis.PerfParams(freq='ME'), title='Unhedged')

    elif local == Locals.PLOT_HEDGE_REPORT:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        usd_assets, fx_spots, domestic_rates = _load_runner_data(local_path=lp.get_resource_path())
        asset_price_local_ccy = usd_assets['IG Bonds']
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)

        run_asset_fx_hedging_report(asset_price_local_ccy=asset_price_local_ccy,
                                    fx_rates_data=fx_rates_data,
                                    local_ccy='USD',
                                    reference_ccy='CHF',
                                    time_period=time_period)

    elif local == Locals.MULTI_ASSET_HEDGE:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        usd_assets, fx_spots, domestic_rates = _load_runner_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        out = compute_multi_asset_fx_hedging(asset_prices=usd_assets,
                                             fx_rates_data=fx_rates_data,
                                             time_period=time_period,
                                             local_ccys='USD',
                                             reference_ccy='CHF')
        print(out)

    elif local == Locals.MULTI_ASSET_HEDGE_REPORT:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        usd_assets, fx_spots, domestic_rates = _load_runner_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        plot_multi_asset_fx_hedging_report(asset_prices=usd_assets,
                                          fx_rates_data=fx_rates_data,
                                          time_period=time_period,
                                          local_ccy='USD',
                                          reference_ccy='CHF')

    elif local == Locals.LOCAL_RATE_ADJUSTMENT:
        usd_assets, fx_spots, domestic_rates = _load_runner_data(local_path=lp.get_resource_path())
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        local_ccys = pd.Series('USD', index=usd_assets.columns)
        usd_assets.loc[:'31Dec2004', 'HY Bonds'] = pd.NA
        local_returns = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=usd_assets,
                local_ccys=local_ccys,
                return_type=qfx.FxReturnType.LOCAL_TOTAL,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                frequency='ME',
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
                zero_return_to_nan=False,
            ),
        )['ME'].values
        excess_returns = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=usd_assets,
                local_ccys=local_ccys,
                return_type=qfx.FxReturnType.LOCAL_EXCESS,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                frequency='ME',
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
                zero_return_to_nan=False,
            ),
        )['ME'].values
        local_rates_df = fx_rates_data.fetch_local_rates(local_ccys=local_ccys, freq='ME', annualise=True)
        print(local_returns)
        print(excess_returns)
        print(local_rates_df)
        local_returns = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=usd_assets,
                local_ccys=pd.Series('USD', index=usd_assets.columns),
                return_type=qfx.FxReturnType.LOCAL_TOTAL,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                frequency='ME',
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
                zero_return_to_nan=False,
            ),
        )['ME'].values
        excess_returns = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=usd_assets,
                local_ccys=pd.Series('USD', index=usd_assets.columns),
                return_type=qfx.FxReturnType.LOCAL_EXCESS,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                frequency='ME',
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
                zero_return_to_nan=False,
            ),
        )['ME'].values
        local_rates_df = fx_rates_data.fetch_local_rates(
            local_ccys=pd.Series('USD', index=usd_assets.columns), freq='ME', annualise=True)
        print(local_returns)
        print(excess_returns)
        print(local_rates_df)

    plt.show()


if __name__ == '__main__':

    run_local(local=Locals.LOCAL_RATE_ADJUSTMENT)
