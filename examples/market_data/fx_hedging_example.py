"""
Examples for the FX hedging research pipeline (run_local dispatcher).

By default, downloads free Yahoo FX spots and USD ETFs. Non-USD interest
rates use the explicitly illustrative spreads in fx_rates_data_yahoo_example.
Pass --input-dir to use your own fx_hedging_data_{fx_spots,domestic_rates,usd_assets}.csv
files instead; those production inputs are not distributed with qis.
"""
import argparse
from pathlib import Path
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


def load_usd_assets(local_path: str,
                    file_name: str = 'fx_hedging_data'
                    ) -> pd.DataFrame:
    """Load the USD benchmark asset prices (example/reporting input).

    These are the USD-denominated benchmark assets (Equities, Govvies, IG, HY)
    written alongside the FX data by the production builder. They are only used
    for examples and reports, so they are loaded here rather than in the core
    ``load_fx_rates_data``.
    """
    data = qis.load_df_dict_from_csv(dataset_keys=['usd_assets'],
                                     file_name=file_name, local_path=local_path,
                                     force_not_found_error=True)
    return data['usd_assets']


def load_example_inputs(input_dir: str = None):
    """Load explicit reader CSVs, or the public-data teaching universe by default."""
    if input_dir is not None:
        for key in ('fx_spots', 'domestic_rates', 'usd_assets'):
            path = Path(input_dir) / f'fx_hedging_data_{key}.csv'
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; omit --input-dir for the Yahoo demo.')
        fx_spots, domestic_rates = load_fx_rates_data(local_path=input_dir)
        usd_assets = load_usd_assets(local_path=input_dir)
    else:
        import yfinance as yf
        from examples.market_data.fx_rates_data_yahoo_example import fetch_fx_rates_data_from_yahoo

        fx = fetch_fx_rates_data_from_yahoo()
        fx_spots, domestic_rates = fx.fx_spots, fx.domestic_rates
        tickers = {'SPY': 'Equities', 'TLT': 'Govt Bonds', 'LQD': 'IG Bonds', 'HYG': 'HY Bonds'}
        usd_assets = yf.download(list(tickers), start='2005-12-31', auto_adjust=True,
                                 progress=False, threads=False)['Close'].reindex(columns=tickers)
        missing = usd_assets.columns[usd_assets.isna().all()].tolist()
        if missing:
            raise ValueError(
                f'Yahoo returned no ETF observations for {missing}; retry the download.')
        usd_assets = usd_assets.rename(columns=tickers)
        print('Yahoo FX/ETF prices; non-USD rates use illustrative differentials.')
    return fx_spots, domestic_rates, usd_assets


class Locals(Enum):
    # The default examples use public FX spots and ETF prices.
    LOAD_DATA = 2
    CHECK_HEDGED_RETURN = 3
    PLOT_HEDGE_REPORT = 5
    MULTI_ASSET_HEDGE = 6
    MULTI_ASSET_HEDGE_REPORT = 7
    LOCAL_RATE_ADJUSTMENT = 8


def run_local(local: Locals, input_dir: str = None):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    fx_spots, domestic_rates, usd_assets = load_example_inputs(input_dir)

    if local == Locals.LOAD_DATA:
        print(usd_assets)
        print(fx_spots)
        print(domestic_rates)

    elif local == Locals.CHECK_HEDGED_RETURN:
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
        carry_fx_nav = fx_rates_data.get_carry_fx_return_nav(
            local_ccy=local_ccy, reference_ccy=reference_ccy, freq=freq)

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
                          local_to_reference_fx_rate.rename(f"{carry_fx_nav.name} spot return"),
                          carry_fx_nav.rename(f"{carry_fx_nav.name} carry return"),
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

    elif local == Locals.PLOT_HEDGE_REPORT:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        asset_price_local_ccy = usd_assets['IG Bonds']
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)

        run_asset_fx_hedging_report(asset_price_local_ccy=asset_price_local_ccy,
                                    fx_rates_data=fx_rates_data,
                                    local_ccy='USD',
                                    reference_ccy='CHF',
                                    time_period=time_period)

    elif local == Locals.MULTI_ASSET_HEDGE:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        out = compute_multi_asset_fx_hedging(asset_prices=usd_assets,
                                             fx_rates_data=fx_rates_data,
                                             time_period=time_period,
                                             local_ccys='USD',
                                             reference_ccy='CHF')
        print(out)

    elif local == Locals.MULTI_ASSET_HEDGE_REPORT:
        time_period = qis.TimePeriod('31Dec2004', '31Oct2025')
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        plot_multi_asset_fx_hedging_report(asset_prices=usd_assets,
                                          fx_rates_data=fx_rates_data,
                                          time_period=time_period,
                                          local_ccy='USD',
                                          reference_ccy='CHF')

    elif local == Locals.LOCAL_RATE_ADJUSTMENT:
        fx_rates_data = FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)
        local_ccys = pd.Series('USD', index=usd_assets.columns)
        usd_assets.loc[:'31Dec2004', 'HY Bonds'] = pd.NA
        local_returns, excess_returns, local_rates_df = (
            fx_rates_data.compute_returns_adjusted_by_local_rate(
                asset_prices=usd_assets, local_ccys=local_ccys))
        print(local_returns)
        local_returns, excess_returns, local_rates_df = (
            fx_rates_data.compute_returns_adjusted_by_local_rate(
                asset_prices=usd_assets,
                local_ccys=pd.Series('USD', index=usd_assets.columns)))
        print(local_returns)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', help='Directory of reader-supplied FX and USD-asset CSVs.')
    run_local(local=Locals.LOCAL_RATE_ADJUSTMENT, input_dir=parser.parse_args().input_dir)
