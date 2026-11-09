"""
Build an FxRatesData container from Bloomberg via bbg-fetch and illustrate it.

This provider-specific teaching basket uses domestic term rates and overnight CAD/SGD
benchmarks. It requires a Bloomberg terminal and ``bbg-fetch`` (which wraps ``blpapi``).
Imports are deferred so the module also imports in an offline/core-only environment.
The consuming pipeline owns ticker selection, benchmark suitability and historical coverage;
this example builds the container in memory and does not persist or update production data.

Conventions (identical to the container contract): ``fx_spots`` columns are units of USD per 1 unit
of the local currency (FX pairs are quoted local/USD on Bloomberg, e.g. 'EURUSD Curncy'); USD is
pinned to 1.0; GBp = 0.01 * GBP; XAU = 1.0. ``domestic_rates`` are short rates as decimals — the
Bloomberg fields are quoted in percent, so they are scaled by 0.01; XAU uses the USD curve as a
financing-cost proxy and GBp shares the GBP curve.
"""

from qis.fx import fx_returns as qfx
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum

from qis.fx import FxRatesData
from qis.fx.reports.fx_hedging_report import (run_asset_fx_hedging_report,
                                                       plot_multi_asset_fx_hedging_report)

# USD-denominated benchmark assets (Bloomberg ticker -> asset name).
USD_ASSETS = {
    'NDUEACWF Index': 'Equities',
    'H03432US Index': 'Govvies',
    'LEGATRUH Index': 'IG Bonds',
    'H23059US Index': 'HY Bonds',
}

# Major currency pairs vs USD, quoted local_ccy/USD (units of USD per 1 unit of local currency).
FX_SPOTS = {
    'EURUSD Curncy': 'EUR',
    'GBPUSD Curncy': 'GBP',
    'CHFUSD Curncy': 'CHF',
    'JPYUSD Curncy': 'JPY',
    'AUDUSD Curncy': 'AUD',
    'CADUSD Curncy': 'CAD',
    'NZDUSD Curncy': 'NZD',
    'HKDUSD Curncy': 'HKD',
    'SGDUSD Curncy': 'SGD',
    'NOKUSD Curncy': 'NOK',
    'SEKUSD Curncy': 'SEK',
}

# Domestic term rates and overnight CAD/SGD benchmarks for carry/forward calculations.
# Bloomberg PX is in percent and is scaled to decimals in the builder below.
DOMESTIC_RATES = {
    'EUR003M Index': 'EUR',
    'BPSWSC BGN Curncy': 'GBP',
    'SSAR3M Index': 'CHF',
    'GJTB3MO Index': 'JPY',
    'BBSW3M Index': 'AUD',
    'usgg3m Index': 'USD',
    'CAONREPO Index': 'CAD',
    'NDSOC BGN Curncy': 'NZD',
    'HIHD03M Index': 'HKD',
    'SIBCSORA Index': 'SGD',  # overnight SORA (SIBOR/SOR discontinued); MAS, history from Jul-2005
    'NIBOR3M Index': 'NOK',   # 3M NIBOR (NoRe/GRSS), still-active critical benchmark; PX in pct
    'STIB3M Index': 'SEK',    # 3M STIBOR (Swedish Financial Benchmark Facility); PX in pct
}

REFERENCE_CCY = 'CHF'   # an investor in this currency views the USD-denominated benchmark panel
FREQ = 'ME'


def fetch_fx_rates_data_from_bloomberg(start_date: pd.Timestamp = pd.Timestamp('31Dec1998')) -> FxRatesData:
    """Build an in-memory FxRatesData container from Bloomberg (PX_LAST) via bbg-fetch.

    Fetches the FX spot pairs
    (already quoted local/USD), pins USD = 1.0, derives GBp = 0.01 * GBP and an XAU = 1.0 mapping;
    fetches the domestic short rates and scales them from percent to decimals, with XAU on the USD
    curve and GBp on the GBP curve. Leading missing rates remain unknown; no later quote
    is copied into the prehistory. This example does not write CSVs.

    Args:
        start_date: First date to request from Bloomberg.

    Returns:
        A ready FxRatesData; ``__post_init__`` forward-fills spots and aligns rates to the spot grid.
    """
    from bbg_fetch import fetch_field_timeseries_per_tickers

    fx_spots = fetch_field_timeseries_per_tickers(tickers=FX_SPOTS, freq='B', field='PX_LAST',
                                                  start_date=start_date).ffill()
    fx_spots['USD'] = 1.0                       # add USD for consistency (1 USD per USD)
    fx_spots['GBp'] = 0.01 * fx_spots['GBP']    # pence -> pounds, per the container contract
    fx_spots['XAU'] = 1.0                       # gold quoted in USD; legacy 1.0 mapping

    # Convert percent to decimals; forward-fill only after the first observed rate.
    domestic_rates = 0.01 * fetch_field_timeseries_per_tickers(tickers=DOMESTIC_RATES, freq='B',
                                                               field='PX_LAST',
                                                               start_date=start_date).ffill()
    domestic_rates['GBp'] = domestic_rates['GBP']   # same sovereign curve as GBP
    domestic_rates['XAU'] = domestic_rates['USD']   # USD rf as the gold financing-cost proxy

    return FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)


def fetch_usd_assets_from_bloomberg(start_date: pd.Timestamp = pd.Timestamp('31Dec1998')) -> pd.DataFrame:
    """Fetch the USD-denominated benchmark price panel (columns renamed to the USD_ASSETS values).

    Used by the translation / hedging cases below; in rosaa this panel is part of
    ``create_fx_rates_data`` and is persisted alongside the FX data.
    """
    from bbg_fetch import fetch_field_timeseries_per_tickers
    return fetch_field_timeseries_per_tickers(tickers=USD_ASSETS, freq='B', field='PX_LAST',
                                              start_date=start_date).ffill()


class Locals(Enum):
    SHOW_FX_DATA = 1
    CROSS_RATES = 2
    FX_TOTAL_RETURN_NAVS = 3
    CARRY_FORWARD_PREMIUM = 4
    CASH_NAVS = 5
    TRANSLATE_ASSET_PANEL = 6
    ASSET_HEDGE_REPORT = 7
    MULTI_ASSET_REPORT = 8


def run_local(local: Locals):
    """Illustrate the FxRatesData container built from Bloomberg data."""
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)

    fx_rates_data = fetch_fx_rates_data_from_bloomberg()

    if local == Locals.SHOW_FX_DATA:
        print("fx_spots — units of USD per 1 unit of local ccy (USD pinned to 1.0):")
        print(fx_rates_data.fx_spots.tail())
        print("\ndomestic_rates — short rates as decimals (term rates; overnight for CAD/SGD):")
        print(fx_rates_data.domestic_rates.tail())

    elif local == Locals.CROSS_RATES:
        # cross rate = units of reference ccy per 1 unit of local ccy
        crosses = pd.concat([fx_rates_data.get_local_to_reference_fx_rate(local_ccy=l, reference_ccy=r)
                             for l, r in (('EUR', 'CHF'), ('GBP', 'USD'), ('EUR', 'GBP'))], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=crosses, ax=ax, title='FX cross rates (reference per local)')

    elif local == Locals.FX_TOTAL_RETURN_NAVS:
        result = qis.compute_fx_returns(fx_rates_data, qis.FxCurrencyReturnsRequest(
            ('EUR', 'GBP', 'CHF', 'JPY', 'AUD'), 'USD', qis.FxReturnType.FX_FORWARD,
            frequency='B'))['B']
        navs = result.to_index(qis.FxIndexNormalization(
            initial_value=1.0, notional_to_capital=1.0, notional_rule='capital_proportional',
            settlement_frequency='B', reinvestment='each_interval',
            cash_treatment='as_returned', missing_returns='skip'))
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(
            prices=navs, ax=ax,
            title='Exact FX forward indices vs USD (capital-normalized, daily reset)')

    elif local == Locals.CARRY_FORWARD_PREMIUM:
        # annualised CIP forward premium of each ccy vs USD (per-period rate * periods-per-year)
        ppy = qis.get_annualization_factor(FREQ)
        premia = pd.concat([fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=l, reference_ccy='USD', freq=FREQ) * ppy
                            for l in ('EUR', 'GBP', 'CHF', 'JPY', 'AUD', 'SGD')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        premia.plot(ax=ax)
        ax.set_title('Annualised CIP forward premium vs USD (supplied term/overnight rates)')
        ax.axhline(0.0, color='black', lw=0.5)

    elif local == Locals.CASH_NAVS:
        # money-market NAV compounding each local short rate
        cash = pd.concat([fx_rates_data.build_local_cash_nav(local_ccy=c, freq='B')
                          for c in ('USD', 'EUR', 'CHF', 'JPY')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=cash, ax=ax, title='Money-market cash NAVs by currency')

    elif local == Locals.TRANSLATE_ASSET_PANEL:
        # translate the USD benchmark panel into a CHF investor's frame, unhedged vs fully hedged
        assets = fetch_usd_assets_from_bloomberg()
        local_ccys = pd.Series('USD', index=assets.columns)
        fx_result_179 = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=assets,
                local_ccys=local_ccys,
                return_type=qfx.FxReturnType.REFERENCE_TOTAL,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                reference_ccy=REFERENCE_CCY,
                hedge_ratios=pd.Series(0.0, index=assets.columns),
                frequency=FREQ,
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
                zero_return_to_nan=False,
            ),
        )[FREQ]
        navs_unhedged = fx_result_179.to_index(qfx.FxIndexNormalization(
            1.0, 1.0, 'capital_proportional', FREQ, 'each_interval', 'as_returned', 'skip'))
        fx_result_182 = qfx.compute_fx_returns(
            fx_rates_data,
            qfx.FxAssetReturnsRequest(
                prices=assets,
                local_ccys=local_ccys,
                return_type=qfx.FxReturnType.REFERENCE_TOTAL,
                coordinate=qfx.FxReturnCoordinate.SIMPLE,
                reference_ccy=REFERENCE_CCY,
                hedge_ratios=pd.Series(1.0, index=assets.columns),
                frequency=FREQ,
                cash_policy=qfx.FxCashPolicy(quote_lag=1),
                zero_return_to_nan=False,
            ),
        )[FREQ]
        navs_hedged = fx_result_182.to_index(qfx.FxIndexNormalization(
            1.0, 1.0, 'capital_proportional', FREQ, 'each_interval', 'as_returned', 'skip'))
        eq = pd.concat([navs_unhedged['Equities'].rename(f'Equities in {REFERENCE_CCY} (unhedged)'),
                        navs_hedged['Equities'].rename(f'Equities in {REFERENCE_CCY} (hedged)')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=eq, ax=ax, title=f'USD Equities in a {REFERENCE_CCY} frame: unhedged vs hedged')
        print(f"unhedged-in-{REFERENCE_CCY} NAVs (tail):\n", navs_unhedged.tail())
        print(f"\nhedged-in-{REFERENCE_CCY} NAVs (tail):\n", navs_hedged.tail())

    elif local == Locals.ASSET_HEDGE_REPORT:
        # single-asset hedging tearsheet for USD Equities viewed in CHF
        assets = fetch_usd_assets_from_bloomberg()
        run_asset_fx_hedging_report(asset_price_local_ccy=assets['Equities'], fx_rates_data=fx_rates_data,
                                    local_ccy='USD', reference_ccy=REFERENCE_CCY,
                                    time_period=qis.TimePeriod('31Dec2005', assets.index[-1]))

    elif local == Locals.MULTI_ASSET_REPORT:
        # multi-asset hedging heatmap report across the USD benchmark panel viewed in CHF
        assets = fetch_usd_assets_from_bloomberg()
        plot_multi_asset_fx_hedging_report(asset_prices=assets, fx_rates_data=fx_rates_data,
                                           local_ccy='USD', reference_ccy=REFERENCE_CCY,
                                           time_period=qis.TimePeriod('31Dec2005', assets.index[-1]))


if __name__ == '__main__':

    run_local(local=Locals.SHOW_FX_DATA)

    plt.show()
