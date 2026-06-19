"""
Build an FxRatesData container from Bloomberg via bbg-fetch and illustrate it.

This is the production data path: real 3M domestic rates (overnight for SGD) and the full rosaa
currency set, in contrast to the Yahoo example (free, but USD-rate-plus-stylised-spreads because
Yahoo carries no usable non-USD short rate). It requires a Bloomberg terminal and the ``bbg-fetch``
package (which wraps ``blpapi``); the import is deferred into the fetch helpers so this module still
imports without a terminal present.

The ticker universe and the spot/rate construction below are extracted from the rosaa production
builder ``create_fx_rates_data``. The difference is scope: ``create_fx_rates_data`` also fetches the
benchmark asset panel and *persists* all three frames to CSV (and stays in rosaa); this example
builds the ``FxRatesData`` container in-memory and exercises the same analytics as the Yahoo example
(cross rates, CIP forward premia, FX total-return NAVs, money-market cash NAVs, reference-currency
translation, and the hedging reports).

Conventions (identical to the container contract): ``fx_spots`` columns are units of USD per 1 unit
of the local currency (FX pairs are quoted local/USD on Bloomberg, e.g. 'EURUSD Curncy'); USD is
pinned to 1.0; GBp = 0.01 * GBP; XAU = 1.0. ``domestic_rates`` are short rates as decimals — the
Bloomberg fields are quoted in percent, so they are scaled by 0.01; XAU uses the USD curve as a
financing-cost proxy and GBp shares the GBP curve.
"""
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum

from qis.market_data import FxRatesData
from qis.market_data.reports.fx_hedging_report import (run_asset_fx_hedging_report,
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

# Short-term domestic rates for the carry/forward calculation (3-month, except SGD = overnight).
# Bloomberg PX is in percent and is scaled to decimals in the builder below.
DOMESTIC_RATES = {
    'EUR003M Index': 'EUR',
    'BPSWSC BGN Curncy': 'GBP',
    'SSAR3M Index': 'CHF',
    'GJTB3MO Index': 'JPY',
    'BBSW3M Index': 'AUD',
    'usgg3m Index': 'USD',
    'CDOR03 Index': 'CAD',
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

    Mirrors the spot/rate construction in rosaa's ``create_fx_rates_data``: fetches the FX spot pairs
    (already quoted local/USD), pins USD = 1.0, derives GBp = 0.01 * GBP and an XAU = 1.0 mapping;
    fetches the domestic short rates and scales them from percent to decimals, with XAU on the USD
    curve and GBp on the GBP curve. Unlike the production builder this does not write CSVs — the
    persisted-CSV path (and the benchmark-asset fetch) stay in rosaa.

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

    # Bloomberg rate fields are in percent -> scale to decimals; bfill the short pre-history head.
    domestic_rates = 0.01 * fetch_field_timeseries_per_tickers(tickers=DOMESTIC_RATES, freq='B',
                                                               field='PX_LAST',
                                                               start_date=start_date).ffill().bfill()
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


class LocalTests(Enum):
    SHOW_FX_DATA = 1
    CROSS_RATES = 2
    FX_TOTAL_RETURN_NAVS = 3
    CARRY_FORWARD_PREMIUM = 4
    CASH_NAVS = 5
    TRANSLATE_ASSET_PANEL = 6
    ASSET_HEDGE_REPORT = 7
    MULTI_ASSET_REPORT = 8


def run_local_test(local_test: LocalTests):
    """Illustrate the FxRatesData container built from Bloomberg data."""
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)

    fx_rates_data = fetch_fx_rates_data_from_bloomberg()

    if local_test == LocalTests.SHOW_FX_DATA:
        print("fx_spots — units of USD per 1 unit of local ccy (USD pinned to 1.0):")
        print(fx_rates_data.fx_spots.tail())
        print("\ndomestic_rates — short rates as decimals (3M, overnight for SGD):")
        print(fx_rates_data.domestic_rates.tail())

    elif local_test == LocalTests.CROSS_RATES:
        # cross rate = units of reference ccy per 1 unit of local ccy
        crosses = pd.concat([fx_rates_data.get_local_to_reference_fx_rate(local_ccy=l, reference_ccy=r)
                             for l, r in (('EUR', 'CHF'), ('GBP', 'USD'), ('EUR', 'GBP'))], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=crosses, ax=ax, title='FX cross rates (reference per local)')

    elif local_test == LocalTests.FX_TOTAL_RETURN_NAVS:
        # spot move + carry, as a total-return NAV for each pair vs USD
        navs = pd.concat([fx_rates_data.get_fx_total_return_nav(local_ccy=l, reference_ccy='USD')
                          for l in ('EUR', 'GBP', 'CHF', 'JPY', 'AUD')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=navs, ax=ax, title='FX total-return NAVs vs USD (spot + carry)')

    elif local_test == LocalTests.CARRY_FORWARD_PREMIUM:
        # annualised CIP forward premium of each ccy vs USD (per-period rate * periods-per-year)
        ppy = qis.get_annualization_factor(FREQ)
        premia = pd.concat([fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=l, reference_ccy='USD', freq=FREQ) * ppy
                            for l in ('EUR', 'GBP', 'CHF', 'JPY', 'AUD', 'SGD')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        premia.plot(ax=ax)
        ax.set_title('Annualised CIP forward premium vs USD (real 3M rates)')
        ax.axhline(0.0, color='black', lw=0.5)

    elif local_test == LocalTests.CASH_NAVS:
        # money-market NAV compounding each local short rate
        cash = pd.concat([fx_rates_data.build_local_cash_nav(local_ccy=c, freq='B')
                          for c in ('USD', 'EUR', 'CHF', 'JPY')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=cash, ax=ax, title='Money-market cash NAVs by currency')

    elif local_test == LocalTests.TRANSLATE_ASSET_PANEL:
        # translate the USD benchmark panel into a CHF investor's frame, unhedged vs fully hedged
        assets = fetch_usd_assets_from_bloomberg()
        local_ccys = pd.Series('USD', index=assets.columns)
        navs_unhedged, _ = fx_rates_data.compute_returns_in_reference_ccy(
            asset_prices=assets, hedge_ratios=pd.Series(0.0, index=assets.columns),
            local_ccys=local_ccys, reference_ccy=REFERENCE_CCY, freq=FREQ)
        navs_hedged, _ = fx_rates_data.compute_returns_in_reference_ccy(
            asset_prices=assets, hedge_ratios=pd.Series(1.0, index=assets.columns),
            local_ccys=local_ccys, reference_ccy=REFERENCE_CCY, freq=FREQ)
        eq = pd.concat([navs_unhedged['Equities'].rename(f'Equities in {REFERENCE_CCY} (unhedged)'),
                        navs_hedged['Equities'].rename(f'Equities in {REFERENCE_CCY} (hedged)')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=eq, ax=ax, title=f'USD Equities in a {REFERENCE_CCY} frame: unhedged vs hedged')
        print(f"unhedged-in-{REFERENCE_CCY} NAVs (tail):\n", navs_unhedged.tail())
        print(f"\nhedged-in-{REFERENCE_CCY} NAVs (tail):\n", navs_hedged.tail())

    elif local_test == LocalTests.ASSET_HEDGE_REPORT:
        # single-asset hedging tearsheet for USD Equities viewed in CHF
        assets = fetch_usd_assets_from_bloomberg()
        run_asset_fx_hedging_report(asset_price_local_ccy=assets['Equities'], fx_rates_data=fx_rates_data,
                                    local_ccy='USD', reference_ccy=REFERENCE_CCY,
                                    time_period=qis.TimePeriod('31Dec2005', assets.index[-1]))

    elif local_test == LocalTests.MULTI_ASSET_REPORT:
        # multi-asset hedging heatmap report across the USD benchmark panel viewed in CHF
        assets = fetch_usd_assets_from_bloomberg()
        plot_multi_asset_fx_hedging_report(asset_prices=assets, fx_rates_data=fx_rates_data,
                                           local_ccy='USD', reference_ccy=REFERENCE_CCY,
                                           time_period=qis.TimePeriod('31Dec2005', assets.index[-1]))


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SHOW_FX_DATA)

    plt.show()
