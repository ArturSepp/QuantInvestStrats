"""
Build an FxRatesData container from free Yahoo data (yfinance) and illustrate it.

The production FxRatesData is sourced from Bloomberg (see the builder in the production layer).
This example provides a self-contained, no-terminal path: it fetches FX spot pairs from Yahoo,
maps them into the container's quoting convention (units of USD per 1 unit of local currency),
attaches domestic 3M rates, and exercises the container's analytics (cross rates, CIP forward
premia, FX total-return NAVs, money-market cash NAVs, and reference-currency translation of an
asset panel).

Domestic rates: the USD 3M is taken from '^IRX' (the 13-week T-bill, real). Yahoo's only reliable
rate series are US Treasuries (^IRX, ^FVX, ^TNX, ^TYX); it has no usable non-USD 3M rate. For CHF
specifically there is a 'SARON.SW' ticker, but SARON is an *overnight* rate (not 3M) and its Yahoo
feed is stale (a negative-rate-era value frozen at an old close), so it is not used here. EUR/GBP/JPY/
AUD/CAD/NZD have no 3M rate on Yahoo at all. The non-USD rates are therefore approximated as the USD
rate plus a stylised, clearly-labelled differential (see _ILLUSTRATIVE_3M_SPREAD_VS_USD) — for
illustration only; production uses Bloomberg 3M rates, per the FxRatesData README. The FX spots, by
contrast, are real Yahoo data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis
from enum import Enum
from typing import Optional

from qis.market_data import FxRatesData

# Yahoo FX tickers quoted directly as USD per 1 unit of local ccy (e.g. EURUSD=X ~ 1.08 USD/EUR).
_DIRECT_USD_PER_LOCAL = {'EUR': 'EURUSD=X', 'GBP': 'GBPUSD=X', 'AUD': 'AUDUSD=X', 'NZD': 'NZDUSD=X'}
# Yahoo FX tickers quoted as local per USD (e.g. USDJPY=X ~ 150 JPY/USD); inverted to USD per local.
_INVERTED_LOCAL_PER_USD = {'JPY': 'USDJPY=X', 'CHF': 'USDCHF=X', 'CAD': 'USDCAD=X'}

# Stylised average 3M rate differentials vs USD (decimals) — ILLUSTRATIVE ONLY.
# Yahoo has no usable non-USD 3M series (CHF's SARON.SW is overnight and its feed is stale; the
# other currencies have nothing), so production sources these from Bloomberg (see README).
_ILLUSTRATIVE_3M_SPREAD_VS_USD = {'USD': 0.0, 'EUR': -0.005, 'GBP': 0.002, 'CHF': -0.010,
                                  'JPY': -0.020, 'AUD': 0.005, 'CAD': 0.001, 'NZD': 0.010}


def _build_illustrative_domestic_rates(usd_rate: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Assemble the domestic_rates panel (decimals) from a real USD rate plus stylised per-ccy spreads.

    The USD column is the supplied real rate; every other 3-letter ccy is usd_rate + its stylised
    differential. The auxiliary columns follow the FxRatesData contract: 'GBp' shares the GBP curve
    (same sovereign) and 'XAU' uses the USD rate as a financing-cost proxy.
    """
    rates = pd.DataFrame({ccy: usd_rate + spread for ccy, spread in _ILLUSTRATIVE_3M_SPREAD_VS_USD.items()},
                         index=index)
    rates['GBp'] = rates['GBP']   # GBp shares the GBP sovereign curve
    rates['XAU'] = rates['USD']   # gold has no native rf; USD proxy, per the container contract
    return rates


def fetch_fx_rates_data_from_yahoo(start_date: str = '2005-12-31',
                                   end_date: Optional[str] = None
                                   ) -> FxRatesData:
    """
    Construct an FxRatesData container from Yahoo data fetched with yfinance.

    FX spots are fetched for the eight rosaa-universe currencies and mapped into the container's
    convention — each column is units of USD per 1 unit of the local currency, with USD pinned to
    1.0. Tickers quoted as USD-per-local (EURUSD=X, GBPUSD=X, AUDUSD=X, NZDUSD=X) are used directly;
    tickers quoted as local-per-USD (USDJPY=X, USDCHF=X, USDCAD=X) are inverted. The auxiliary
    mappings from the FxRatesData contract are added: GBp = 0.01 * GBP and XAU = 1.0.

    Domestic 3M rates use the real USD rate ('^IRX' / 100) and stylised per-ccy differentials for
    the rest (see module docstring) — illustrative only.

    Parameters
    ----------
    start_date, end_date : str
        Yahoo download window (end_date=None fetches up to today).

    Returns
    -------
    FxRatesData
        ready to use; __post_init__ forward-fills spots and aligns rates to the spot calendar.
    """
    fx_tickers = list(_DIRECT_USD_PER_LOCAL.values()) + list(_INVERTED_LOCAL_PER_USD.values())
    raw = yf.download(fx_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']

    fx_spots = pd.DataFrame(index=raw.index)
    fx_spots['USD'] = 1.0
    for ccy, ticker in _DIRECT_USD_PER_LOCAL.items():
        fx_spots[ccy] = raw[ticker]                 # already USD per local
    for ccy, ticker in _INVERTED_LOCAL_PER_USD.items():
        fx_spots[ccy] = 1.0 / raw[ticker]           # invert local-per-USD -> USD per local
    fx_spots = fx_spots.ffill().dropna(how='all')
    fx_spots['GBp'] = 0.01 * fx_spots['GBP']        # pence -> pounds, per the container contract
    fx_spots['XAU'] = 1.0                           # gold quoted in USD; legacy 1.0 mapping

    irx = yf.download('^IRX', start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    if isinstance(irx, pd.DataFrame):
        irx = irx.iloc[:, 0]
    usd_rate = (irx / 100.0).reindex(fx_spots.index).ffill()
    domestic_rates = _build_illustrative_domestic_rates(usd_rate=usd_rate, index=fx_spots.index)

    return FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)


class LocalTests(Enum):
    SHOW_FX_DATA = 1
    CROSS_RATES = 2
    FX_TOTAL_RETURN_NAVS = 3
    CARRY_FORWARD_PREMIUM = 4
    CASH_NAVS = 5
    TRANSLATE_ASSET_PANEL = 6


def run_local_test(local_test: LocalTests):
    """Illustrate the FxRatesData container built from Yahoo data."""
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)

    fx_rates_data = fetch_fx_rates_data_from_yahoo(start_date='2005-12-31')

    if local_test == LocalTests.SHOW_FX_DATA:
        print("fx_spots — units of USD per 1 unit of local ccy (USD pinned to 1.0):")
        print(fx_rates_data.fx_spots.tail())
        print("\ndomestic_rates — 3M rates as decimals (USD real, others illustrative):")
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
                          for l in ('EUR', 'GBP', 'CHF', 'JPY')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=navs, ax=ax, title='FX total-return NAVs vs USD (spot + carry)')

    elif local_test == LocalTests.CARRY_FORWARD_PREMIUM:
        # annualised CIP forward premium of each ccy vs USD (per-period rate * periods-per-year)
        ppy = qis.get_annualization_factor('ME')
        premia = pd.concat([fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=l, reference_ccy='USD', freq='ME') * ppy
                            for l in ('EUR', 'GBP', 'CHF', 'JPY')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        premia.plot(ax=ax)
        ax.set_title('Annualised CIP forward premium vs USD')
        ax.axhline(0.0, color='black', lw=0.5)

    elif local_test == LocalTests.CASH_NAVS:
        # synthetic money-market NAV compounding each local short rate
        cash = pd.concat([fx_rates_data.build_local_cash_nav(local_ccy=c, freq='B')
                          for c in ('USD', 'EUR', 'CHF', 'JPY')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=cash, ax=ax, title='Money-market cash NAVs by currency')

    elif local_test == LocalTests.TRANSLATE_ASSET_PANEL:
        # translate a USD asset panel into a CHF investor's frame, unhedged vs fully hedged
        assets = yf.download(['SPY', 'TLT', 'GLD'], start='2010-12-31', auto_adjust=True,
                             progress=False)['Close'][['SPY', 'TLT', 'GLD']].dropna()
        local_ccys = pd.Series('USD', index=assets.columns)
        navs_unhedged, _ = fx_rates_data.compute_returns_in_reference_ccy(
            asset_prices=assets, hedge_ratios=pd.Series(0.0, index=assets.columns),
            local_ccys=local_ccys, reference_ccy='CHF', freq='ME')
        navs_hedged, _ = fx_rates_data.compute_returns_in_reference_ccy(
            asset_prices=assets, hedge_ratios=pd.Series(1.0, index=assets.columns),
            local_ccys=local_ccys, reference_ccy='CHF', freq='ME')
        spy = pd.concat([navs_unhedged['SPY'].rename('SPY in CHF (unhedged)'),
                         navs_hedged['SPY'].rename('SPY in CHF (hedged)')], axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=spy, ax=ax, title='USD asset (SPY) in a CHF frame: unhedged vs hedged')
        print("unhedged-in-CHF NAVs (tail):\n", navs_unhedged.tail())
        print("\nhedged-in-CHF NAVs (tail):\n", navs_hedged.tail())


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.TRANSLATE_ASSET_PANEL)

    plt.show()
