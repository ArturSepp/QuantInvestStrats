"""
Illustrate the covered-interest-parity (CIP) invariant of FxRatesData on real Yahoo data.

This is the data-driven companion to fx_cip_identity_test. The invariant: for a USD-denominated
asset, the CHF-reference EXCESS return at a full hedge (h=1) must equal the USD-reference excess
return (h=0), up to forward-rate discretisation — the hedge converts the asset's USD rf into the
CHF rf exactly under CIP, so the two excess paths coincide. Unhedged (h=0 viewed in CHF), the CHF
investor instead absorbs the full USD/CHF spot drift, and the two paths diverge by the FX vol.

The FX spots here are real Yahoo data; domestic rates are USD-real plus stylised differentials (see
fx_rates_data_yahoo_example) — the identity is a property of the container's rate accounting and
holds regardless of whether the rates are real or stylised, exactly as the unit test checks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis
from enum import Enum

from qis.examples.market_data.fx_rates_data_yahoo_example import fetch_fx_rates_data_from_yahoo

ASSET = 'SPY'            # a USD-denominated asset
REFERENCE_CCY = 'CHF'    # non-USD investor frame
FREQ = 'ME'


def _excess_return(fx_rates_data, prices, hedge_ratio: float, reference_ccy: str) -> pd.Series:
    """Per-period reference-ccy EXCESS return of a USD asset at a given hedge ratio."""
    res = fx_rates_data.compute_fx_adjusted_returns(
        prices=prices,
        hedge_ratios=pd.Series({ASSET: hedge_ratio}),
        local_ccys=pd.Series({ASSET: 'USD'}),
        reference_ccy=reference_ccy,
        freq=FREQ,
        is_log_returns=False,
        is_excess_returns=True)
    return res[FREQ][ASSET].dropna()


class LocalTests(Enum):
    CIP_HEDGED_IDENTITY = 1     # CHF-hedged excess == USD excess (CIP holds)
    UNHEDGED_DIVERGENCE = 2     # CHF-unhedged excess diverges by FX vol


def run_local_test(local_test: LocalTests):
    pd.set_option('display.width', 180)

    fx_rates_data = fetch_fx_rates_data_from_yahoo(start_date='2005-12-31')
    prices = yf.download([ASSET], start='2005-12-31', auto_adjust=True, progress=False)['Close'][[ASSET]].dropna()

    usd_excess = _excess_return(fx_rates_data, prices, hedge_ratio=0.0, reference_ccy='USD')

    if local_test == LocalTests.CIP_HEDGED_IDENTITY:
        chf_hedged_excess = _excess_return(fx_rates_data, prices, hedge_ratio=1.0, reference_ccy=REFERENCE_CCY)
        common = usd_excess.index.intersection(chf_hedged_excess.index)
        gap_bp = abs(usd_excess.loc[common].mean() - chf_hedged_excess.loc[common].mean()) * 12 * 1e4
        rate_diff_bp = (fx_rates_data.domestic_rates['USD'].mean()
                        - fx_rates_data.domestic_rates[REFERENCE_CCY].mean()) * 1e4
        print("CIP IDENTITY (USD asset, h=1 hedge to CHF, excess returns):")
        print(f"  USD excess mean        : {usd_excess.loc[common].mean()*12*1e4:+8.2f} bp p.a.")
        print(f"  CHF-hedged excess mean : {chf_hedged_excess.loc[common].mean()*12*1e4:+8.2f} bp p.a.")
        print(f"  Mean gap               : {gap_bp:8.2f} bp p.a.  (a gap ~ the {rate_diff_bp:+.0f} bp")
        print(f"                            USD-CHF rate differential would signal a rate-accounting bug)")
        navs = pd.concat([qis.returns_to_nav(usd_excess.loc[common]).rename(f"{ASSET} USD excess (h=0)"),
                          qis.returns_to_nav(chf_hedged_excess.loc[common]).rename(f"{ASSET} CHF-hedged excess (h=1)")],
                         axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=navs, ax=ax, title='CIP: CHF-hedged excess overlays USD excess')

    elif local_test == LocalTests.UNHEDGED_DIVERGENCE:
        chf_unhedged_excess = _excess_return(fx_rates_data, prices, hedge_ratio=0.0, reference_ccy=REFERENCE_CCY)
        common = usd_excess.index.intersection(chf_unhedged_excess.index)
        diff_std_bp = (usd_excess.loc[common] - chf_unhedged_excess.loc[common]).std() * np.sqrt(12) * 1e4
        print("UNHEDGED DIVERGENCE (USD asset, h=0 in both frames, excess returns):")
        print(f"  std of (USD excess - CHF-unhedged excess): {diff_std_bp:8.1f} bp p.a.")
        print(f"  -> non-trivial (≈ USD/CHF FX vol): the CHF investor carries the spot exposure.")
        navs = pd.concat([qis.returns_to_nav(usd_excess.loc[common]).rename(f"{ASSET} USD excess (h=0)"),
                          qis.returns_to_nav(chf_unhedged_excess.loc[common]).rename(f"{ASSET} CHF-unhedged excess (h=0)")],
                         axis=1)
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        qis.plot_prices(prices=navs, ax=ax, title='Unhedged: CHF excess diverges from USD excess by FX')


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CIP_HEDGED_IDENTITY)

    plt.show()
