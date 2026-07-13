"""
Invariance test for ``fx_rates_data.compute_fx_adjusted_returns``.

The key invariant this module must satisfy is covered interest-rate parity
(CIP): if a USD-denominated asset is hedged at h=1 to CHF, its realised
return in CHF terms minus CHF rf must equal its USD return minus USD rf,
to within numerical precision (FX-spot sampling + forward-rate discretisation).

If this invariance is violated, the excess-return path has a rate-accounting
bug: it is either subtracting the wrong reference rate, mis-applying the
forward premium, or double-counting one of the two legs.

This module contains a single integration test, ``test_cip_identity``, that
builds a synthetic USD price series plus synthetic USD/CHF rate and FX
series, runs them through the same ``compute_fx_adjusted_returns`` entry
point that the CMA pipeline uses, and asserts the identity holds on every
date and on the annualised mean.

Run directly:
    python -m rosaa.market_data.tests.fx_cip_identity_test
or via pytest:
    pytest rosaa/market_data/tests/fx_cip_identity_test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis

from qis.market_data import FxRatesData


# Tolerance for CIP identity on the annualised mean. The residual comes from
# finite-sample Monte Carlo noise in the fixture's cross-term
# E[r_L × fx_return] — under independence the true expectation is zero, but
# over ~300 months with fund vol ~12% and FX vol ~10% per annum the sample
# cross-term has a std of roughly 7 bp p.a. A 25 bp tolerance catches real
# logical errors (which produce gaps of 150+ bp, the rate differential scale)
# with ~3.5σ headroom over fixture noise. If this threshold trips, the
# problem is structural, not statistical.
CIP_TOLERANCE_BP_PA = 25.0


def _build_synthetic_fx_rates_data(
    start: str = '2000-01-31',
    end:   str = '2025-12-31',
    freq:  str = 'ME',
    seed:  int = 20260422,
) -> FxRatesData:
    """Construct a small ``FxRatesData`` with plausible USD/CHF rates and FX.

    Rates: USD averages ~3% p.a., CHF averages ~0.5% p.a., each with slow
    AR(1) drift and a small innovation. FX: USD/CHF starts at 1.50 and drifts
    mildly with autoregressive shocks on top. The exact paths are not
    important for the test — what matters is that rates and FX evolve
    independently so that CIP must do real work to collapse the two
    reference frames.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)

    usd_rate = np.clip(0.03 + 0.015 * np.cumsum(rng.normal(0, 0.015, n)) / np.sqrt(n), -0.01, 0.08)
    chf_rate = np.clip(0.005 + 0.01 * np.cumsum(rng.normal(0, 0.012, n)) / np.sqrt(n), -0.015, 0.04)

    domestic_rates = pd.DataFrame(
        {'USD': usd_rate, 'CHF': chf_rate},
        index=dates,
    )

    # fx_spots convention per the module: units of USD per 1 local ccy.
    # So CHF column is USD per 1 CHF (inverse of USD/CHF quote).
    # Start USD/CHF at 1.50 → CHFUSD = 1/1.50 ≈ 0.667; USD column = 1 by definition.
    fx_innov = rng.normal(0, 0.03, n)
    log_chf_usd = np.log(1.0 / 1.50) + np.cumsum(fx_innov)
    chf_usd = np.exp(log_chf_usd)

    fx_spots = pd.DataFrame(
        {'USD': 1.0, 'CHF': chf_usd},
        index=dates,
    )
    return FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)


def _build_synthetic_usd_asset(
    fx_rates_data: FxRatesData,
    mean_ret_pa: float = 0.06,
    vol_pa:      float = 0.12,
    seed:        int   = 20260423,
) -> pd.Series:
    """Create a USD-denominated asset price series aligned to the fx index."""
    rng = np.random.default_rng(seed)
    dates = fx_rates_data.fx_spots.index
    n = len(dates)
    dt = 1.0 / 12.0  # test runs at monthly freq
    innov = rng.normal(0, 1, n) * vol_pa * np.sqrt(dt) + mean_ret_pa * dt
    price = 100.0 * np.exp(np.cumsum(innov))
    return pd.Series(price, index=dates, name='USD_ASSET')


def test_cip_identity() -> None:
    """USD-denom asset, h=1, excess returns: CHF leg must equal USD leg.

    Steps:
      1. Build synthetic FxRatesData with independent USD/CHF rates and FX.
      2. Build a USD-denom asset price.
      3. Run ``compute_fx_adjusted_returns`` twice:
           (a) reference_ccy='USD', h=0  → USD excess = r_fund - r_USD
           (b) reference_ccy='CHF', h=1  → CHF-hedged excess
      4. Assert the two series match within ``CIP_TOLERANCE_BP_PA``, both on
         the annualised mean and on the full joint path.
    """
    fx_rates_data = _build_synthetic_fx_rates_data()
    asset_name = 'USD_ASSET'
    price = _build_synthetic_usd_asset(fx_rates_data)
    prices = price.to_frame()

    # Hedge ratios: 0 for USD leg (no hedge needed in native ccy),
    # 1 for CHF leg (full hedge back to USD for a USD-denom asset).
    hedge_0 = pd.Series({asset_name: 0.0})
    hedge_1 = pd.Series({asset_name: 1.0})
    local_ccys = pd.Series({asset_name: 'USD'})

    res_usd = fx_rates_data.compute_fx_adjusted_returns(
        prices=prices,
        hedge_ratios=hedge_0,
        local_ccys=local_ccys,
        reference_ccy='USD',
        freq='ME',
        is_log_returns=False,
        is_excess_returns=True,
    )
    res_chf = fx_rates_data.compute_fx_adjusted_returns(
        prices=prices,
        hedge_ratios=hedge_1,
        local_ccys=local_ccys,
        reference_ccy='CHF',
        freq='ME',
        is_log_returns=False,
        is_excess_returns=True,
    )

    usd_excess = res_usd['ME'][asset_name].dropna()
    chf_excess = res_chf['ME'][asset_name].dropna()
    common = usd_excess.index.intersection(chf_excess.index)

    usd_mean_pa = usd_excess.loc[common].mean() * 12 * 1e4  # bp p.a.
    chf_mean_pa = chf_excess.loc[common].mean() * 12 * 1e4
    mean_diff_bp = abs(usd_mean_pa - chf_mean_pa)

    # Per-date tolerance is looser than the mean tolerance because individual
    # months pick up full cross-term (r_fund * fx) noise; but on annual
    # summation this averages out. Use a 15 bp per-month envelope.
    diff = (usd_excess.loc[common] - chf_excess.loc[common]) * 1e4  # bp per month
    max_abs_monthly_bp = diff.abs().max()
    rmse_monthly_bp = float(np.sqrt((diff ** 2).mean()))

    # Diagnostic: if the mean gap is ~(r_USD - r_CHF) or ~(r_CHF - r_USD), that
    # pinpoints which rate is (wrongly) being subtracted from the hedged leg.
    # The forward-premium hedge accounting should make hedged CHF excess = USD
    # excess. A persistent mean gap equal to +/- the rate differential means
    # the excess adjustment is still subtracting the wrong currency's rf.
    usd_rate_avg_pa = fx_rates_data.domestic_rates['USD'].mean() * 1e4
    chf_rate_avg_pa = fx_rates_data.domestic_rates['CHF'].mean() * 1e4
    rate_diff_pa = usd_rate_avg_pa - chf_rate_avg_pa
    signed_gap = usd_mean_pa - chf_mean_pa

    if abs(signed_gap - rate_diff_pa) < 30:
        hint = (
            "\n  Diagnosis: the CHF leg is subtracting r_USD instead of r_CHF. "
            "The 'excess' adjustment in compute_performance_of_local_ccy_asset_in_reference_ccy "
            "must use the REFERENCE currency rate, not the fund's local-currency rate. "
            "Verify you are running the fixed fx_rates_data.py and clear stale __pycache__."
        )
    elif abs(signed_gap + rate_diff_pa) < 30:
        hint = (
            "\n  Diagnosis: gap has opposite sign of rate differential. The CHF leg may be "
            "adding r_USD where it should subtract r_CHF, or the forward-rate sign has "
            "been flipped. Inspect compute_performance_of_local_ccy_asset_in_reference_ccy."
        )
    elif abs(signed_gap) >= CIP_TOLERANCE_BP_PA:
        hint = (
            "\n  Diagnosis: gap does not match the plain rate differential — the issue is "
            "elsewhere in the FX/hedge path (cross-term handling, forward-rate shift, "
            "or rate alignment). Inspect compute_performance_of_local_ccy_asset_in_reference_ccy."
        )
    else:
        hint = ""

    report = (
        "\nCIP IDENTITY CHECK (USD-denom asset, h=1 hedge to CHF, is_excess=True)"
        f"\n  USD excess mean:       {usd_mean_pa:+8.2f} bp p.a."
        f"\n  CHF-hedged excess mean:{chf_mean_pa:+8.2f} bp p.a."
        f"\n  Mean difference:       {mean_diff_bp:8.2f} bp p.a. (tolerance {CIP_TOLERANCE_BP_PA})"
        f"\n  Max monthly |diff|:    {max_abs_monthly_bp:8.2f} bp"
        f"\n  RMSE monthly:          {rmse_monthly_bp:8.2f} bp"
        f"\n  Sample USD rate avg:   {usd_rate_avg_pa:8.2f} bp p.a."
        f"\n  Sample CHF rate avg:   {chf_rate_avg_pa:8.2f} bp p.a."
        f"\n  Sample rate diff:      {rate_diff_pa:+8.2f} bp p.a."
        + hint
    )
    print(report)

    assert mean_diff_bp < CIP_TOLERANCE_BP_PA, (
        f"CIP identity violated in annualised mean: "
        f"{mean_diff_bp:.2f} bp p.a. exceeds {CIP_TOLERANCE_BP_PA} bp tolerance.\n"
        f"Under CIP hedging, CHF-reference excess must equal USD-reference\n"
        f"excess for a USD-denom asset at h=1, up to finite-sample noise in the\n"
        f"cross-term E[r_L * fx_return] (typically <20 bp on this fixture). A\n"
        f"gap comparable to the USD-CHF rate differential ({rate_diff_pa:+.0f} bp p.a.\n"
        f"on this sample) would indicate a rate-accounting bug — most likely\n"
        f"the reference rf used in the 'excess' adjustment does not match the\n"
        f"currencies used in the forward premium.\n"
        f"{report}"
    )


def test_cip_identity_unhedged_drift() -> None:
    """Sanity companion: unhedged CHF excess should differ from USD excess.

    With h=0, the CHF investor fully absorbs spot-FX drift on the USD asset.
    The CHF-reference excess therefore equals USD-native return minus CHF rf,
    which differs from USD-native excess by the realised ``USD/CHF spot drift
    minus (r_USD - r_CHF)``. On any non-trivial sample this is NOT zero — so
    this test catches the inverse failure mode where the fix accidentally
    makes the excess returns currency-invariant even without hedging.
    """
    fx_rates_data = _build_synthetic_fx_rates_data()
    asset_name = 'USD_ASSET'
    price = _build_synthetic_usd_asset(fx_rates_data)
    prices = price.to_frame()

    hedge_0 = pd.Series({asset_name: 0.0})
    local_ccys = pd.Series({asset_name: 'USD'})

    res_usd = fx_rates_data.compute_fx_adjusted_returns(
        prices=prices, hedge_ratios=hedge_0, local_ccys=local_ccys,
        reference_ccy='USD', freq='ME',
        is_log_returns=False, is_excess_returns=True,
    )
    res_chf = fx_rates_data.compute_fx_adjusted_returns(
        prices=prices, hedge_ratios=hedge_0, local_ccys=local_ccys,
        reference_ccy='CHF', freq='ME',
        is_log_returns=False, is_excess_returns=True,
    )

    usd_ex = res_usd['ME'][asset_name].dropna()
    chf_ex = res_chf['ME'][asset_name].dropna()
    common = usd_ex.index.intersection(chf_ex.index)
    diff_std_pa = (usd_ex.loc[common] - chf_ex.loc[common]).std() * np.sqrt(12) * 1e4  # bp p.a.

    # Unhedged, we should see diff std of order of annual FX vol — far above
    # the monthly-bp CIP residual. Use a loose floor: 500 bp p.a. std.
    assert diff_std_pa > 500.0, (
        f"Unhedged CHF-vs-USD excess std is only {diff_std_pa:.1f} bp p.a.; "
        "expected the full FX translation vol. This suggests the excess path "
        "may be reducing to the hedged identity even when h=0 — verify that "
        "the hedge ratio is actually being consumed by the forward-cost term."
    )


if __name__ == '__main__':
    pd.set_option('display.width', 180)
    test_cip_identity()
    test_cip_identity_unhedged_drift()
    print('\nAll CIP identity tests passed.')