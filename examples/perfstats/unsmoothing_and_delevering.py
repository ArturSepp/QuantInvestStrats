"""
End-to-end de-levering and return-unsmoothing walkthrough on real BDC /
private-credit data: Oaktree Specialty Lending (OCSL — listed levered BDC)
vs Oaktree Global Credit Fund (GCF — unlisted unlevered analogue).

The dataset is bundled at ``examples/_helpers/data_cache.parquet``: weekly
rebased NAVs for OCSL, GCF, SPX, US HY, US Agg from 2018-11-09 to 2026-04-17.

Demonstrates:
  1. ``qis.delever_returns``        — recover asset-level returns from a
                                      levered NAV given debt/equity and
                                      financing rate.
  2. ``qis.implied_leverage``       — back out the implicit leverage of a
                                      levered vehicle vs an unlevered
                                      analogue via OLS slope.
  3. ``qis.unsmooth_returns_ar1_ewma`` — rolling EWMA AR(1) unsmoothing for
                                      appraisal-based NAV series.
  4. ``qis.unsmooth_returns_glm``   — static AR(q) Getmansky-Lo-Makarov (2004)
                                      unsmoothing with severity diagnostics.

What to look for in the output: the two raw observed series (OCSL levered,
GCF observed) sit at opposite extremes of the Sharpe range — OCSL inflated
by leverage-and-discount noise on the listed BDC, GCF deflated (or in this
sample, mildly deflated) by quarterly NAV smoothing. After de-levering OCSL
and unsmoothing GCF, the three "underlying credit asset" Sharpe estimates
land in a much tighter band, demonstrating that both adjustments move the
estimates towards a common implied risk-adjusted return.

Caveat on the GLM step: with only ~30 quarterly GCF observations, the AR(3)
fit can produce small or negative ``theta_sum``, in which case the
methodology's vol-inflation interpretation does not apply and the EWMA AR(1)
result is the one to trust. The example prints the diagnostics so the
severity is visible.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

import qis as qis
from qis import PerfStat


# Path to the bundled dataset, resolved relative to this file.
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '_helpers',
    'data_cache.parquet',
)

# OCSL leverage and financing assumptions (from public 10-Q filings, FY2024).
OCSL_LEVERAGE = 1.07     # debt / equity
OCSL_FINANCING = 0.061   # weighted average annualised debt cost

PERF_PARAMS = qis.PerfParams(freq='QE')

PERF_COLS = (PerfStat.PA_RETURN,
             PerfStat.VOL,
             PerfStat.SHARPE_RF0,
             PerfStat.MAX_DD,
             PerfStat.SKEWNESS,
             PerfStat.KURTOSIS)


def load_data() -> pd.DataFrame:
    """Load the bundled OCSL / GCF / benchmarks weekly NAV panel."""
    return pd.read_parquet(DATA_PATH)


def main() -> None:
    prices = load_data()
    print(f"Loaded {prices.shape[0]} weekly NAVs from "
          f"{prices.index.min():%Y-%m-%d} to {prices.index.max():%Y-%m-%d}")
    print(f"Columns: {list(prices.columns)}\n")

    # ── 1. Raw NAVs: levered BDC vs unlevered private-credit fund ───────
    fig1, ax = plt.subplots(1, 1, figsize=(10, 5))
    qis.plot_prices(
        prices=prices[['OCSL', 'Oaktree GCF']],
        title="Levered BDC (OCSL) vs unlevered private-credit fund (GCF)",
        ax=ax,
    )

    # ── 2. De-lever OCSL ────────────────────────────────────────────────
    # Convert weekly NAVs to weekly returns.
    weekly_returns = qis.to_returns(prices, freq='W-FRI', drop_first=True)
    ocsl_returns = weekly_returns['OCSL']

    # Apply de-levering identity:  r_asset = (r_portfolio + L * r_f) / (1 + L)
    ocsl_unlev_returns = qis.delever_returns(
        returns=ocsl_returns,
        leverage=OCSL_LEVERAGE,
        financing_rate=OCSL_FINANCING,
        periods_per_year=52,
    )
    ocsl_unlev_nav = qis.returns_to_nav(returns=ocsl_unlev_returns).rename(
        f'OCSL de-levered (L={OCSL_LEVERAGE:.2f}x, fin={OCSL_FINANCING:.1%})'
    )

    # Stack levered, de-levered, GCF for visual comparison.
    fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
    delev_panel = pd.concat([
        prices['OCSL'].reindex(ocsl_unlev_nav.index, method='ffill').rename('OCSL (levered)'),
        ocsl_unlev_nav,
        prices['Oaktree GCF'].reindex(ocsl_unlev_nav.index, method='ffill').rename('GCF (observed)'),
    ], axis=1)
    qis.plot_prices(prices=delev_panel, ax=ax,
                    title="OCSL de-levered vs GCF (rebased to 100)")

    # ── 3. Implied leverage from regression ─────────────────────────────
    # Sanity check: regress levered OCSL on unlevered GCF; slope - 1 ≈ L.
    # Use quarterly returns: GCF's weekly observations are mostly forward-filled
    # appraisals and a weekly regression is dominated by quarter-end jumps.
    q_returns = qis.to_quarterly_returns(returns=weekly_returns)
    ocsl_q_returns = q_returns['OCSL']
    gcf_q_returns = q_returns['Oaktree GCF']

    L_hat = qis.implied_leverage(
        levered_returns=ocsl_q_returns,
        unlevered_returns=gcf_q_returns,
    )
    print(f"OCSL implied leverage vs GCF (quarterly regression): {L_hat:.2f}x")
    print(f"OCSL stated leverage (filings):                      {OCSL_LEVERAGE:.2f}x\n")
    # Note: the regression L_hat is contaminated by mark-to-market vs appraisal
    # differences (OCSL is listed daily, GCF is appraised quarterly). The
    # gap relative to the filings number is informative about the smoothing.

    # ── 4. Unsmooth GCF — quarterly observations are the natural frequency ─
    # GCF's NAV is appraised quarterly. We've already resampled above; just
    # reuse the quarterly returns.

    # 4a. Rolling EWMA AR(1) unsmoothing (the qis-recommended default).
    gcf_q_returns_df = gcf_q_returns.to_frame()
    unsm_returns_ewma, betas, r2 = qis.unsmooth_returns_ar1_ewma(
        returns=gcf_q_returns_df,
        span=20,                  # ~5 years of quarterly data
        max_value_for_beta=0.75,
        min_value_for_beta=-0.25,
    )
    gcf_unsm_ewma_nav = qis.returns_to_nav(unsm_returns_ewma).iloc[:, 0].rename(
        'GCF unsmoothed — rolling EWMA AR(1)'
    )

    # 4b. Static AR(3) Getmansky-Lo-Makarov for comparison.
    gcf_unsm_glm, glm_diag = qis.unsmooth_returns_glm(
        returns=gcf_q_returns,
        ar_order=3,
        return_diagnostics=True,
    )
    gcf_unsm_glm_nav = qis.returns_to_nav(gcf_unsm_glm).rename(
        'GCF unsmoothed — static GLM AR(3)'
    )

    print("GLM AR(3) unsmoothing diagnostics for GCF:")
    print(f"  theta              = {glm_diag.theta.round(3)}")
    print(f"  theta_sum          = {glm_diag.theta_sum:.3f}")
    print(f"  vol inflation      = {glm_diag.vol_inflation_factor:.2f}x")
    print(f"  is_severe (>0.95)  = {glm_diag.is_severe}\n")

    # ── 5. Side-by-side: smoothed vs unsmoothed quarterly NAVs ──────────
    gcf_observed_q = qis.returns_to_nav(returns=gcf_q_returns).rename(
        'GCF observed (quarterly NAV)'
    )
    fig3, ax = plt.subplots(1, 1, figsize=(10, 5))
    unsm_panel = pd.concat([gcf_observed_q,
                            gcf_unsm_ewma_nav,
                            gcf_unsm_glm_nav], axis=1).dropna()
    qis.plot_prices(prices=unsm_panel,
                    title="GCF unsmoothing: observed vs AR(1) EWMA vs static GLM AR(3)",
                    ax=ax)

    # ── 6. Quarterly performance summary ────────────────────────────────
    # Resample everything to QE for an apples-to-apples comparison.
    qe = prices.asfreq('QE', method='ffill')
    summary = pd.concat([
        qe['OCSL'].rename('OCSL (levered)'),
        ocsl_unlev_nav.asfreq('QE', method='ffill').rename('OCSL de-levered'),
        qe['Oaktree GCF'].rename('GCF (observed)'),
        gcf_unsm_ewma_nav.rename('GCF unsm. EWMA AR(1)'),
        gcf_unsm_glm_nav.rename('GCF unsm. GLM AR(3)'),
        qe['SPX'], qe['US HY'], qe['US Agg'],
    ], axis=1).dropna()

    fig4, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)
    qis.plot_ra_perf_table(prices=summary,
                           perf_params=PERF_PARAMS,
                           perf_columns=PERF_COLS,
                           title="Risk-adjusted performance — quarterly returns",
                           ax=ax)

    # Print the key Sharpe comparison inline for narrative.
    perf = qis.compute_ra_perf_table(prices=summary, perf_params=PERF_PARAMS)
    sharpe_col = PerfStat.SHARPE_RF0.to_str()
    vol_col = PerfStat.VOL.to_str()
    print("Quarterly Sharpe / vol comparison:")
    for name in summary.columns:
        s = perf.loc[name, sharpe_col]
        v = perf.loc[name, vol_col]
        print(f"  {name:30s}  Sharpe = {s:+.2f}   Vol = {v:.1%}")

    plt.show()


if __name__ == '__main__':
    main()
