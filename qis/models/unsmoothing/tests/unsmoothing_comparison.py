"""
Tests comparing the two unsmoothing methods (EWMA AR(1) vs static GLM) on:
  (1) synthetic smoothed data with known ground truth
  (2) real HF-like data with structural breaks
  (3) cross-method agreement on steady-state series

Run standalone:
    python unsmoothing_comparison.py
Or with pytest:
    pytest unsmoothing_comparison.py -v
"""
import numpy as np
import pandas as pd
from enum import Enum

from qis.models.unsmoothing.ar_lag import (unsmooth_returns_ar1_ewma,
                                           unsmooth_returns_glm)


class ComparisonTests(Enum):
    SYNTHETIC_CONSTANT_SMOOTHING = 1        # both methods should recover true vol
    SYNTHETIC_REGIME_CHANGE = 2              # EWMA should beat GLM under structural break
    REAL_DATA_ON_SMOOTHED_HF = 3             # apply both to appraisal-like series
    CROSS_METHOD_AGREEMENT_LIQUID = 4        # on liquid series, both should do ~nothing
    VOL_INFLATION_COMPARISON = 5             # compare implied smoothing severity


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def make_true_returns(n: int = 500,
                      mu_ann: float = 0.08,
                      vol_ann: float = 0.15,
                      periods_per_year: int = 52,
                      seed: int = 42) -> pd.Series:
    """Generate synthetic 'true' (unsmoothed) weekly returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2015-01-02', periods=n, freq='W-FRI')
    vals = rng.normal(loc=mu_ann / periods_per_year,
                      scale=vol_ann / np.sqrt(periods_per_year),
                      size=n)
    return pd.Series(vals, index=idx, name='true')


def apply_constant_smoothing(true_returns: pd.Series,
                             theta: np.ndarray = np.array([0.6, 0.3, 0.1])
                             ) -> pd.Series:
    """Apply a constant smoothing kernel to produce observed returns.

    r_obs_t = theta_0 * r_true_t + theta_1 * r_true_{t-1} + theta_2 * r_true_{t-2}
    """
    assert abs(theta.sum() - 1.0) < 1e-10, "theta must sum to 1"
    obs = true_returns.copy()
    vals = true_returns.values
    q = len(theta)
    for t in range(q - 1, len(vals)):
        obs.iloc[t] = sum(theta[i] * vals[t - i] for i in range(q))
    return obs


def apply_regime_smoothing(true_returns: pd.Series,
                           breakpoint_frac: float = 0.5) -> pd.Series:
    """Apply regime-changing smoothing: heavy smoothing early, light smoothing late.

    Simulates a fund that improved its valuation discipline mid-sample.
    """
    obs = true_returns.copy()
    vals = true_returns.values
    bp = int(len(vals) * breakpoint_frac)
    # heavy smoothing for first half: theta = [0.4, 0.35, 0.25]
    theta_early = np.array([0.4, 0.35, 0.25])
    # light smoothing for second half: theta = [0.85, 0.10, 0.05]
    theta_late = np.array([0.85, 0.10, 0.05])
    for t in range(2, len(vals)):
        theta = theta_early if t < bp else theta_late
        obs.iloc[t] = sum(theta[i] * vals[t - i] for i in range(3))
    return obs


# =============================================================================
# METRIC HELPERS
# =============================================================================

def compute_metrics(returns: pd.Series, label: str) -> dict:
    """Compute summary statistics for a return series."""
    clean = returns.dropna()
    return {
        'label': label,
        'mean_ann': clean.mean() * 52,
        'vol_ann': clean.std() * np.sqrt(52),
        'sharpe': (clean.mean() / clean.std()) * np.sqrt(52) if clean.std() > 0 else np.nan,
        'ar1_corr': clean.autocorr(lag=1),
        'n_obs': len(clean),
    }


def print_metrics_table(metrics_list: list):
    """Print a comparison table of metrics."""
    df = pd.DataFrame(metrics_list).set_index('label')
    print(df.round(4).to_string())


# =============================================================================
# TEST IMPLEMENTATIONS
# =============================================================================

def test_synthetic_constant_smoothing():
    """Both methods should recover true-series volatility when smoothing is constant.

    Under static smoothing, the GLM method is the correctly-specified model, so it
    should achieve perfect recovery. The EWMA AR(1) method is slightly mis-specified
    (only uses 1 lag instead of 3) but should still get close.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Synthetic data with constant 3-lag smoothing")
    print("=" * 70)

    true_ret = make_true_returns(n=1000, seed=1)
    obs_ret = apply_constant_smoothing(true_ret, theta=np.array([0.6, 0.3, 0.1]))

    # apply both methods
    glm_unsmoothed, glm_diag = unsmooth_returns_glm(obs_ret, ar_order=3, return_diagnostics=True)

    # EWMA requires DataFrame input — wrap Series in single-column DataFrame
    obs_df = obs_ret.to_frame()
    ewma_unsmoothed_df, betas, r2 = unsmooth_returns_ar1_ewma(
        returns=obs_df, span=40, warmup_period=20,
    )
    ewma_unsmoothed = ewma_unsmoothed_df.iloc[:, 0]

    metrics = [
        compute_metrics(true_ret, 'true'),
        compute_metrics(obs_ret, 'observed (smoothed)'),
        compute_metrics(glm_unsmoothed, 'GLM unsmoothed'),
        compute_metrics(ewma_unsmoothed, 'EWMA AR(1) unsmoothed'),
    ]
    print_metrics_table(metrics)

    print(f"\nGLM diagnostics:")
    print(f"  theta estimated:   {glm_diag.theta.round(4)}")
    print(f"  theta true:        [0.6 0.3 0.1]")
    print(f"  theta_sum:         {glm_diag.theta_sum:.4f}")
    print(f"  vol inflation:     {glm_diag.vol_inflation_factor:.2f}x")

    # key check: both unsmoothed vols should be closer to true vol than observed
    true_vol = true_ret.std()
    obs_vol = obs_ret.std()
    glm_vol = glm_unsmoothed.dropna().std()
    ewma_vol = ewma_unsmoothed.dropna().std()
    print(f"\nVol recovery (true = {true_vol:.5f}):")
    print(f"  observed:  {obs_vol:.5f}  (deficit: {(true_vol - obs_vol)/true_vol:.1%})")
    print(f"  GLM:       {glm_vol:.5f}  (error:   {(glm_vol - true_vol)/true_vol:+.1%})")
    print(f"  EWMA:      {ewma_vol:.5f}  (error:   {(ewma_vol - true_vol)/true_vol:+.1%})")


def test_synthetic_regime_change():
    """EWMA should outperform GLM when smoothing weights change mid-sample.

    The static GLM method must compromise between the two regimes, producing a
    single theta that's wrong in both. The EWMA method adapts locally.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Synthetic data with mid-sample smoothing regime change")
    print("=" * 70)

    true_ret = make_true_returns(n=1000, seed=2)
    obs_ret = apply_regime_smoothing(true_ret, breakpoint_frac=0.5)

    glm_unsmoothed = unsmooth_returns_glm(obs_ret, ar_order=3)
    ewma_unsmoothed_df, betas, r2 = unsmooth_returns_ar1_ewma(
        returns=obs_ret.to_frame(), span=40, warmup_period=20,
    )
    ewma_unsmoothed = ewma_unsmoothed_df.iloc[:, 0]

    # compare per-regime vol recovery
    n = len(true_ret)
    first_half = slice(None, n // 2)
    second_half = slice(n // 2, None)

    print("\nVol recovery by regime:")
    for label, sl in [('first half (heavy smoothing)', first_half),
                      ('second half (light smoothing)', second_half)]:
        true_vol = true_ret.iloc[sl].std()
        obs_vol = obs_ret.iloc[sl].std()
        glm_vol = glm_unsmoothed.iloc[sl].dropna().std()
        ewma_vol = ewma_unsmoothed.iloc[sl].dropna().std()
        print(f"\n{label}:")
        print(f"  true vol:  {true_vol:.5f}")
        print(f"  observed:  {obs_vol:.5f}  (gap: {(true_vol-obs_vol)/true_vol:+.1%})")
        print(f"  GLM:       {glm_vol:.5f}  (error: {(glm_vol-true_vol)/true_vol:+.1%})")
        print(f"  EWMA:      {ewma_vol:.5f}  (error: {(ewma_vol-true_vol)/true_vol:+.1%})")

    # show how EWMA beta tracks the regime change
    beta_series = betas.iloc[:, 0]
    print(f"\nEWMA beta evolution:")
    print(f"  first-half mean beta:  {beta_series.iloc[:n//2].mean():.4f}")
    print(f"  second-half mean beta: {beta_series.iloc[n//2:].mean():.4f}")
    print(f"  (first-half theta should give higher AR(1) beta; EWMA should show this)")


def test_cross_method_agreement_liquid():
    """On liquid (already unsmoothed) data, both methods should barely change the series.

    A liquid return series has near-zero AR(1) autocorrelation, so:
      - GLM: theta ≈ 0 → r_true ≈ r_obs
      - EWMA: beta ≈ 0 → r_true ≈ r_obs

    Both methods should produce unsmoothed series that are essentially
    identical to the input.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Liquid (uncorrelated) data — both methods should be near-identity")
    print("=" * 70)

    liquid_ret = make_true_returns(n=1000, seed=3)

    glm_unsmoothed = unsmooth_returns_glm(liquid_ret, ar_order=3)
    ewma_unsmoothed_df, _, _ = unsmooth_returns_ar1_ewma(
        returns=liquid_ret.to_frame(), span=40, warmup_period=20,
    )
    ewma_unsmoothed = ewma_unsmoothed_df.iloc[:, 0]

    # after the warmup, unsmoothed should closely match original
    aligned = pd.concat([liquid_ret, glm_unsmoothed.rename('glm'),
                         ewma_unsmoothed.rename('ewma')], axis=1).dropna()
    corr_glm = aligned['true'].corr(aligned['glm'])
    corr_ewma = aligned['true'].corr(aligned['ewma'])
    vol_ratio_glm = aligned['glm'].std() / aligned['true'].std()
    vol_ratio_ewma = aligned['ewma'].std() / aligned['true'].std()

    print(f"\nGLM vs input:    corr = {corr_glm:.4f}, vol ratio = {vol_ratio_glm:.4f}")
    print(f"EWMA vs input:   corr = {corr_ewma:.4f}, vol ratio = {vol_ratio_ewma:.4f}")
    print(f"\nBoth should be close to (corr=1.0, vol_ratio=1.0) — indicating no over-correction.")


def test_real_data_smoothed_hf_proxy():
    """Apply both methods to a constructed 'HF-like' proxy.

    We construct a fake HF series by applying realistic smoothing to a public
    equity index, then compare how each method reverses the smoothing.
    """
    print("\n" + "=" * 70)
    print("TEST 4: HF-like proxy (smoothed equity returns)")
    print("=" * 70)

    # simulate an equity-like base series
    base_ret = make_true_returns(n=500, mu_ann=0.10, vol_ann=0.18, seed=4)

    # apply moderate HF-like smoothing
    theta_hf = np.array([0.5, 0.3, 0.15, 0.05])
    assert abs(theta_hf.sum() - 1.0) < 1e-10
    obs_ret = base_ret.copy()
    for t in range(3, len(base_ret)):
        obs_ret.iloc[t] = sum(theta_hf[i] * base_ret.iloc[t - i] for i in range(4))

    glm_unsmoothed, glm_diag = unsmooth_returns_glm(obs_ret, ar_order=4,
                                                    return_diagnostics=True)
    ewma_unsmoothed_df, _, _ = unsmooth_returns_ar1_ewma(
        returns=obs_ret.to_frame(), span=40, warmup_period=20,
    )
    ewma_unsmoothed = ewma_unsmoothed_df.iloc[:, 0]

    metrics = [
        compute_metrics(base_ret, 'base (true)'),
        compute_metrics(obs_ret, 'smoothed observed'),
        compute_metrics(glm_unsmoothed, 'GLM unsmoothed'),
        compute_metrics(ewma_unsmoothed, 'EWMA unsmoothed'),
    ]
    print_metrics_table(metrics)
    print(f"\nGLM detected smoothing severity (vol inflation): {glm_diag.vol_inflation_factor:.2f}x")


def test_vol_inflation_comparison():
    """Compare how each method implies smoothing severity across several series."""
    print("\n" + "=" * 70)
    print("TEST 5: Smoothing severity across heterogeneous series")
    print("=" * 70)

    scenarios = {
        'liquid':         np.array([1.0]),                        # no smoothing
        'mild':           np.array([0.85, 0.15]),                 # light
        'moderate':       np.array([0.6, 0.3, 0.1]),              # moderate
        'heavy':          np.array([0.4, 0.3, 0.2, 0.1]),         # heavy
    }

    print(f"\n{'Scenario':<12} | {'GLM vol inflation':<20} | {'EWMA mean beta':<20}")
    print("-" * 60)

    for name, theta in scenarios.items():
        true_ret = make_true_returns(n=1000, seed=hash(name) % 2**32)
        obs_ret = apply_constant_smoothing(true_ret, theta=theta) if len(theta) > 1 else true_ret

        glm_unsmoothed, glm_diag = unsmooth_returns_glm(
            obs_ret, ar_order=max(len(theta), 2), return_diagnostics=True,
        )
        _, betas, _ = unsmooth_returns_ar1_ewma(
            returns=obs_ret.to_frame(), span=40, warmup_period=20,
        )
        mean_beta = betas.iloc[:, 0].dropna().mean()

        print(f"{name:<12} | {glm_diag.vol_inflation_factor:<20.3f} | {mean_beta:<20.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_comparison_test(test: ComparisonTests):
    """Dispatch to specific test."""
    if test == ComparisonTests.SYNTHETIC_CONSTANT_SMOOTHING:
        test_synthetic_constant_smoothing()
    elif test == ComparisonTests.SYNTHETIC_REGIME_CHANGE:
        test_synthetic_regime_change()
    elif test == ComparisonTests.CROSS_METHOD_AGREEMENT_LIQUID:
        test_cross_method_agreement_liquid()
    elif test == ComparisonTests.REAL_DATA_ON_SMOOTHED_HF:
        test_real_data_smoothed_hf_proxy()
    elif test == ComparisonTests.VOL_INFLATION_COMPARISON:
        test_vol_inflation_comparison()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Run all comparison tests
    for test in ComparisonTests:
        run_comparison_test(test)