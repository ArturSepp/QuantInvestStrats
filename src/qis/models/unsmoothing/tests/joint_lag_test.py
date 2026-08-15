"""
Unit tests for adjust_returns_with_joint_unsmoothing (qis.models.unsmoothing.joint_lag).

Self-contained on synthetic appraisal-smoothed series, no data fixtures. Mirrors
the run_local_test(case) idiom used for the other unsmoothing engines.

Cases:
    COEFFICIENT_RECOVERY  recover an imposed phi_1, beta_1; mean preserved; vol lifts
    INVERSION_CONSISTENCY the returned series equals the closed-form inversion
    NO_SMOOTHING          on data with only a contemporaneous beta, phi_1, beta_1 ~ 0
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple
# qis / project
from qis.models.unsmoothing.joint_lag import adjust_returns_with_joint_unsmoothing


def _make_smoothed_series(phi_1: float,
                          beta_1: float,
                          mu: float,
                          n: int,
                          seed: int,
                          sigma_f: float = 0.08,
                          sigma_e: float = 0.04,
                          ) -> Tuple[pd.DataFrame, pd.Series]:
    """generate one column from r_t = mu(1-phi) + phi_1 r_{t-1} + beta_1 F_{t-1} + e_t."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2000-03-31', periods=n, freq='QE')
    f = pd.Series(rng.normal(0.0, sigma_f, n), index=idx, name='Equity')
    eps = rng.normal(0.0, sigma_e, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = mu * (1.0 - phi_1) + phi_1 * r[t - 1] + beta_1 * f.iloc[t - 1] + eps[t]
    return pd.Series(r, index=idx, name='asset').to_frame(), f


class UnitTests(Enum):
    COEFFICIENT_RECOVERY = 1
    INVERSION_CONSISTENCY = 2
    NO_SMOOTHING = 3


def run_local_test(case: UnitTests) -> None:

    if case == UnitTests.COEFFICIENT_RECOVERY:
        phi_1, beta_1, mu = 0.40, 0.50, 0.02
        returns, factor = _make_smoothed_series(phi_1, beta_1, mu, n=800, seed=1)
        corrected, phi1, beta1 = adjust_returns_with_joint_unsmoothing(
            returns, factor, span=120, return_diagnostics=True)

        phi_hat = float(phi1['asset'].iloc[-200:].mean())
        beta_hat = float(beta1['asset'].iloc[-200:].mean())
        mean_shift = float(abs(corrected['asset'].mean() - returns['asset'].mean()))
        vol_ratio = float(corrected['asset'].std() / returns['asset'].std())

        print(f"[recovery] phi_1 {phi_1:.2f}->{phi_hat:.2f}  beta_1 {beta_1:.2f}->{beta_hat:.2f}  "
              f"mean shift {mean_shift:.4f}  vol x{vol_ratio:.2f}")
        assert abs(phi_hat - phi_1) < 0.12, f"phi_1 not recovered: {phi_hat:.3f}"
        assert abs(beta_hat - beta_1) < 0.15, f"beta_1 not recovered: {beta_hat:.3f}"
        assert mean_shift < 0.002, f"mean not preserved: shift {mean_shift:.4f}"
        assert vol_ratio > 1.15, f"vol did not lift on real smoothing: x{vol_ratio:.3f}"

    elif case == UnitTests.INVERSION_CONSISTENCY:
        # the implementation must equal the documented mean-preserving inversion
        max_ar = 0.9
        returns, factor = _make_smoothed_series(0.45, 0.50, 0.015, n=500, seed=7)
        corrected, phi1, beta1 = adjust_returns_with_joint_unsmoothing(
            returns, factor, span=80, max_ar_coeff=max_ar, return_diagnostics=True)

        phi_l = phi1.shift(1)
        beta_l = beta1.shift(1)
        f_diff = factor - factor.shift(1)
        one_minus_phi = (1.0 - phi_l).clip(lower=1.0 - max_ar)
        rebuilt = ((returns - phi_l.multiply(returns.shift(1))
                    + beta_l.multiply(f_diff, axis=0))
                   .divide(one_minus_phi).where(returns.notna()))
        max_abs = float((corrected - rebuilt).abs().max().max())
        print(f"[inversion] max|engine - closed_form| = {max_abs:.2e}")
        assert max_abs < 1e-12, f"engine deviates from the inversion formula: {max_abs:.2e}"

    elif case == UnitTests.NO_SMOOTHING:
        # only a contemporaneous beta, no own-lag and no lagged factor -> phi_1, beta_1 ~ 0
        rng = np.random.default_rng(3)
        n = 600
        idx = pd.date_range('2000-03-31', periods=n, freq='QE')
        factor = pd.Series(rng.normal(0.0, 0.08, n), index=idx, name='Equity')
        returns = (0.6 * factor + pd.Series(rng.normal(0.01, 0.05, n), index=idx)).rename('asset').to_frame()
        corrected, phi1, beta1 = adjust_returns_with_joint_unsmoothing(
            returns, factor, span=120, return_diagnostics=True)

        phi_hat = float(phi1['asset'].iloc[-200:].mean())
        beta_hat = float(beta1['asset'].iloc[-200:].mean())
        vol_ratio = float(corrected['asset'].std() / returns['asset'].std())
        print(f"[no-smoothing] phi_1~{phi_hat:+.3f}  beta_1~{beta_hat:+.3f}  vol x{vol_ratio:.2f} "
              f"(residual lift is variance injection from estimation noise)")
        assert abs(phi_hat) < 0.15, f"spurious own-lag on unsmoothed data: {phi_hat:.3f}"
        assert abs(beta_hat) < 0.15, f"spurious lagged beta on unsmoothed data: {beta_hat:.3f}"

    else:
        raise ValueError(f"unknown case: {case!r}")

    print(f"{case.name}: PASS")


if __name__ == '__main__':

    case = UnitTests.COEFFICIENT_RECOVERY

    is_run_all_tests = True
    if is_run_all_tests:
        for case in UnitTests:
            run_local_test(case=case)
    else:
        run_local_test(case=case)
