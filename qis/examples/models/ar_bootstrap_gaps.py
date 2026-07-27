"""
what a gap does to an AR(1) fitted for the residual bootstrap.

`qis.bootstrap_ar_process` fits an AR(1) per column, keeps the residuals, and resamples
them. The fit is the part a missing observation corrupts, and it corrupts it quietly: the
run completes and returns a plausible number.

Two ways to handle a gap when forming the lag pairs (y_t, y_{t-1}):

  drop the missing observations first
      the observations either side of a gap become adjacent, so a step of two or three
      periods is fitted as a step of one. An AR(1) at spacing k has persistence theta^k,
      so the estimate is dragged towards zero.

  drop the pairs that straddle the gap
      each missing observation removes the pair ending at it and the pair starting from
      it. Every surviving pair is a genuine one-period step.

`qis` does the second. This example measures the difference, and reports the two
quantities that make the choice visible: the deviation from the gap-free estimate under
each rule, and the number of residual rows the draw is taken over.

Runner for the numbers quoted in qis/models/bootstrap/tests/test_bootstrap_ar.py.
"""

# packages
import numpy as np
import pandas as pd
from enum import Enum
# qis
import qis as qis
from qis.models.bootstrap.bootstrap_numba import compute_ar_residuals

THETA = 0.7  # true AR(1) persistence
NUM_OBSERVATIONS = 3000
MISSING_FRACTION = 0.30  # share of observations blanked
NUM_PATTERNS = 20  # independent gap patterns
SEED = 3


def generate_ar1(theta: float,
                 n: int = NUM_OBSERVATIONS,
                 sigma: float = 0.02,  # innovation standard deviation
                 seed: int = SEED
                 ) -> pd.Series:
    """draw a seeded AR(1) on a quarter-end index."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = theta * values[t - 1] + rng.normal(0.0, sigma)
    # unit='s' because NUM_OBSERVATIONS is 3000 quarters, which is 750 years: a nanosecond
    # DatetimeIndex tops out at 2262 and pandas 2.x raises OutOfBoundsDatetime there
    return pd.Series(values, name='a',
                     index=pd.date_range('1950-03-31', periods=n, freq='QE', unit='s'))


def estimate_by_collapsing_gaps(series: pd.Series) -> float:
    """fit the AR(1) after dropping the missing observations.

    This is the rule qis used before 5.1.1 and it is the subject of the example, not a
    defect in the code below. Written with numpy rather than a library call so that the
    only difference from the qis path is the treatment of the gap.
    """
    values = series.dropna().to_numpy()
    target, regressor = values[1:], values[:-1]
    return float(np.cov(target, regressor, ddof=1)[0, 1] / np.var(regressor, ddof=1))


class LocalTest(Enum):
    """the three things worth seeing."""

    GAP_BIAS = 1  # deviation from the gap-free estimate under each rule
    RESIDUAL_ROWS = 2  # how many rows the draw is taken over
    END_TO_END = 3  # the bootstrap runs on gapped data and returns finite paths


def run_local_test(local_test: LocalTest) -> None:
    """run one case."""

    if local_test is LocalTest.GAP_BIAS:
        series = generate_ar1(THETA)
        _, _, beta_full = compute_ar_residuals(series)
        print(f"gap-free estimate: beta = {beta_full[0]:.4f}, true theta = {THETA}")
        print(f"{NUM_PATTERNS} gap patterns, {MISSING_FRACTION:.0%} of "
              f"{NUM_OBSERVATIONS} observations blanked\n")

        kept_pairs, collapsed = [], []
        for k in range(NUM_PATTERNS):
            holed = series.copy()
            holed[np.random.default_rng(100 + k).random(len(series)) < MISSING_FRACTION] = np.nan
            _, _, beta_kept = compute_ar_residuals(holed)
            kept_pairs.append(abs(beta_kept[0] - beta_full[0]))
            collapsed.append(abs(estimate_by_collapsing_gaps(holed) - beta_full[0]))

        kept_pairs, collapsed = np.array(kept_pairs), np.array(collapsed)
        print(f"{'deviation from gap-free beta':<32}{'min':>8}{'max':>8}{'mean':>8}")
        print(f"{'pairs straddling a gap dropped':<32}{kept_pairs.min():>8.3f}"
              f"{kept_pairs.max():>8.3f}{kept_pairs.mean():>8.3f}")
        print(f"{'gaps collapsed':<32}{collapsed.min():>8.3f}"
              f"{collapsed.max():>8.3f}{collapsed.mean():>8.3f}")
        print(f"\nthe ranges do not overlap: {kept_pairs.max():.3f} < {collapsed.min():.3f}")

    elif local_test is LocalTest.RESIDUAL_ROWS:
        series = generate_ar1(THETA, n=120)
        holed = series.copy()
        holed.iloc[[10, 50, 90]] = np.nan
        residuals_full, _, _ = compute_ar_residuals(series)
        residuals_holed, _, _ = compute_ar_residuals(holed)
        print(f"{len(series)} observations, no gaps -> {len(residuals_full)} residual rows")
        print(f"{len(series)} observations, 3 gaps  -> {len(residuals_holed)} residual rows, "
              f"NaN count {int(np.isnan(residuals_holed).sum())}")
        print(f"each interior gap removes 2 pairs: {len(series) - 1} - 6 "
              f"= {len(series) - 7}")
        print("the draw is taken over the residual rows, so no index reaches past the end")

    elif local_test is LocalTest.END_TO_END:
        series = generate_ar1(THETA, n=200)
        holed = series.copy()
        holed.iloc[[10, 50, 90]] = np.nan
        paths = qis.bootstrap_ar_process(data=holed, num_samples=5, index_length=100,
                                         block_size=20, seed=1)
        arrays = [np.asarray(path) for path in paths]
        print(f"{len(arrays)} paths of length {len(arrays[0])} from data with 3 gaps")
        print(f"all finite: {all(np.isfinite(a).all() for a in arrays)}")


if __name__ == '__main__':
    for case in LocalTest:
        print(f"\n--- {case.name} ---")
        run_local_test(case)
