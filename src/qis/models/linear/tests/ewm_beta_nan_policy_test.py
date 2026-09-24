"""Missing-observation policy tests for the EWM beta recursion."""

import numpy as np
import pytest

from qis.models.linear.ewm import NanBackfill, compute_ewm_xy_beta_tensor


@pytest.mark.parametrize(
    ("nan_backfill", "expected"),
    [
        (NanBackfill.FFILL, [2.0, 2.0 / 3.0, 2.0 / 3.0, 134.0 / 73.0]),
        (NanBackfill.DEFLATED_FFILL, [2.0, 2.0 / 3.0, 2.0 / 3.0, 262.0 / 137.0]),
        (NanBackfill.ZERO_FILL, [2.0, 2.0 / 3.0, 0.0, 2.0]),
        (NanBackfill.NAN_FILL, [2.0, 2.0 / 3.0, 0.0, 2.0]),
    ],
)
def test_missing_factor_applies_policy_to_both_beta_moments(
    nan_backfill: NanBackfill,
    expected: list[float],
) -> None:
    """Each policy updates the numerator and denominator from the same factor row."""
    factor = np.array([1.0, 2.0, np.nan, 4.0])
    asset = np.array([2.0, 1.0, 3.0, 8.0])

    actual = compute_ewm_xy_beta_tensor(
        x=factor,
        y=asset,
        ewm_lambda=0.5,
        warmup_period=-1,
        nan_backfill=nan_backfill,
    )[:, 0, 0]

    # These ratios come from hand-updating E[x*y] and E[x*x] before dividing.
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)


def test_ffill_preserves_multifactor_multiasset_beta_across_factor_gap() -> None:
    """Correlated multi-factor betas hold when the entire factor row is missing."""
    coefficients = np.array([[2.0, -1.0], [0.5, 3.0]])
    observed_factors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    observed_assets = observed_factors @ coefficients
    factors = np.vstack((observed_factors, [np.nan, np.nan]))
    assets = np.vstack((observed_assets, [7.0, -2.0]))

    actual = compute_ewm_xy_beta_tensor(
        x=factors,
        y=assets,
        ewm_lambda=0.5,
        warmup_period=-1,
        is_x_correlated=True,
        nan_backfill=NanBackfill.FFILL,
    )

    # For y = x @ coefficients, E[x*x]^-1 @ E[x*y] equals coefficients.
    np.testing.assert_allclose(actual[-2:], np.broadcast_to(coefficients, (2, 2, 2)))
