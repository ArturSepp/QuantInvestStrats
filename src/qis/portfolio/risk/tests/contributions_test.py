"""Tests for covariance-implied asset and group risk contributions."""

import numpy as np
import pandas as pd
import pytest

import qis


ASSETS = pd.Index(["a", "b", "c"])
COVAR = pd.DataFrame(
    [[0.04, 0.01, 0.00], [0.01, 0.09, 0.01], [0.00, 0.01, 0.16]],
    index=ASSETS,
    columns=ASSETS,
)
WEIGHTS = pd.Series([0.50, 0.30, 0.20], index=ASSETS)
GROUPS = pd.Series(["rates", "credit", "credit"], index=ASSETS)


def test_risk_contribution_ratios_normalise_existing_euler_identity() -> None:
    """Ratios must equal absolute Euler contributions divided by portfolio volatility."""
    absolute = qis.compute_portfolio_risk_contributions(w=WEIGHTS, covar=COVAR)
    ratios = qis.compute_portfolio_risk_contribution_ratios(weights=WEIGHTS, covar=COVAR)

    pd.testing.assert_series_equal(ratios, absolute / absolute.sum())
    assert ratios.sum() == pytest.approx(1.0, abs=1e-15)


def test_risk_contribution_ratios_align_labels_and_guard_zero_risk() -> None:
    """The covariance universe controls alignment and zero risk maps to zero ratios."""
    extended = pd.concat([WEIGHTS, pd.Series({"outside": 9.0})])
    actual = qis.compute_portfolio_risk_contribution_ratios(weights=extended, covar=COVAR)
    expected = qis.compute_portfolio_risk_contribution_ratios(weights=WEIGHTS, covar=COVAR)
    pd.testing.assert_series_equal(actual, expected)

    zero = qis.compute_portfolio_risk_contribution_ratios(
        weights=pd.Series(0.0, index=ASSETS), covar=COVAR,
    )
    pd.testing.assert_series_equal(zero, pd.Series(0.0, index=ASSETS))


def test_risk_contribution_ratios_support_arrays_and_validate_dimensions() -> None:
    """The array path preserves type, normalizes risk, and rejects ambiguous inputs."""
    weights = WEIGHTS.to_numpy()
    covar = COVAR.to_numpy()
    actual = qis.compute_portfolio_risk_contribution_ratios(weights=weights, covar=covar)
    absolute = qis.compute_portfolio_risk_contributions(w=weights, covar=covar)
    np.testing.assert_allclose(actual, absolute / absolute.sum(), rtol=0.0, atol=1e-15)

    zero = qis.compute_portfolio_risk_contribution_ratios(
        weights=np.zeros_like(weights), covar=covar,
    )
    np.testing.assert_array_equal(zero, np.zeros_like(weights))
    with pytest.raises(AssertionError):
        qis.compute_portfolio_risk_contribution_ratios(
            weights=np.ones(2), covar=covar,
        )
    with pytest.raises(ValueError, match="unsupported types"):
        qis.compute_portfolio_risk_contribution_ratios(
            weights=WEIGHTS, covar=covar,
        )


def test_group_risk_contribution_ratios_aggregate_in_first_seen_order() -> None:
    """Group ratios must aggregate asset ratios without changing their signed total."""
    actual = qis.compute_group_portfolio_risk_contribution_ratios(
        weights=WEIGHTS, covar=COVAR, groups=GROUPS,
    )
    asset = qis.compute_portfolio_risk_contribution_ratios(weights=WEIGHTS, covar=COVAR)
    expected = asset.groupby(GROUPS, sort=False).sum().rename("risk_contribution")

    pd.testing.assert_series_equal(actual, expected)
    assert actual.index.tolist() == ["rates", "credit"]
    assert actual.sum() == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    ("covar", "groups", "error", "match"),
    [
        (pd.DataFrame(), pd.Series(dtype=object), ValueError, "non-empty and square"),
        (
            COVAR.rename(columns={"c": "x"}),
            GROUPS,
            ValueError,
            "index and columns",
        ),
        (COVAR, pd.Series(["x", "y"], index=["a", "b"]), ValueError, "classify every"),
    ],
)
def test_group_risk_contribution_ratios_validate_partition(
        covar: pd.DataFrame,
        groups: pd.Series,
        error: type[Exception],
        match: str,
        ) -> None:
    """Malformed covariance labels or incomplete group mappings must fail explicitly."""
    with pytest.raises(error, match=match):
        qis.compute_group_portfolio_risk_contribution_ratios(
            weights=WEIGHTS, covar=covar, groups=groups,
        )


def test_group_risk_contribution_ratios_require_series_groups() -> None:
    """A labelled Series is required so group aggregation cannot be positional."""
    with pytest.raises(TypeError, match="groups must be a pandas Series"):
        qis.compute_group_portfolio_risk_contribution_ratios(
            weights=WEIGHTS, covar=COVAR, groups=np.array(["x", "y", "y"]),
        )

    with pytest.raises(ValueError, match="group asset labels must be unique"):
        qis.compute_group_portfolio_risk_contribution_ratios(
            weights=WEIGHTS,
            covar=COVAR,
            groups=pd.Series(["x", "y", "z"], index=["a", "a", "c"]),
        )
