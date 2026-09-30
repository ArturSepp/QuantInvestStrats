"""EWM parameter domains, precedence, and column-wise validation.

The decay recursion is stable only for finite ``span >= 1`` or ``0 <= ewm_lambda < 1``.
The boundary tests matter because invalid scalars and one invalid column previously reached the
recursion and returned plausible-looking output instead of rejecting the complete request.
"""

import numpy as np
import pandas as pd
import pytest

from qis.models.linear.ewm import (
    compute_ewm,
    compute_ewm_covar,
    compute_ewm_covar_newey_west,
    compute_ewm_covar_tensor,
    compute_ewm_covar_tensor_vol_norm_returns,
    compute_ewm_vol,
    compute_ewm_xy_beta_tensor,
    ewm_recursion,
)


_VALUES = np.array([1.0, 3.0, 3.0, 5.0])
_PANEL = np.column_stack((_VALUES, _VALUES + 1.0))


@pytest.mark.parametrize("span", [0.0, -2.0, np.nan, np.inf, True])
def test_compute_ewm_rejects_invalid_span_before_recursion(span) -> None:
    """Every invalid scalar span is rejected before it can generate an EWM path."""
    with pytest.raises(ValueError, match="span must be finite and >= 1"):
        compute_ewm(_VALUES, span=span)


@pytest.mark.parametrize("ewm_lambda", [-0.2, 1.0, 1.2, np.nan, np.inf, True])
def test_compute_ewm_vol_rejects_invalid_decay_before_recursion(ewm_lambda) -> None:
    """Every invalid scalar decay is rejected before variance recursion or square root."""
    with pytest.raises(ValueError, match=r"ewm_lambda must be finite and in \[0, 1\)"):
        compute_ewm_vol(_VALUES, ewm_lambda=ewm_lambda)


@pytest.mark.parametrize(
    "parameter",
    [
        {"span": np.array([3.0, 0.0])},
        {"ewm_lambda": np.array([0.5, 1.0])},
        {"span": np.array([3.0, np.nan])},
        {"ewm_lambda": np.array([0.5, np.inf])},
        {"span": np.array([True, False])},
    ],
)
def test_compute_ewm_rejects_mixed_invalid_per_column_parameters(parameter) -> None:
    """One invalid column rejects the panel instead of partially updating recursive state."""
    with pytest.raises(ValueError):
        compute_ewm(pd.DataFrame(_PANEL, columns=["a", "b"]), **parameter)


@pytest.mark.parametrize("span", [np.array([True, False]), np.array([True, True])])
def test_ewm_recursion_rejects_boolean_span_arrays(span) -> None:
    """Direct recursion reports the invalid span domain instead of a Numba typing error."""
    with pytest.raises(ValueError, match="span must be finite and >= 1"):
        ewm_recursion(_PANEL, init_value=np.zeros(2), span=span)


def test_ewm_recursion_accepts_valid_per_column_span_arrays() -> None:
    """Array spans retain their documented independent column recursions."""
    actual = ewm_recursion(_PANEL, init_value=_PANEL[0], span=np.array([1.0, 3.0]))
    expected = np.array([[1.0, 2.0], [3.0, 3.0], [3.0, 3.5], [5.0, 4.75]])
    np.testing.assert_allclose(actual, expected)


def test_compute_ewm_validates_decay_shape_before_recursion() -> None:
    """A one-dimensional input cannot silently use only the first of several decays."""
    with pytest.raises(ValueError, match="one decay value"):
        compute_ewm(_VALUES, ewm_lambda=np.array([0.5, 0.8]))


def test_compute_ewm_xy_beta_tensor_requires_scalar_smoothing_parameter() -> None:
    """Matrix recursions reject per-column smoothing that cannot preserve symmetric moments."""
    with pytest.raises(ValueError, match="require a scalar smoothing parameter"):
        compute_ewm_xy_beta_tensor(_PANEL[:, :1], _PANEL[:, 1:], ewm_lambda=np.array([0.5, 0.8]))


def test_compute_ewm_preserves_span_precedence_and_span_one_containers() -> None:
    """A valid span overrides decay, and span one remains the exact pass-through boundary."""
    series = pd.Series(_VALUES, name="returns")
    expected = compute_ewm(series, span=3.0)
    pd.testing.assert_series_equal(compute_ewm(series, span=3.0, ewm_lambda=1.2), expected)
    pd.testing.assert_series_equal(compute_ewm(series, span=1.0), series)
    np.testing.assert_array_equal(compute_ewm(_VALUES, span=1.0), _VALUES)


def test_compute_ewm_applies_valid_per_column_spans_independently() -> None:
    """Each column uses its own decay without changing labels, order, or container type."""
    data = pd.DataFrame(_PANEL, columns=["pass_through", "half_decay"])
    expected = pd.DataFrame({"pass_through": _VALUES, "half_decay": [2.0, 3.0, 3.5, 4.75]})
    pd.testing.assert_frame_equal(compute_ewm(data, span=np.array([1.0, 3.0])), expected)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ewm_recursion(_VALUES, init_value=_VALUES[0], span=0.0),
        lambda: compute_ewm_covar(_PANEL, span=0.0),
        lambda: compute_ewm_covar_newey_west(_PANEL, span=0.0),
        lambda: compute_ewm_covar_tensor(_PANEL, span=0.0),
        lambda: compute_ewm_covar_tensor_vol_norm_returns(_PANEL, span=0.0),
        lambda: compute_ewm_xy_beta_tensor(_PANEL[:, :1], _PANEL[:, 1:], span=0.0),
    ],
    ids=["recursion", "covariance", "newey-west", "tensor", "normalised-tensor", "beta"],
)
def test_ewm_public_numerical_paths_reject_invalid_span(operation) -> None:
    """Public kernels that bypass the shared wrapper enforce the same stable span domain."""
    with pytest.raises(ValueError, match="span must be finite and >= 1"):
        operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ewm_recursion(_VALUES, init_value=_VALUES[0], ewm_lambda=1.0),
        lambda: compute_ewm_covar(_PANEL, ewm_lambda=1.0),
        lambda: compute_ewm_covar_newey_west(_PANEL, ewm_lambda=1.0),
        lambda: compute_ewm_covar_tensor(_PANEL, ewm_lambda=1.0),
        lambda: compute_ewm_covar_tensor_vol_norm_returns(_PANEL, ewm_lambda=1.0),
        lambda: compute_ewm_xy_beta_tensor(_PANEL[:, :1], _PANEL[:, 1:], ewm_lambda=1.0),
    ],
    ids=["recursion", "covariance", "newey-west", "tensor", "normalised-tensor", "beta"],
)
def test_ewm_public_numerical_paths_reject_invalid_decay(operation) -> None:
    """Public kernels that bypass the shared wrapper enforce the same stable decay domain."""
    with pytest.raises(ValueError, match=r"ewm_lambda must be finite and in \[0, 1\)"):
        operation()
