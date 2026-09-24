"""Regression tests for label-aligned error-bar magnitudes."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qis.plots.errorbar import plot_errorbar  # noqa: E402


def _capture_yerr(
    estimates: pd.Series | pd.DataFrame,
    errors: pd.Series | pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> list[np.ndarray]:
    """Return the error magnitudes supplied to Matplotlib."""
    captured: list[np.ndarray] = []
    fig, ax = plt.subplots()
    # Capture the renderer boundary so label association is independent of styling details.
    monkeypatch.setattr(ax, "errorbar", lambda **kwargs: captured.append(kwargs["yerr"]))
    try:
        plot_errorbar(df=estimates, y_std_errors=errors, ax=ax, var_format=None)
    finally:
        plt.close(fig)
    return captured


def test_plot_errorbar_aligns_series_errors_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach each shared error magnitude to its estimate label."""
    estimates = pd.Series([10.0, 20.0], index=["a", "b"], name="estimate")
    errors = pd.Series([2.0, 1.0], index=["b", "a"])
    estimates_before = estimates.copy(deep=True)
    errors_before = errors.copy(deep=True)

    captured = _capture_yerr(estimates, errors, monkeypatch)

    assert [values.tolist() for values in captured] == [[1.0, 2.0]]
    pd.testing.assert_series_equal(estimates, estimates_before)
    pd.testing.assert_series_equal(errors, errors_before)


def test_plot_errorbar_aligns_frame_errors_by_both_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Align independently permuted error rows and columns for every estimate series."""
    estimates = pd.DataFrame(
        {"left": [10.0, 20.0], "right": [30.0, 40.0]},
        index=["a", "b"],
    )
    # Independent row and column permutations expose either omitted alignment operation.
    errors = pd.DataFrame(
        {"right": [4.0, 3.0], "left": [2.0, 1.0]},
        index=["b", "a"],
    )
    estimates_before = estimates.copy(deep=True)
    errors_before = errors.copy(deep=True)

    captured = _capture_yerr(estimates, errors, monkeypatch)

    assert [values.tolist() for values in captured] == [[1.0, 2.0], [3.0, 4.0]]
    pd.testing.assert_frame_equal(estimates, estimates_before)
    pd.testing.assert_frame_equal(errors, errors_before)


@pytest.mark.parametrize("invalid_kind", ("missing", "extra", "duplicate"))
def test_plot_errorbar_rejects_inexact_series_error_labels(
    invalid_kind: str,
) -> None:
    """Reject ambiguous shared-error indexes before constructing artists."""
    estimates = pd.Series([10.0, 20.0], index=["a", "b"], name="estimate")
    if invalid_kind == "missing":
        errors = pd.Series([1.0], index=["a"])
    elif invalid_kind == "extra":
        errors = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    else:
        errors = pd.Series([1.0, 2.0], index=["a", "a"])

    with pytest.raises(ValueError, match="y_std_errors index"):
        plot_errorbar(df=estimates, y_std_errors=errors)


@pytest.mark.parametrize(
    ("axis_name", "invalid_kind"),
    (
        ("index", "missing"),
        ("index", "extra"),
        ("index", "duplicate"),
        ("columns", "missing"),
        ("columns", "extra"),
        ("columns", "duplicate"),
    ),
)
def test_plot_errorbar_rejects_inexact_frame_error_labels(
    axis_name: str,
    invalid_kind: str,
) -> None:
    """Reject ambiguous error-frame axes before constructing artists."""
    estimates = pd.DataFrame(
        {"left": [10.0, 20.0], "right": [30.0, 40.0]},
        index=["a", "b"],
    )
    errors = pd.DataFrame(
        {"left": [1.0, 2.0], "right": [3.0, 4.0]},
        index=["a", "b"],
    )
    if axis_name == "index":
        if invalid_kind == "missing":
            errors = errors.iloc[:1]
        elif invalid_kind == "extra":
            errors.loc["c"] = [5.0, 6.0]
        else:
            errors.index = ["a", "a"]
    elif invalid_kind == "missing":
        errors = errors[["left"]]
    elif invalid_kind == "extra":
        errors["extra"] = [5.0, 6.0]
    else:
        errors = pd.concat([errors[["left"]], errors[["left"]]], axis=1)

    with pytest.raises(ValueError, match=f"y_std_errors {axis_name}"):
        plot_errorbar(df=estimates, y_std_errors=errors)
