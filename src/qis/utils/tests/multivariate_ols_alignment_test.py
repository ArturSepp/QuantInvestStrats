"""Keep multivariate OLS observations paired before their pandas labels are discarded.

An apparently perfect fit can describe the wrong dated relationship. Rejection tests therefore
exercise index identity, while literal coefficient and prediction oracles protect valid fits.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis


def _dated_line():
    """Provide the independently specified line y = 1 + 2x on five named dates."""
    index = pd.date_range("2024-01-31", periods=5, freq="ME", name="observation")
    x = pd.DataFrame({"factor": [-2.0, -1.0, 0.0, 1.0, 2.0]}, index=index)
    y = pd.Series([-3.0, -1.0, 1.0, 3.0, 5.0], index=index, name="response")
    return x, y


@pytest.mark.parametrize("storage", ["float64", "Float64"])
@pytest.mark.parametrize(
    "mismatch", ["reversed", "shifted", "shorter", "longer", "duplicates", "timezone"]
)
def test_fit_multivariate_ols_rejects_mismatched_observation_indexes(storage, mismatch):
    """Reject incompatible row identities without changing either caller-owned input.

    Nullable storage must not hide the index error behind a later positional-conversion failure.
    This does not promise new support for fitting an aligned nullable design.
    """
    x, y = _dated_line()
    x, y = x.astype(storage), y.astype(storage)
    if mismatch == "reversed":
        y = y.iloc[::-1]
    elif mismatch == "shifted":
        y = y.set_axis(y.index + pd.Timedelta(days=1))
    elif mismatch == "shorter":
        y = y.iloc[:-1]
    elif mismatch == "longer":
        extra = pd.Series([7.0], index=[pd.Timestamp("2024-06-30")], dtype=storage)
        y = pd.concat([y, extra])
    elif mismatch == "duplicates":
        y = y.set_axis(y.index.take([0, 0, 2, 3, 4]))
    else:
        y = y.set_axis(y.index.tz_localize("UTC"))
    x_before, y_before = x.copy(deep=True), y.copy(deep=True)

    with pytest.raises(ValueError, match="x and y indices must be aligned"):
        qis.fit_multivariate_ols(x, y, verbose=False)

    pd.testing.assert_frame_equal(x, x_before)
    pd.testing.assert_series_equal(y, y_before)


def test_fit_multivariate_ols_rejects_before_conversion_can_hide_a_filtered_mismatch(monkeypatch):
    """Missing rows cannot make originally mismatched observation labels acceptable."""
    x, y = _dated_line()
    x.iloc[:2, 0] = np.nan
    y.iloc[:2] = np.nan
    y = y.set_axis(y.index.take([1, 0, 2, 3, 4]))
    x_before, y_before = x.copy(deep=True), y.copy(deep=True)

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("misaligned observations must be rejected before positional conversion")

    # Both mismatched rows would disappear in the finite sample; validate the original labels.
    with monkeypatch.context() as conversion_guard:
        conversion_guard.setattr(pd.DataFrame, "to_numpy", unexpected_conversion)
        conversion_guard.setattr(pd.Series, "to_numpy", unexpected_conversion)
        with pytest.raises(ValueError, match="x and y indices must be aligned"):
            qis.fit_multivariate_ols(x, y, verbose=False)

    pd.testing.assert_frame_equal(x, x_before)
    pd.testing.assert_series_equal(y, y_before)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("rows", ["dated", "joint-permutation", "duplicates", "metadata"])
def test_fit_multivariate_ols_preserves_aligned_coefficients_predictions_and_labels(
    fit_intercept, rows
):
    """Preserve both fitting branches and the native Index.equals compatibility boundary.

    Equal duplicate labels and joint permutations still pair observations correctly. Index names
    and frequency attributes need not match: they are not part of the existing native row guard.
    """
    index = pd.date_range("2024-01-31", periods=7, freq="ME", name="observation")
    x = pd.DataFrame(
        {
            "linear": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "curvature": [9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0],
        },
        index=index,
    )
    # Literal responses implement y = 2*linear - 3*curvature, optionally plus an intercept of 1.
    y = pd.Series([-33.0, -16.0, -5.0, 0.0, -1.0, -8.0, -21.0], index=index)
    if fit_intercept:
        y = y + 1.0
    if rows == "joint-permutation":
        positions = [4, 1, 6, 0, 5, 3, 2]
        x, y = x.iloc[positions], y.iloc[positions]
    elif rows == "duplicates":
        duplicated = index.take([0, 1, 1, 3, 3, 5, 6])
        x, y = x.set_axis(duplicated), y.set_axis(duplicated)
    elif rows == "metadata":
        x = x.set_axis(index.rename("factor_date"))
        y = y.set_axis(pd.DatetimeIndex(index.to_numpy(), name="response_date"))
    x_before, y_before = x.copy(deep=True), y.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        prediction, params, label = qis.fit_multivariate_ols(x, y, fit_intercept, verbose=False)

    expected_params = pd.Series([2.0, -3.0], index=["linear", "curvature"])
    expected_label = "y=+2.00*linear-3.00*curvature, R²=100%"
    if fit_intercept:
        expected_params = pd.Series([1.0, 2.0, -3.0], index=["intercept", "linear", "curvature"])
        expected_label = "y=+1.00+2.00*linear-3.00*curvature, R²=100%"
    pd.testing.assert_series_equal(params, expected_params, rtol=0.0, atol=1e-12)
    pd.testing.assert_series_equal(prediction, y, rtol=0.0, atol=1e-12)
    assert label == expected_label
    pd.testing.assert_frame_equal(x, x_before)
    pd.testing.assert_series_equal(y, y_before)


def test_fit_multivariate_ols_preserves_joint_finite_sample_and_prediction_dates():
    """Missingness in either factor or the response drops only the explicitly identified rows."""
    index = pd.date_range("2024-01-31", periods=7, freq="ME", name="observation")
    x = pd.DataFrame(
        {
            "linear": [-3.0, np.nan, -1.0, 0.0, 1.0, 2.0, 3.0],
            "curvature": [9.0, 4.0, 1.0, 0.0, np.inf, 4.0, 9.0],
        },
        index=index,
    )
    y = pd.Series([-32.0, -15.0, np.nan, 1.0, 0.0, -7.0, -20.0], index=index)
    x_before, y_before = x.copy(deep=True), y.copy(deep=True)

    prediction, params, label = qis.fit_multivariate_ols(x, y, verbose=False)

    expected = pd.Series([-32.0, 1.0, -7.0, -20.0], index=index.take([0, 3, 5, 6]))
    pd.testing.assert_series_equal(prediction, expected, rtol=0.0, atol=1e-12)
    pd.testing.assert_series_equal(
        params,
        pd.Series([1.0, 2.0, -3.0], index=["intercept", "linear", "curvature"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert label == "y=+1.00+2.00*linear-3.00*curvature, R²=100%"
    pd.testing.assert_frame_equal(x, x_before)
    pd.testing.assert_series_equal(y, y_before)


def test_fit_multivariate_ols_keeps_public_scatter_prediction_on_the_fitted_coordinates():
    """The plotting caller constructs an aligned design; its overlay must remain drawable."""
    import matplotlib.pyplot as plt

    x, y = _dated_line()
    frame = x.assign(response=y).iloc[[4, 1, 3, 0, 2]]
    before = frame.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        qis.plot_multivariate_scatter_with_prediction(
            frame, x=["factor"], y="response", x_axis_column="factor", verbose=False, ax=ax
        )
        fig.canvas.draw()
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [-3.0, -1.0, 1.0, 3.0, 5.0], atol=1e-12)
        assert ax.get_legend().get_texts()[0].get_text() == (
            "prediction y=+1.00+2.00*factor, R²=100%"
        )
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(fig)
