"""Point-in-time EWMA beta/alpha estimator tests."""
import numpy as np
import pandas as pd

from qis.models.linear.ewm import (
    InitType,
    compute_ewm_beta_alpha_forecast,
)


def _ewma(values: np.ndarray, decay: float, initial_value: float) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    output[0] = initial_value
    for idx in range(1, len(values)):
        output[idx] = decay * output[idx - 1] + (1.0 - decay) * values[idx]
    return output


def test_beta_init_value_is_one_observation_point_in_time_prior() -> None:
    """A beta seed owns the first estimate and then decays through observed moments."""
    index = pd.date_range('2025-01-31', periods=5, freq='ME')
    x = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01], index=index, name='Benchmark')
    y = pd.DataFrame({'Portfolio': [0.01, -0.008, 0.02, -0.012, 0.009]}, index=index)
    span = 3

    betas, alphas, *_ = compute_ewm_beta_alpha_forecast(
        x_data=x,
        y_data=y,
        span=span,
        init_type=InitType.X0,
        beta_init_value=1.0,
    )

    decay = 1.0 - 2.0 / (span + 1.0)
    x_values = x.to_numpy()
    y_values = y['Portfolio'].to_numpy()
    x_var = _ewma(
        np.square(x_values), decay=decay, initial_value=x_values[0] ** 2)
    xy = _ewma(
        x_values * y_values,
        decay=decay,
        initial_value=x_values[0] ** 2,
    )
    expected_beta = xy / x_var
    residual = y_values - expected_beta * x_values
    expected_alpha = _ewma(residual, decay=decay, initial_value=residual[0])

    np.testing.assert_allclose(betas['Portfolio'], expected_beta)
    np.testing.assert_allclose(alphas['Portfolio'], expected_alpha)
    assert betas.at[index[0], 'Portfolio'] == 1.0


def test_omitting_beta_init_value_preserves_existing_x0_estimator() -> None:
    """The optional seed cannot alter the historical default covariance-ratio path."""
    index = pd.date_range('2025-01-31', periods=4, freq='ME')
    x = pd.Series([0.02, -0.01, 0.03, -0.02], index=index, name='Benchmark')
    y = pd.DataFrame({'Portfolio': [0.01, -0.008, 0.02, -0.012]}, index=index)
    span = 3

    betas, *_ = compute_ewm_beta_alpha_forecast(
        x_data=x,
        y_data=y,
        span=span,
        init_type=InitType.X0,
    )

    decay = 1.0 - 2.0 / (span + 1.0)
    x_values = x.to_numpy()
    y_values = y['Portfolio'].to_numpy()
    x_var = _ewma(
        np.square(x_values), decay=decay, initial_value=x_values[0] ** 2)
    xy = _ewma(
        x_values * y_values,
        decay=decay,
        initial_value=x_values[0] * y_values[0],
    )
    np.testing.assert_allclose(betas['Portfolio'], xy / x_var)
