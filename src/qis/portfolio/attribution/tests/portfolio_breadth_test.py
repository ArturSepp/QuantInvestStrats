"""Numerical contracts for point-in-time portfolio-breadth analytics."""

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.attribution.portfolio_breadth import (
    CAPITAL_UTILISATION,
    EFFECTIVE_CAPITAL_COUNT,
    EFFECTIVE_RISK_COUNT,
    EFFECTIVE_UNIVERSE_COUNT,
    INVESTABLE_COUNT,
    INVESTED_COUNT,
    RISK_BREADTH_EFFICIENCY,
    SELECTION_COVERAGE,
    SIZING_EVENNESS,
    UNAVAILABLE_GROSS_WEIGHT,
    UNAVAILABLE_INVESTED_COUNT,
    compute_portfolio_breadth,
)


def _monthly_returns() -> pd.DataFrame:
    """Return a small deterministic monthly return panel."""
    index = pd.date_range("2024-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, -0.02],
            "B": [0.02, 0.01, -0.01, 0.00],
            "C": [0.00, 0.02, 0.01, -0.01],
        },
        index=index,
    )


def test_hand_computed_counts_capital_breadth_and_exact_efficiency_decomposition() -> None:
    """Capital breadth must be the inverse HHI of absolute material weights."""
    returns = _monthly_returns()
    date = returns.index[-1]
    weights = pd.DataFrame([[0.6, 0.3, 0.1]], index=[date], columns=returns.columns)
    covar = pd.DataFrame(np.eye(3), index=returns.columns, columns=returns.columns)

    result = compute_portfolio_breadth(
        returns=returns,
        weights=weights,
        covar_dict={date: covar},
    )

    row = result.metrics.loc[date]
    expected_effective_count = 1.0 / (0.6**2 + 0.3**2 + 0.1**2)
    assert row[INVESTABLE_COUNT] == 3.0
    assert row[INVESTED_COUNT] == 3.0
    assert row[EFFECTIVE_UNIVERSE_COUNT] == pytest.approx(3.0)
    assert row[EFFECTIVE_CAPITAL_COUNT] == pytest.approx(expected_effective_count)
    risk_shares = np.square([0.6, 0.3, 0.1])
    risk_shares = risk_shares / risk_shares.sum()
    expected_effective_risk_count = 1.0 / np.square(risk_shares).sum()
    assert row[EFFECTIVE_RISK_COUNT] == pytest.approx(expected_effective_risk_count)
    assert row[SELECTION_COVERAGE] == 1.0
    assert row[SIZING_EVENNESS] == pytest.approx(expected_effective_count / 3.0)
    assert row[CAPITAL_UTILISATION] == pytest.approx(expected_effective_count / 3.0)
    assert row[RISK_BREADTH_EFFICIENCY] == pytest.approx(expected_effective_risk_count / 3.0)
    assert row[CAPITAL_UTILISATION] == pytest.approx(
        row[SELECTION_COVERAGE] * row[SIZING_EVENNESS]
    )
    assert result.absolute_weight_shares.loc[date].sum() == pytest.approx(1.0)
    assert result.absolute_risk_contribution_shares.loc[date].sum() == pytest.approx(1.0)


def test_provided_covariance_controls_availability_as_of_each_weight_date() -> None:
    """Only positive finite variances from the latest past covariance define availability."""
    returns = _monthly_returns()
    dates = returns.index[[1, 3]]
    weights = pd.DataFrame(0.25, index=dates, columns=returns.columns)
    weights["D"] = 0.25
    returns["D"] = 0.01
    covar_1 = pd.DataFrame(
        np.diag([1.0, np.nan, 0.0, 1.0]),
        index=returns.columns,
        columns=returns.columns,
    )
    covar_2 = pd.DataFrame(np.eye(4), index=returns.columns, columns=returns.columns)

    result = compute_portfolio_breadth(
        returns=returns,
        weights=weights,
        covar_dict={dates[0]: covar_1, dates[1]: covar_2},
    )

    assert result.availability.loc[dates[0]].to_dict() == {
        "A": True,
        "B": False,
        "C": False,
        "D": True,
    }
    assert result.metrics.loc[dates[0], INVESTABLE_COUNT] == 2.0
    assert result.metrics.loc[dates[0], UNAVAILABLE_INVESTED_COUNT] == 2.0
    assert result.metrics.loc[dates[0], UNAVAILABLE_GROSS_WEIGHT] == pytest.approx(0.5)
    assert result.availability.loc[dates[1]].all()


def test_future_covariance_is_never_used() -> None:
    """An evaluation before the first covariance has no investable assets or risk breadth."""
    returns = _monthly_returns()
    evaluation_date = returns.index[1]
    future_date = returns.index[2]
    weights = pd.DataFrame([[0.5, 0.5, 0.0]], index=[evaluation_date], columns=returns.columns)
    covar = pd.DataFrame(np.eye(3), index=returns.columns, columns=returns.columns)

    result = compute_portfolio_breadth(
        returns=returns,
        weights=weights,
        covar_dict={future_date: covar},
    )

    assert not result.availability.loc[evaluation_date].any()
    assert pd.isna(result.covariance_dates.loc[evaluation_date])
    assert result.metrics.loc[evaluation_date, INVESTABLE_COUNT] == 0.0
    assert result.metrics.loc[evaluation_date, EFFECTIVE_RISK_COUNT] == 0.0


def test_return_inferred_covariance_and_availability_are_point_in_time() -> None:
    """Changing future returns must not alter an earlier breadth observation."""
    returns = _monthly_returns()
    returns.loc[returns.index[:2], "C"] = np.nan
    evaluation_dates = returns.index[[1, 3]]
    weights = pd.DataFrame(
        [[0.5, 0.5, 0.0], [1.0 / 3.0] * 3],
        index=evaluation_dates,
        columns=returns.columns,
    )
    changed_future = returns.copy()
    changed_future.loc[returns.index[2]:, ["A", "B", "C"]] = [100.0, -50.0, 25.0]

    original = compute_portfolio_breadth(returns=returns, weights=weights, span=3)
    changed = compute_portfolio_breadth(returns=changed_future, weights=weights, span=3)

    pd.testing.assert_series_equal(
        original.metrics.loc[evaluation_dates[0]],
        changed.metrics.loc[evaluation_dates[0]],
    )
    assert original.availability.loc[evaluation_dates[0]].to_dict() == {
        "A": True,
        "B": True,
        "C": False,
    }
    assert original.availability.loc[evaluation_dates[1], "C"]
    assert original.covariance_dates.loc[evaluation_dates[0]] == returns.index[1]


def test_long_short_weights_use_absolute_shares_and_cash_is_only_explicitly_excluded() -> None:
    """Gross-normalized shares make capital and risk breadth safe for long-short portfolios."""
    returns = _monthly_returns().rename(columns={"C": "Cash"})
    date = returns.index[-1]
    weights = pd.DataFrame([[0.5, -0.5, 0.4]], index=[date], columns=returns.columns)
    covar = pd.DataFrame(np.eye(3), index=returns.columns, columns=returns.columns)

    with_cash = compute_portfolio_breadth(
        returns=returns,
        weights=weights,
        covar_dict={date: covar},
    )
    without_cash = compute_portfolio_breadth(
        returns=returns,
        weights=weights,
        covar_dict={date: covar},
        cash_columns=["Cash"],
    )

    assert with_cash.metrics.loc[date, INVESTED_COUNT] == 3.0
    assert without_cash.metrics.loc[date, INVESTED_COUNT] == 2.0
    assert without_cash.metrics.loc[date, EFFECTIVE_CAPITAL_COUNT] == pytest.approx(2.0)
    assert without_cash.metrics.loc[date, EFFECTIVE_RISK_COUNT] == pytest.approx(2.0)
    assert without_cash.absolute_weight_shares.loc[date].to_dict() == {
        "A": 0.5,
        "B": 0.5,
    }


def test_effective_universe_count_is_inverse_hhi_of_correlation_eigenvalues() -> None:
    """Two perfectly correlated assets represent one independent universe direction."""
    returns = _monthly_returns().loc[:, ["A", "B"]]
    date = returns.index[-1]
    weights = pd.DataFrame([[0.5, 0.5]], index=[date], columns=returns.columns)
    perfectly_correlated = pd.DataFrame(
        [[0.04, 0.06], [0.06, 0.09]],
        index=returns.columns,
        columns=returns.columns,
    )

    result = compute_portfolio_breadth(
        returns=returns,
        weights=weights,
        covar_dict={date: perfectly_correlated},
    )

    assert result.metrics.loc[date, EFFECTIVE_UNIVERSE_COUNT] == pytest.approx(1.0)
    assert result.metrics.loc[date, EFFECTIVE_CAPITAL_COUNT] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("returns", "weights", "message"),
    [
        (pd.Series(dtype=float), pd.DataFrame(), "returns must be a pandas DataFrame"),
        (_monthly_returns(), pd.Series(dtype=float), "weights must be a pandas DataFrame"),
    ],
)
def test_input_container_validation(returns: object, weights: object, message: str) -> None:
    """The public computation rejects containers without labelled two-dimensional axes."""
    with pytest.raises(TypeError, match=message):
        compute_portfolio_breadth(returns=returns, weights=weights)


def test_weight_assets_must_be_in_return_universe_and_parameters_are_validated() -> None:
    """Typos in target assets and invalid numerical controls must fail clearly."""
    returns = _monthly_returns()
    weights = pd.DataFrame([[1.0]], index=[returns.index[-1]], columns=["Unknown"])

    with pytest.raises(ValueError, match="weights columns missing from returns"):
        compute_portfolio_breadth(returns=returns, weights=weights)
    with pytest.raises(ValueError, match="span must be a positive integer"):
        compute_portfolio_breadth(
            returns=returns,
            weights=pd.DataFrame(
                [[1.0, 0.0, 0.0]], index=[returns.index[-1]], columns=returns.columns
            ),
            span=0,
        )
