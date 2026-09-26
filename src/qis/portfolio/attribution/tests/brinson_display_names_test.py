"""Regression checks for role-preserving Brinson summary display labels."""

import pandas as pd
import pytest

from qis import compute_brinson_attribution_table


@pytest.mark.parametrize(
    ("is_linked", "return_label", "strategy_return", "benchmark_return"),
    [
        (False, "Return Sum", 0.20, 0.065),
        (True, "Return Total", 1.06 * 1.14 - 1.0, 1.025 * 1.04 - 1.0),
    ],
)
@pytest.mark.parametrize("is_exclude_interaction_term", [True, False])
def test_compute_brinson_attribution_table_preserves_matching_name_roles(
    is_linked: bool,
    return_label: str,
    strategy_return: float,
    benchmark_return: float,
    is_exclude_interaction_term: bool,
) -> None:
    """Matching display names retain both portfolio roles and reconciliation."""
    index = pd.date_range("2025-01-31", periods=2, freq="ME")
    columns = ["Equity"]
    strategy_weights = pd.DataFrame([[0.60], [0.70]], index=index, columns=columns)
    benchmark_weights = pd.DataFrame([[0.50], [0.40]], index=index, columns=columns)
    strategy_pnl = pd.DataFrame([[0.06], [0.14]], index=index, columns=columns)
    benchmark_pnl = pd.DataFrame([[0.025], [0.04]], index=index, columns=columns)

    totals = compute_brinson_attribution_table(
        benchmark_pnl=benchmark_pnl,
        strategy_pnl=strategy_pnl,
        strategy_weights=strategy_weights,
        benchmark_weights=benchmark_weights,
        asset_class_data=pd.Series({"Equity": "Equity"}),
        is_exclude_interaction_term=is_exclude_interaction_term,
        strategy_name="Same",
        benchmark_name="Same",
        is_linked=is_linked,
    )[0]

    expected_columns = [
        "Same (Strategy)\nWeight Ave",
        "Same (Benchmark)\nWeight Ave",
        f"Same (Strategy)\n{return_label}",
        f"Same (Benchmark)\n{return_label}",
        "Asset\nAllocation",
        "Instrument\nSelection",
    ]
    if not is_exclude_interaction_term:
        expected_columns.append("Interaction")
    expected_columns.append("Total\nActive")
    assert totals.columns.to_list() == expected_columns

    total = totals.loc["Total Sum"]
    assert total["Same (Strategy)\nWeight Ave"] == pytest.approx(0.65)
    assert total["Same (Benchmark)\nWeight Ave"] == pytest.approx(0.45)
    assert total[f"Same (Strategy)\n{return_label}"] == pytest.approx(strategy_return)
    assert total[f"Same (Benchmark)\n{return_label}"] == pytest.approx(benchmark_return)
    assert total["Total\nActive"] == pytest.approx(strategy_return - benchmark_return)
