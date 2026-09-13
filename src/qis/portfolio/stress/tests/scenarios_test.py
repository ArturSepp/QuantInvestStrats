"""Factor-family splitting and joint conditioning are different numerical operations."""

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.risk.factor_groups import FactorGroupSpec
from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
from qis.portfolio.stress.portfolio import PortfolioHolding
from qis.portfolio.stress.scenarios import ScenarioMode, ShockConvention, StressScenarios


def grouped_portfolio(market):
    """Attach generic family metadata to a fitted model, without a MATF dependency."""
    h = PortfolioHolding(
        "fund", "Fund", 100.0, (InstrumentLeg(InstrumentType.DELTA_1, "proxy_quote", 1.0),)
    )
    p = market([h])
    model = replace(
        p.risk_model,
        factor_groups={"credit_family": FactorGroupSpec("credit_family", ("Credit", "Credit EM"))},
    )
    return replace(p, risk_model=model)


def test_credit_total_simple_bump_split_precedes_log_and_joint_conditioning(market):
    """A -10% family bump produces two -5% returns, then one joint covariance solve."""
    p = grouped_portfolio(market)
    request = StressScenarios(
        pd.DataFrame({"credit_family": [-0.1]}, index=["Credit down"]),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    z = request.resolve(p.risk_model, p.risk_date).iloc[0]
    member = math.log(0.95)
    assert z.Credit == pytest.approx(member)
    assert z["Credit EM"] == pytest.approx(member)
    assert z.Equity == pytest.approx((0.006 + 0.002) / 0.014 * member)
    assert z.FX == pytest.approx(0.0)
    # This must differ from splitting the logarithm of a total 10% move.
    assert z.Credit != pytest.approx(math.log(0.9) / 2.0)
    grid = StressScenarios(
        pd.DataFrame(
            {"credit_family": [-0.1, 0.0, 0.1]},
            index=pd.Index([-0.1, 0.0, 0.1], name="Total Credit bump"),
        ),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    resolved = grid.resolve(p.risk_model, p.risk_date)
    assert resolved.index.equals(grid.anchors.index)
    np.testing.assert_allclose(resolved.loc[-0.1], z)


def test_group_exposure_sum_differs_from_split_bump_derivative(market):
    """Economic exposure aggregation is not the sensitivity to an allocated unit bump."""
    p = grouped_portfolio(market)
    x = p.response_jacobian().sum()
    groups = p.risk_model.compute_factor_group_exposures_at_date(x, p.risk_date)
    assert groups.loc["credit_family", "exposure_sum"] == 200.0
    assert groups.loc["credit_family", "split_bump_exposure"] == 100.0


def test_explicit_zero_and_full_vectors_preserved(market):
    """Conditional completion honors actual zero anchors and fully specified rows."""
    p = grouped_portfolio(market)
    f = p.risk_model.factor_loadings[p.risk_date].columns
    request = StressScenarios(
        pd.DataFrame([[-0.2, 0.0, 0.03, -0.01]], columns=f), ScenarioMode.CONDITIONAL
    )
    pd.testing.assert_frame_equal(request.resolve(p.risk_model, p.risk_date), request.anchors)
    partial = StressScenarios(
        pd.DataFrame({"Equity": [-0.2], "FX": [0.0]}), ScenarioMode.CONDITIONAL
    )
    assert partial.resolve(p.risk_model, p.risk_date).iloc[0].FX == 0.0


@pytest.mark.parametrize(
    "anchors",
    [
        {"credit_family": [-0.1], "Credit": [0.0]},
        {"unknown": [0.0]},
        {"Credit": [-1.0]},
        {"Credit": [np.inf]},
        {"Credit": [np.nan]},
    ],
)
def test_conflicting_unknown_or_invalid_anchors_fail(market, anchors):
    """Unknown or duplicate economic instructions cannot quietly double-count a bump."""
    p = grouped_portfolio(market)
    with pytest.raises(ValueError):
        StressScenarios(pd.DataFrame(anchors), convention=ShockConvention.SIMPLE).resolve(
            p.risk_model, p.risk_date
        )


@pytest.mark.parametrize(
    "group",
    [
        FactorGroupSpec("Credit", ("Credit", "Credit EM")),
        FactorGroupSpec("credit_family", ("Credit", "unknown")),
    ],
)
def test_risk_model_rejects_ambiguous_or_unknown_families(market, group):
    """Factor membership stays valid at the canonical model boundary."""
    p = grouped_portfolio(market)
    with pytest.raises(ValueError):
        replace(p.risk_model, factor_groups={group.group_id: group})


@pytest.mark.parametrize(
    "members,weights",
    [
        ((), None),
        (("a", "a"), None),
        (("a", "b"), (0.2, 0.2)),
        (("a", "b"), (-1.0, 2.0)),
        (("a", "b"), (np.nan, np.nan)),
    ],
)
def test_invalid_group_weights_not_silently_normalized(members, weights):
    """One total bump has explicit nonnegative allocation weights summing to one."""
    with pytest.raises(ValueError):
        FactorGroupSpec("group", members, weights)


def test_price_target_can_pin_conditional_completion_on_both_comparison_pages(market):
    """Generic row policy preserves a caller's conditional level-target interpretation."""
    p = grouped_portfolio(market)
    request = StressScenarios(
        pd.DataFrame({"Equity": [-0.2, -0.2]}, index=["return", "level target"]),
        scenario_modes={"level target": ScenarioMode.CONDITIONAL},
    )
    first = request.resolve(p.risk_model, p.risk_date, ScenarioMode.INDEPENDENT)
    second = request.resolve(p.risk_model, p.risk_date, ScenarioMode.CONDITIONAL)
    assert first.loc["return", "Credit"] == 0.0
    assert first.loc["level target", "Credit"] != 0.0
    pd.testing.assert_series_equal(first.loc["level target"], second.loc["level target"])
