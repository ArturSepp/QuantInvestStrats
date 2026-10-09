"""Factor-loss historical selection retains crisis months and complete vectors."""

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress.analytics import StressTestConfig, run_portfolio_stress_test
from qis.portfolio.stress.historical import HistoricalScenarioSelection
from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
from qis.portfolio.stress.portfolio import PortfolioHolding
from qis.portfolio.stress.tests.analytics_test import request


def history(p):
    """Use a long monthly panel with explicit crisis dates and a future shock."""
    h = pd.DataFrame(0., index=pd.date_range("2005-12-31", "2026-09-30", freq="ME"),
                     columns=p.risk_model.factor_covar[p.risk_date].columns)
    h.loc["2005-12-31", "Equity"] = -1.
    h.loc["2008-10-31", ["Equity", "Credit"]] = [-.3, -.1]
    h.loc["2020-03-31", ["Equity", "Credit"]] = [-.2, -.03]
    h.loc["2026-07-31", "Equity"] = .5
    h.loc["2026-09-30", "Equity"] = -2.
    return h


def test_named_factor_ranking_is_independent_of_current_portfolio_pnl(market):
    p = market([PortfolioHolding("short", "Short stock", -100.,
        (InstrumentLeg(InstrumentType.DELTA_1, "actual", -1.),))])
    h = history(p)
    result = run_portfolio_stress_test(p, request(), h,
        config=StressTestConfig(historical_selection=HistoricalScenarioSelection(count=2)))
    dates = pd.to_datetime(["2008-10-31", "2020-03-31"])
    assert result.historical_ranking.index.tolist() == dates.tolist()
    assert result.historical_ranking.portfolio_pnl.gt(0).all()
    pd.testing.assert_frame_equal(result.historical.factor_log_shocks.loc[dates], h.loc[dates])
    assert pd.Timestamp("2005-12-31") not in result.historical.pnl.index
    assert pd.Timestamp("2026-09-30") not in result.historical.pnl.index
    assert result.metadata["historical_count"] == 2
    legacy = run_portfolio_stress_test(p, request(), h, config=StressTestConfig(historical_count=2))
    assert legacy.historical_ranking.index[0] == pd.Timestamp("2026-07-31")


@pytest.mark.parametrize("column,match", [("Equity", "ranking-factor"), ("Credit", "complete")])
def test_missing_crisis_data_raises_instead_of_substituting_a_milder_month(market, column, match):
    p = market([PortfolioHolding("stock", "Stock", 100.,
        (InstrumentLeg(InstrumentType.DELTA_1, "actual", 1.),))])
    h = history(p)
    h.loc["2008-10-31", column] = np.nan
    with pytest.raises(ValueError, match=match):
        run_portfolio_stress_test(p, request(), h,
            config=StressTestConfig(historical_selection=HistoricalScenarioSelection(count=2)))
