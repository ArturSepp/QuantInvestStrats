"""Single observed-factor stresses retain native points and conditional residual risk."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from qis.perfstats.config import ReturnTypes
from qis.portfolio.risk.risk_model import RiskModel
from qis.portfolio.stress.analytics import run_portfolio_stress_test, StressTestConfig
from qis.portfolio.stress.historical import HistoricalScenarioSelection
from qis.portfolio.stress.instruments import (
    InstrumentLeg, InstrumentType, Underlying, ResponseBasis,
)
from qis.portfolio.stress.portfolio import PortfolioHolding
from qis.portfolio.stress.scenarios import StressScenarios, ShockConvention
from qis.portfolio.stress.reporting import StressReportConfig
from qis.portfolio.stress._figures import report_pages
import matplotlib.pyplot as plt


def test_single_native_factor_end_to_end(market):
    """Five VIX points yield the exact funded payoff; every report page renders."""
    date = pd.Timestamp('2026-08-31')
    beta = pd.DataFrame({'VIX':[-.002]}, index=['stock'])
    cov = pd.DataFrame([[100.]],index=['VIX'],columns=['VIX'])
    residual = pd.Series([.0001],index=['stock'])
    model = RiskModel({date:beta@cov@beta.T+np.diag(residual)},
        {date:beta},{date:cov},{date:residual},factor_return_types={'VIX':ReturnTypes.DIFFERENCE})
    holding = PortfolioHolding('fund', 'Fund', 100.,
                               (InstrumentLeg(InstrumentType.DELTA_1, 'actual', 1.),))
    portfolio = market([holding],model_override=model, underlyings={
        'actual':Underlying('actual',100.,'USD','stock',ResponseBasis.REFERENCE)})
    shocks = StressScenarios(pd.DataFrame({'VIX':[0.,5.,10.]},index=['flat','up5','up10']),
        convention=ShockConvention.MODEL)
    history = pd.DataFrame({'VIX':[3.,-4.,8.,1.,-1.,2.,1.,-2.]},
        index=pd.date_range('2026-01-31',periods=8,freq='ME'))
    selection = HistoricalScenarioSelection('VIX','2026-01-01',2,ascending=False,
        return_type=ReturnTypes.DIFFERENCE)
    result = run_portfolio_stress_test(portfolio,shocks,history,
        config=StressTestConfig(historical_selection=selection))
    np.testing.assert_allclose(result.valuations['requested'].portfolio_pnl,
                                100*np.expm1(-.002*np.array([0,5,10])))
    pd.testing.assert_frame_equal(result.valuations['requested'].factor_log_shocks,
                                  result.valuations['conditional'].factor_log_shocks)
    with pytest.raises(ValueError,match='DIFFERENCE|native|MODEL'):
        replace(shocks,convention=ShockConvention.SIMPLE).resolve(model,date)
    titles=[]
    for title,fig in report_pages(result,StressReportConfig(model_name='VIX',portfolio_r2=.5)):
        titles.append(title)
        fig.canvas.draw()
        plt.close(fig)
    assert any('methodology' in s for s in titles)
    assert not any('co-moves' in s for s in titles)
