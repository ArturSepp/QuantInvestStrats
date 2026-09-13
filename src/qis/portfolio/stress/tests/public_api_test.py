"""Public portfolio types must be usable identically across stack layers."""

from pathlib import Path
import re

import qis
import qis.portfolio as portfolio
import qis.portfolio.stress as stress


def test_public_portfolio_contract():
    """Canonical, portfolio and root imports refer to the same objects."""
    names = (
        "Underlying",
        "InstrumentLeg",
        "InstrumentType",
        "ResponseBasis",
        "KinkPolicy",
        "HoldingPayoff",
        "PayoffContext",
        "PortfolioHolding",
        "InstrumentPortfolio",
        "PortfolioValuationResult",
        "FactorGroupSpec",
        "ScenarioMode",
        "ShockConvention",
        "StressScenarios",
        "StressTestConfig",
        "PortfolioStressResult",
        "run_portfolio_stress_test",
        "StressReportConfig",
        "StressReportArtifacts",
        "generate_portfolio_stress_report",
    )
    for name in names:
        assert getattr(stress, name) is getattr(portfolio, name)
        assert getattr(stress, name) is getattr(qis, name)
        assert name in qis.__all__
    assert qis.PortfolioData is portfolio.PortfolioData


def test_shipped_public_example_executes():
    """The installed guide contains runnable synthetic inputs, not proposed symbols."""
    path = Path(qis.__file__).parent / "docs" / "portfolio_stress.md"
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
    assert blocks
    exec("\n".join(blocks), {})
