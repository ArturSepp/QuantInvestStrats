"""The marginal-active-risk runner reconciles its contributions with tracking error.

``calculate_marginal_active_risk`` returns the gradient of active variance, m = 2 S d, so
sum_i d_i m_i = 2 TE^2. The runner must divide d_i m_i by 2 TE to report contributions that sum
to TE, and its verification must print True.
"""

import pytest

contributions_run = pytest.importorskip(
    "qis.portfolio.risk.run_local.contributions_run",
    reason="development runners are intentionally excluded from installed wheels",
)


def test_runner_contributions_sum_to_tracking_error(capsys: pytest.CaptureFixture) -> None:
    """The printed verification holds and the percentage contributions sum to 100%.

    Args:
        capsys: Pytest fixture capturing the runner's printed output.
    """
    contributions_run.run_local(local=contributions_run.Locals.MARGINAL_ACTIVE_RISK)
    out = capsys.readouterr().out

    assert 'Risk contributions sum correctly: True' in out
    assert 'Risk decomposition verified: True' in out
    assert 'Sum of percentage contributions: 100.00%' in out
