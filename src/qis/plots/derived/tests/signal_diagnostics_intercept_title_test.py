"""The composite signal-diagnostics figure names the regression that was actually fitted."""
# packages
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# qis
from qis.perfstats.signal_diagnostics import estimate_signal_diagnostics  # noqa: E402
from qis.plots.derived.signal_diagnostics_plot import plot_signal_diagnostics  # noqa: E402


def _inputs():
    rng = np.random.default_rng(2)
    dates = pd.date_range('2018-01-31', periods=48, freq='ME')
    names = [f'A{i:02d}' for i in range(12)]
    returns = pd.DataFrame(0.05 * rng.standard_normal((48, 12)), index=dates, columns=names)
    signal = pd.DataFrame(rng.standard_normal((48, 12)), index=dates, columns=names)
    return returns, signal


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_result_records_the_intercept_choice(fit_intercept):
    returns, signal = _inputs()
    result = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1,),
                                         fit_intercept=fit_intercept)
    assert result.fit_intercept is fit_intercept


@pytest.mark.parametrize('fit_intercept, present, absent', [
    (False, '(no intercept)', r'\alpha'),
    (True, '(with intercept)', '(no intercept)'),
])
def test_default_suptitle_matches_the_fit(fit_intercept, present, absent):
    returns, signal = _inputs()
    result = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1,),
                                         fit_intercept=fit_intercept)
    fig = plot_signal_diagnostics(result)
    try:
        title = fig._suptitle.get_text()
        assert present in title
        assert absent not in title
    finally:
        plt.close(fig)
