"""Calendar-consistent hedge costs and frequency-consistent risk initialisation."""

import numpy as np
import pandas as pd
import pytest
import qis
from qis.fx import fx_hedging


@pytest.mark.parametrize("freq,start", [("ME", "2024-01-31"), ("QE", "2023-12-31"),
                                        ("B", "2024-03-01"), ("YE", "2023-12-31")])
def test_cost_annualises_the_priced_interval(monkeypatch, freq, start):
    """Independent covered-interest delivery prices include leap days and weekends."""
    dates = pd.date_range(start, periods=5, freq=freq)
    boundaries = pd.date_range(start, periods=6, freq=freq)
    tau = pd.Series(np.diff(boundaries.values) / np.timedelta64(1, "D") / 365, index=dates)
    spots = pd.DataFrame({"EUR": 1.1, "USD": 1.0}, index=dates)
    rates = pd.DataFrame({"EUR": 0.07, "USD": 0.02}, index=dates)
    fx = qis.FxRatesData(spots, rates)
    premium = fx.get_forward_rate_for_local_ccy("EUR", "USD", freq=freq)
    def known_risk(**kwargs):
        """Hold risk fixed so the oracle measures only cash day count."""
        return pd.Series(0.2, index=dates), pd.Series(0.3, index=dates)
    monkeypatch.setattr(fx_hedging, "compute_fx_vol_beta", known_risk)
    _, carry, _ = fx_hedging.compute_fx_optimal_hedge(
        pd.Series(100., index=dates), spots.EUR, premium, freq=freq,
        risk_aversion_lambda=2., min_max_hedge=None)
    delivery = (1 + 0.02 * tau) / (1 + 0.07 * tau)
    expected = 1 - (1 - delivery) / tau / (4 * 0.2**2)
    np.testing.assert_allclose(carry, expected, atol=2e-13, rtol=0)


@pytest.mark.parametrize("freq", ["B", "ME", "QE", "YE"])
def test_annual_volatility_seed_is_eight_percent(monkeypatch, freq):
    """A flat path starts at the same annual risk prior on every supported grid."""
    dates = pd.date_range("2020-12-31", periods=15, freq=freq)
    def flat_returns(**kwargs):
        """Supply finite returns to isolate the variance prior from missing-price policy."""
        return pd.Series(0., index=dates), pd.Series(0., index=dates)
    monkeypatch.setattr(fx_hedging, "compute_local_and_fx_return", flat_returns)
    vol, _ = qis.compute_fx_vol_beta(pd.Series(100., index=dates),
                                    pd.Series(1.1, index=dates), freq=freq)
    np.testing.assert_allclose(vol, 0.08 * (35 / 37) ** (0.5 * np.arange(1, 16)),
                               rtol=0, atol=1e-14)


def test_account_rejects_unimplemented_monthly_cash_policy():
    """The recorded policy must describe the actual valuation-grid cash ledger."""
    with pytest.raises(ValueError, match="PERIOD_OPENING"):
        qis.FxAccountSpec("USD", 100., "local", "fixed", "reference",
                          cash_policy=qis.FxCashPolicy(accrual=qis.FxCashAccrual.MONTHLY_INDEX))


def test_mixed_currency_plot_keeps_all_foreign_assets():
    """The plotting wrapper accepts the same asset-currency mapping as the computation."""
    import matplotlib.pyplot as plt
    dates = pd.date_range("2020-12-31", periods=48, freq="ME")
    t = np.arange(len(dates))
    prices = pd.DataFrame({"US asset": 100*np.exp(.01*t),
                           "EU asset": 100*np.exp(.008*t)}, index=dates)
    spots = pd.DataFrame({"USD": 1., "EUR": 1.1*np.exp(.01*np.sin(t)),
                          "CHF": 1.05*np.exp(.02*np.cos(t))}, index=dates)
    rates = pd.DataFrame({"USD": .04, "EUR": .03, "CHF": .01}, index=dates)
    fig = qis.plot_multi_asset_fx_hedging_report(
        prices, qis.FxRatesData(spots, rates),
        local_ccy=pd.Series({"US asset": "USD", "EU asset": "EUR"}), reference_ccy="CHF")
    try:
        assert len(fig.axes) == 3
        for ax in fig.axes:
            cells = [cell.get_text().get_text() for table in ax.tables
                     for cell in table.get_celld().values()]
            assert "US asset" in cells and "EU asset" in cells
    finally:
        plt.close(fig)
