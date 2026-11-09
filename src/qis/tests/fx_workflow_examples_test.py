"""Offline checks for the public FX teaching workflows and provider boundary."""
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    not (ROOT / "examples").is_dir(), reason="Examples are not distributed in the wheel")


def test_bloomberg_example_keeps_unknown_opening_rates(monkeypatch):
    """A provider adapter must not create historical quotes using later observations."""
    monkeypatch.syspath_prepend(str(ROOT))
    from examples.fx import fx_rates_data_bloomberg_example as example

    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    calls = []

    def fetch(tickers, **kwargs):
        """Supply a missing rate head, followed by a percent quote and a missing tail."""
        calls.append(tickers)
        values = [np.nan, 4., np.nan] if tickers is example.DOMESTIC_RATES else [1., 1.1, 1.2]
        return pd.DataFrame({ccy: values for ccy in tickers.values()}, index=dates)

    provider = ModuleType("bbg_fetch")
    provider.fetch_field_timeseries_per_tickers = fetch
    monkeypatch.setitem(sys.modules, "bbg_fetch", provider)
    result = example.fetch_fx_rates_data_from_bloomberg(start_date=dates[0])
    assert result.domestic_rates.iloc[0].isna().all()
    np.testing.assert_allclose(result.domestic_rates.iloc[1:], .04)
    assert example.DOMESTIC_RATES["CAONREPO Index"] == "CAD"
    assert calls == [example.FX_SPOTS, example.DOMESTIC_RATES]
    assert result.fx_spots.USD.eq(1.).all()
    np.testing.assert_allclose(result.fx_spots.GBp, .01 * result.fx_spots.GBP)


@pytest.fixture
def offline_examples(monkeypatch):
    """Refuse network access while executing the real public examples."""
    import socket
    import matplotlib.pyplot as plt

    def refuse(*args, **kwargs):
        """Reject a network dependency in a supposedly offline example."""
        raise AssertionError("Offline examples must not connect to a network")

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield
    plt.close("all")


def test_typed_returns_monthly_cash_and_local_financing(offline_examples):
    """Run independent price, deposit, two-sweep and delayed-settlement checks."""
    from examples.fx.fx_returns_and_accounting_example import run_example

    result = run_example(verbose=False)
    assert round(float(result["account"].nav.iloc[-1]), 6) == 91.520313
    monthly = result["monthly_cash"]
    # Fixture arithmetic independent of the index implementation and example assertions.
    np.testing.assert_allclose(monthly["cash_index"].iloc[-1],
                               (1+.02*31/365)*(1+.08*29/365)*(1+.03*31/365), atol=1e-14)
    assert set(result["local_accounts"]) == {"zero_net", "nonzero_net"}


def test_coverage_workflow_renders_and_preserves_page_one(offline_examples, tmp_path):
    """Exercise actual factsheet routing and render the explicit coverage audit."""
    from examples.factsheets.holdings_alpha_coverage import run_example

    result = run_example()
    reports = result["reports"]
    for pages in reports.values():
        for page in pages:
            page.canvas.draw()
    np.testing.assert_array_equal(np.asarray(reports["strict"][0].canvas.buffer_rgba()),
                                  np.asarray(reports["assisted"][0].canvas.buffer_rgba()))
    for position, page in enumerate(reports["assisted"]):
        if "coverage audit" in page.get_suptitle():
            page.savefig(tmp_path / f"coverage-{position}.png", dpi=120)
