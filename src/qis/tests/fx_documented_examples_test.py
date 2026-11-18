"""Execute the FX README, source runner and provider examples with supplied offline inputs."""
import importlib
import re
import socket
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_prices

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    not (ROOT / "examples").is_dir(), reason="Repository examples are not shipped in the wheel")


@pytest.fixture
def supplied_inputs(monkeypatch, tmp_path):
    """Supply deterministic Yahoo/Bloomberg responses while refusing all network access."""
    panel = generate_synthetic_prices(start="2001-12-31", apply_quirks=False)

    def refusal(*args, **kwargs):
        """Reject external data/configuration access from the controlled workflows."""
        raise AssertionError("This FX example must use supplied data only")

    def download(tickers, **kwargs):
        """Preserve Yahoo's Close MultiIndex and the annual-percent rate quote."""
        names = tickers.split() if isinstance(tickers, str) else list(tickers)
        close = pd.DataFrame({
            name: panel.iloc[:, sum(map(ord, name)) % panel.shape[1]] for name in names
        })
        if "^IRX" in close:
            close["^IRX"] = 4.0
        return pd.concat({"Close": close}, axis=1)

    def fetch(tickers, **kwargs):
        """Return synthetic prices, spots or annual-percent rates by requested labels."""
        currencies = {"USD", "EUR", "GBP", "CHF", "JPY", "AUD", "CAD", "NZD", "HKD",
                      "SGD", "NOK", "SEK"}
        if set(tickers.values()).issubset(currencies):
            is_spot = all("USD Curncy" in ticker for ticker in tickers)
            return pd.DataFrame({
                name: (panel["SBD_TSY"] / 100 if is_spot else 4.0)
                for name in tickers.values()
            }, index=panel.index)
        return pd.DataFrame({name: panel.iloc[:, position % panel.shape[1]]
                             for position, name in enumerate(tickers.values())})

    vendor = ModuleType("yfinance")
    vendor.download = download
    bbg = ModuleType("bbg_fetch")
    bbg.fetch_field_timeseries_per_tickers = fetch
    monkeypatch.setitem(sys.modules, "yfinance", vendor)
    monkeypatch.setitem(sys.modules, "bbg_fetch", bbg)
    monkeypatch.setattr(socket.socket, "connect", refusal)
    monkeypatch.setattr(socket, "create_connection", refusal)
    monkeypatch.setattr(socket, "getaddrinfo", refusal)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    spots = pd.DataFrame({"USD": 1.0, "CHF": panel["SBD_TSY"] / 100}, index=panel.index)
    rates = pd.DataFrame({"USD": .04, "CHF": .02}, index=panel.index)
    assets = panel.iloc[:, :4].copy()
    assets.columns = ["Equities", "Govt Bonds", "IG Bonds", "HY Bonds"]
    yield assets, spots, rates, fetch, vendor
    plt.close("all")


def test_fx_readme_quick_start_needs_no_private_files(supplied_inputs, monkeypatch):
    """The displayed quick start runs offline and matches independent EUR terminal wealth."""
    text = (ROOT / "src/qis/fx/README.md").read_text(encoding="utf-8")
    quick_start = text.split("## Quick start", 1)[1]
    blocks = re.findall(r"```python\s*\n(.*?)```", quick_start, re.S)
    assert blocks, "The README must keep an executable quick start"

    def no_files(*args, **kwargs):
        """Prohibit hidden CSV/YAML requirements in the advertised offline quick start."""
        raise AssertionError("The FX quick start must not load private data")

    monkeypatch.setattr(qis.FxRatesData, "load", no_files)
    namespace = {}
    exec(compile("\n".join(blocks), "FX README quick start", "exec"), namespace)
    result = namespace["returns"]
    assert result.iloc[0].isna().all()
    usd_cash = 1.0 + .05 * 29 / 365
    eur_cash = 1.0 + .03 * 29 / 365
    forward = 1.10 * usd_cash / eur_cash
    closing_wealth = 102.0 * 1.12 + 100.0 * (forward - 1.12)
    expected = np.log(closing_wealth / 110.0) - np.log(usd_cash)
    np.testing.assert_allclose(result.loc["2024-02-29", "EU_equity"], expected, atol=1e-14)
    np.testing.assert_allclose(expected, .017746756654550708, atol=1e-14)
    assert namespace["results"]["ME"].normalization is None


def test_runner_quarter_end_uses_latest_prior_quote(supplied_inputs, monkeypatch):
    """Weekend quarter-ends preserve Friday observations instead of the previous quarter."""
    assets, spots, rates, fetch, _ = supplied_inputs
    runner = importlib.import_module("qis.fx.run_local.fx_rates_data_run")
    monkeypatch.setattr(runner.lp, "get_resource_path", lambda: "supplied")
    monkeypatch.setattr(runner, "load_fx_rates_data", lambda **kwargs: (spots, rates))
    captured = []
    monkeypatch.setattr(qis, "plot_prices_with_dd", lambda **kwargs: captured.append(kwargs))
    runner.run_local(runner.Locals.CHECK_CHF)
    assert len(captured) == 2
    pairs = [{"LGCPTRUH Index": "IG USD", "LGCPTRCH Index": "IG CHF Hedged"},
             {"NDUEACWF Index": "ACWI USD", "MEWD Index": "ACWI CHF"}]
    for capture, pair in zip(captured, pairs):
        source = fetch(pair)
        actual = capture["prices"][list(pair.values())]
        expected = pd.DataFrame({column: [source.loc[:date, column].dropna().iloc[-1]
                                          for date in actual.index] for column in source},
                                index=actual.index)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert pd.Timestamp("2023-12-31") in actual.index
        np.testing.assert_array_equal(actual.loc["2023-12-31"], source.loc["2023-12-29"])
        assert actual.notna().all().all()


@pytest.mark.parametrize("module", ["qis.fx.run_local.fx_rates_data_run",
                                    "examples.fx.fx_hedging_example"])
def test_local_rate_example_computes_each_request_once(supplied_inputs, monkeypatch, module):
    """Duplicated printing must not hide a second redundant pair of FX requests."""
    assets, spots, rates, _, _ = supplied_inputs
    example = importlib.import_module(module)
    if "run_local" in module:
        monkeypatch.setattr(example.lp, "get_resource_path", lambda: "supplied")
        monkeypatch.setattr(example, "_load_runner_data",
                            lambda **kwargs: (assets.copy(), spots, rates))
    else:
        monkeypatch.setattr(example, "load_example_inputs",
                            lambda *args, **kwargs: (spots, rates, assets.copy()))
    from qis.fx import fx_returns
    original = qis.compute_fx_returns
    calls = []

    def capture(data, request):
        """Execute the real request and record its economic kind."""
        calls.append(request.return_type)
        return original(data, request)

    monkeypatch.setattr(qis, "compute_fx_returns", capture)
    monkeypatch.setattr(fx_returns, "compute_fx_returns", capture)
    example.run_local(example.Locals.LOCAL_RATE_ADJUSTMENT)
    assert calls == [qis.FxReturnType.LOCAL_TOTAL, qis.FxReturnType.LOCAL_EXCESS]


@pytest.mark.parametrize("module", [
    "fx_rates_data_yahoo_example", "fx_rates_data_bloomberg_example",
    "fx_hedging_yahoo_example", "fx_hedging_example",
])
def test_provider_workflow_dispatchers_with_supplied_data(supplied_inputs, monkeypatch, module):
    """Exercise every public example case against real analytics and synthetic vendor inputs."""
    vendor = supplied_inputs[-1]
    example = importlib.import_module(f"examples.fx.{module}")
    # Other tests may already have imported the module with their own synthetic provider.
    yahoo = importlib.import_module("examples.fx.fx_rates_data_yahoo_example")
    monkeypatch.setattr(yahoo, "yf", vendor)
    if hasattr(example, "yf"):
        monkeypatch.setattr(example, "yf", vendor)
    if module.startswith("fx_rates_data"):
        assert "FX_FORWARD_INDICES" in example.Locals.__members__
    for case in example.Locals:
        example.run_local(case)
        for figure_number in plt.get_fignums():
            plt.figure(figure_number).canvas.draw()
        plt.close("all")
