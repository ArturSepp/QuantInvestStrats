"""Tests for the local ETF price-cache runner."""

# packages
import sys
from types import ModuleType

import numpy as np
import pandas as pd

# qis
from qis.run_local import price_data_run


def test_fetch_etf_prices_replaces_placeholder_with_user_cache(tmp_path, monkeypatch) -> None:
    """Use a created machine-local cache when settings still contain the shipped placeholder."""
    local_app_data = tmp_path.joinpath("local-app-data")
    downloaded_prices = pd.DataFrame(
        {"SPY": [100.0, 101.0], "TLT": [90.0, 89.5]},
        index=pd.date_range("2026-01-02", periods=2),
    )
    download_result = pd.concat({"Close": downloaded_prices}, axis=1)
    fake_yfinance = ModuleType("yfinance")
    fake_yfinance.download = lambda **kwargs: download_result

    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(
        price_data_run.local_path,
        "get_paths",
        lambda: {"RESOURCE_PATH": "C:\\Users\\...\\"},
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    price_data_run.run_local(local=price_data_run.Locals.FETCH_ETF_PRICES)

    cache_file = local_app_data.joinpath("qis", "resources", "etf_prices.csv")
    assert cache_file.is_file()
    actual_prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(actual_prices, downloaded_prices, check_freq=False)
    assert np.isfinite(actual_prices.to_numpy()).all()
