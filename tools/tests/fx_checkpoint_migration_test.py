"""Offline migration preserves nested typed FX data and original archive bytes."""

import hashlib
import pickle
from pathlib import Path
import runpy

import pandas as pd
import pytest
import qis


MIGRATE = runpy.run_path(str(Path(__file__).parents[1] / "migrate_fx_checkpoint.py"))[
    "migrate_checkpoint"
]


def test_nested_data_migration_preserves_values_and_records_paths(tmp_path):
    """Rewritten class identities preserve panels, enums, policies and normalization."""
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    fx = qis.FxRatesData(
        pd.DataFrame({"USD": 1.0, "CHF": [1.1, 1.2, 1.0]}, index=dates),
        pd.DataFrame({"USD": 0.05, "CHF": 0.03}, index=dates),
    )
    request = qis.FxCurrencyReturnsRequest(("CHF",), "USD")
    payload = {
        "fx": fx,
        "factors": qis.FactorsData(fx.fx_spots),
        "request": request,
        "result": qis.compute_fx_returns(fx, request)["ME"],
    }
    raw = pickle.dumps(payload, protocol=0)
    for new, old in [
        (b"qis.fx.fx_rates_data", b"qis.market_data.fx_rates_data"),
        (b"qis.fx.fx_returns", b"qis.market_data.fx_returns"),
        (b"qis.portfolio.risk.factors_data", b"qis.market_data.factors_data"),
    ]:
        raw = raw.replace(new, old)
    source, target = tmp_path / "old.pkl", tmp_path / "new.pkl"
    source.write_bytes(raw)
    receipt = MIGRATE(source, target)
    actual = pickle.loads(target.read_bytes())
    assert source.read_bytes() == raw
    assert receipt["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["destination_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert {row["class"] for row in receipt["remapped"]} >= {
        "FxRatesData",
        "FactorsData",
        "FxCurrencyReturnsRequest",
        "FxReturnsData",
    }
    pd.testing.assert_frame_equal(actual["fx"].fx_spots, fx.fx_spots, check_exact=True)
    pd.testing.assert_frame_equal(actual["fx"].domestic_rates, fx.domestic_rates, check_exact=True)
    pd.testing.assert_frame_equal(actual["factors"].get_prices(), fx.fx_spots, check_exact=True)
    pd.testing.assert_frame_equal(
        actual["result"].values, payload["result"].values, check_exact=True
    )
    assert actual["request"] == request
    assert actual["fx"].__class__.__module__ == "qis.fx.fx_rates_data"
    assert target.with_suffix(".pkl.migration.json").exists()
    with pytest.raises(FileExistsError):
        MIGRATE(source, target)
    with pytest.raises(FileExistsError):
        MIGRATE(source, source)


def test_unknown_retired_identity_fails_without_output(tmp_path):
    """An unmapped old callable cannot be silently redirected to different economics."""
    source, target = tmp_path / "old.pkl", tmp_path / "new.pkl"
    source.write_bytes(b"cqis.market_data.fx_rates_data\nUnknownClass\n.")
    with pytest.raises(pickle.UnpicklingError, match="Unmapped retired"):
        MIGRATE(source, target)
    assert not target.exists()
