"""Legacy URL coverage and build-time redirect-destination contracts."""
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXTENSION = ROOT / "docs" / "_ext" / "qis_indexing.py"
if not EXTENSION.is_file():
    pytest.skip("Documentation sources are not shipped in the wheel.", allow_module_level=True)
SEO = runpy.run_path(str(EXTENSION))


def test_manifest_keeps_reviewed_legacy_and_compiled_entry_points():
    """Preserve retired APIs and handbook fallbacks for documented compiled kernels."""
    manifest = SEO["read_redirect_manifest"]()
    sources = {r["from_url"] for r in manifest["redirects"]}
    names = {
        "LastLabel",
        "OhlcEstimatorType",
        "TrainLivePeriod",
        "TrainLiveSamples",
        "backtest_rebalanced_portfolio",
        "compute_ewm_covar",
        "compute_ewm_covar_newey_west",
        "compute_ewm_covar_tensor",
        "compute_ewm_covar_tensor_vol_norm_returns",
        "compute_ewm_long_short",
        "compute_ewm_matrix_autocorr",
        "compute_path_autocorr",
        "compute_path_autocorr_given_lags",
        "compute_path_corr",
        "compute_path_lagged_corr",
        "compute_path_lagged_corr_given_lags",
        "compute_portfolio_var_np",
        "estimate_hf_ohlc_vol",
        "estimate_ohlc_var",
        "ewm_recursion",
        "limit_weights_to_max_var_limit",
    }
    assert sources == {f"/api/generated/qis.{name}.html" for name in names}
    assert all(r["http_status"] == 301 and r["force"] is False for r in manifest["redirects"])


@pytest.mark.parametrize("change", ["duplicate", "forced", "temporary", "self", "query"])
def test_manifest_rejects_unsafe_or_inconsistent_hosting_rules(tmp_path, change):
    """Reject loops, duplicates, forced redirects and accidental temporary responses."""
    manifest = copy.deepcopy(SEO["read_redirect_manifest"]())
    rule = manifest["redirects"][0]
    if change == "duplicate":
        manifest["redirects"].append(copy.deepcopy(rule))
    elif change == "forced":
        rule["force"] = True
    elif change == "temporary":
        rule["http_status"] = 302
    elif change == "self":
        rule["to_url"] = manifest["site"] + "/en/stable" + rule["from_url"]
    else:
        rule["to_url"] = rule["to_url"].split("#")[0] + "?token=not-allowed"
    file = tmp_path / "redirects.json"
    file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        SEO["read_redirect_manifest"](file)


def test_build_guard_rejects_a_removed_local_target_or_anchor(tmp_path, monkeypatch):
    """A future documentation move must not silently break an existing redirect."""
    manifest = {
        "site": "https://quantinveststrats.readthedocs.io",
        "redirects": [{"to_url": "https://quantinveststrats.readthedocs.io/en/stable/new.html#section"}],
    }
    callback = SEO["validate_redirect_targets"]
    monkeypatch.setitem(callback.__globals__, "read_redirect_manifest", lambda path=None: manifest)
    target = tmp_path / "new.html"
    target.write_text(
        '<html><body><h1 id="section">Replacement</h1></body></html>', encoding="utf-8",
    )
    app = SimpleNamespace(builder=SimpleNamespace(name="html"), outdir=str(tmp_path))
    callback(app, None)
    assert list(tmp_path.iterdir()) == [target]
    target.write_text("<html><body>No preserved anchor</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="anchor"):
        callback(app, None)
    target.unlink()
    with pytest.raises(ValueError, match="destination"):
        callback(app, None)


@pytest.mark.parametrize(
    "builder,error", [("latex", None), ("html", RuntimeError("earlier failure"))],
)
def test_failed_or_non_html_build_does_not_validate_missing_outputs(tmp_path, builder, error):
    """Do not obscure an earlier build failure or require HTML from a different builder."""
    app = SimpleNamespace(builder=SimpleNamespace(name=builder), outdir=str(tmp_path))
    SEO["validate_redirect_targets"](app, error)
