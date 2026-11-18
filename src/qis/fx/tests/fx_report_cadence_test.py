"""Independent sampling contracts for FX report statistics and rendered policies."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.fx.reports import fx_hedging_report as reports


@pytest.fixture(scope="module")
def report_data():
    """Use clean paths from the frozen public generator, with explicitly synthetic FX."""
    panel = generate_synthetic_prices(start="2004-12-31", end="2025-12-31", apply_quirks=False)
    assets = panel[["SEQ_US", "SBD_TSY"]].copy()
    spots = pd.DataFrame({
        "USD": 1.0,
        "CHF": 0.95 * panel["SCM_GLD"] / panel["SCM_GLD"].iloc[0],
        "EUR": 1.15 * panel["SCM_BCOM"] / panel["SCM_BCOM"].iloc[0],
    }, index=panel.index)
    rates = pd.DataFrame({
        "USD": 0.035 + 0.01 * np.sin(np.arange(len(panel)) / 300.0),
        "CHF": 0.01,
        "EUR": 0.02,
    }, index=panel.index)
    return assets, qis.FxRatesData(fx_spots=spots, domestic_rates=rates)


def independent_statistics(prices, benchmark, freq):
    """Sample complete as-of boundaries and use NumPy dispersion/OLS, without QIS stats."""
    annual_factor = {"ME": 12, "QE": 4, "W-WED": 52}[freq]
    dates = pd.date_range(prices.index.min(), prices.index.max(), freq=freq)
    sampled = prices.sort_index().ffill().reindex(dates, method="ffill")
    for column in prices:
        observed = prices[column].dropna()
        sampled.loc[sampled.index > observed.index[-1], column] = np.nan
    log_returns = np.log(sampled).diff()
    simple_returns = sampled.pct_change(fill_method=None)
    result = {}
    for column in prices:
        observed = prices[column].dropna()
        periodic = sampled[column].dropna()
        native_pa = (observed.iloc[-1] / observed.iloc[0]) ** (
            365.25 / (observed.index[-1] - observed.index[0]).days
        ) - 1.0
        sampled_pa = (periodic.iloc[-1] / periodic.iloc[0]) ** (
            365.25 / (periodic.index[-1] - periodic.index[0]).days
        ) - 1.0
        vol = np.std(log_returns[column].dropna().to_numpy(), ddof=1) * np.sqrt(annual_factor)
        joint = simple_returns[[benchmark, column]].dropna()
        design = np.column_stack([np.ones(len(joint)), joint.iloc[:, 0].to_numpy()])
        alpha, beta = np.linalg.lstsq(design, joint.iloc[:, 1].to_numpy(), rcond=None)[0]
        result[column] = {
            qis.PerfStat.PA_RETURN.to_str(): native_pa,
            qis.PerfStat.VOL.to_str(): vol,
            qis.PerfStat.SHARPE_RF0.to_str(): sampled_pa / vol,
            qis.PerfStat.BETA.to_str(): beta,
            qis.PerfStat.ALPHA_AN.to_str(): annual_factor * alpha,
            qis.PerfStat.MAX_DD.to_str(): (observed / observed.cummax() - 1.0).min(),
        }
    return pd.DataFrame.from_dict(result, orient="index")


@pytest.mark.parametrize("freq", ["ME", "QE", "W-WED"])
@pytest.mark.parametrize("filtered", [False, True])
def test_multi_asset_statistics_use_requested_boundaries(report_data, monkeypatch, freq, filtered):
    """Every reported strategy matches independent risk, return, OLS and drawdown calculations."""
    assets, fx = report_data
    original = qis.compute_ra_perf_table_with_benchmark
    captures = []

    def capture(**kwargs):
        """Keep the actual NAVs and calculated table for the independent reference."""
        table = original(**kwargs)
        captures.append((kwargs, table))
        return table

    monkeypatch.setattr(qis, "compute_ra_perf_table_with_benchmark", capture)
    period = qis.TimePeriod("13Feb2011", "17Oct2025") if filtered else None
    out = reports.compute_multi_asset_fx_hedging(
        assets, fx, local_ccys="USD", reference_ccy="CHF", freq=freq, time_period=period
    )
    for asset, (kwargs, actual) in zip(assets.columns, captures):
        expected = independent_statistics(kwargs["prices"], "LocalAsset", freq)
        np.testing.assert_allclose(actual[expected.columns], expected, rtol=2e-12, atol=2e-14)
        for key, metric in [("pas", qis.PerfStat.PA_RETURN), ("vols", qis.PerfStat.VOL),
                            ("sharpes", qis.PerfStat.SHARPE_RF0), ("betas", qis.PerfStat.BETA)]:
            np.testing.assert_allclose(out[key].loc[asset], expected[metric.to_str()],
                                       rtol=2e-12, atol=2e-14)
        policy = kwargs["perf_params"]
        assert policy.freq_vol == policy.freq_reg == freq
        assert policy.freq_drawdown == "D"
        assert policy.rates_data is None
    assert len(captures) == len(assets.columns)


@pytest.mark.parametrize("freq", ["ME", "QE", "W-WED"])
def test_single_asset_table_and_panels_share_policy(report_data, monkeypatch, freq):
    """The rendered table, Sharpe bars and NAV/DD panels use the same explicit policy."""
    assets, fx = report_data
    captures = {}

    def spy(name):
        """Wrap a real plot call without replacing its statistics or rendering."""
        original = getattr(qis, name)

        def capture(*args, **kwargs):
            """Record the actual policy and rendered result."""
            result = original(*args, **kwargs)
            captures[name] = (kwargs, result)
            return result

        monkeypatch.setattr(qis, name, capture)

    for name in ("plot_ra_perf_table_benchmark", "plot_prices_with_dd", "plot_ra_perf_bars"):
        spy(name)
    fig = reports.run_asset_fx_hedging_report(assets.iloc[:, 0], fx, freq=freq)
    try:
        fig.canvas.draw()
        table_kwargs, (_, rendered) = captures["plot_ra_perf_table_benchmark"]
        policy = table_kwargs.get("perf_params")
        assert policy is not None, "The table must receive the report's explicit cadence policy"
        assert policy.freq_vol == policy.freq_reg == freq
        assert policy.freq_drawdown == "D"
        assert policy.rates_data is None
        assert policy.return_type == qis.ReturnTypes.LOG
        assert policy.freq_skewness == "ME"
        for name in ("plot_prices_with_dd", "plot_ra_perf_bars"):
            assert captures[name][0]["perf_params"] is policy
        assert captures["plot_ra_perf_bars"][0]["perf_column"] == qis.PerfStat.SHARPE_RF0
        actual = qis.compute_ra_perf_table_with_benchmark(
            table_kwargs["prices"], benchmark=assets.columns[0], perf_params=policy
        )
        expected = independent_statistics(table_kwargs["prices"], assets.columns[0], freq)
        np.testing.assert_allclose(actual[expected.columns], expected, rtol=2e-12, atol=2e-14)
        cells = {cell.get_text().get_text() for cell in fig.axes[0].tables[0].get_celld().values()}
        assert set(rendered.to_numpy().ravel()).issubset(cells)
        assert f"Statistics: {freq}" in fig._suptitle.get_text()
        assert f"Asset hedge resets/estimation: {freq}" in fig._suptitle.get_text()
        assert "daily resets" in fig._suptitle.get_text()
        assert "drawdowns: D" in fig._suptitle.get_text()
        assert "Skewness: ME" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("freq", ["ME", "QE", "W-WED"])
def test_multi_asset_rendered_tables_match_statistics(report_data, freq):
    """The user-facing comparison renders the full-precision result with its cadence label."""
    assets, fx = report_data
    out = reports.compute_multi_asset_fx_hedging(assets, fx, freq=freq)
    fig = reports.plot_multi_asset_fx_hedging_report(assets, fx, freq=freq)
    try:
        fig.canvas.draw()
        for ax, key, fmt in zip(fig.axes, ("pas", "vols", "sharpes"), (".1%", ".1%", ".2f")):
            cells = {cell.get_text().get_text() for cell in ax.tables[0].get_celld().values()}
            expected = {format(value, fmt) for value in out[key].to_numpy().ravel()}
            assert expected.issubset(cells)
        assert f"Statistics: {freq}" in fig._suptitle.get_text()
        assert f"Asset hedge resets/estimation: {freq}" in fig._suptitle.get_text()
        assert "rf=0" in fig.axes[2].get_title()
    finally:
        plt.close(fig)


def test_global_perf_params_defaults_are_unchanged():
    """FX reports must not alter statistics defaults for unrelated callers."""
    policy = qis.PerfParams()
    assert policy.freq_vol == "ME"
    assert policy.freq_reg == "QE"
    assert policy.freq_drawdown == "D"
    assert policy.rates_data is None
