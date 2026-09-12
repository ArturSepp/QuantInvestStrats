"""Offline factor stresses, joint conditioning and analytical one-month prediction bands.

Run from the checkout: python -m examples.portfolios.factor_stress_testing
Pass --output-dir <directory> to retain CSV tables and PNG/PDF figures.
Synthetic factor histories come from qis.datasets.synthetic; loadings and positions are
explicit teaching inputs, not fitted or vendor data. No files are written by default.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qis
from qis.datasets.synthetic import generate_synthetic_universe


def build_inputs():
    """Build annual factor covariance and explicit synthetic portfolio inputs."""
    universe = generate_synthetic_universe(
        start="2014-01-02", end="2025-12-31", apply_quirks=False)
    factor_tickers = {
        "SEQ_US": "Equity", "SBD_TSY": "Rates", "SBD_IG": "Credit IG",
        "SBD_HY": "Credit HY", "SCM_GLD": "Gold",
    }
    prices = universe.prices[list(factor_tickers)].rename(columns=factor_tickers)
    covariances = qis.estimate_rolling_ewma_covar(
        prices=prices, returns_freq="W-WED", rebalancing_freq="QE", span=52,
        demean=True, apply_an_factor=True)
    date = max(covariances)
    covariance = covariances[date]
    assets = ["Equity fund", "Balanced fund", "Credit fund", "Gold hedge"]
    betas = pd.DataFrame(
        [[1., 0., 0., 0., 0.], [.45, .50, .20, .10, .05],
         [.10, .25, .65, .35, 0.], [0., 0., 0., 0., 1.]],
        index=assets, columns=covariance.columns)
    amounts = pd.Series([55., 35., 15., -5.], index=assets) * 1_000_000
    residual_variances = pd.Series([.06, .04, .025, .02], index=assets).pow(2)
    return date, covariance, betas, amounts, residual_variances, 100_000_000.


def requested_scenarios(covariance, betas, amounts):
    """Compare isolated shocks with joint conditional co-moves and preserve a zero anchor."""
    anchors = {
        "Equity -20%": {"Equity": qis.return_log_shock(-.20)},
        "Yields -50bp": {"Rates": qis.duration_log_shock(-.005, duration=8.)},
        "Gold 2500 to 3000": {"Gold": qis.price_target_log_shock(2500., 3000.)},
        "Both credit -10%": {
            "Credit IG": qis.return_log_shock(-.10),
            "Credit HY": qis.return_log_shock(-.10)},
        "Equity -20%, rates fixed": {
            "Equity": qis.return_log_shock(-.20), "Rates": 0.},
    }
    direct = pd.DataFrame(0., index=list(anchors), columns=covariance.columns)
    for scenario, values in anchors.items():
        direct.loc[scenario, list(values)] = list(values.values())
    correlated = pd.DataFrame(
        {name: qis.conditional_factor_shock(covariance, values)
         for name, values in anchors.items()}).T
    isolated_pnl = qis.project_factor_scenarios(betas, amounts, direct)
    correlated_pnl = qis.project_factor_scenarios(betas, amounts, correlated)
    return direct, correlated, isolated_pnl, correlated_pnl


def sensitivity_grids(covariance, betas, amounts, residual_variances, nav):
    """Condition each grid jointly and include unanchored factor and residual risk."""
    panels = {
        "Equity": (["Equity"], 30),
        "Rates": (["Rates"], 20),
        "Joint credit": (["Credit IG", "Credit HY"], 20),
        "Gold": (["Gold"], 20),
    }
    return {
        name: qis.compute_factor_sensitivity(
            covariance=covariance, betas=betas, residual_variances=residual_variances,
            amounts=amounts, nav=nav, anchors=anchors,
            grid=pd.Index(np.arange(-limit, limit + 1) / 100., name="anchor_return"),
            horizon_years=1 / 12, confidence=.95)
        for name, (anchors, limit) in panels.items()
    }


def plot_results(comparison, grids):
    """Use qis bars/scatters; shade analytical prediction bands without regression CIs."""
    scenario_fig, scenario_ax = plt.subplots(figsize=(12, 5), layout="constrained")
    qis.plot_bars(
        comparison, stacked=False, title="Synthetic portfolio: direct and correlated shocks",
        yvar_format="{:.1%}", ylabel="Portfolio NAV return", x_rotation=15,
        legend_loc="upper right", ax=scenario_ax)
    sensitivity_fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, (name, result) in zip(axes.flat, grids.items()):
        summary = result.band.summary
        low, high = summary.lower_bound.min(), summary.upper_bound.max()
        margin = .05 * (high - low)
        qis.plot_scatter(
            summary.reset_index(), x="anchor_return", y="portfolio_return",
            full_sample_order=0, markersize=14, add_universe_model_label=False,
            xlabel="Anchored simple factor return", ylabel="Portfolio NAV return",
            title=name, xvar_format="{:.0%}", yvar_format="{:.1%}",
            y_limits=(low - margin, high + margin), ax=ax)
        ax.fill_between(
            summary.index, summary.lower_bound, summary.upper_bound,
            color="steelblue", alpha=.2, zorder=0, label="One-month 95% prediction band")
        ax.set_xticks(np.linspace(summary.index.min(), summary.index.max(), 5))
        ax.legend(loc="best", fontsize=8)
    sensitivity_fig.suptitle(
        "Correlated factor shocks: exact model valuation with analytical risk bands")
    return {"requested_scenarios": scenario_fig, "factor_sensitivities": sensitivity_fig}


def run_example(output_dir=None):
    """Run offline; optionally export figures, full shocks, attribution and band tables."""
    date, covariance, betas, amounts, residual_variances, nav = build_inputs()
    direct, correlated, isolated_pnl, correlated_pnl = requested_scenarios(
        covariance, betas, amounts)
    comparison = pd.DataFrame({
        "Direct": isolated_pnl.asset_pnl.sum(axis=1) / nav,
        "Correlated": correlated_pnl.asset_pnl.sum(axis=1) / nav})
    grids = sensitivity_grids(covariance, betas, amounts, residual_variances, nav)
    # This identity catches dropped holdings or accidentally using net-P&L percentages.
    np.testing.assert_allclose(
        correlated_pnl.factor_attribution.sum(axis=1),
        correlated_pnl.asset_pnl.sum(axis=1), atol=1e-7, rtol=1e-12)
    figures = plot_results(comparison, grids)
    print(f"Synthetic covariance date: {date:%Y-%m-%d}; annualised W-WED EWMA span 52")
    print("Scenario P&L, percent of USD 100m NAV:")
    print((100 * comparison).round(2).to_string())
    print("All attribution reconciles; prediction horizon: one month; confidence: 95%.")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "factor_covariance": covariance, "asset_betas": betas,
            "amounts": amounts, "annual_residual_variances": residual_variances,
            "direct_log_shocks": direct, "correlated_log_shocks": correlated,
            "scenario_nav_returns": comparison, "asset_pnl": correlated_pnl.asset_pnl,
            "factor_attribution_nav": correlated_pnl.factor_attribution / nav,
            "asset_attribution_nav": correlated_pnl.asset_pnl / nav,
        }
        for name, result in grids.items():
            prefix = name.lower().replace(" ", "_")
            tables.update({
                f"{prefix}_summary": result.band.summary,
                f"{prefix}_log_shocks": result.factor_log_shocks,
                f"{prefix}_conditional_covariance": result.band.conditional_factor_covariance,
            })
        for name, table in tables.items():
            table.to_csv(output_dir / f"{name}.csv")
        for name, figure in figures.items():
            figure.savefig(output_dir / f"{name}.png", dpi=150)
            figure.savefig(output_dir / f"{name}.pdf")
    for figure in figures.values():
        plt.close(figure)
    return comparison, grids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    run_example(parser.parse_args().output_dir)
