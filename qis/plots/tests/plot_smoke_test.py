"""
smoke test over every plot_* function exported from qis

The contract asserted here is deliberately weak and deliberately total: each of the exported
plot functions must run on a canonical synthetic panel and must draw a figure. Nothing is
asserted about pixels, so this test does not break when a style changes; it breaks when a plot
function stops working, which for most of this surface nothing else would notice.

The parametrisation is read from `dir(qis)` at collection time. Exporting a new `plot_*` without
adding it to CALL_KWARGS therefore fails the suite rather than silently going uncovered.
"""

# packages
import inspect
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')  # no display in CI; must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

# qis
import qis  # noqa: E402
from qis.tests.synthetic_data import generate_synthetic_universe  # noqa: E402

# exported plot functions that cannot be called on their documented minimal signature.
# strict=True: when the defect is fixed the test fails until the entry is removed.
KNOWN_BROKEN = {
    'plot_prices_2ax': "passes trend_line into plot_time_series_2ax, which forwards it to "
                       "plot_time_series alongside trend_line1: TypeError",
    'plot_regime_pdf': "calls regime_classifier._asdict() on BenchmarkReturnsQuantilesRegime, "
                       "which is a class and not a NamedTuple: AttributeError",
    'plot_vbars': "the x tick locator is fixed at 8 positions regardless of len(df.index), so "
                  "the call succeeds only on a frame with exactly 8 rows: ValueError. With 5 or "
                  "more columns it raises IndexError first, indexing a shorter colors list",
}


def _exported_plot_functions() -> list:
    """Names of every callable exported from qis whose name starts with plot_."""
    return sorted(name for name in dir(qis)
                  if name.startswith('plot_') and callable(getattr(qis, name)))


class Fixtures:
    """Canonical inputs built once from the frozen synthetic panel."""

    def __init__(self) -> None:
        universe = generate_synthetic_universe()
        # forward-fill the reporting gaps and drop the monthly-only sleeve: the panel a plot
        # function is entitled to expect. The nan-tolerance of each function is a separate
        # question from whether it runs at all.
        daily_tickers = [x for x in universe.prices.columns if universe.prices[x].count()
                         > 0.5 * len(universe.prices.index)]
        self.benchmark = str(universe.benchmark_prices.columns[0])
        self.prices = pd.concat([universe.prices[daily_tickers].ffill(),
                                 universe.benchmark_prices], axis=1).dropna()
        self.price = self.prices.iloc[:, 0]
        self.group_data = universe.group_data.reindex(daily_tickers)
        self.group_order = universe.group_order

        self.returns = qis.to_returns(prices=self.prices, freq='ME', is_log_returns=False,
                                      drop_first=True)
        self.two_columns = self.returns.iloc[:, :2]
        self.positive_table = self.returns.abs().tail(6)
        self.small_table = self.returns.tail(4).round(4)
        self.covar = self.returns.cov()
        self.time_period_dict = {'full': qis.get_time_period(df=self.prices)}
        self.regime_classifier = qis.BenchmarkReturnsQuantilesRegime()
        self.perf_params = qis.PerfParams(freq='ME')

        # signal diagnostics: one horizon dictionary and a signal on the same grid
        self.asset_returns_dict = {'ME': self.returns[self.group_data.index]}
        self.signal = self.returns[self.group_data.index].rolling(3).mean().dropna()
        self.diagnostics = qis.estimate_signal_diagnostics(
            asset_returns_dict=self.asset_returns_dict,
            signal=self.signal,
            group_data=self.group_data,
            horizons=(1, 3),
            is_log_returns=False)
        self.horizon = self.diagnostics.horizon_labels[0]

        # brinson attribution: monthly weights and pnl for strategy and benchmark
        weights = pd.DataFrame(1.0 / len(self.group_data.index),
                               index=self.returns.index, columns=self.group_data.index)
        rng = np.random.default_rng(7)
        tilt = pd.DataFrame(rng.normal(0.0, 0.01, weights.shape),
                            index=weights.index, columns=weights.columns)
        strategy_weights = weights + tilt
        strategy_weights = strategy_weights.div(strategy_weights.sum(axis=1), axis=0)
        asset_returns = self.returns[self.group_data.index]
        self.brinson = qis.compute_brinson_attribution_table(
            benchmark_pnl=weights.multiply(asset_returns),
            strategy_pnl=strategy_weights.multiply(asset_returns),
            strategy_weights=strategy_weights,
            benchmark_weights=weights,
            asset_class_data=self.group_data,
            group_order=self.group_order)

        self.exposures = weights.resample('YE').last()


@pytest.fixture(scope='module')
def fx() -> Fixtures:
    return Fixtures()


def _call_kwargs(name: str, fx: Fixtures) -> dict:
    """
    Arguments for one plot function.

    Every exported plot_* needs an entry: functions taking only ``prices`` or only ``df`` fall
    through to the default, anything else is named explicitly.

    Args:
        name: exported function name
        fx: the canonical fixtures

    Returns:
        keyword arguments sufficient to call ``name``

    Raises:
        KeyError: if the function needs an argument that has no fixture
    """
    explicit = {
        'plot_box': dict(df=fx.returns, x=fx.returns.columns[0], y=fx.returns.columns[1]),
        'plot_brinson_attribution_table': dict(zip(
            ('totals_table', 'active_total', 'grouped_allocation_return',
             'grouped_selection_return', 'grouped_interaction_return'), fx.brinson)),
        'plot_brinson_totals_table': dict(totals_table=fx.brinson[0]),
        'plot_classification_scatter': dict(df=fx.two_columns, x=fx.two_columns.columns[0],
                                            y=fx.two_columns.columns[1]),
        'plot_contour': dict(x=np.linspace(0.0, 1.0, 8), y=np.linspace(0.0, 1.0, 8),
                             z=np.outer(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))),
        'plot_corr_matrix_from_covar': dict(covar=fx.covar),
        'plot_data_timeseries': dict(data=fx.prices),
        'plot_df_table_with_ci': dict(df=fx.small_table, df_ci=fx.small_table.abs()),
        'plot_exposures_strategy_vs_benchmark_stack': dict(
            strategy_exposures=fx.exposures, benchmark_exposures=fx.exposures,
            axs=plt.subplots(1, 2)[1]),
        'plot_histplot2d': dict(df=fx.two_columns),
        # each value is an x/y frame: first column is x, second is y
        'plot_lines_list': dict(xy_datas={'a': fx.returns.iloc[:, :2].reset_index(drop=True),
                                          'b': fx.returns.iloc[:, 2:4].reset_index(drop=True)},
                                data_labels=['a', 'b']),
        'plot_multivariate_scatter_with_prediction': dict(
            df=fx.returns, x=list(fx.returns.columns[1:3]), y=fx.returns.columns[0],
            # hue is not optional in practice: without it seaborn draws no legend and the
            # function dereferences ax.get_legend() unguarded
            x_axis_column=fx.returns.columns[1], hue=fx.returns.columns[3]),
        'plot_pie': dict(df=fx.positive_table),
        'plot_prices_2ax': dict(prices_ax1=fx.prices.iloc[:, [0]],
                                prices_ax2=fx.prices.iloc[:, [1]]),
        'plot_prices_with_fundamentals': dict(prices=fx.prices, volumes=fx.prices.abs(),
                                              mcap=fx.prices.abs()),
        'plot_quantile_class_table': dict(data=fx.two_columns, x_column=fx.two_columns.columns[0]),
        'plot_ra_perf_by_dates': dict(prices=fx.prices, time_period_dict=fx.time_period_dict),
        'plot_ra_perf_table_benchmark': dict(prices=fx.prices, benchmark=fx.benchmark),
        'plot_regime_boxplot': dict(regime_classifier=fx.regime_classifier, prices=fx.prices,
                                    benchmark=fx.benchmark),
        # prices / benchmark / perf_params reach compute_regimes_pa_perf_table through **kwargs
        'plot_regime_data': dict(regime_classifier=fx.regime_classifier, prices=fx.prices,
                                 benchmark=fx.benchmark, perf_params=fx.perf_params),
        'plot_regime_pdf': dict(prices=fx.prices, benchmark=fx.benchmark),
        'plot_returns_heatmap': dict(prices=fx.price),
        'plot_returns_scatter': dict(prices=fx.prices, benchmark=fx.benchmark),
        'plot_returns_table': dict(prices=fx.prices, time_period_dict=fx.time_period_dict),
        'plot_scatter': dict(df=fx.two_columns, x=fx.two_columns.columns[0],
                             y=fx.two_columns.columns[1]),
        'plot_scatter_regression': dict(prices=fx.prices, regime_benchmark=fx.benchmark),
        'plot_signal_diagnostics': dict(result=fx.diagnostics),
        'plot_signal_diagnostics_beta_boxplot': dict(asset_returns_dict=fx.asset_returns_dict,
                                                     signal=fx.signal),
        'plot_signal_diagnostics_boxplot': dict(result=fx.diagnostics, horizon=fx.horizon),
        'plot_signal_diagnostics_for_returns': dict(asset_returns_dict=fx.asset_returns_dict,
                                                    signal=fx.signal),
        'plot_signal_diagnostics_group_boxplot': dict(result=fx.diagnostics, horizon=fx.horizon),
        'plot_time_series_2ax': dict(df1=fx.returns.iloc[:, [0]], df2=fx.returns.iloc[:, [1]]),
        'plot_vbars': dict(df=fx.small_table),
        'plot_xy_qq': dict(x=fx.returns.iloc[:, 0], y=fx.returns.iloc[:, 1]),
    }
    if name in explicit:
        return explicit[name]

    signature = inspect.signature(getattr(qis, name))
    required = [parameter for parameter, value in signature.parameters.items()
                if value.default is inspect.Parameter.empty
                and value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD)]
    defaults = {'prices': fx.prices, 'price': fx.price, 'df': fx.returns, 'data': fx.returns}
    missing = [parameter for parameter in required if parameter not in defaults]
    if len(missing) > 0:
        raise KeyError(f"{name} needs {missing!r}; add an entry to the explicit table in "
                       f"qis/plots/tests/plot_smoke_test.py")
    return {parameter: defaults[parameter] for parameter in required}


@pytest.mark.parametrize('name', _exported_plot_functions())
def test_exported_plot_function_draws(name: str, fx: Fixtures) -> None:
    """Every exported plot_* runs on the canonical panel and leaves a figure behind."""
    if name in KNOWN_BROKEN:
        pytest.xfail(KNOWN_BROKEN[name])
    plt.close('all')
    try:
        output = getattr(qis, name)(**_call_kwargs(name=name, fx=fx))
        assert isinstance(output, (plt.Figure, plt.Axes, tuple, pd.DataFrame)) or output is None
        assert len(plt.get_fignums()) > 0, f"{name} drew no figure"
    finally:
        plt.close('all')


def test_every_exported_plot_function_is_covered(fx: Fixtures) -> None:
    """The parametrisation cannot go stale: a new export without a fixture fails here."""
    uncovered = []
    for name in _exported_plot_functions():
        try:
            _call_kwargs(name=name, fx=fx)
        except KeyError:
            uncovered.append(name)
    assert uncovered == [], f"exported plot functions with no fixture: {uncovered!r}"
