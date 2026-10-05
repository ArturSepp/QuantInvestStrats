"""Brinson display precision preserves cumulative effects and panel ownership.

The report previously honored precision in its table but rounded every cumulative
legend to whole percentages. These tests compare the rendered text with independently
calculated effects, so a visually plausible but misleading rounded legend cannot pass.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.portfolio.tests.brinson_wrapper_test import make_portfolios


def _attribution_inputs(dtype):
    """Isolate display formatting with hand-authored attribution increments.

    Equity finishes at 0.40% allocation and 0.20% selection: both round to 0%,
    while their 0.60% grouped effect rounds to 1%. Bonds and Cash ensure precision
    preserves negative effects and genuine zeros too. These are renderer inputs,
    not a second implementation of the Brinson calculation.
    """
    index = pd.date_range("2025-01-31", periods=3, freq="ME", name="Date")
    allocation = pd.DataFrame(
        {
            "Equity": [0.001, 0.0015, 0.0015],
            "Bonds": [-0.001, 0.0, -0.001],
            "Cash": [0.0, 0.0, 0.0],
        },
        index=index,
        dtype=dtype,
    )
    selection = allocation / 2
    interaction = allocation / 4
    for frame in (allocation, selection, interaction):
        frame["Total Sum"] = frame.sum(axis=1)
    active = pd.DataFrame(
        {"Allocation Total": allocation["Total Sum"], "Selection Total": selection["Total Sum"]},
        index=index,
        dtype=dtype,
    )
    totals = pd.DataFrame({"Allocation": allocation.sum(), "Selection": selection.sum()})
    return totals, active, allocation, selection, interaction


def _assert_time_series(axis, increments, var_format, interaction=False):
    """Check plotted effects and legend text against an independent display oracle.

    NumPy accumulation and literal Python formatting avoid using the report's
    pandas accumulation or QIS legend builder to derive their own expected output.
    Comparing line data separately prevents correct-looking labels from concealing
    a change to the effects being plotted.
    """
    cumulative = np.cumsum(increments.to_numpy(dtype=float), axis=0)
    expected = []
    for position, column in enumerate(increments.columns):
        values = cumulative[:, position]
        if interaction:
            # This panel intentionally keeps the plotting helper's AVG_LAST default.
            expected.append(
                f"{column}: avg={var_format.format(sum(values) / len(values))}, "
                f"last={var_format.format(values[-1])}"
            )
        else:
            expected.append(f"{column} = {var_format.format(values[-1])}")
        np.testing.assert_allclose(axis.lines[position].get_ydata(), values, atol=1e-15)
    assert [text.get_text() for text in axis.get_legend().get_texts()] == expected
    assert axis.yaxis.get_major_formatter()(0.004, 0) == var_format.format(0.004)
    if interaction:
        # Precision must not remove the endpoint trend lines native to this panel.
        trend_lines = [line for line in axis.lines if line.get_linestyle() == "--"]
        assert len(trend_lines) == len(increments.columns)


@pytest.mark.parametrize("merged", [True, False], ids=["merged", "split"])
@pytest.mark.parametrize("supplied_axes", [False, True], ids=["new-figures", "supplied-axes"])
@pytest.mark.parametrize("var_format", [None, "{:.1%}", "{:.2%}"], ids=["default", "one", "two"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_plot_brinson_attribution_table_honors_var_format(
    merged,
    supplied_axes,
    var_format,
    dtype,
):
    """Keep one formatting contract across both panel layouts and ownership paths.

    Omitting the option protects the existing whole-percent default. Explicit
    precision must reach both newly created figures and axes supplied by a caller
    composing a larger report, for ordinary and nullable floating-point inputs.
    """
    inputs = _attribution_inputs(dtype)
    before = tuple(frame.copy(deep=True) for frame in inputs)
    original_figures = set(plt.get_fignums())
    options = {"is_exclude_interaction_term": merged}
    if var_format is not None:
        options["var_format"] = var_format
    effective_format = var_format or "{:.0%}"
    if supplied_axes:
        figure, axes = plt.subplots(5, 1, figsize=(9, 15))
        options["axs"] = list(axes)
    try:
        results = qis.plot_brinson_attribution_table(*inputs, **options)
        if supplied_axes:
            # Native plot helpers return None when they borrow axes; creating extra
            # figures here would detach these panels from the caller's report.
            assert results == (None,) * 5
            assert set(plt.get_fignums()) == original_figures | {figure.number}
            assert list(figure.axes) == list(axes)
            figures = [figure]
        else:
            assert len(results) == 5
            assert len({id(result) for result in results}) == 5
            figures = list(results)
            axes = [result.axes[0] for result in results]
            assert len(set(plt.get_fignums()) - original_figures) == 5
        # Drawing reaches deferred Matplotlib work, not just artist construction.
        for figure in figures:
            figure.canvas.draw()
        by_title = {axis.get_title(): axis for axis in axes[1:]}
        table_cells = [
            cell.get_text().get_text() for cell in axes[0].tables[0].get_celld().values()
        ]
        for row in inputs[0].to_numpy(dtype=float):
            for value in row:
                assert effective_format.format(value) in table_cells
        for title, increments in (
            ("Cumulative Active Attribution Effects", inputs[1]),
            ("Cumulative Asset Class Allocation Effects", inputs[2]),
            ("Cumulative Instrument Selection Effects", inputs[3]),
        ):
            _assert_time_series(by_title[title], increments, effective_format)
        if merged:
            # Merged selection already contains interaction. Grouped active effects
            # add allocation and selection only, and omit the portfolio-total curve.
            grouped = pd.DataFrame(
                inputs[2].iloc[:, :-1].to_numpy(dtype=float)
                + inputs[3].iloc[:, :-1].to_numpy(dtype=float),
                index=inputs[2].index,
                columns=inputs[2].columns[:-1],
            )
            final_title = "Total Cumulative Active Effects by Groups"
            _assert_time_series(by_title[final_title], grouped, effective_format)
        else:
            final_title = "Cumulative Asset Class Interaction Effects"
            _assert_time_series(
                by_title[final_title], inputs[4], effective_format, interaction=True
            )
        assert len(by_title) == 4
        # Presentation must not modify caller-owned attribution data, including
        # labels and dtypes that downstream tables or reports may reuse.
        for actual, expected in zip(inputs, before):
            pd.testing.assert_frame_equal(actual, expected)
    finally:
        for number in set(plt.get_fignums()) - original_figures:
            plt.close(number)


@pytest.mark.parametrize("merged", [True, False], ids=["merged", "split"])
def test_multi_portfolio_data_plot_brinson_attribution_forwards_var_format(merged):
    """Exercise kwargs delegation through the real public portfolio wrapper.

    Direct renderer coverage cannot prove that a wrapper forwards the requested
    precision. Real synthetic portfolios also let us check that rendering leaves
    their computed attribution unchanged in both interaction modes.
    """
    strategy, benchmark = make_portfolios()
    multi = qis.MultiPortfolioData([strategy, benchmark])
    effects = multi.compute_brinson_attribution(is_exclude_interaction_term=merged)
    figure, axes = plt.subplots(5, 1, figsize=(9, 15))
    try:
        results = multi.plot_brinson_attribution(
            axs=list(axes),
            var_format="{:.2%}",
            is_exclude_interaction_term=merged,
        )
        assert results == (None,) * 5
        figure.canvas.draw()
        by_title = {axis.get_title(): axis for axis in axes[1:]}
        _assert_time_series(
            by_title["Cumulative Asset Class Allocation Effects"],
            effects[2],
            "{:.2%}",
        )
        after = multi.compute_brinson_attribution(is_exclude_interaction_term=merged)
        for actual, expected in zip(after, effects):
            pd.testing.assert_frame_equal(actual, expected)
    finally:
        plt.close(figure)
