"""
The documented core of the public API must stay documented.

``qis/api.py`` records which exported symbols are core: the ones a published package or qis's
own documentation calls. This file turns that record into a promise. A core symbol must resolve
from the top-level namespace and carry an ``Args`` or ``Attributes`` block, so that "the core
API is documented" is a claim a reader can check rather than an intention.

``PENDING_DOCSTRINGS`` is the backlog, and it is a ratchet in both directions: a core symbol
outside it must be documented, and a symbol inside it must still be undocumented. Writing a
docstring therefore fails the suite until the name is removed from the list, which is what keeps
the list from going stale and quietly hiding finished work.

Same shape as ``qis/plots/tests/plot_smoke_test.py`` and ``qis/tests/test_examples.py``: an
enforced invariant rather than a convention nobody runs.
"""
# packages
import inspect
from typing import Any, Dict, FrozenSet
import pytest
# qis / project
import qis
from qis.api import CORE_API, core_api_names

# core symbols still awaiting an Args/Attributes block. This list only shrinks.
PENDING_DOCSTRINGS: FrozenSet[str] = frozenset({
    'EwmLinearModel', 'bootstrap_data', 'bootstrap_price_data',
    'compute_ewm_covar_tensor_vol_norm_returns', 'compute_ewm_long_short_filtered_ra_returns',
    'compute_masked_covar_corr', 'covar_to_corr', 'df_abssum', 'df_abssum_negative',
    'df_abssum_positive', 'df_boxplot_by_classification_var', 'df_boxplot_by_hue_var',
    'df_last_row', 'df_nanmean', 'df_nanmean_clip', 'df_nanmean_positive', 'df_nanmedian',
    'df_nansum', 'df_nansum_clip', 'df_nansum_negative', 'df_nansum_positive',
    'df_to_equal_weight_allocation', 'df_to_long_only_allocation_sum1',
    'df_to_weight_allocation_sum1', 'estimate_hf_ohlc_vol', 'estimate_rolling_ewma_covar',
    'fetch_default_report_kwargs', 'fit_multivariate_ols', 'generate_fixed_maturity_rolls',
    'generate_multi_portfolio_factsheet', 'generate_strategy_benchmark_factsheet_plt',
    'get_group_dict', 'get_ra_perf_columns', 'get_resource_path', 'get_time_period',
    'get_time_period_label', 'infer_annualisation_factor_from_df',
    'interpolate_infrequent_returns', 'load_df_dict_from_csv', 'load_df_from_csv',
    'load_df_from_excel', 'np_array_to_df_columns', 'plot_bars', 'plot_classification_scatter',
    'plot_df_table', 'plot_exposures_strategy_vs_benchmark_stack', 'plot_heatmap', 'plot_qq',
    'plot_scatter', 'save_df_dict_to_csv', 'save_df_to_csv', 'save_fig', 'save_figs_to_pdf',
    'series_nansum_weighted', 'set_suptitle', 'split_df_by_groups', 'timer',
    'truncate_prior_to_start', 'unsmooth_returns_ar1_ewma', 'update_kwargs',
})


# core symbols where an Args/Attributes block is the wrong form, with the reason. These are
# large compositional enums: the members are generated from a scheme, and a block restating
# every member would be longer than the enum and would tell a reader nothing the name does not.
PROSE_DOCUMENTED: Dict[str, str] = {
    'PerfStat': 'about 60 ColVar members; the docstring documents the ColVar scheme instead',
    'LegendStats': '29 compositional members; the docstring documents the naming vocabulary',
}


def _is_documented(obj: Any) -> bool:
    """
    True when the documentation carries an argument or attribute block.

    Google style puts constructor arguments on ``__init__`` rather than on the class, and
    ``inspect.getdoc`` on a class does not reach them, so both are checked. A numba ``@njit``
    function carries the docstring of its wrapped ``py_func``, which ``getdoc`` does find.
    """
    docs = [inspect.getdoc(obj) or '']
    if inspect.isclass(obj):
        docs.append(inspect.getdoc(getattr(obj, '__init__', None)) or '')
    return any(('Args:' in doc) or ('Attributes:' in doc) for doc in docs)


def _documentable(name: str) -> bool:
    """
    True for a core symbol that can carry an Args/Attributes block at all.

    A numba ``@njit`` function is a ``CPUDispatcher`` rather than a function, and it carries the
    docstring of the wrapped ``py_func``; excluding it would silently drop the EWM kernels from
    the check. A module-level constant that happens to be a dataclass instance is documented on
    its class, not on the name, so it is excluded.
    """
    obj = getattr(qis, name)
    if inspect.isclass(obj) or inspect.isfunction(obj):
        return True
    return hasattr(obj, 'py_func')  # numba dispatcher


def test_core_api_is_exported() -> None:
    """every core name resolves from the top-level namespace."""
    missing = [name for name in core_api_names() if not hasattr(qis, name)]
    assert not missing, f"qis/api.py lists names that qis does not export: {missing}"


def test_core_api_has_no_duplicates() -> None:
    """a symbol belongs to one capability, so the reference cannot list it twice."""
    names = core_api_names()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"listed under more than one capability: {duplicates}"


def test_pending_list_is_a_subset_of_the_core() -> None:
    """the backlog cannot name something that is not core."""
    stray = sorted(PENDING_DOCSTRINGS - set(core_api_names()))
    assert not stray, f"PENDING_DOCSTRINGS names non-core symbols: {stray}"


@pytest.mark.parametrize('name', sorted(set(core_api_names()) - PENDING_DOCSTRINGS
                                        - set(PROSE_DOCUMENTED)))
def test_core_symbol_is_documented(name: str) -> None:
    """a core symbol carries an Args or Attributes block."""
    if not _documentable(name):
        pytest.skip(f"{name} is a constant, not a callable")
    assert _is_documented(getattr(qis, name)), (
        f"qis.{name} is core but has no Args:/Attributes: block. Write it, or move the name "
        f"out of CORE_API if it is not core after all")


@pytest.mark.parametrize('name', sorted(PENDING_DOCSTRINGS))
def test_pending_symbol_is_still_undocumented(name: str) -> None:
    """
    the backlog must shrink when work lands.

    A name here that is now documented means the docstring was written and the list not updated.
    Remove it from PENDING_DOCSTRINGS; the check above then holds it documented forever.
    """
    if not _documentable(name):
        pytest.skip(f"{name} is a constant, not a callable")
    assert not _is_documented(getattr(qis, name)), (
        f"qis.{name} is documented now - remove it from PENDING_DOCSTRINGS")


def test_capability_groups_are_named() -> None:
    """every capability group has a title and at least one symbol."""
    empty = [title for title, names in CORE_API.items() if len(names) == 0]
    assert not empty, f"empty capability groups: {empty}"


@pytest.mark.parametrize('name', sorted(PROSE_DOCUMENTED))
def test_prose_documented_symbol_has_a_substantive_docstring(name: str) -> None:
    """
    the Args/Attributes exemption is not an exemption from documentation.

    These symbols are excused the block form, not the obligation. A one-line summary is not
    enough to describe an enum with dozens of members, so the docstring must actually explain
    the scheme.
    """
    assert name in set(core_api_names()), f"{name} is exempted but is not core"
    doc = inspect.getdoc(getattr(qis, name)) or ''
    assert len(doc.splitlines()) >= 6, (
        f"qis.{name} is exempted from the Args/Attributes rule because "
        f"{PROSE_DOCUMENTED[name]}, so its prose has to carry the description")
