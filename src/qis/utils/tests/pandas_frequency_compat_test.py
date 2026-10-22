"""QIS keeps frequency strings and analytical conventions across pandas versions."""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis


@pytest.mark.parametrize(('freq', 'annualisation'), [
    ('B', 252), ('D', 365), ('W-WED', 52), ('ME', 12), ('2BME', 6), ('QE-DEC', 4),
])
def test_frequency_inference_preserves_units_without_transition_warnings(freq, annualisation):
    index = pd.date_range('2023-01-01', periods=12, freq=freq, tz='Europe/Zurich')
    values = pd.Series(np.arange(12, dtype=float), index=index)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        assert qis.infer_annualisation_factor_from_df(values) == annualisation
        actual = qis.df_asfreq(values, freq=freq)
    pd.testing.assert_series_equal(actual, values)


def test_frequency_inference_preserves_pandas_offset_option():
    option = 'future.infer_freq_returns_offset'
    try:
        before = pd.get_option(option)
    except pd.errors.OptionError:
        pytest.skip('Pandas predates the offset inference option')
    index = pd.date_range('2023-01-01', periods=12, freq='ME')
    values = pd.Series(np.arange(12, dtype=float), index=index)
    with pd.option_context(option, True), warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        assert qis.infer_annualisation_factor_from_df(values) == 12
        pd.testing.assert_series_equal(qis.df_asfreq(values, freq='ME'), values)
        assert pd.get_option(option) is True
    assert pd.get_option(option) == before


@pytest.mark.parametrize('category', [UserWarning, FutureWarning, DeprecationWarning])
def test_frequency_adapter_keeps_unrelated_warnings_visible(monkeypatch, category):
    from qis.utils._pandas_compat import infer_freq

    def warning_inference(index):
        warnings.warn('Unrelated inference warning', category)
        return 'D'

    monkeypatch.setattr(pd, 'infer_freq', warning_inference)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(category, match='Unrelated inference warning'):
            infer_freq(pd.date_range('2024-01-01', periods=3))


@pytest.mark.parametrize('kind', ['short', 'irregular', 'timedelta'])
def test_frequency_adapter_retains_native_edge_cases(kind):
    from qis.utils._pandas_compat import infer_freq

    if kind == 'short':
        with pytest.raises(ValueError, match='at least 3'):
            infer_freq(pd.date_range('2024-01-01', periods=2))
    elif kind == 'irregular':
        assert infer_freq(pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-08'])) is None
    else:
        assert infer_freq(pd.timedelta_range(pd.Timedelta(0), periods=5, freq='2h')) == '2h'
