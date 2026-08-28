"""Contracts for calendar-period sampling and removal of obsolete helpers."""
import pandas as pd
import pytest

import qis
from qis.utils import sampling
from qis.utils.sampling import split_to_samples


def _monthly_boundary_data(as_series: bool = False) -> pd.DataFrame | pd.Series:
    index = pd.to_datetime([
        '2024-01-31',
        '2024-02-15',
        '2024-02-29',
        '2024-03-15',
        '2024-03-31',
        '2024-04-15',
        '2024-04-30',
        '2024-05-15',
    ])
    series = pd.Series(
        [10.0, 11.0, 12.0, 15.0, 18.0, 21.0, 24.0, 30.0],
        index=index,
        name='asset',
    )
    return series if as_series else series.to_frame()


def _assert_pandas_equal(
        actual: pd.DataFrame | pd.Series,
        expected: pd.DataFrame | pd.Series,
) -> None:
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    else:
        pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize('as_series', [False, True])
def test_split_to_samples_preserves_calendar_boundary_contract(as_series: bool) -> None:
    data = _monthly_boundary_data(as_series=as_series)

    samples = split_to_samples(data=data, sample_freq='ME')

    assert list(samples) == [pd.Timestamp('2024-03-31'), pd.Timestamp('2024-04-30')]
    expected_march = data.loc['2024-02-29':'2024-03-31']
    expected_april = data.loc['2024-03-31':'2024-04-30']
    _assert_pandas_equal(samples[pd.Timestamp('2024-03-31')], expected_march)
    _assert_pandas_equal(samples[pd.Timestamp('2024-04-30')], expected_april)


@pytest.mark.parametrize('as_series', [False, True])
def test_split_to_samples_normalizes_each_sample_without_mutating_input(
        as_series: bool,
) -> None:
    data = _monthly_boundary_data(as_series=as_series)
    original = data.copy(deep=True)

    samples = split_to_samples(data=data, sample_freq='ME', start_to_one=True)

    for sample in samples.values():
        expected = original.loc[sample.index].divide(original.loc[sample.index].iloc[0])
        _assert_pandas_equal(sample, expected)
    _assert_pandas_equal(data, original)


def test_obsolete_sampling_reshapers_are_removed() -> None:
    assert not hasattr(sampling, 'split_to_train_live_samples')
    assert not hasattr(sampling, 'get_data_samples_df')


def test_obsolete_sampling_types_are_removed() -> None:
    for name in ('TrainLivePeriod', 'TrainLiveSamples'):
        assert not hasattr(sampling, name)
        assert not hasattr(qis, name)
