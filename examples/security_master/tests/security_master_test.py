"""Identity, ingestion integrity and point-in-time prices for the example data layer."""

from dataclasses import replace
import sqlite3

import pandas as pd
import pytest
import qis
from qis.datasets.synthetic import generate_synthetic_prices

from examples.security_master.store import (
    IdentifierAssignment, IdentifierScheme, PriceBasis, PriceObservation, PriceSeries,
    SecurityMaster, SecurityType,
)


def ingest(master, accepted_at, provider='DEMO'):
    return master.ingest_batch(provider=provider, accepted_at=accepted_at,
                               raw_payload=b'Fictional test feed', source_ref='synthetic:test')


def alias(value, listing_id, start='2020-01-02', end=None, namespace='DEMO'):
    return IdentifierAssignment(IdentifierScheme.TICKER, namespace, value, listing_id, start, end)


def series(listing_id, **changes):
    definition = PriceSeries(f'PX_{listing_id}', listing_id, 'DEMO', 'close', PriceBasis.RAW_CLOSE,
                             'USD', 'SYNTHETIC_BUSINESS_DAYS', 'America/New_York')
    return replace(definition, **changes)


def initialise(master):
    with ingest(master, '2020-01-01T12:00:00Z') as batch:
        for suffix in ('1', '2'):
            batch.add_entity(f'E{suffix}', f'Synthetic issuer {suffix}')
            batch.add_security(f'S{suffix}', f'E{suffix}', security_type=SecurityType.COMMON_STOCK,
                               share_class='ordinary')
            batch.add_listing(f'L{suffix}', f'S{suffix}', mic='XNYS', currency='USD')
            batch.add_price_series(series(f'L{suffix}'))
        batch.add_identifier(alias('ABC', 'L1'))


@pytest.fixture
def master():
    with SecurityMaster() as store:
        initialise(store)
        yield store


@pytest.fixture
def source():
    return generate_synthetic_prices(start='2020-01-02', end='2020-01-10')


def resolve(master, ticker, day, cutoff='2020-01-31T22:00:00Z', **context):
    scope = dict(namespace='DEMO', mic='XNYS', currency='USD')
    scope.update(context)
    return master.resolve_ticker(ticker, effective_date=day, known_at=cutoff, **scope)


def rename(master, *, accepted_at='2020-01-03T12:00:00Z', effective_date='2020-01-06'):
    with ingest(master, accepted_at) as batch:
        batch.rename_ticker('L1', namespace='DEMO', old_ticker='ABC', new_ticker='XYZ',
                            effective_date=effective_date)


def test_rename_boundaries_and_identity(master):
    rename(master)
    assert resolve(master, 'ABC', '2020-01-05') == 'L1'
    assert resolve(master, 'XYZ', '2020-01-06') == 'L1'
    for ticker, day in [('ABC', '2020-01-06'), ('XYZ', '2020-01-05')]:
        with pytest.raises(LookupError):
            resolve(master, ticker, day)


def test_reuse_selects_different_listing_and_does_not_fill_gap(master):
    rename(master)
    with ingest(master, '2020-01-07T12:00:00Z') as batch:
        batch.add_identifier(alias('ABC', 'L2', start='2020-01-08'))
    assert resolve(master, 'ABC', '2020-01-03') == 'L1'
    assert resolve(master, 'ABC', '2020-01-08') == 'L2'
    assert resolve(master, 'XYZ', '2020-01-08') == 'L1'
    with pytest.raises(LookupError):
        resolve(master, 'ABC', '2020-01-07')


def test_knowledge_boundary_and_utc_offsets(master):
    rename(master, accepted_at='2020-01-03T13:00:00+01:00')
    before = '2020-01-03T11:59:59.999999Z'
    boundary = '2020-01-03T12:00:00Z'
    assert resolve(master, 'ABC', '2020-01-06', before) == 'L1'
    with pytest.raises(LookupError):
        resolve(master, 'XYZ', '2020-01-06', before)
    assert resolve(master, 'XYZ', '2020-01-06', boundary) == 'L1'
    with pytest.raises(LookupError):
        resolve(master, 'ABC', '2020-01-06', boundary)


def test_late_rename_notice_does_not_rewrite_earlier_knowledge(master):
    rename(master, accepted_at='2020-01-09T12:00:00Z')
    assert resolve(master, 'ABC', '2020-01-07', '2020-01-08T22:00:00Z') == 'L1'
    assert resolve(master, 'XYZ', '2020-01-07', '2020-01-09T22:00:00Z') == 'L1'


def test_rename_at_interval_start_has_no_empty_old_interval(master):
    rename(master, effective_date='2020-01-02')
    assert resolve(master, 'XYZ', '2020-01-02') == 'L1'
    with pytest.raises(LookupError):
        resolve(master, 'ABC', '2020-01-02')


def test_ticker_context_and_security_level_identifier(master):
    with ingest(master, '2020-01-02T12:00:00Z') as batch:
        batch.add_listing('L3', 'S1', mic='XNAS', currency='USD')
        batch.add_identifier(alias('ABC', 'L3'))
        batch.add_identifier(alias('ABC', 'L2', namespace='SECOND_FEED'))
        batch.add_identifier(IdentifierAssignment(
            IdentifierScheme.PROVIDER_SECURITY, 'DEMO', 'SECURITY_ALPHA', 'S1', '2020-01-02'))
    assert resolve(master, 'ABC', '2020-01-02', mic='XNAS') == 'L3'
    assert resolve(master, 'ABC', '2020-01-02', namespace='SECOND_FEED') == 'L2'
    assert master.resolve_identifier(
        IdentifierScheme.PROVIDER_SECURITY, 'DEMO', 'SECURITY_ALPHA',
        effective_date='2020-01-02', known_at='2020-01-02T22:00:00Z') == 'S1'
    with pytest.raises(LookupError):
        resolve(master, 'ABC', '2020-01-02', currency='EUR')


def test_overlapping_alias_is_rejected(master):
    with pytest.raises(sqlite3.IntegrityError, match='overlapping identifier'):
        with ingest(master, '2020-01-03T12:00:00Z') as batch:
            batch.add_identifier(alias('ABC', 'L2'))
    assert resolve(master, 'ABC', '2020-01-10') == 'L1'


def test_conflicting_rename_rolls_back_even_if_caller_catches_error(master):
    with ingest(master, '2020-01-02T12:00:00Z') as batch:
        batch.add_identifier(alias('XYZ', 'L2'))
    with ingest(master, '2020-01-03T12:00:00Z') as batch:
        with pytest.raises(sqlite3.IntegrityError, match='overlapping identifier'):
            batch.rename_ticker('L1', namespace='DEMO', old_ticker='ABC', new_ticker='XYZ',
                                effective_date='2020-01-06')
    assert resolve(master, 'ABC', '2020-01-10') == 'L1'
    assert resolve(master, 'XYZ', '2020-01-10') == 'L2'


def test_stable_price_columns_and_independent_returns_through_rename_reuse(master, source):
    expected = source[['SEQ_US', 'SEQ_EU']].copy()
    expected.columns = pd.Index(['L1', 'L2'], name='listing_id')
    expected.loc[expected.index < '2020-01-08', 'L2'] = float('nan')
    for date, values in expected.iterrows():
        day = date.strftime('%Y-%m-%d')
        if day == '2020-01-03':
            rename(master)
        if day == '2020-01-07':
            with ingest(master, f'{day}T12:00:00Z') as batch:
                batch.add_identifier(alias('ABC', 'L2', start='2020-01-08'))
        with ingest(master, f'{day}T21:05:00Z') as batch:
            for listing_id, value in values.items():
                if pd.notna(value):
                    batch.add_prices(f'PX_{listing_id}', [PriceObservation(
                        day, f'{day}T21:00:00Z', value)])
    actual = master.load_price_panel(['PX_L1', 'PX_L2'], sessions=source.index,
                                     known_at='2020-01-10T22:00:00Z')
    pd.testing.assert_frame_equal(actual, expected)
    returns = qis.to_returns(actual, is_log_returns=False, ffill_nans=False)
    reference = expected.div(expected.shift(1)).sub(1.0)
    pd.testing.assert_frame_equal(returns, reference)
    assert pd.notna(returns.loc['2020-01-06', 'L1'])
    assert pd.isna(returns.loc['2020-01-08', 'L2'])
    assert actual.attrs['series_ids'] == ['PX_L1', 'PX_L2']


def test_prices_follow_acceptance_time_and_preserve_missing_sessions(master, source):
    day = '2020-01-02'
    value = source.loc[day, 'SEQ_US']
    with ingest(master, '2020-01-03T12:00:00Z') as batch:
        batch.add_prices('PX_L1', [PriceObservation(day, f'{day}T21:00:00Z', value)])
    early = master.load_price_panel(['PX_L2', 'PX_L1'], sessions=source.index,
                                    known_at='2020-01-03T11:59:59.999999Z')
    assert early.isna().all().all()
    assert early.columns.tolist() == ['L2', 'L1']
    boundary = master.load_price_panel(['PX_L1'], sessions=source.index,
                                       known_at='2020-01-03T12:00:00Z')
    assert boundary.loc[day, 'L1'] == value
    assert boundary.loc[boundary.index > day, 'L1'].isna().all()


def test_price_correction_and_withdrawal_retain_historical_vintages(master, source):
    # Two canonical synthetic observations stand in for the two feed revisions.
    day, observed = '2020-01-02', '2020-01-02T21:00:00Z'
    original, corrected = source['SEQ_US'].iloc[:2]
    with ingest(master, '2020-01-02T21:05:00Z') as batch:
        batch.add_prices('PX_L1', [PriceObservation(day, observed, original)])
    with ingest(master, '2020-01-03T12:00:00Z') as batch:
        batch.add_prices('PX_L1', [PriceObservation(day, observed, corrected)])
    with ingest(master, '2020-01-06T12:00:00Z') as batch:
        batch.add_prices('PX_L1', [PriceObservation(day, observed, None, is_deleted=True)])
    for cutoff, expected in [('2020-01-02T22:00:00Z', original),
                             ('2020-01-03T12:00:00Z', corrected)]:
        panel = master.load_price_panel(['PX_L1'], sessions=source.index, known_at=cutoff)
        assert panel.loc[day, 'L1'] == expected
    withdrawn = master.load_price_panel(['PX_L1'], sessions=source.index,
                                        known_at='2020-01-06T12:00:00Z')
    assert withdrawn['L1'].isna().all()


def test_failed_batch_leaves_no_partial_rows_and_writer_expires(master):
    with pytest.raises(RuntimeError, match='simulate'):
        with ingest(master, '2020-01-02T12:00:00Z') as batch:
            batch.add_identifier(alias('BETA', 'L2'))
            raise RuntimeError('simulate an interrupted feed')
    with pytest.raises(LookupError):
        resolve(master, 'BETA', '2020-01-02')
    with pytest.raises(RuntimeError, match='no longer active'):
        batch.add_entity('E3', 'Should not be written')
    # Failed acceptance did not advance the committed ledger.
    with ingest(master, '2020-01-02T12:00:00Z') as committed:
        committed.add_identifier(alias('BETA', 'L2'))
    with pytest.raises(RuntimeError, match='no longer active'):
        committed.add_identifier(alias('LATE', 'L2'))


@pytest.mark.parametrize('cutoff', ['2020-01-01T11:00:00Z', '2020-01-01T12:00:00Z'])
def test_batches_cannot_be_backdated_or_reopened(master, cutoff):
    with pytest.raises(ValueError, match='strictly increase'):
        with ingest(master, cutoff):
            pass


def test_nested_batches_and_close_during_batch_fail(master):
    with ingest(master, '2020-01-02T12:00:00Z'):
        with pytest.raises(RuntimeError, match='Nested'):
            with ingest(master, '2020-01-03T12:00:00Z'):
                pass
        with pytest.raises(RuntimeError, match='active ingestion'):
            master.close()


def test_bulk_prices_are_atomic_on_duplicate_even_if_error_is_caught(master, source):
    observation = PriceObservation('2020-01-02', '2020-01-02T21:00:00Z', source['SEQ_US'].iloc[0])
    with ingest(master, '2020-01-02T22:00:00Z') as batch:
        with pytest.raises(sqlite3.IntegrityError):
            batch.add_prices('PX_L1', [observation, observation])
    actual = master.load_price_panel(['PX_L1'], sessions=source.index,
                                     known_at='2020-01-02T22:00:00Z')
    assert actual.isna().all().all()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -1.0, None])
def test_invalid_price_values_are_rejected(master, value):
    with ingest(master, '2020-01-02T22:00:00Z') as batch:
        with pytest.raises(ValueError, match='finite non-negative'):
            batch.add_prices('PX_L1', [PriceObservation(
                '2020-01-02', '2020-01-02T21:00:00Z', value)])


@pytest.mark.parametrize('provider,observed', [
    ('OTHER', '2020-01-02T21:00:00Z'), ('DEMO', '2020-01-02T23:00:00Z')])
def test_price_provenance_and_future_observations_are_rejected(master, source, provider, observed):
    with ingest(master, '2020-01-02T22:00:00Z', provider=provider) as batch:
        with pytest.raises(sqlite3.IntegrityError, match='time or provider'):
            batch.add_prices('PX_L1', [PriceObservation(
                '2020-01-02', observed, source['SEQ_US'].iloc[0])])


def test_observation_date_uses_exchange_timezone(master, source):
    # 00:30 UTC on Jan 3 is still Jan 2 in New York.
    with ingest(master, '2020-01-03T01:00:00Z') as batch:
        value = source['SEQ_US'].iloc[0]
        batch.add_prices('PX_L1', [PriceObservation('2020-01-02', '2020-01-03T00:30:00Z', value)])
        with pytest.raises(ValueError, match='local session date'):
            batch.add_prices('PX_L2', [PriceObservation(
                '2020-01-03', '2020-01-03T00:30:00Z', value)])


def test_loading_does_not_change_the_callers_calendar(master, source):
    sessions = source.index.rename('caller_session')
    result = master.load_price_panel(['PX_L1'], sessions=sessions,
                                     known_at='2020-01-02T22:00:00Z')
    assert sessions.name == 'caller_session'
    assert result.index.name == 'date'


def test_explicit_units_and_panel_conventions(master, source):
    value = source['SEQ_US'].iloc[0]
    with ingest(master, '2020-01-02T22:00:00Z') as batch:
        batch.add_price_series(series('L2', series_id='CENTS', provider_field='minor_units',
                                      unit_multiplier=0.01))
        batch.add_price_series(series('L2', series_id='EUR', currency='EUR'))
        batch.add_price_series(series('L2', series_id='TRI', basis=PriceBasis.TOTAL_RETURN_INDEX))
        batch.add_prices('CENTS', [PriceObservation('2020-01-02', '2020-01-02T21:00:00Z', value)])
    actual = master.load_price_panel(['CENTS'], sessions=source.index,
                                     known_at='2020-01-02T22:00:00Z')
    assert actual.loc['2020-01-02', 'L2'] == pytest.approx(value * 0.01)
    for selection in [['PX_L1', 'EUR'], ['PX_L1', 'TRI']]:
        with pytest.raises(ValueError, match='one currency and one price basis'):
            master.load_price_panel(selection, sessions=source.index,
                                    known_at='2020-01-03T22:00:00Z')
    with pytest.raises(ValueError, match='one explicitly selected series'):
        master.load_price_panel(['PX_L2', 'CENTS'], sessions=source.index,
                                known_at='2020-01-03T22:00:00Z')


@pytest.mark.parametrize('selection', [[], ['PX_L1', 'PX_L1'], ['UNKNOWN'], 'PX_L1'])
def test_invalid_series_selection_is_rejected(master, source, selection):
    with pytest.raises(ValueError):
        master.load_price_panel(selection, sessions=source.index, known_at='2020-01-02T22:00:00Z')


@pytest.mark.parametrize('sessions', [
    pd.DatetimeIndex([]), pd.DatetimeIndex(['2020-01-02', '2020-01-02']),
    pd.DatetimeIndex(['2020-01-03', '2020-01-02']), pd.DatetimeIndex(['2020-01-02T12:00:00']),
    pd.DatetimeIndex(['2020-01-02'], tz='UTC'), pd.DatetimeIndex([pd.NaT]), ['2020-01-02']])
def test_invalid_session_grids_are_rejected(master, sessions):
    with pytest.raises(ValueError, match='session-date'):
        master.load_price_panel(['PX_L1'], sessions=sessions, known_at='2020-01-02T22:00:00Z')


@pytest.mark.parametrize('stamp', ['2020-01-02T12:00:00', 'NaT', '2020-01-02T12:00:00.000000001Z'])
def test_timestamps_require_timezone_and_supported_precision(master, stamp):
    with pytest.raises(ValueError):
        resolve(master, 'ABC', '2020-01-02', cutoff=stamp)


def test_persistent_database_reopens_and_rejects_unknown_schema(tmp_path):
    database = tmp_path / 'master.sqlite'
    with SecurityMaster(database) as store:
        initialise(store)
    with SecurityMaster(database) as store:
        assert resolve(store, 'ABC', '2020-01-02') == 'L1'
        assert store._con.execute('PRAGMA foreign_key_check').fetchall() == []
    connection = sqlite3.connect(database)
    try:
        connection.execute('PRAGMA user_version = 999')
    finally:
        connection.close()
    with pytest.raises(ValueError, match='unsupported'):
        SecurityMaster(database)


def test_unrelated_database_is_not_overwritten(tmp_path):
    database = tmp_path / 'unrelated.sqlite'
    connection = sqlite3.connect(database)
    try:
        connection.execute('CREATE TABLE unrelated (value TEXT)')
    finally:
        connection.close()
    with pytest.raises(ValueError, match='Unrelated'):
        SecurityMaster(database)


def test_database_guards_retain_history(master, source):
    with ingest(master, '2020-01-02T22:00:00Z') as batch:
        batch.add_prices('PX_L1', [PriceObservation(
            '2020-01-02', '2020-01-02T21:00:00Z', source['SEQ_US'].iloc[0])])
    for statement in ["UPDATE price_revision SET value = 1", 'DELETE FROM price_revision',
                      "UPDATE identifier_assignment SET value = 'BROKEN'",
                      'DELETE FROM identifier_assignment', 'DELETE FROM ingest_batch']:
        with pytest.raises(sqlite3.IntegrityError):
            master._con.execute(statement)
    assert resolve(master, 'ABC', '2020-01-02') == 'L1'


def test_example_runs_without_files(capsys, tmp_path, monkeypatch):
    from examples.security_master.ticker_history import run_local

    monkeypatch.chdir(tmp_path)
    run_local()
    text = capsys.readouterr().out
    assert '2020-01-08: ABC -> L2' in text
    assert 'after the January 13 withdrawal: nan' in text
    assert list(tmp_path.iterdir()) == []
