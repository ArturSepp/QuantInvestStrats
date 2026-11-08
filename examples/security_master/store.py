"""SQLite security master for repository-based QIS research workflows.

Identity, effective dates and knowledge timestamps are separate. Writes belong to
atomic ingestion batches; reads require an explicit knowledge cutoff. This module
uses only the Python standard library and pandas. It is not part of qis.__all__.
See README.md for scope, storage paths, schema and the executable example.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
from pathlib import Path
import re
import sqlite3
from typing import Iterator, Sequence

import pandas as pd


SCHEMA_VERSION = 1
APPLICATION_ID = 0x5149534D  # QISM; reject unrelated and original proposal databases.
FOREVER_DATE = '9999-12-31'
FOREVER_TIME = '9999-12-31T23:59:59.999999Z'


class SecurityType(str, Enum):
    """Supported cash-equity instruments."""

    COMMON_STOCK = 'COMMON_STOCK'
    ETF = 'ETF'
    ADR = 'ADR'


class IdentifierScheme(str, Enum):
    """Identifier schemes with an explicit entity, security or venue-level target."""

    TICKER = 'TICKER'
    LEI = 'LEI'
    ISIN = 'ISIN'
    CUSIP = 'CUSIP'
    FIGI_SHARE_CLASS = 'FIGI_SHARE_CLASS'
    FIGI_VENUE = 'FIGI_VENUE'
    PROVIDER_ENTITY = 'PROVIDER_ENTITY'
    PROVIDER_SECURITY = 'PROVIDER_SECURITY'
    PROVIDER_LISTING = 'PROVIDER_LISTING'


class PriceBasis(str, Enum):
    """Price definitions that must not be mixed silently in an analysis panel."""

    RAW_CLOSE = 'RAW_CLOSE'
    SPLIT_ADJUSTED = 'SPLIT_ADJUSTED'
    TOTAL_RETURN_INDEX = 'TOTAL_RETURN_INDEX'


@dataclass(frozen=True)
class IdentifierAssignment:
    """An identifier's effective interval, independent of when it became known.

    Attributes:
        scheme: Identifier scheme; determines the target's level.
        namespace: Exchange or vendor naming convention, kept case-sensitive.
        value: Exact identifier spelling; punctuation is not normalised.
        target_id: Entity, security or listing ID appropriate for the scheme.
        valid_from: Inclusive effective session date, YYYY-MM-DD.
        valid_to: Exclusive effective date; None means no known end.
    """

    scheme: IdentifierScheme
    namespace: str
    value: str
    target_id: str
    valid_from: str
    valid_to: str | None = None


@dataclass(frozen=True)
class PriceSeries:
    """Immutable definition of a provider's daily series for one listing.

    Attributes:
        series_id: Internal immutable series ID.
        listing_id: Internal listing ID.
        provider: Must equal the provider on batches supplying its observations.
        provider_field: Provider field or source series name.
        basis: Raw close, split-adjusted price or total-return index.
        currency: ISO-style three-letter currency code.
        calendar_name: Name/version of the caller's session calendar.
        timezone: Exchange timezone used to validate observation session dates.
        unit_multiplier: Multiplier into major currency units, or index units for TRI.
    """

    series_id: str
    listing_id: str
    provider: str
    provider_field: str
    basis: PriceBasis
    currency: str
    calendar_name: str
    timezone: str
    unit_multiplier: float = 1.0


@dataclass(frozen=True)
class PriceObservation:
    """One observation, correction or withdrawal in an ingestion batch.

    Attributes:
        session_date: Exchange session label, YYYY-MM-DD.
        observed_at: Aware timestamp of the original observation, not its correction.
        value: Finite non-negative value; None only for an explicit withdrawal.
        is_deleted: Withdraw the observation without reviving an earlier revision.
    """

    session_date: str
    observed_at: str | datetime
    value: float | None
    is_deleted: bool = False


def _label(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('Identifiers and labels must be nonempty strings without edge whitespace')
    return value


def _code(value: str, pattern: str) -> str:
    if not re.fullmatch(pattern, _label(value)):
        raise ValueError(f'Invalid venue or currency code: {value!r}')
    return value


def _day(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Session dates must use YYYY-MM-DD')
    datetime.strptime(value, '%Y-%m-%d')
    return value


def _utc(value: str | datetime) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.nanosecond:
        raise ValueError('Timestamps must be timezone-aware with at most microsecond precision')
    return stamp.to_pydatetime().astimezone(timezone.utc).isoformat(
        timespec='microseconds').replace('+00:00', 'Z')


def _target_column(scheme: IdentifierScheme) -> str:
    if scheme in (IdentifierScheme.LEI, IdentifierScheme.PROVIDER_ENTITY):
        return 'entity_id'
    if scheme in (IdentifierScheme.ISIN, IdentifierScheme.CUSIP,
                  IdentifierScheme.FIGI_SHARE_CLASS, IdentifierScheme.PROVIDER_SECURITY):
        return 'security_id'
    return 'listing_id'


class SecurityMaster:
    """Own a local SQLite connection with atomic writes and explicit as-of reads.

    Args:
        database: Existing/new database filename, with an existing parent directory,
            or ':memory:'. Pass a C-local path explicitly on this Windows setup.

    Raises:
        ValueError: The existing database has an unrelated or unsupported schema.
    """

    def __init__(self, database: str | Path = ':memory:') -> None:
        location = (str(database) if str(database) == ':memory:'
                    else str(Path(database).expanduser()))
        self._con = sqlite3.connect(location, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._closed = False
        self._writer = None
        try:
            self._con.execute('PRAGMA foreign_keys = ON')
            # Serialise initialisation against another process opening the same fresh file.
            self._con.execute('BEGIN IMMEDIATE')
            version = self._con.execute('PRAGMA user_version').fetchone()[0]
            app_id = self._con.execute('PRAGMA application_id').fetchone()[0]
            tables = self._con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            if version == 0 and app_id == 0 and not tables:
                # Execute individual complete statements: executescript would commit our lock.
                statement = ''
                for line in Path(__file__).with_name('schema.sql').read_text(
                        encoding='utf-8').splitlines(keepends=True):
                    statement += line
                    if sqlite3.complete_statement(statement):
                        self._con.execute(statement)
                        statement = ''
                self._con.execute(f'PRAGMA application_id = {APPLICATION_ID}')
                self._con.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            elif (version, app_id) != (SCHEMA_VERSION, APPLICATION_ID):
                raise ValueError('Unrelated or unsupported security-master database schema')
            self._con.commit()
        except BaseException:
            self._con.rollback()
            self._con.close()
            self._closed = True
            raise

    def __enter__(self) -> 'SecurityMaster':
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError('Security master is closed')

    def close(self) -> None:
        """Close the connection; finish an active ingestion batch first."""
        if self._writer is not None:
            raise RuntimeError('Finish the active ingestion batch before closing')
        if not self._closed:
            self._con.close()
            self._closed = True

    @contextmanager
    def ingest_batch(self, *, provider: str, raw_payload: bytes, source_ref: str,
                     accepted_at: str | datetime | None = None) -> Iterator['IngestBatch']:
        """Commit a batch together, or roll it all back if its context raises.

        Args:
            provider: Provider namespace for provenance and price-source validation.
            raw_payload: Original bytes; their SHA-256 is recorded, not the bytes themselves.
            source_ref: Non-secret locator of the caller's archived payload.
            accepted_at: Acceptance time, defaulting to UTC now. Explicit times are for
                trusted archived ingestion evidence or simulation. Must strictly increase
                across committed batches; never substitute the observation date.

        Yields:
            Writer valid only inside this context. Nested batches are rejected.
        """
        self._ensure_open()
        if self._writer is not None:
            raise RuntimeError('Nested ingestion batches are not supported')
        provider, source_ref = _label(provider), _label(source_ref)
        if not isinstance(raw_payload, bytes):
            raise TypeError('raw_payload must be the original bytes')
        cutoff = _utc(accepted_at if accepted_at is not None else datetime.now(timezone.utc))
        self._con.execute('BEGIN IMMEDIATE')
        try:
            latest = self._con.execute('SELECT MAX(accepted_at) FROM ingest_batch').fetchone()[0]
            if latest is not None and cutoff <= latest:
                raise ValueError('Batch acceptance times must strictly increase')
            cursor = self._con.execute(
                'INSERT INTO ingest_batch(provider, accepted_at, raw_sha256, source_ref) '
                'VALUES (?, ?, ?, ?)',
                (provider, cutoff, hashlib.sha256(raw_payload).hexdigest(), source_ref))
            writer = IngestBatch(self, cursor.lastrowid, cutoff)
            self._writer = writer
            yield writer
            self._con.commit()
        except BaseException:
            self._con.rollback()
            raise
        finally:
            self._writer = None

    @contextmanager
    def _read_snapshot(self):
        self._ensure_open()
        owns_transaction = not self._con.in_transaction
        if owns_transaction:
            self._con.execute('BEGIN')
        try:
            yield
        finally:
            if owns_transaction:
                self._con.rollback()

    def resolve_identifier(self, scheme: IdentifierScheme, namespace: str, value: str, *,
                           effective_date: str, known_at: str | datetime,
                           mic: str | None = None, currency: str | None = None) -> str:
        """Resolve exactly one entity/security/listing ID, or raise LookupError.

        Args:
            scheme: Determines the target level of the returned ID.
            namespace: Exact vendor or exchange naming convention.
            value: Exact identifier spelling.
            effective_date: Effective session date; interval ends are exclusive.
            known_at: Explicit knowledge cutoff; timestamp starts are inclusive.
            mic: Required for ticker lookups; omit for other schemes.
            currency: Required for ticker lookups; omit for other schemes.
        """
        self._ensure_open()
        scheme = IdentifierScheme(scheme)
        scope = ''
        if scheme == IdentifierScheme.TICKER:
            scope = f'{_code(mic, "[A-Z0-9]{4}")}:{_code(currency, "[A-Z]{3}")}'
        elif mic is not None or currency is not None:
            raise ValueError('Venue/currency scope is only used for tickers')
        day, cutoff = _day(effective_date), _utc(known_at)
        target = _target_column(scheme)
        rows = self._con.execute(
            f'SELECT {target} FROM identifier_assignment WHERE scheme = ? '
            'AND namespace = ? AND scope = ? AND value = ? '
            'AND valid_from <= ? AND ? < valid_to '
            'AND known_from <= ? AND ? < known_to',
            (scheme.value, _label(namespace), scope, _label(value), day, day, cutoff, cutoff)
        ).fetchall()
        if len(rows) != 1:
            raise LookupError(f'{value}: expected one accepted mapping, got {len(rows)}')
        return rows[0][0]

    def resolve_ticker(self, ticker: str, *, namespace: str, mic: str, currency: str,
                       effective_date: str, known_at: str | datetime) -> str:
        """Resolve a ticker to a listing ID; use the same timing contract as resolve_identifier.

        Args:
            ticker: Exact ticker spelling.
            namespace: Exchange or vendor naming convention.
            mic: Venue MIC.
            currency: Trading currency.
            effective_date: Session date to resolve.
            known_at: Explicit knowledge cutoff.
        """
        return self.resolve_identifier(
            IdentifierScheme.TICKER, namespace, ticker, mic=mic, currency=currency,
            effective_date=effective_date, known_at=known_at)

    def load_price_panel(self, series_ids: Sequence[str], *, sessions: pd.DatetimeIndex,
                         known_at: str | datetime) -> pd.DataFrame:
        """Load daily prices as known at ONE decision time, retaining missing observations.

        Args:
            series_ids: Explicit selection, one series per listing, in output column order.
            sessions: Increasing unique timezone-naive midnight labels from a versioned
                calendar. This method does not infer sessions or forward-fill gaps.
            known_at: Knowledge cutoff. Replay must reload at each decision time rather
                than reuse a panel assembled with an end-of-backtest cutoff.

        Returns:
            DataFrame indexed by session date, with listing IDs as columns. Values are
            multiplied by their series' unit multipliers. attrs records selected series,
            currency, price basis and the normalised knowledge cutoff.

        Raises:
            ValueError: Missing/duplicate series, mixed bases/currencies or invalid sessions.
        """
        if isinstance(series_ids, str) or not series_ids or len(set(series_ids)) != len(series_ids):
            raise ValueError('Select nonempty distinct series IDs')
        series_ids = [_label(value) for value in series_ids]
        if (not isinstance(sessions, pd.DatetimeIndex) or sessions.empty or sessions.hasnans
                or not sessions.is_unique or not sessions.is_monotonic_increasing
                or sessions.tz is not None or not sessions.equals(sessions.normalize())):
            raise ValueError('Supply increasing unique timezone-naive session-date labels')
        cutoff = _utc(known_at)
        placeholders = ','.join('?' for _ in series_ids)
        with self._read_snapshot():
            meta = pd.read_sql_query(
                f'SELECT * FROM price_series WHERE series_id IN ({placeholders})',
                self._con, params=series_ids)
            if len(meta) != len(series_ids) or meta['listing_id'].duplicated().any():
                raise ValueError('Need exactly one explicitly selected series per listing')
            if meta['currency'].nunique() != 1 or meta['basis'].nunique() != 1:
                raise ValueError('Select one currency and one price basis per panel')
            meta = meta.set_index('series_id').loc[series_ids]
            rows = pd.read_sql_query(
                f'''WITH versions AS (
                    SELECT p.*, ROW_NUMBER() OVER (
                        PARTITION BY p.series_id, p.session_date
                        ORDER BY b.accepted_at DESC, b.batch_id DESC
                    ) AS version_rank
                    FROM price_revision p JOIN ingest_batch b ON b.batch_id = p.batch_id
                    WHERE p.series_id IN ({placeholders})
                      AND p.session_date BETWEEN ? AND ? AND b.accepted_at <= ?
                ) SELECT series_id, session_date, value FROM versions
                  WHERE version_rank = 1 AND is_deleted = 0''',
                self._con, params=[*series_ids, sessions[0].strftime('%Y-%m-%d'),
                                   sessions[-1].strftime('%Y-%m-%d'), cutoff],
                parse_dates=['session_date'])
        # Remove withdrawals AFTER ranking, preventing resurrection of the old value.
        panel = rows.pivot(index='session_date', columns='series_id', values='value')
        panel = panel.reindex(index=sessions, columns=series_ids).astype(float)
        panel = panel.mul(meta['unit_multiplier'], axis='columns')
        panel.columns = pd.Index(meta['listing_id'], name='listing_id')
        panel = panel.rename_axis(index='date')
        panel.attrs = {'known_at': cutoff, 'series_ids': series_ids,
                       'currency': meta['currency'].iloc[0], 'basis': meta['basis'].iloc[0]}
        return panel


class IngestBatch:
    """Scoped writer yielded by SecurityMaster.ingest_batch; do not construct directly.

    Args:
        master: Owning master.
        batch_id: Batch row in the active transaction.
        accepted_at: Canonical UTC timestamp of this batch.
    """

    def __init__(self, master: SecurityMaster, batch_id: int, accepted_at: str) -> None:
        self._master = master
        self._batch_id = batch_id
        self._accepted_at = accepted_at

    @property
    def _con(self) -> sqlite3.Connection:
        if self._master._writer is not self:
            raise RuntimeError('Ingestion writer is no longer active')
        return self._master._con

    @contextmanager
    def _atomic(self):
        con = self._con
        con.execute('SAVEPOINT batch_operation')
        try:
            yield
            con.execute('RELEASE batch_operation')
        except BaseException:
            con.execute('ROLLBACK TO batch_operation')
            con.execute('RELEASE batch_operation')
            raise

    def add_entity(self, entity_id: str, display_name: str) -> None:
        """Register an immutable entity ID and a descriptive, nonhistorical label.

        Args:
            entity_id: Internal immutable ID.
            display_name: Label for display; not point-in-time fundamentals.
        """
        self._con.execute('INSERT INTO entity VALUES (?, ?)',
                          (_label(entity_id), _label(display_name)))

    def add_security(self, security_id: str, issuer_id: str, *, security_type: SecurityType,
                     share_class: str, underlying_security_id: str | None = None) -> None:
        """Register one security, separate from any venue where it trades.

        Args:
            security_id: Internal immutable security ID.
            issuer_id: Existing entity ID.
            security_type: Supported cash-equity type.
            share_class: Descriptive class label.
            underlying_security_id: Existing underlying security, for example for an ADR.
        """
        self._con.execute('INSERT INTO security VALUES (?, ?, ?, ?, ?)',
                          (_label(security_id), _label(issuer_id),
                           SecurityType(security_type).value, _label(share_class),
                           underlying_security_id))

    def add_listing(self, listing_id: str, security_id: str, *, mic: str, currency: str) -> None:
        """Register an immutable venue/currency trading line.

        Args:
            listing_id: Internal immutable listing ID.
            security_id: Existing security ID.
            mic: Four-character venue MIC.
            currency: Three-letter trading currency.
        """
        self._con.execute('INSERT INTO listing VALUES (?, ?, ?, ?)',
                          (_label(listing_id), _label(security_id), _code(mic, '[A-Z0-9]{4}'),
                           _code(currency, '[A-Z]{3}')))

    def add_identifier(self, assignment: IdentifierAssignment) -> int:
        """Insert an alias with this batch's knowledge time; overlapping aliases fail.

        Args:
            assignment: Effective interval and correctly typed target identity.

        Returns:
            Assignment row ID, for audit purposes.
        """
        scheme = IdentifierScheme(assignment.scheme)
        target = _target_column(scheme)
        scope = ''
        if scheme == IdentifierScheme.TICKER:
            listing = self._con.execute('SELECT mic, currency FROM listing WHERE listing_id = ?',
                                        (_label(assignment.target_id),)).fetchone()
            if listing is None:
                raise ValueError('Unknown listing ID')
            scope = f'{listing[0]}:{listing[1]}'
        start = _day(assignment.valid_from)
        end = FOREVER_DATE if assignment.valid_to is None else _day(assignment.valid_to)
        if start >= end:
            raise ValueError('Identifier effective intervals must be nonempty')
        cursor = self._con.execute(
            'INSERT INTO identifier_assignment(scheme, namespace, scope, value, '
            f'{target}, valid_from, valid_to, known_from, source_batch_id) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (scheme.value, _label(assignment.namespace), scope, _label(assignment.value),
             _label(assignment.target_id), start, end, self._accepted_at, self._batch_id))
        return cursor.lastrowid

    def rename_ticker(self, listing_id: str, *, namespace: str, old_ticker: str,
                      new_ticker: str, effective_date: str) -> None:
        """Atomically split a current alias's validity while retaining earlier knowledge.

        Args:
            listing_id: Listing whose identity is retained.
            namespace: Vendor/exchange naming convention.
            old_ticker: Current alias at effective_date.
            new_ticker: Replacement alias; must be available for this interval.
            effective_date: Inclusive start of the new ticker. Can precede acceptance
                for a late notice, but the correction is known only from this batch.

        Raises:
            LookupError: No single current alias covers effective_date.
            ValueError: No actual rename, or the old alias was created in this same batch.
            sqlite3.IntegrityError: The new ticker conflicts; the operation is rolled back.
        """
        day = _day(effective_date)
        if _label(new_ticker) == _label(old_ticker):
            raise ValueError('A rename requires different old and new tickers')
        with self._atomic():
            rows = self._con.execute(
                "SELECT * FROM identifier_assignment WHERE scheme = 'TICKER' "
                'AND listing_id = ? AND namespace = ? AND value = ? AND known_to = ? '
                'AND valid_from <= ? AND ? < valid_to',
                (_label(listing_id), _label(namespace), old_ticker, FOREVER_TIME, day, day)
            ).fetchall()
            if len(rows) != 1:
                raise LookupError('No single current ticker assignment covers the rename date')
            old = rows[0]
            if old['known_from'] >= self._accepted_at:
                raise ValueError('Rename must occur in a later batch than the original assignment')
            self._con.execute(
                'UPDATE identifier_assignment SET known_to = ? WHERE assignment_id = ?',
                (self._accepted_at, old['assignment_id']))
            if old['valid_from'] < day:
                self.add_identifier(IdentifierAssignment(
                    IdentifierScheme.TICKER, namespace, old_ticker, listing_id,
                    old['valid_from'], day))
            self.add_identifier(IdentifierAssignment(
                IdentifierScheme.TICKER, namespace, new_ticker, listing_id, day, old['valid_to']))

    def add_price_series(self, series: PriceSeries) -> None:
        """Register an immutable daily-series definition.

        Args:
            series: Explicit provider, adjustment convention, units and calendar metadata.
        """
        scale = float(series.unit_multiplier)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('unit_multiplier must be finite and positive')
        try:
            pd.Timestamp('2000-01-01', tz=_label(series.timezone))
        except (ValueError, KeyError) as exc:
            raise ValueError('Invalid series timezone') from exc
        self._con.execute('INSERT INTO price_series VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                          (_label(series.series_id), _label(series.listing_id),
                           _label(series.provider), _label(series.provider_field),
                           PriceBasis(series.basis).value, _code(series.currency, '[A-Z]{3}'),
                           scale, _label(series.calendar_name), _label(series.timezone)))

    def add_prices(self, series_id: str, observations: Sequence[PriceObservation]) -> None:
        """Append revisions atomically, rejecting nonfinite values and future observations.

        Args:
            series_id: Existing series ID supplied by this batch's provider.
            observations: Daily observations/corrections/withdrawals. Missing feed rows
                should be omitted; a withdrawal requires is_deleted=True and value=None.
        """
        series_id = _label(series_id)
        definition = self._con.execute(
            'SELECT timezone FROM price_series WHERE series_id = ?', (series_id,)).fetchone()
        if definition is None:
            raise ValueError('Unknown price series')
        rows = []
        for item in observations:
            if type(item.is_deleted) is not bool:
                raise ValueError('is_deleted must be a boolean')
            value = None if item.value is None else float(item.value)
            if (item.is_deleted and value is not None) or (not item.is_deleted and (
                    value is None or not math.isfinite(value) or value < 0)):
                raise ValueError('Use a finite non-negative value or an explicit withdrawal')
            day, observed_at = _day(item.session_date), _utc(item.observed_at)
            local_stamp = pd.Timestamp(observed_at).tz_convert(definition['timezone'])
            local_day = local_stamp.date().isoformat()
            if day != local_day:
                raise ValueError('Observation timestamp must belong to its local session date')
            rows.append((series_id, day, observed_at, self._batch_id, value, int(item.is_deleted)))
        with self._atomic():
            self._con.executemany('INSERT INTO price_revision VALUES (?, ?, ?, ?, ?, ?)', rows)
