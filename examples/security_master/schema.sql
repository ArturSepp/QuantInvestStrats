-- Venue-level cash-equity/ETF security master, SQLite, schema version 1.
-- Not a migration of qis.sql_engine and not a complete corporate-action engine.
-- All intervals are [from, to). All dates are YYYY-MM-DD; timestamps are UTC
-- YYYY-MM-DDTHH:MM:SS.ffffffZ (normalised by store.py). Never backdate knowledge.
-- Every new connection must enable foreign keys before opening a transaction.
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE entity (
    entity_id TEXT PRIMARY KEY NOT NULL,
    display_name TEXT NOT NULL  -- descriptive label, not historical fundamentals
);

CREATE TABLE security (
    security_id TEXT PRIMARY KEY NOT NULL,
    issuer_id TEXT NOT NULL REFERENCES entity(entity_id),
    security_type TEXT NOT NULL CHECK (security_type IN ('COMMON_STOCK', 'ETF', 'ADR')),
    share_class TEXT NOT NULL,
    underlying_security_id TEXT REFERENCES security(security_id)
);

CREATE TABLE listing (
    listing_id TEXT PRIMARY KEY NOT NULL,
    security_id TEXT NOT NULL REFERENCES security(security_id),
    mic TEXT NOT NULL CHECK (length(mic) = 4),
    currency TEXT NOT NULL CHECK (length(currency) = 3)
    -- Stable trading-line identity. Do not mutate venue/currency/parent links.
    -- An actual venue/currency migration gets a new line plus an explicit event.
);

CREATE TABLE ingest_batch (
    batch_id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL CHECK (length(raw_sha256) = 64),
    source_ref TEXT NOT NULL
    -- accepted_at is when this system accepted the data, not the observation date.
    -- raw_sha256 addresses the preserved input; source_ref is a non-secret locator.
);

CREATE UNIQUE INDEX batch_acceptance ON ingest_batch(accepted_at);

CREATE TABLE identifier_assignment (
    assignment_id INTEGER PRIMARY KEY,
    scheme TEXT NOT NULL CHECK (scheme IN (
        'TICKER', 'LEI', 'ISIN', 'CUSIP', 'FIGI_SHARE_CLASS', 'FIGI_VENUE',
        'PROVIDER_ENTITY', 'PROVIDER_SECURITY', 'PROVIDER_LISTING'
    )),
    namespace TEXT NOT NULL,  -- exchange/vendor naming convention, not just 'ticker'
    scope TEXT NOT NULL,      -- TICKER uses MIC:currency; global identifiers use ''
    value TEXT NOT NULL,
    entity_id TEXT REFERENCES entity(entity_id),
    security_id TEXT REFERENCES security(security_id),
    listing_id TEXT REFERENCES listing(listing_id),
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL DEFAULT '9999-12-31',
    known_from TEXT NOT NULL,
    known_to TEXT NOT NULL DEFAULT '9999-12-31T23:59:59.999999Z',
    source_batch_id INTEGER NOT NULL REFERENCES ingest_batch(batch_id),
    CHECK (valid_from < valid_to),
    CHECK (known_from < known_to),
    CHECK ((entity_id IS NOT NULL) + (security_id IS NOT NULL)
           + (listing_id IS NOT NULL) = 1),
    CHECK (
        (scheme IN ('LEI', 'PROVIDER_ENTITY') AND entity_id IS NOT NULL) OR
        (scheme IN ('ISIN', 'CUSIP', 'FIGI_SHARE_CLASS', 'PROVIDER_SECURITY')
            AND security_id IS NOT NULL) OR
        (scheme IN ('TICKER', 'FIGI_VENUE', 'PROVIDER_LISTING')
            AND listing_id IS NOT NULL)
    )
);

CREATE INDEX identifier_lookup ON identifier_assignment
    (scheme, namespace, scope, value, valid_from, known_from);

-- One exact alias must have at most one accepted answer in BOTH time dimensions.
-- Conflicting vendor assertions belong in staging until reconciled.
CREATE TRIGGER identifier_no_overlap
BEFORE INSERT ON identifier_assignment
WHEN EXISTS (
    SELECT 1 FROM identifier_assignment AS old
    WHERE old.scheme = NEW.scheme AND old.namespace = NEW.namespace
      AND old.scope = NEW.scope AND old.value = NEW.value
      AND old.valid_from < NEW.valid_to AND NEW.valid_from < old.valid_to
      AND old.known_from < NEW.known_to AND NEW.known_from < old.known_to
)
BEGIN
    SELECT RAISE(ABORT, 'overlapping identifier assignment');
END;

CREATE TRIGGER identifier_check_provenance
BEFORE INSERT ON identifier_assignment
WHEN NEW.known_from != (
    SELECT accepted_at FROM ingest_batch WHERE batch_id = NEW.source_batch_id
)
BEGIN
    SELECT RAISE(ABORT, 'knowledge time must equal batch acceptance time');
END;

CREATE TRIGGER identifier_check_ticker_scope
BEFORE INSERT ON identifier_assignment
WHEN NEW.scheme = 'TICKER' AND NEW.scope != (
    SELECT mic || ':' || currency FROM listing WHERE listing_id = NEW.listing_id
)
BEGIN
    SELECT RAISE(ABORT, 'ticker scope differs from listing venue/currency');
END;

-- The sole allowed UPDATE closes knowledge time; insert replacement facts in
-- the same transaction. This can only shrink an interval, never create overlap.
CREATE TRIGGER identifier_close_only
BEFORE UPDATE ON identifier_assignment
WHEN OLD.known_to != '9999-12-31T23:59:59.999999Z' OR NEW.known_to >= OLD.known_to
  OR NEW.assignment_id IS NOT OLD.assignment_id OR NEW.scheme IS NOT OLD.scheme
  OR NEW.namespace IS NOT OLD.namespace OR NEW.scope IS NOT OLD.scope
  OR NEW.value IS NOT OLD.value OR NEW.entity_id IS NOT OLD.entity_id
  OR NEW.security_id IS NOT OLD.security_id OR NEW.listing_id IS NOT OLD.listing_id
  OR NEW.valid_from IS NOT OLD.valid_from OR NEW.valid_to IS NOT OLD.valid_to
  OR NEW.known_from IS NOT OLD.known_from OR NEW.source_batch_id IS NOT OLD.source_batch_id
BEGIN
    SELECT RAISE(ABORT, 'close knowledge interval and insert a new version');
END;

CREATE TRIGGER identifier_no_delete BEFORE DELETE ON identifier_assignment
BEGIN
    SELECT RAISE(ABORT, 'identifier history must be retained');
END;

CREATE TABLE price_series (
    series_id TEXT PRIMARY KEY NOT NULL,
    listing_id TEXT NOT NULL REFERENCES listing(listing_id),
    provider TEXT NOT NULL,
    provider_field TEXT NOT NULL,
    basis TEXT NOT NULL CHECK (basis IN ('RAW_CLOSE', 'SPLIT_ADJUSTED', 'TOTAL_RETURN_INDEX')),
    currency TEXT NOT NULL CHECK (length(currency) = 3),
    unit_multiplier REAL NOT NULL CHECK (unit_multiplier > 0),
    calendar_name TEXT NOT NULL,
    timezone TEXT NOT NULL,
    UNIQUE (listing_id, provider, provider_field, basis, currency)
    -- Values * unit_multiplier are in major currency units, or index units for TRI.
    -- A change in provider, adjustment convention, currency or units starts a new series.
);

CREATE TABLE price_revision (
    series_id TEXT NOT NULL REFERENCES price_series(series_id),
    session_date TEXT NOT NULL,
    observed_at TEXT NOT NULL,  -- actual observation/close timestamp, in UTC
    batch_id INTEGER NOT NULL REFERENCES ingest_batch(batch_id),
    value REAL,
    is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
    PRIMARY KEY (series_id, session_date, batch_id),
    CHECK ((is_deleted = 0 AND value IS NOT NULL AND value >= 0)
        OR (is_deleted = 1 AND value IS NULL))
);

CREATE TRIGGER price_check_batch BEFORE INSERT ON price_revision
WHEN (SELECT accepted_at FROM ingest_batch WHERE batch_id = NEW.batch_id) < NEW.observed_at
  OR (SELECT provider FROM ingest_batch WHERE batch_id = NEW.batch_id) !=
     (SELECT provider FROM price_series WHERE series_id = NEW.series_id)
BEGIN
    SELECT RAISE(ABORT, 'price batch has incompatible time or provider');
END;

CREATE TRIGGER price_no_update BEFORE UPDATE ON price_revision
BEGIN
    SELECT RAISE(ABORT, 'append a price correction instead');
END;

CREATE TRIGGER price_no_delete BEFORE DELETE ON price_revision
BEGIN
    SELECT RAISE(ABORT, 'append a price tombstone instead');
END;

-- Preserve foreign-key meaning and all acceptance timestamps once inserted.
CREATE TRIGGER entity_no_update BEFORE UPDATE ON entity
BEGIN SELECT RAISE(ABORT, 'reference identities are immutable'); END;
CREATE TRIGGER security_no_update BEFORE UPDATE ON security
BEGIN SELECT RAISE(ABORT, 'reference identities are immutable'); END;
CREATE TRIGGER listing_no_update BEFORE UPDATE ON listing
BEGIN SELECT RAISE(ABORT, 'reference identities are immutable'); END;
CREATE TRIGGER series_no_update BEFORE UPDATE ON price_series
BEGIN SELECT RAISE(ABORT, 'series definitions are immutable'); END;
CREATE TRIGGER batch_no_update BEFORE UPDATE ON ingest_batch
BEGIN SELECT RAISE(ABORT, 'ingestion provenance is immutable'); END;

CREATE TRIGGER entity_no_delete BEFORE DELETE ON entity
BEGIN SELECT RAISE(ABORT, 'reference identities must be retained'); END;
CREATE TRIGGER security_no_delete BEFORE DELETE ON security
BEGIN SELECT RAISE(ABORT, 'reference identities must be retained'); END;
CREATE TRIGGER listing_no_delete BEFORE DELETE ON listing
BEGIN SELECT RAISE(ABORT, 'reference identities must be retained'); END;
CREATE TRIGGER series_no_delete BEFORE DELETE ON price_series
BEGIN SELECT RAISE(ABORT, 'series definitions must be retained'); END;
CREATE TRIGGER batch_no_delete BEFORE DELETE ON ingest_batch
BEGIN SELECT RAISE(ABORT, 'ingestion provenance must be retained'); END;

-- Add before production equity backtesting:
-- corporate_action_revision(event_id, security_id, successor_security_id, event_type,
--   ex_date, pay_date, cash_amount, currency, share_ratio, source_batch_id, is_deleted)
-- universe_membership_revision(universe_id, listing_id, valid_from, valid_to,
--   known_from, known_to, source_batch_id)
-- dataset_release(release_id, created_at, knowledge_cutoff, database_backup_sha256,
--   raw_manifest_sha256, selected_series_json, calendar_version, code_revision)
-- Scope extension: SEDOL/composite FIGI require explicit country/composite objects;
-- do not force them to be unique venue aliases. ISIN -> security -> listings is one-to-many.
