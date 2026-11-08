# Security master for QIS research

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A security master separates an instrument's identity from its changing market and vendor labels.
[store.py](store.py) implements a reusable local SQLite store; [schema.sql](schema.sql) defines its
tables and integrity constraints. [ticker_history.py](ticker_history.py) demonstrates ingestion and
QIS integration on the frozen synthetic data. These modules belong to the repository's examples;
they are not exported by `qis` or included in its wheel. No optional dependencies are required.

## Data contract

| Table | Meaning |
|---|---|
| `entity` | Immutable issuer ID and descriptive label. |
| `security` | Security/share-class ID linked to an issuer; an ADR can link to its underlying. |
| `listing` | Immutable security/venue MIC/trading-currency combination with its own ID. |
| `identifier_assignment` | Typed and namespaced alias with effective and knowledge intervals. |
| `ingest_batch` | Provider, acceptance time, original-payload SHA-256 and archival source reference. |
| `price_series` | Immutable provider field, listing, currency, basis, units, calendar and timezone. |
| `price_revision` | Daily observation or withdrawal, versioned by ingestion batch. |

Ticker scope is `MIC:currency`; exact spelling and namespace are retained. A security-level
identifier resolves to a security ID, not to an arbitrarily selected listing. Supported identifier
schemes are enumerated by `IdentifierScheme`. SEDOL and composite FIGI require additional market
or composite objects and are intentionally not forced into venue-level slots.

Intervals use inclusive starts and exclusive ends:

```text
valid_from <= effective_date < valid_to
known_from <= known_at < known_to
```

Dates are exchange session labels in `YYYY-MM-DD` format. Aware timestamps are normalised to UTC
with six fractional digits; naive and sub-microsecond timestamps are rejected. The caller supplies
the session calendar. There is no implied annualisation or calendar inference.

Knowledge time means acceptance by this store. Batches must have strictly increasing acceptance
timestamps, defaulting to UTC now. Explicit past timestamps are suitable only for simulations or
verified archived ingestion evidence. Downloading history today does not establish that it was
available to a strategy in the past. A rename notice received late corrects effective history only
for knowledge cutoffs on or after its acceptance.

## Ingestion and lookup

Create `SecurityMaster()` for an in-memory store. To persist it, pass a database filename with an
existing parent directory. On this Windows setup use C-local storage below the configured
`AGENT_LOCAL_ROOT`; do not place the live database or generated outputs in OneDrive. Caller paths
can be resolved through `qis.get_resource_path(local_path=..., create=True)` without changing the
distributed YAML. Archive raw inputs separately: the store records their hash and locator, not
their bytes. Back up the database and raw-input manifest for each frozen research release.

```python
from examples.security_master.store import (
    IdentifierAssignment, IdentifierScheme, SecurityMaster, SecurityType,
)

with SecurityMaster() as master:
    with master.ingest_batch(
        provider="EXAMPLE_FEED",
        raw_payload=b"Fictional initial record",
        source_ref="synthetic:initial",
        accepted_at="2020-01-01T12:00:00Z",
    ) as batch:
        batch.add_entity("E1", "Synthetic Alpha")
        batch.add_security("S1", "E1", security_type=SecurityType.COMMON_STOCK,
                           share_class="ordinary")
        batch.add_listing("L1", "S1", mic="XNYS", currency="USD")
        batch.add_identifier(IdentifierAssignment(
            IdentifierScheme.TICKER, "EXAMPLE_FEED", "ABC", "L1", "2020-01-02"))

    with master.ingest_batch(
        provider="EXAMPLE_FEED", raw_payload=b"Fictional rename notice",
        source_ref="synthetic:rename", accepted_at="2020-01-03T12:00:00Z",
    ) as batch:
        batch.rename_ticker("L1", namespace="EXAMPLE_FEED", old_ticker="ABC",
                            new_ticker="XYZ", effective_date="2020-01-06")

    listing_id = master.resolve_ticker(
        "XYZ", namespace="EXAMPLE_FEED", mic="XNYS", currency="USD",
        effective_date="2020-01-06", known_at="2020-01-06T22:00:00Z",
    )  # L1
```

An ingestion context commits all its writes or rolls them back. Its writer cannot be used after
exit. Renames and bulk price inserts also use savepoints, so catching an error inside a batch does
not retain a partially applied operation. Unknown/ambiguous aliases raise `LookupError`;
conflicting assignments and relational-integrity failures raise `sqlite3.IntegrityError`.
Invalid input or unsupported database versions raise `ValueError`. Schema version 1 is initialised
only in an empty database. The earlier proposal database is not silently migrated.

## Prices and QIS

Register a `PriceSeries` once. Supply observations with `batch.add_prices(series_id, observations)`
using `PriceObservation`. An observation's timestamp must fall on its session date in the series'
timezone and must not exceed its batch acceptance time. The batch must belong to the series'
provider. Corrections append revisions; withdrawals use
`value=None, is_deleted=True`. NaN feed values should be omitted, not used as withdrawal markers.

```python
import qis
from examples.security_master.store import SecurityMaster
from examples.security_master.ticker_history import populate_demo

with SecurityMaster() as master:
    sessions = populate_demo(master)
    prices = master.load_price_panel(
        ["PX_L1", "PX_L2"], sessions=sessions, known_at="2020-01-10T22:00:00Z")
    simple_returns = qis.to_returns(prices, is_log_returns=False, ffill_nans=False)
```

Columns are stable listing IDs. Select exactly one series per listing and one common currency and
price basis per panel. Values are multiplied by their explicit unit multipliers. No missing
observations are filled. `DataFrame.attrs` retains the cutoff, series IDs, currency and basis.
For a replay, reload each decision's lookback with that decision's cutoff; using a final-date
cutoff for all decisions would expose later corrections. The returned metadata is descriptive,
not a historical issuer or investment-universe filter.

The example is daily, synthetic and raw-close based. Alpha changes from `ABC` to `XYZ` on January 6,
2020 while retaining `L1`; Beta starts using `ABC` on January 8 with `L2`. There is no return bridging
these two issuers. A withdrawal accepted January 13 leaves the January 10 data vintage unchanged.
No dividends, delisting payoffs, portfolio trades, FX conversion or annualisation are calculated.

## Run and verify

First apply the repository's mandatory environment setup in the same PowerShell session:

```powershell
. "$env:USERPROFILE\OneDrive\analytics\my_github\ArturSepp\scripts\repo_governance\Enter-AgentRepo.ps1"
& 'C:\Python\QuantInvestStrats312\Scripts\python.exe' -m examples.security_master.ticker_history
```

For tests, use a C-local source export with this directory present and the setup above applied.
Run from that export's root, with its `src` directory on `PYTHONPATH`. Tests are inside this example
directory as requested; the repository's default `testpaths` excludes them, so name the file:

```powershell
& 'C:\Python\QuantInvestStrats312\Scripts\python.exe' -m pytest examples/security_master/tests/security_master_test.py -o "cache_dir=$env:AGENT_LOCAL_ROOT\cache\pytest"
& 'C:\Python\QuantInvestStrats312\Scripts\python.exe' -m ruff check examples/security_master
```

The tests cover rename/reuse boundaries, late notices, context, overlap rejection, atomic rollback,
price publication lag, revisions, withdrawals, units, missing observations, invalid input and
persistent reopening. The QIS return adapter is compared with direct arithmetic on the canonical
synthetic source panel. This validates identity and version handling, not total-return accounting.

## Scope

The store handles venue-level cash equities, ETFs and ADRs. Issuer relations, series definitions and
descriptive labels are immutable in this version; metadata corrections require an explicit future
migration rather than in-place edits. It does not fetch vendor data, reconcile fuzzy matches,
construct corporate-action-adjusted histories, track historical index membership or choose a
primary listing. Those must be supplied explicitly upstream of an investment backtest.

See [SQLite foreign-key enforcement](https://www.sqlite.org/foreignkeys.html) and
[appropriate SQLite deployments](https://www.sqlite.org/whentouse.html) for storage behaviour.
