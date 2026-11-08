"""Offline ticker rename/reuse example with point-in-time daily price revisions.

Run from the checkout: python -m examples.security_master.ticker_history.
All identifiers and events are fictional. Prices use the frozen QIS synthetic data.
The default run uses an in-memory database and writes no output files.
"""

from enum import Enum

import pandas as pd
import qis
from qis.datasets.synthetic import generate_synthetic_prices

from examples.security_master.store import (
    IdentifierAssignment, IdentifierScheme, PriceBasis, PriceObservation, PriceSeries,
    SecurityMaster, SecurityType,
)


class Locals(Enum):
    """Available offline demonstrations."""

    TICKER_HISTORY = 1


def populate_demo(master: SecurityMaster) -> pd.DatetimeIndex:
    """Populate an empty master with fictional, chronologically accepted feed batches.

    Args:
        master: Empty master owned by the caller; memory or an explicit local file.

    Returns:
        Synthetic business-day session grid. This is not a real exchange calendar.
    """
    source = generate_synthetic_prices(start='2020-01-02', end='2020-01-10')
    prices = source[['SEQ_US', 'SEQ_EU']].set_axis(['L1', 'L2'], axis=1)
    prices.loc[prices.index < '2020-01-08', 'L2'] = float('nan')
    with master.ingest_batch(provider='DEMO', raw_payload=b'Initial fictional identities',
                             source_ref='synthetic:initial',
                             accepted_at='2020-01-01T12:00:00Z') as batch:
        for suffix, name in [('1', 'Synthetic Alpha'), ('2', 'Synthetic Beta')]:
            batch.add_entity(f'E{suffix}', name)
            batch.add_security(f'S{suffix}', f'E{suffix}', security_type=SecurityType.COMMON_STOCK,
                               share_class='ordinary')
            batch.add_listing(f'L{suffix}', f'S{suffix}', mic='XNYS', currency='USD')
            batch.add_price_series(PriceSeries(
                f'PX_L{suffix}', f'L{suffix}', 'DEMO', 'close', PriceBasis.RAW_CLOSE,
                'USD', 'SYNTHETIC_BUSINESS_DAYS', 'America/New_York'))
        batch.add_identifier(IdentifierAssignment(
            IdentifierScheme.TICKER, 'DEMO', 'ABC', 'L1', '2020-01-02'))

    for date, values in prices.iterrows():
        day = date.strftime('%Y-%m-%d')
        if day == '2020-01-03':
            with master.ingest_batch(provider='DEMO', raw_payload=b'Alpha rename notice',
                                     source_ref='synthetic:rename',
                                     accepted_at=f'{day}T12:00:00Z') as batch:
                batch.rename_ticker('L1', namespace='DEMO', old_ticker='ABC', new_ticker='XYZ',
                                    effective_date='2020-01-06')
        if day == '2020-01-07':
            with master.ingest_batch(provider='DEMO', raw_payload=b'Beta reuses ABC',
                                     source_ref='synthetic:reuse',
                                     accepted_at=f'{day}T12:00:00Z') as batch:
                batch.add_identifier(IdentifierAssignment(
                    IdentifierScheme.TICKER, 'DEMO', 'ABC', 'L2', '2020-01-08'))
        with master.ingest_batch(provider='DEMO', raw_payload=values.to_json().encode(),
                                 source_ref=f'synthetic:prices:{day}',
                                 accepted_at=f'{day}T21:05:00Z') as batch:
            for listing_id, value in values.items():
                if pd.notna(value):
                    batch.add_prices(f'PX_{listing_id}', [PriceObservation(
                        day, f'{day}T21:00:00Z', value)])

    with master.ingest_batch(provider='DEMO', raw_payload=b'Withdraw Alpha Jan 3 close',
                             source_ref='synthetic:withdrawal',
                             accepted_at='2020-01-13T12:00:00Z') as batch:
        batch.add_prices('PX_L1', [PriceObservation(
            '2020-01-03', '2020-01-03T21:00:00Z', None, is_deleted=True)])
    return prices.index


def run_local(local: Locals) -> None:
    """Print identities, an as-of panel and the later effect of a withdrawal.

    Args:
        local: Offline demonstration to run.
    """
    if local != Locals.TICKER_HISTORY:
        raise ValueError(f'Unsupported example: {local}')
    with SecurityMaster() as master:
        sessions = populate_demo(master)
        for ticker, day in [('ABC', '2020-01-03'), ('XYZ', '2020-01-06'), ('ABC', '2020-01-08')]:
            listing_id = master.resolve_ticker(
                ticker, namespace='DEMO', mic='XNYS', currency='USD', effective_date=day,
                known_at=f'{day}T22:00:00Z')
            print(f'{day}: {ticker} -> {listing_id}')
        prices = master.load_price_panel(['PX_L1', 'PX_L2'], sessions=sessions,
                                         known_at='2020-01-10T22:00:00Z')
        returns = qis.to_returns(prices, is_log_returns=False, ffill_nans=False)
        print('\nRaw-close panel as known January 10 (no filling of missing data):')
        print(prices.round(4).to_string())
        print('\nSimple raw-price returns; dividends/corporate actions are not modelled:')
        print(returns.round(4).to_string())
        later = master.load_price_panel(['PX_L1', 'PX_L2'], sessions=sessions,
                                        known_at='2020-01-13T22:00:00Z')
        print('\nJanuary 3 L1 price after the January 13 withdrawal:',
              later.loc['2020-01-03', 'L1'])


if __name__ == '__main__':
    run_local(local=Locals.TICKER_HISTORY)
