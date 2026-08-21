"""Development runner extracted from ``qis.models.linear.ra_returns``."""

from enum import Enum

from qis.models.linear.ra_returns import (
    compute_ra_returns,
    compute_returns_transform,
)

class Locals(Enum):
    RA_RETURNS = 1
    TRANSFORM = 2

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()
    returns = prices.pct_change()

    if local == Locals.RA_RETURNS:
        df = compute_ra_returns(returns=returns)
        print(df)

    elif local == Locals.TRANSFORM:
        df = compute_returns_transform(returns=returns)
        print(df)

if __name__ == "__main__":
    run_local(local=Locals.RA_RETURNS)
