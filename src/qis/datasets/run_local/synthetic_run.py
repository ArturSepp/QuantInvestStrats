"""Development runner extracted from ``qis.datasets.synthetic``."""

import numpy as np
import pandas as pd
from enum import Enum

from qis.datasets.synthetic import (
    SYNTHETIC_UNIVERSE,
    TRADING_DAYS_PER_YEAR,
    generate_synthetic_prices,
    generate_synthetic_universe,
)

class Locals(Enum):
    """Enumeration of available local test cases."""
    UNIVERSE_SUMMARY = 1
    QUIRK_DIAGNOSTICS = 2

def run_local(local: Locals) -> None:
    """
    Run local tests for development and debugging purposes.

    Args:
        local: which test case to run
    """
    if local == Locals.UNIVERSE_SUMMARY:
        universe = generate_synthetic_universe()
        print(universe.prices.tail())
        print(universe.benchmark_prices.tail())
        print(universe.group_data)

    elif local == Locals.QUIRK_DIAGNOSTICS:
        prices = generate_synthetic_prices(apply_quirks=True)
        clean_prices = generate_synthetic_prices(apply_quirks=False)
        # both volatilities are annualised on the business-day grid, so the reported figure is
        # comparable to the clean figure only for daily-reporting instruments
        quirks = [instrument.quirk.value for instrument in SYNTHETIC_UNIVERSE]
        report = pd.DataFrame({'quirk': quirks,
                               'target_vol': [x.vol for x in SYNTHETIC_UNIVERSE],
                               'clean_vol': (np.log(clean_prices).diff().std()
                                             * np.sqrt(TRADING_DAYS_PER_YEAR)),
                               'num_reported': prices.count(),
                               'num_nans': prices.isna().sum(),
                               'first_valid': prices.apply(lambda x: x.first_valid_index()),
                               'last_valid': prices.apply(lambda x: x.last_valid_index())})
        print(report.to_string())

if __name__ == "__main__":
    run_local(local=Locals.QUIRK_DIAGNOSTICS)
