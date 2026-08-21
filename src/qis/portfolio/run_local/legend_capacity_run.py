"""Development runner extracted from ``qis.portfolio.tests.legend_capacity_test``."""

import pandas as pd
from qis.portfolio.reports.config import (LEGEND_ROW_HEIGHT_PER_FONTSIZE)

from qis.portfolio.tests.legend_capacity_test import (
    measure_legend_height,
)

class Locals:
    """runnable checks, not part of the pytest suite"""
    LEGEND_HEIGHT_TABLE = 1

def run_local(local: Locals):
    if local == Locals.LEGEND_HEIGHT_TABLE:
        rows = []
        for fontsize in (2.5, 3.5, 5.0, 8.0):
            for n_entries in (5, 10, 15, 20, 30):
                rows.append(dict(fontsize=fontsize,
                                 n_entries=n_entries,
                                 measured=measure_legend_height(n_entries=n_entries,
                                                                fontsize=fontsize),
                                 model=fontsize * LEGEND_ROW_HEIGHT_PER_FONTSIZE * n_entries))
        print(pd.DataFrame(rows))

if __name__ == "__main__":
    run_local(local=Locals.LEGEND_HEIGHT_TABLE)
