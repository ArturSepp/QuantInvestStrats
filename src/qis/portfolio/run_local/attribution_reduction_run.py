"""Development runner extracted from ``qis.portfolio.tests.attribution_reduction_test``."""

import pandas as pd
from qis.plots.utils import (estimate_bar_label_capacity)

class Locals:
    """runnable checks, not part of the pytest suite"""
    CAPACITY_TABLE = 1

def run_local(local: Locals):
    if local == Locals.CAPACITY_TABLE:
        rows = []
        for axis_width in (2.0, 4.09, 6.0, 8.19):
            for fontsize in (3.0, 4.0, 5.0, 8.0):
                rows.append(dict(axis_width=axis_width, fontsize=fontsize,
                                 capacity=estimate_bar_label_capacity(axis_width=axis_width,
                                                                      fontsize=fontsize)))
        print(pd.DataFrame(rows).pivot(index='axis_width', columns='fontsize', values='capacity'))

if __name__ == "__main__":
    run_local(local=Locals.CAPACITY_TABLE)
