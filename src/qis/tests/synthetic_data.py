"""
compatibility shim: the seeded panel moved to ``qis.datasets.synthetic`` in 5.3.0.

The generator was documented at this path in the 5.2 quickstart, so the path keeps working.
Import from ``qis.datasets`` in new code; ``tests`` is not a namespace a user should be told to
import from, and test modules are the first thing a redistributor drops from a wheel.

Nothing is reimplemented here. The module is the same object under a second name, so the seeds,
the draw order and every golden pinned to it are unchanged.
"""
# qis / project
from qis.datasets.synthetic import (  # noqa: F401
    BENCHMARK_TICKER,
    BENCHMARK_WEIGHTS,
    DELISTED_FRACTION,
    FAT_TAIL_DF,
    GAP_PROBABILITY,
    GROUP_ORDER,
    INITIAL_PRICE,
    LATE_START_FRACTION,
    SMOOTHING_AR1,
    STALE_RUN_PROBABILITY,
    TRADING_DAYS_PER_YEAR,
    DataQuirk,
    SyntheticInstrument,
    SyntheticUniverseData,
    SYNTHETIC_UNIVERSE,
    generate_synthetic_prices,
    generate_synthetic_universe,
)
