"""
seeded data a reader can run, shipped with the package.

``qis.datasets.synthetic`` draws a ten-instrument multi-asset panel from a fixed seed. Every
documented example, the quickstart and the test suite use it, so nothing in the documentation
needs a network connection, a data file or a vendor licence to reproduce.

It lived at ``qis.tests.synthetic_data`` until 5.3.0. That path still works and re-exports this
module, but a namespace called ``tests`` is not somewhere a user should be told to import from,
and test modules are the first thing a redistributor drops.
"""
# qis / project
from qis.datasets.synthetic import (
    BENCHMARK_TICKER,
    BENCHMARK_WEIGHTS,
    GROUP_ORDER,
    DataQuirk,
    SyntheticInstrument,
    SyntheticUniverseData,
    SYNTHETIC_UNIVERSE,
    generate_synthetic_prices,
    generate_synthetic_universe,
)
