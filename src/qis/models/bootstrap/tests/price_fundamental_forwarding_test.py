"""Argument forwarding and price weighting in ``bootstrap_price_fundamental_data``.

Guards two defects, each failing on its own:

  - the function had no ``init_to_end`` and always continued prices from their last level,
    although ``bootstrap_price_data`` offers alternative histories from the first row;
  - with ``SERIES_TO_DF`` output and ``is_price_weighted_fundamentals=True`` it zipped two
    DataFrames, which iterates column labels, and raised ``TypeError``.

It also pins the positivity switch forwarded to ``bootstrap_ar_process``.
"""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from qis.models.bootstrap.bootstrap_numba import (BootstrapOutput,
                                                  BootstrapType,
                                                  bootstrap_ar_process,
                                                  bootstrap_price_data,
                                                  bootstrap_price_fundamental_data,
                                                  generate_bootstrapped_indices)

N = 120
DATES = pd.date_range('2010-01-31', periods=N, freq='ME')
_RNG = np.random.default_rng(21)
PRICES = pd.Series(100.0 * np.cumprod(1.0 + _RNG.normal(0.005, 0.04, N)), index=DATES,
                   name='asset')
_YIELD = np.full(N, 0.02)
for _t in range(1, N):
    _YIELD[_t] = 0.02 + 0.95 * (_YIELD[_t - 1] - 0.02) + _RNG.normal(0.0, 0.002)
YIELDS = pd.Series(_YIELD, index=DATES, name='asset')
KWARGS = dict(bootstrap_type=BootstrapType.STATIONARY, num_samples=4, index_length=60,
              block_size=6, seed=3)


def _shared_indices() -> np.ndarray:
    """The index array the function draws internally, over the N-1 return rows."""
    return generate_bootstrapped_indices(num_data_index=N - 1, min_block_size=1, **KWARGS)


@pytest.mark.parametrize('init_to_end', [True, False])
def test_init_to_end_is_forwarded_to_the_price_paths(init_to_end):
    prices, _ = bootstrap_price_fundamental_data(
        price_datas={'p': PRICES.to_frame()}, fundamental_datas={'f': YIELDS.to_frame()},
        init_to_end=init_to_end, **KWARGS)
    expected = bootstrap_price_data(prices=PRICES.to_frame(), init_to_end=init_to_end,
                                    bootstrapped_indices=_shared_indices())
    anchor = PRICES.iloc[-1] if init_to_end else PRICES.iloc[0]
    for actual, reference in zip(prices['p'], expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-13)
        assert actual[0, 0] == pytest.approx(anchor, rel=1e-13)


def test_series_output_multiplies_prices_and_fundamentals():
    prices, fundamentals = bootstrap_price_fundamental_data(
        price_datas={'p': PRICES}, fundamental_datas={'f': YIELDS},
        bootstrap_output=BootstrapOutput.SERIES_TO_DF, is_price_weighted_fundamentals=True,
        **KWARGS)
    unweighted = bootstrap_ar_process(YIELDS, bootstrap_output=BootstrapOutput.SERIES_TO_DF,
                                      bootstrapped_indices=_shared_indices())
    assert isinstance(fundamentals['f'], pd.DataFrame)
    assert fundamentals['f'].columns.tolist() == prices['p'].columns.tolist()
    np.testing.assert_allclose(fundamentals['f'].to_numpy(),
                               prices['p'].to_numpy() * unweighted.to_numpy(), rtol=1e-13)


@pytest.mark.parametrize('is_positive', [True, False])
def test_is_positive_is_forwarded_to_the_ar_paths(is_positive):
    _, fundamentals = bootstrap_price_fundamental_data(
        price_datas={'p': PRICES.to_frame()}, fundamental_datas={'f': YIELDS.to_frame()},
        is_positive=is_positive, **KWARGS)
    expected = bootstrap_ar_process(YIELDS.to_frame(), is_positive=is_positive,
                                    bootstrapped_indices=_shared_indices())
    for actual, reference in zip(fundamentals['f'], expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-13)
