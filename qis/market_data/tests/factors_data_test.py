"""Unit tests for FactorsData: construction, validation, accessors, round-trip."""
import numpy as np
import pandas as pd
import pytest
from enum import Enum

import qis as qis
from qis.market_data import FactorsData


class _Factors(str, Enum):
    EQUITY = 'Equity'
    RATES = 'Rates'
    CREDIT = 'Credit'


def _sample_prices() -> pd.DataFrame:
    idx = pd.date_range('2020-01-31', periods=24, freq='ME')
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, size=(len(idx), 3))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=['Equity', 'Rates', 'Credit'])


def test_construct_and_accessors() -> None:
    fd = FactorsData(factors_prices=_sample_prices(), factors=_Factors)
    assert fd.factor_names == ['Equity', 'Rates', 'Credit']
    assert fd.get_factor_prices(_Factors.EQUITY).equals(fd.factors_prices['Equity'])
    assert fd.get_factor_prices('Rates').equals(fd.factors_prices['Rates'])


def test_missing_column_raises() -> None:
    prices = _sample_prices().drop(columns=['Credit'])
    with pytest.raises(ValueError):
        FactorsData(factors_prices=prices, factors=_Factors)


def test_generic_without_factors() -> None:
    fd = FactorsData(factors_prices=_sample_prices())  # factors=None -> generic
    assert fd.factors is None
    assert len(fd.get_prices()) == 24


def test_csv_round_trip(tmp_path) -> None:
    prices = _sample_prices()
    qis.save_df_to_csv(df=prices, file_name='futures_risk_factors', local_path=f"{tmp_path}/")
    fd = FactorsData.load(local_path=f"{tmp_path}/", factors=_Factors)
    pd.testing.assert_frame_equal(fd.factors_prices, prices)


if __name__ == '__main__':
    test_construct_and_accessors()
    test_generic_without_factors()
    print("ok")
