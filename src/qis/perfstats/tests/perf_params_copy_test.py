"""``PerfParams.copy`` keeps every field it is not asked to change.

The copy previously rebuilt the object without ``freq_skewness`` and ``freq``, so both fell back
to 'ME': a copy of ``PerfParams(freq_skewness='QE')`` computed skewness and kurtosis on months,
and a copy of ``PerfParams(freq='QE')`` reported ``freq='ME'``.
"""

import pandas as pd

# qis
from qis.perfstats.config import PerfParams, ReturnTypes, SharpeConvention


def _fields(params: PerfParams) -> dict:
    """All dataclass fields except the rate series, which is compared by identity."""
    return {key: value for key, value in vars(params).items() if key != 'rates_data'}


def test_copy_without_arguments_is_an_equal_object() -> None:
    """Every field survives a plain copy, including freq and freq_skewness."""
    rates = pd.Series(0.01, index=pd.date_range('2020-01-01', periods=3), name='rate')
    params = PerfParams(freq_vol='W-WED', freq_skewness='QE', freq_drawdown='B', freq_reg='ME',
                        freq_excess_return='QE', return_type=ReturnTypes.RELATIVE,
                        sharpe_convention=SharpeConvention.LOG, rates_data=rates)
    copy = params.copy()
    assert _fields(copy) == _fields(params)
    assert copy.rates_data is rates
    assert copy.freq_skewness == 'QE'


def test_copy_keeps_the_frequency_shortcut() -> None:
    """A copy of PerfParams(freq='QE') keeps freq='QE' and the fields it set."""
    params = PerfParams(freq='QE')
    copy = params.copy()
    assert _fields(copy) == _fields(params)
    assert copy.freq == 'QE' and copy.freq_vol == 'QE' and copy.freq_reg == 'QE'


def test_copy_overrides_only_the_requested_fields() -> None:
    """Named arguments replace their fields; freq_skewness can now be replaced too."""
    params = PerfParams(freq_skewness='QE')
    copy = params.copy(freq_vol='QE', freq_skewness='YE')
    assert copy.freq_vol == 'QE' and copy.freq_skewness == 'YE'
    assert copy.freq_reg == params.freq_reg and copy.freq_drawdown == params.freq_drawdown
