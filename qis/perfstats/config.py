"""
configuration of performance stats and regime params
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from qis.utils.generic import ColVar, ValueType


class ReturnTypes(Enum):
    RELATIVE = 'Relative'  # =S1/S0-1
    LOG = 'Log'  # =ln(S1/S0)
    DIFFERENCE = 'Diff'  # =S1-S0
    LEVEL = 'Level'  # =S1
    LEVEL0 = 'Level0'  # =S0


class RegimeType(Enum):
    RETURNS = 'returns'
    LEVELS = 'levels'
    AVG = 'averages'


class RegimeData(Enum):
    REGIME_AVG = 'Average'
    REGIME_PA = 'P.a.'
    REGIME_SHARPE = 'Sharpe'


class PerfStat(ColVar, Enum):
    """
    performance data outputs are computed as df(index=assets, columns=PerfStat)
    """
    START_DATE = ColVar(name='Start date', short_n='Start\ndate', value_type=ValueType.DATE)
    END_DATE = ColVar(name='End date', short_n='End\ndate', value_type=ValueType.DATE)
    START_PRICE = ColVar(name='Start', short_n='Start\nprice', value_type=ValueType.PRICE)
    END_PRICE = ColVar(name='End', short_n='End\nprice', value_type=ValueType.PRICE)
    NUM_OBS = ColVar(name='Num Obs', short_n='Num\nObs', value_type=ValueType.INT)

    # computed in compute_pa_return_dict
    TOTAL_RETURN = ColVar(name='Total', short_n='Total return', value_type=ValueType.PERCT)
    PA_RETURN = ColVar(name='P.a. return', short='P.a.', short_n='P.a.\nreturn', value_type=ValueType.PERCT)
    AN_LOG_RETURN = ColVar(name='An. log return', short_n='An. log-return', value_type=ValueType.PERCT)
    AN_LOG_RETURN_EXCESS = ColVar(name='An. log return ex', short_n='An. log-return ex', value_type=ValueType.PERCT)
    AVG_AN_RETURN = ColVar(name='Avg. An return', short_n='Avg. An return', value_type=ValueType.PERCT)
    APR = ColVar(name='APR', short_n='APR', value_type=ValueType.PERCT)
    PA_EXCESS_RETURN = ColVar(name='P.a. excess return', short_n='P.a.\nexcess', value_type=ValueType.PERCT)
    NAV1 = ColVar(name='1$ Invested', short_n='1$ Invested', value_type=ValueType.FLOAT)
    NUM_YEARS = ColVar(name='Num Years', short_n='Num\nYears', value_type=ValueType.FLOAT)

    VOL = ColVar(name='Vol', short_n='An. vol', value_type=ValueType.PERCT)
    DOWNSIDE_VOL = ColVar(name='DownVol', short_n='DownVol', value_type=ValueType.PERCT)
    AVG_LOG_RETURN = ColVar(name='AvgLogReturn', short_n='AvgReturn', value_type=ValueType.PERCT)
    SHARPE_RF0 = ColVar(name='Sharpe (rf=0)', short='Sharpe(rf=0)', short_n='Sharpe\n(rf=0)', value_type=ValueType.SHARPE)  # compounded returns with rate = 0.0
    SHARPE_EXCESS = ColVar(name='Ex. Sharpe', short='Ex.Sharpe', short_n='Excess\nSharpe', value_type=ValueType.SHARPE)  # compunded with given rate
    SHARPE_LOG_AN = ColVar(name='An. Log Sharpe', short_n='An. Log Sharpe', value_type=ValueType.SHARPE)  # log return
    SHARPE_AVG = ColVar(name='Sharpe Avg', short_n='Sharpe Avg', value_type=ValueType.SHARPE)  # using avg return
    SHARPE_LOG_EXCESS = ColVar(name='Log Sharpe', short_n='Log Sharpe', value_type=ValueType.SHARPE)
    SHARPE_APR = ColVar(name='APR Sharpe', short_n='APR\nSharpe', value_type=ValueType.SHARPE)
    MARGINAL_SHARPE = ColVar(name='Marginal Sharpe', short_n='Marginal\nSharpe', value_type=ValueType.SHARPE)
    MARGINAL_SHARPE_RATIO = ColVar(name='Marginal Sharpe Ratio', short_n='Marginal\nSharpe Ratio', value_type=ValueType.FLOAT)

    SORTINO_RATIO = ColVar(name='Sortino', short_n='Sortino', value_type=ValueType.SHARPE)
    CALMAR_RATIO = ColVar(name='Calmar', short_n='Calmar', value_type=ValueType.SHARPE)

    MAX_DD = ColVar(name='Max DD', short='MaxDD', short_n='Max DD', value_type=ValueType.PERCT0)
    MAX_DD_VOL = ColVar(name='Max DD/Vol', short='MaxDD/Vol', short_n='Max DD\n/Vol', value_type=ValueType.FLOAT)
    SKEWNESS = ColVar(name='Skewness', short='Skew', short_n='Skew', value_type=ValueType.FLOAT)
    KURTOSIS = ColVar(name='Kurtosis', short_n='Kurt', value_type=ValueType.FLOAT)
    NORMTEST = ColVar(name='P-val', short_n='P-val', value_type=ValueType.FLOAT4)
    WORST = ColVar(name='Worst', short_n='Worst', value_type=ValueType.PERCT)
    BEST = ColVar(name='Best', short_n='Best', value_type=ValueType.PERCT)
    POSITIVE = ColVar(name='Positive', short_n='Positive', value_type=ValueType.PERCT)

    BEAR_AVG = ColVar(name='Bear Avg', short_n='Bear\nAvg', value_type=ValueType.PERCT)
    NORMAL_AVG = ColVar(name='Normal Avg', short_n='Normal\nAvg', value_type=ValueType.PERCT)
    BULL_AVG = ColVar(name='Bull Avg', short_n='Bull\nAvg', value_type=ValueType.PERCT)

    BEAR_PA = ColVar(name='Bear P.a.', short_n='Bear\nP.a.', value_type=ValueType.PERCT)
    NORMAL_PA = ColVar(name='Normal P.a.', short_n='Normal\nP.a.', value_type=ValueType.PERCT)
    BULL_PA = ColVar(name='Bull P.a.', short_n='Bull\nP.a.', value_type=ValueType.PERCT)

    BEAR_SHARPE = ColVar(name='Bear Sharpe', short_n='Bear\nSharpe', value_type=ValueType.SHARPE)
    NORMAL_SHARPE = ColVar(name='Normal Sharpe', short_n='Normal\nSharpe', value_type=ValueType.SHARPE)
    BULL_SHARPE = ColVar(name='Bull Sharpe', short_n='Bull\nSharpe', value_type=ValueType.SHARPE)

    BEAR_CORR = ColVar(name='Bear Correlation', short_n='Bear\nCorrelation', value_type=ValueType.FLOAT)
    NORMAL_CORR = ColVar(name='Normal Correlation', short_n='Normal\nCorrelation', value_type=ValueType.FLOAT)
    BULL_CORR = ColVar(name='Bull Correlation', short_n='Bull\nCorrelation', value_type=ValueType.FLOAT)
    TOTAL_CORR = ColVar(name='Total Correlation', short_n='Total\nCorrelation', value_type=ValueType.FLOAT)

    AVG = ColVar(name='Avg', short_n='Avg', value_type=ValueType.FLOAT)
    T_STAT = ColVar(name='T-stat', short_n='T-stat', value_type=ValueType.FLOAT)
    STD = ColVar(name='Std', short_n='Std', value_type=ValueType.FLOAT)
    STD_AN = ColVar(name='Std An', short_n='Std AN', value_type=ValueType.FLOAT)
    MEDIAN = ColVar(name='Median', short_n='50-Q', value_type=ValueType.FLOAT)
    MIN = ColVar(name='Min', short_n='Min', value_type=ValueType.FLOAT)
    MAX = ColVar(name='Max', short_n='Max', value_type=ValueType.FLOAT)
    QUANT_M_1STD = ColVar(name='-1std', short_n='16-Q', value_type=ValueType.FLOAT)
    QUANT_P1_STD = ColVar(name='+1std', short_n='84-Q', value_type=ValueType.FLOAT)
    LAST = ColVar(name='Last', short_n='Last', value_type=ValueType.PERCT)
    RANK = ColVar(name='Rank', short_n='Rank', value_type=ValueType.PERCT)

    # tre and ir
    TE = ColVar(name='TE', short_n='TE', value_type=ValueType.PERCT)
    IR = ColVar(name='IR', short_n='IR', value_type=ValueType.SHARPE)

    # linear ml
    ALPHA = ColVar(name='Alpha', short_n='Alpha', value_type=ValueType.PERCT)
    ALPHA_AN = ColVar(name='An Alpha', short_n='Alpha', value_type=ValueType.PERCT)
    BETA = ColVar(name='Beta', short_n='Beta', value_type=ValueType.FLOAT2)
    R2 = ColVar(name='R2', short_n='R2', value_type=ValueType.PERCT0)


"""
Most common table columns
"""

FULL_TABLE_COLUMNS = (PerfStat.START_DATE,
                      PerfStat.END_DATE,
                      PerfStat.NUM_OBS,
                      PerfStat.AVG_LOG_RETURN,
                      PerfStat.PA_RETURN,
                      PerfStat.VOL,
                      PerfStat.SHARPE_RF0,
                      PerfStat.MAX_DD,
                      PerfStat.MAX_DD_VOL,
                      PerfStat.SKEWNESS,
                      PerfStat.KURTOSIS,
                      PerfStat.WORST,
                      PerfStat.BEST)


RA_TABLE_COLUMNS = (PerfStat.START_DATE,
                    PerfStat.END_DATE,
                    PerfStat.PA_RETURN,
                    PerfStat.VOL,
                    PerfStat.SHARPE_RF0,
                    PerfStat.MAX_DD,
                    PerfStat.MAX_DD_VOL,
                    PerfStat.SKEWNESS,
                    PerfStat.KURTOSIS,
                    PerfStat.WORST,
                    PerfStat.BEST)


SD_PERF_COLUMNS = (PerfStat.START_DATE,
                   PerfStat.END_DATE,
                   PerfStat.PA_RETURN,
                   PerfStat.VOL,
                   PerfStat.SHARPE_RF0,
                   PerfStat.BEAR_SHARPE,
                   PerfStat.NORMAL_SHARPE,
                   PerfStat.BULL_SHARPE,
                   PerfStat.MAX_DD,
                   PerfStat.MAX_DD_VOL,
                   PerfStat.SKEWNESS)


RA_TABLE_COMPACT_COLUMNS = (PerfStat.PA_RETURN,
                            PerfStat.VOL,
                            PerfStat.SHARPE_RF0,
                            PerfStat.MAX_DD,
                            PerfStat.MAX_DD_VOL,
                            PerfStat.SKEWNESS,
                            PerfStat.KURTOSIS)


TRE_TABLE_COLUMNS = (PerfStat.TOTAL_RETURN,
                     PerfStat.AVG_AN_RETURN,
                     PerfStat.TE,
                     PerfStat.IR,
                     PerfStat.MAX_DD,
                     PerfStat.MAX_DD_VOL,
                     PerfStat.SKEWNESS,
                     PerfStat.KURTOSIS,
                     PerfStat.WORST,
                     PerfStat.BEST)


@dataclass
class PerfParams:
    """
    contain key parameters for computing risk adjusted performance
    """
    freq: str = None
    freq_vol: str = 'ME'  # volatility of Sharpe
    freq_drawdown: str = 'D'
    freq_reg: str = 'QE'  # for quadratic/linear regressions
    freq_excess_return: str = 'ME'
    return_type: ReturnTypes = ReturnTypes.LOG  # for vol computation
    rates_data: Optional[pd.Series] = None  # to compute EXCESS returns
    alpha_an_factor: float = 4.0  # to annualise alpha in linear regression, linked to frequency of freq_reg
    # alpha_an_factor = 12, 4 for freq_reg='ME', 'QE' and so

    def __post_init__(self):
        if self.freq is not None:  # global parameter
            self.freq = self.freq
            self.freq_vol = self.freq
            self.freq_drawdown = self.freq_drawdown or self.freq
            self.freq_excess_return = self.freq
        else:
            self.freq = 'ME'
            self.freq_vol = self.freq_vol
            self.freq_drawdown = self.freq_drawdown or self.freq_vol
            self.freq_excess_return = self.freq_excess_return

    def print(self):
        print(f"freq_reg: {self.freq_reg}")
        print(f"freq_vol: {self.freq_vol}")
        print(f"return_type: {self.return_type.name}")
        print(f"freq_drawdown: {self.freq_drawdown}")
        print(f"freq_excess_return: {self.freq_excess_return}")
        if self.rates_data is not None:
            print(f"rate_data:\n{self.rates_data}")

    def copy(self,
             freq_reg: str = None,
             freq_vol: str = None,
             freq_drawdown: str = None,
             freq_excess_return: str = None,
             return_type: ReturnTypes = None,
             rates_data: pd.Series = None,
             **kwargs
             ) -> PerfParams:
        this_copy = PerfParams(freq_reg=freq_reg or self.freq_reg,
                               freq_vol=freq_vol or self.freq_vol,
                               freq_drawdown=freq_drawdown or self.freq_drawdown,
                               freq_excess_return=freq_excess_return or self.freq_excess_return,
                               return_type=return_type or self.return_type,
                               rates_data=rates_data if rates_data is not None else self.rates_data)
        return this_copy


class UnitTests(Enum):
    PERFORMANCE_STAT = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.PERFORMANCE_STAT:
        print(PerfStat.TOTAL_RETURN)


if __name__ == '__main__':

    unit_test = UnitTests.PERFORMANCE_STAT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
