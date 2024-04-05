"""
implementation of abstract
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from matplotlib._color_data import CSS4_COLORS as mcolors
from typing import NamedTuple, Union, Dict, List, Tuple, Any
from enum import Enum

import qis.utils.dates as da
import qis.utils.df_cut as dfc
import qis.perfstats.perf_stats as pt
import qis.perfstats.returns as ret
from qis.perfstats.config import ReturnTypes, RegimeData, PerfParams, PerfStat


def compute_mean_freq_regimes(sampled_returns_with_regime_id: pd.DataFrame):
    regime_groups = sampled_returns_with_regime_id.groupby([RegimeClassifier.REGIME_COLUMN], observed=False)
    regime_means = regime_groups.mean()
    regime_dims = regime_groups.count().iloc[:, 0]  # count regimes
    # replace nans
    regime_dims[np.isnan(regime_dims)] = 0.0
    norm_sum = np.sum(regime_dims)
    if np.isclose(np.sum(regime_dims), 0.0):
        norm_q = np.zeros_like(regime_dims)
    else:
        norm_q = regime_dims / norm_sum
    return regime_means, norm_q


def compute_regime_avg(sampled_returns_with_regime_id: pd.DataFrame,
                       freq: str,
                       is_report_pa_returns: bool = True,
                       regime_ids: List[str] = None,
                       **kwargs
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    """
    compute conditional means by the regime ids
    compute normalized prices attributions = af*freq*cvar
    """
    # compute mean by regimes
    regime_means, norm_q = compute_mean_freq_regimes(sampled_returns_with_regime_id=sampled_returns_with_regime_id)
    _, af_mult = da.get_period_days(freq=freq)

    if is_report_pa_returns:
        regime_pa = np.expm1(regime_means.multiply(af_mult * norm_q, axis=0))
    else:
        regime_pa = regime_means.multiply(af_mult * norm_q, axis=0)

    # transpose: index = regime, column = assets return
    regime_means = regime_means.T
    regime_pa = regime_pa.T

    # arrange columns according to given labels
    if regime_ids is not None:
        regime_means = regime_means[regime_ids]
        regime_pa = regime_pa[regime_ids]

    return regime_means, regime_pa, norm_q


def compute_regimes_pa_perf_table(sampled_returns_with_regime_id: pd.DataFrame,
                                  prices: pd.DataFrame,
                                  benchmark: str,
                                  perf_params: PerfParams,
                                  freq: str,
                                  is_use_benchmark_means: bool = False,
                                  is_add_ra_perf_table: bool = True,
                                  drop_benchmark: bool = False,
                                  additive_pa_returns_to_pa_total: bool = True,
                                  regime_ids: List[str] = None,  # define regime order
                                  **kwargs
                                  ) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:

    """
    compute regime conditional returns, regime conditional Sharpes and total performance
    """
    regime_avg, regime_pa, norm_q = compute_regime_avg(sampled_returns_with_regime_id=sampled_returns_with_regime_id,
                                                       regime_ids=regime_ids,
                                                       freq=freq,
                                                       **kwargs)

    # align data by regime labels for consitent colors
    given_columns = regime_avg.columns.to_list()
    regime_avg = regime_avg[given_columns]
    regime_avg.columns = [f"{x} {RegimeData.REGIME_AVG.value}" for x in given_columns]

    # average regime returns
    regime_pa = regime_pa[given_columns]
    regime_pa.columns = [f"{x} {RegimeData.REGIME_PA.value}" for x in given_columns]
    regime_pa_columns = regime_pa.columns

    # compute stadradized ra _ perf table
    ra_perf_table = pt.compute_ra_perf_table(prices=prices, perf_params=perf_params)
    if additive_pa_returns_to_pa_total:  # use pa return to normalize conditional an returns

        total_sum = regime_pa[regime_pa_columns].sum(1)
        total_to_match = ra_perf_table[PerfStat.PA_RETURN.to_str()]

        # adjust regimes to add to total
        total_pa_diff = (total_to_match - total_sum)
        weighted_diff = pd.DataFrame(np.tile(total_pa_diff, (len(norm_q.to_numpy()), 1)).T,
                                     index=total_pa_diff.index, columns=norm_q.index) # = matrix[rows[c by asset]]
        regime_pa_diff = weighted_diff.multiply(norm_q, axis=1)  # each row is regime freq [number of regime] * adjustment[asset]
        regime_pa1 = regime_pa[regime_pa_columns].add(regime_pa_diff.to_numpy(), axis=0)

    else:  # need regime_pa1 for level benchmarks
        regime_pa1 = regime_pa

    if is_use_benchmark_means and benchmark is not None:
        regime_pa1.loc[benchmark][regime_pa_columns] = regime_avg.loc[benchmark]

    # compute simple norm sharpes, using vols from pa returns
    vols_for_sharpe_pa = ra_perf_table[PerfStat.VOL.to_str()]
    regime_sharpe = regime_pa1.divide(vols_for_sharpe_pa, axis=0)[regime_pa_columns]

    regime_sharpe.columns = [f"{x} {RegimeData.REGIME_SHARPE.value}" for x in given_columns]
    if is_add_ra_perf_table:
        cond_perf_table = pd.concat([regime_avg, regime_pa1, regime_sharpe, ra_perf_table], axis=1)
    else:
        cond_perf_table = pd.concat([regime_avg, regime_pa1, regime_sharpe], axis=1)

    regime_datas = {RegimeData.REGIME_AVG: regime_avg,
                    RegimeData.REGIME_PA: regime_pa1,
                    RegimeData.REGIME_SHARPE: regime_sharpe}

    if drop_benchmark:
        cond_perf_table = cond_perf_table.drop(benchmark, axis=0)

    return cond_perf_table, regime_datas


class RegimeClassifier(ABC):
    """
    Abstract class for regime classification which is part of performance params
    and regime-conditional performance attribution
    """
    REGIME_COLUMN = 'regime'

    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_sampled_returns_with_regime_id(self, **kwargs) -> pd.DataFrame:
        """
        abstract method for getting df with params and regime ids
        """
        pass

    @abstractmethod
    def get_regime_ids_colors(self) -> Dict[str, str]:
        """
        important for proper visualizations
        """

    def compute_regimes_pa_perf_table(self,
                                      regime_id_func_kwargs: Dict[str, Any],
                                      prices: pd.DataFrame,
                                      benchmark: str,
                                      freq: str,
                                      perf_params: PerfParams,
                                      is_use_benchmark_means: bool = False,
                                      is_add_ra_perf_table: bool = True,
                                      drop_benchmark: bool = False,
                                      additive_pa_returns_to_pa_total: bool = True,
                                      regime_ids: List[str] = None,  # define regime order
                                      **kwargs
                                      ) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:

        """
        compute regime conditional returns, regime conditional Sharpes and total performance
        """
        sampled_returns_with_regime_id = self.compute_sampled_returns_with_regime_id(**regime_id_func_kwargs)
        cond_perf_table, regime_datas = compute_regimes_pa_perf_table(sampled_returns_with_regime_id=sampled_returns_with_regime_id,
                                                                      prices=prices,
                                                                      benchmark=benchmark,
                                                                      perf_params=perf_params,
                                                                      freq=freq,
                                                                      is_use_benchmark_means=is_use_benchmark_means,
                                                                      is_add_ra_perf_table=is_add_ra_perf_table,
                                                                      drop_benchmark=drop_benchmark,
                                                                      regime_ids=regime_ids)
        return cond_perf_table, regime_datas

    def class_data_to_colors(self,
                             regime_data: pd.Series
                             ) -> pd.Series:
        map_id_into_color = self.get_regime_ids_colors()
        regime_id_color = regime_data.map(map_id_into_color)
        regime_id_color = regime_id_color.astype(str)  # change category index
        regime_id_color.loc[np.in1d(regime_id_color.to_numpy(), 'nan')] = '#FFFFFF'  #just in case put white for non-mapped data
        return regime_id_color
    
    def get_regime_ids(self) -> List[str]:
        return list(self.get_regime_ids_colors().keys())


####################################
###   Implementation of returns quantile regime
###################################


class BenchmarkReturnsQuantileRegimeSpecs(NamedTuple):
    freq: str = 'QE'  # frequency of returns
    return_type: ReturnTypes = ReturnTypes.RELATIVE  # return type
    q: Union[np.ndarray, int] = np.array([0.0, 0.17, 0.83, 1.0])  # quantiles = q[1:] - q[:-1]
    regime_ids_colors: Dict[str, str] = {'Bear': mcolors['salmon'], 'Normal': mcolors['yellowgreen'], 'Bull': mcolors['darkgreen']}

    def get_regime_ids_colors(self) -> Dict[str, str]:
        return self.regime_ids_colors


class BenchmarkReturnsQuantilesRegime(RegimeClassifier):

    def __init__(self,
                 regime_params: BenchmarkReturnsQuantileRegimeSpecs = BenchmarkReturnsQuantileRegimeSpecs()
                 ):
        self.regime_params = regime_params
        super().__init__()

    def compute_sampled_returns_with_regime_id(self,
                                               prices: Union[pd.DataFrame, pd.Series],
                                               benchmark: str,
                                               include_start_date: bool = True,
                                               include_end_date: bool = True,
                                               **kwargs
                                               ) -> pd.DataFrame:
        """
        implementation of abstract method for getting df with params and regime ids
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        sampled_returns_with_regime_id = ret.to_returns(prices=prices,
                                                        freq=self.regime_params.freq,
                                                        return_type=self.regime_params.return_type,
                                                        include_start_date=include_start_date,
                                                        include_end_date=include_end_date)
        if len(sampled_returns_with_regime_id.index) <= 3:
            raise ValueError(f"need to have more than 3 returns in time series: {sampled_returns_with_regime_id.index}\nDecrease regime frequency")
        x = sampled_returns_with_regime_id[benchmark]
        quant0 = pd.qcut(x=x, q=self.regime_params.q, labels=self.get_regime_ids())
        sampled_returns_with_regime_id[self.REGIME_COLUMN] = quant0
        return sampled_returns_with_regime_id

    def get_regime_ids_colors(self) -> Dict[str, str]:
        return self.regime_params.get_regime_ids_colors()

    def compute_regimes_pa_perf_table(self,
                                      prices: pd.DataFrame,
                                      benchmark: str,
                                      perf_params: PerfParams,
                                      **kwargs
                                      ) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:

        regime_id_func_kwargs = dict(prices=prices, benchmark=benchmark,
                                     include_start_date=True, include_end_date=True)

        return super().compute_regimes_pa_perf_table(regime_id_func_kwargs=regime_id_func_kwargs,
                                                     prices=prices,
                                                     benchmark=benchmark,
                                                     perf_params=perf_params,
                                                     freq=self.regime_params.freq,
                                                     is_report_pa_returns=True,
                                                     is_use_benchmark_means=False,
                                                     regime_ids=self.get_regime_ids())


####################################
### Implementation of vols quantile regime
###################################


class VolQuantileRegimeSpecs(NamedTuple):
    freq: str = 'QE'  # frequency of vol sampling
    return_type: ReturnTypes = ReturnTypes.RELATIVE  # return type
    q: int = 4  # 4 qiantiles


class BenchmarkVolsQuantilesRegime(RegimeClassifier):

    def __init__(self,
                 regime_params: VolQuantileRegimeSpecs = VolQuantileRegimeSpecs()
                 ):
        self.regime_params = regime_params
        self.regime_colors: Dict[str, str]  # will be computed in the call compute_sampled_returns_with_regime_id
        super().__init__()

    def compute_sampled_returns_with_regime_id(self,
                                               prices: pd.DataFrame,
                                               benchmark: str,
                                               include_start_date: bool = True,
                                               include_end_date: bool = True,
                                               **kwargs
                                               ) -> pd.DataFrame:
        """
        implementation of abstract method for getting df with params and regime ids
        """
        vols = ret.compute_sampled_vols(prices=prices[benchmark],
                                        freq_vol=self.regime_params.freq,
                                        include_start_date=include_start_date, include_end_date=include_end_date)

        hue_name = f"{benchmark} vol"
        classificator, labels = dfc.add_quantile_classification(df=vols.to_frame(), x_column=benchmark,
                                                                num_buckets=self.regime_params.q,
                                                                hue_name=hue_name,
                                                                xvar_format='{:.0%}',
                                                                bucket_prefix=hue_name)
        classificator = classificator.sort_index()
        sampled_returns_with_regime_id = ret.to_returns(prices=prices,
                                                        freq=self.regime_params.freq,
                                                        return_type=self.regime_params.return_type,
                                                        include_start_date=include_start_date,
                                                        include_end_date=include_end_date)
        sampled_returns_with_regime_id[self.REGIME_COLUMN] = classificator[hue_name]

        cmap = plt.cm.get_cmap('RdYlGn', len(labels))
        colors = [cmap(n_) for n_ in range(len(labels))]
        self.regime_colors = {k: v for k, v in zip(labels, colors)}
        return sampled_returns_with_regime_id
    
    def get_regime_ids_colors(self) -> Dict[str, str]:
        return self.regime_colors
    
    def compute_regimes_pa_perf_table(self,
                                      prices: pd.DataFrame,
                                      benchmark: str,
                                      perf_params: PerfParams,
                                      **kwargs
                                      ) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:

        regime_id_func_kwargs = dict(prices=prices, benchmark=benchmark,
                                     include_start_date=True, include_end_date=True)

        return super().compute_regimes_pa_perf_table(regime_id_func_kwargs=regime_id_func_kwargs,
                                                     prices=prices,
                                                     benchmark=benchmark,
                                                     perf_params=perf_params,
                                                     freq=self.regime_params.freq,
                                                     is_report_pa_returns=True,
                                                     is_use_benchmark_means=False,
                                                     regime_ids=self.get_regime_ids())

    def get_regime_colors(self) -> List[str]:
        return list(self.regime_colors.values())


####################################
###   Implementation of plotting figs
###################################


def compute_bnb_regimes_pa_perf_table(prices: pd.DataFrame,
                                      benchmark: str,
                                      regime_params: BenchmarkReturnsQuantileRegimeSpecs = None,
                                      perf_params: PerfParams = None,
                                      **kwargs
                                      ) -> pd.DataFrame:

        """
        compute regime conditional returns, regime conditional Sharpes and total performance
        """
        if regime_params is None:
            regime_params = BenchmarkReturnsQuantileRegimeSpecs()
        regime_classifier = BenchmarkReturnsQuantilesRegime(regime_params=regime_params)

        regimes_pa_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(prices=prices,
                                                                                              benchmark=benchmark,
                                                                                              perf_params=perf_params,
                                                                                              **regime_params._asdict())
        return regimes_pa_perf_table


class UnitTests(Enum):
    BNB_REGIME = 1
    BNB_PERF_TABLE = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    perf_params = PerfParams()

    if unit_test == UnitTests.BNB_REGIME:
        regime_params = BenchmarkReturnsQuantileRegimeSpecs()
        regime_classifier = BenchmarkReturnsQuantilesRegime(regime_params=regime_params)
        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices, benchmark='SPY')
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(prices=prices,
                                                                                        benchmark='SPY',
                                                                                        perf_params=perf_params)
        print(f"regime_means:\n{cond_perf_table}")
        print(f"regime_pa:\n{regime_datas}")

    elif unit_test == UnitTests.BNB_PERF_TABLE:
        df = compute_bnb_regimes_pa_perf_table(prices=prices,
                                               benchmark='SPY',
                                               regime_params=BenchmarkReturnsQuantileRegimeSpecs(),
                                               perf_params=PerfParams())
        print(df)


    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.BNB_PERF_TABLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
