"""
Regime classification for performance attribution analysis.

This module provides abstract and concrete implementations of regime classifiers
that partition time periods based on market conditions (e.g., bull/bear markets,
high/low volatility) for conditional performance analysis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from matplotlib._color_data import CSS4_COLORS as mcolors
from typing import Union, Dict, List, Tuple, Any
from enum import Enum

import qis.utils.df_cut as dfc
import qis.perfstats.perf_stats as pt
import qis.perfstats.returns as ret
from qis.perfstats.config import ReturnTypes, RegimeData, PerfParams, PerfStat
from qis.utils.annualisation import get_annualization_factor

# ============================================================================
# Core computation functions
# ============================================================================

def compute_mean_freq_regimes(sampled_returns_with_regime_id: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute mean returns and frequency for each regime.

    Args:
        sampled_returns_with_regime_id: DataFrame with returns and regime column

    Returns:
        Tuple of (regime means DataFrame, normalized regime frequencies Series)
    """
    regime_groups = sampled_returns_with_regime_id.groupby([RegimeClassifier.REGIME_COLUMN], observed=False)
    regime_means = regime_groups.mean()
    regime_dims = regime_groups.count().iloc[:, 0]

    # Replace nans and normalize
    regime_dims[np.isnan(regime_dims)] = 0.0
    norm_sum = np.sum(regime_dims)

    if np.isclose(norm_sum, 0.0):
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
    """Compute conditional means and annualized contributions by regime.

    Args:
        sampled_returns_with_regime_id: DataFrame with returns and regime classification
        freq: Sampling frequency for annualization
        is_report_pa_returns: If True, report as per annum returns
        regime_ids: Optional ordered list of regime IDs

    Returns:
        Tuple of (regime means, regime PA contributions, regime frequencies)
    """
    regime_means, norm_q = compute_mean_freq_regimes(
        sampled_returns_with_regime_id=sampled_returns_with_regime_id
    )

    af_mult = get_annualization_factor(freq=freq)

    if is_report_pa_returns:
        regime_pa = np.expm1(regime_means.multiply(af_mult * norm_q, axis=0))
    else:
        regime_pa = regime_means.multiply(af_mult * norm_q, axis=0)

    # Transpose: index = assets, columns = regimes
    regime_means = regime_means.T
    regime_pa = regime_pa.T

    # Arrange columns by specified regime order
    if regime_ids is not None:
        regime_means = regime_means[regime_ids]
        regime_pa = regime_pa[regime_ids]
    return regime_means, regime_pa, norm_q


def compute_regimes_pa_perf_table_from_sampled_returns(
        sampled_returns_with_regime_id: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark: str,
        perf_params: PerfParams,
        freq: str,
        is_use_benchmark_means: bool = False,
        is_add_ra_perf_table: bool = True,
        drop_benchmark: bool = False,
        additive_pa_returns_to_pa_total: bool = True,
        regime_ids: List[str] = None,
        **kwargs) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:
    """Compute comprehensive regime-conditional performance table.

    Args:
        sampled_returns_with_regime_id: Returns with regime classification
        prices: Asset price series
        benchmark: Benchmark asset name
        perf_params: Performance calculation parameters
        freq: Sampling frequency
        is_use_benchmark_means: Use benchmark means for normalization
        is_add_ra_perf_table: Include risk-adjusted performance table
        drop_benchmark: Exclude benchmark from final table
        additive_pa_returns_to_pa_total: Adjust regime PA to sum to total
        regime_ids: Ordered list of regime IDs

    Returns:
        Tuple of (performance table, regime data dictionary)
    """
    regime_avg, regime_pa, norm_q = compute_regime_avg(
        sampled_returns_with_regime_id=sampled_returns_with_regime_id,
        regime_ids=regime_ids,
        freq=freq,
        **kwargs
    )

    # Label columns consistently
    given_columns = regime_avg.columns.to_list()
    regime_avg = regime_avg[given_columns]
    regime_avg.columns = [f"{x} {RegimeData.REGIME_AVG.value}" for x in given_columns]

    regime_pa = regime_pa[given_columns]
    regime_pa.columns = [f"{x} {RegimeData.REGIME_PA.value}" for x in given_columns]
    regime_pa_columns = regime_pa.columns

    # Compute risk-adjusted performance table
    ra_perf_table = pt.compute_ra_perf_table_with_benchmark(
        prices=prices,
        benchmark=benchmark,
        perf_params=perf_params
    )

    if additive_pa_returns_to_pa_total:
        # Adjust regime PA to match total PA return
        total_sum = regime_pa[regime_pa_columns].sum(1)
        total_to_match = ra_perf_table[PerfStat.PA_RETURN.to_str()]
        total_pa_diff = total_to_match - total_sum

        weighted_diff = pd.DataFrame(
            np.tile(total_pa_diff, (len(norm_q.to_numpy()), 1)).T,
            index=total_pa_diff.index,
            columns=norm_q.index
        )
        regime_pa_diff = weighted_diff.multiply(norm_q, axis=1)
        regime_pa1 = regime_pa[regime_pa_columns].add(regime_pa_diff.to_numpy(), axis=0)
    else:
        regime_pa1 = regime_pa

    if is_use_benchmark_means and benchmark is not None:
        regime_pa1.loc[benchmark][regime_pa_columns] = regime_avg.loc[benchmark]

    # Compute regime Sharpe ratios
    vols_for_sharpe_pa = ra_perf_table[PerfStat.VOL.to_str()]
    regime_sharpe = regime_pa1.divide(vols_for_sharpe_pa, axis=0)[regime_pa_columns]
    regime_sharpe.columns = [f"{x}{RegimeData.REGIME_SHARPE.value}" for x in given_columns]

    # Combine into performance table
    if is_add_ra_perf_table:
        cond_perf_table = pd.concat([regime_avg, regime_pa1, regime_sharpe, ra_perf_table], axis=1)
    else:
        cond_perf_table = pd.concat([regime_avg, regime_pa1, regime_sharpe], axis=1)

    regime_datas = {
        RegimeData.REGIME_AVG: regime_avg,
        RegimeData.REGIME_PA: regime_pa1,
        RegimeData.REGIME_SHARPE: regime_sharpe
    }

    if drop_benchmark:
        cond_perf_table = cond_perf_table.drop(benchmark, axis=0)

    return cond_perf_table, regime_datas


# ============================================================================
# Abstract base class
# ============================================================================

class RegimeClassifier(ABC):
    """Abstract base class for regime classification.

    Regime classifiers partition time periods based on market conditions
    for conditional performance attribution analysis.
    """

    REGIME_COLUMN = 'regime'

    def __init__(self):
        self.regime_ids_colors: Dict[str, str] = {}
        super().__init__()

    @abstractmethod
    def compute_sampled_returns_with_regime_id(self, **kwargs) -> pd.DataFrame:
        """Compute returns with regime classification.

        Returns:
            DataFrame with returns and regime ID column
        """
        pass

    def get_regime_ids_colors(self) -> Dict[str, str]:
        """Get mapping of regime IDs to visualization colors.

        Returns:
            Dictionary mapping regime ID strings to color codes
        """
        return self.regime_ids_colors

    def get_regime_ids(self) -> List[str]:
        """Get ordered list of regime IDs.

        Returns:
            List of regime ID strings
        """
        return list(self.get_regime_ids_colors().keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert regime parameters to dictionary.

        Generic implementation that extracts public attributes (not starting with '_')
        from the instance, excluding methods and the REGIME_COLUMN class variable.
        Subclasses can override to customize serialization.

        Returns:
            Dictionary of parameter names to values

        Examples:
            >>> classifier = BenchmarkReturnsQuantilesRegime(freq='QE', q=4)
            >>> classifier.to_dict()
            {'freq': 'QE', 'return_type': <ReturnTypes.RELATIVE: ...>, 'q': 4, ...}
        """
        params = {}
        for key, value in self.__dict__.items():
            # Skip private attributes and methods
            if not key.startswith('_') and not callable(value):
                params[key] = value
        return params

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
                                      regime_ids: List[str] = None,
                                      **kwargs) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:
        """Compute regime-conditional performance attribution table.

        Args:
            regime_id_func_kwargs: Arguments for regime ID computation
            prices: Asset price series
            benchmark: Benchmark asset name
            freq: Sampling frequency
            perf_params: Performance parameters
            is_use_benchmark_means: Use benchmark means for normalization
            is_add_ra_perf_table: Include risk-adjusted performance
            drop_benchmark: Exclude benchmark from results
            additive_pa_returns_to_pa_total: Adjust regime PA to sum to total
            regime_ids: Ordered list of regime IDs

        Returns:
            Tuple of (performance table, regime data dictionary)
        """
        sampled_returns_with_regime_id = self.compute_sampled_returns_with_regime_id(
            **regime_id_func_kwargs
        )

        cond_perf_table, regime_datas = compute_regimes_pa_perf_table_from_sampled_returns(
            sampled_returns_with_regime_id=sampled_returns_with_regime_id,
            prices=prices,
            benchmark=benchmark,
            perf_params=perf_params,
            freq=freq,
            is_use_benchmark_means=is_use_benchmark_means,
            is_add_ra_perf_table=is_add_ra_perf_table,
            drop_benchmark=drop_benchmark,
            regime_ids=regime_ids
        )

        return cond_perf_table, regime_datas

    def class_data_to_colors(self, regime_data: pd.Series) -> pd.Series:
        """Map regime IDs to colors for visualization.

        Args:
            regime_data: Series of regime IDs

        Returns:
            Series of color codes
        """
        map_id_into_color = self.get_regime_ids_colors()
        regime_id_color = regime_data.map(map_id_into_color).astype(str)
        regime_id_color = regime_id_color.replace({'nan': '#FFFFFF'}) # type: ignore
        return regime_id_color


# ============================================================================
# Concrete implementations
# ============================================================================

class BenchmarkReturnsQuantilesRegime(RegimeClassifier):
    """Regime classifier based on benchmark return quantiles.

    Classifies periods into regimes based on quantiles of benchmark returns,
    enabling analysis of performance in different market environments
    (e.g., bear, normal, bull markets).
    """

    def __init__(self,
                 freq: str = 'QE',
                 return_type: ReturnTypes = ReturnTypes.RELATIVE,
                 q: Union[np.ndarray, int] = None,
                 regime_ids_colors: Dict[str, str] = None):
        """Initialize benchmark returns quantiles regime classifier.

        Args:
            freq: Sampling frequency (default: 'QE' for quarter-end)
            return_type: Type of returns to compute
            q: Quantile boundaries or number of quantiles (default: [0.0, 0.17, 0.83, 1.0])
            regime_ids_colors: Mapping of regime names to colors
        """
        super().__init__()
        self.freq = freq
        self.return_type = return_type
        self.q = q if q is not None else np.array([0.0, 0.17, 0.83, 1.0])
        self.regime_ids_colors = regime_ids_colors or {
            'Bear': mcolors['salmon'],
            'Normal': mcolors['yellowgreen'],
            'Bull': mcolors['darkgreen']
        }

    def compute_sampled_returns_with_regime_id(self,
                                               prices: Union[pd.DataFrame, pd.Series],
                                               benchmark: str,
                                               include_start_date: bool = True,
                                               include_end_date: bool = True,
                                               **kwargs) -> pd.DataFrame:
        """Classify periods by benchmark return quantiles.

        Args:
            prices: Asset prices
            benchmark: Benchmark column name
            include_start_date: Include first period
            include_end_date: Include last period

        Returns:
            DataFrame with returns and regime classification

        Raises:
            ValueError: If insufficient data for classification
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        sampled_returns_with_regime_id = ret.to_returns(
            prices=prices,
            freq=self.freq,
            return_type=self.return_type,
            include_start_date=include_start_date,
            include_end_date=include_end_date
        )

        if len(sampled_returns_with_regime_id.index) < 3:
            raise ValueError(
                f"Need at least 3 returns in time series: {sampled_returns_with_regime_id.index}\n"
                f"Decrease regime frequency"
            )

        x = sampled_returns_with_regime_id[benchmark]
        quant0 = pd.qcut(x=x, q=self.q, labels=self.get_regime_ids())
        sampled_returns_with_regime_id[self.REGIME_COLUMN] = quant0

        return sampled_returns_with_regime_id

    def compute_regimes_pa_perf_table(self,
                                      prices: pd.DataFrame,
                                      benchmark: str,
                                      perf_params: PerfParams,
                                      drop_benchmark: bool = False,
                                      **kwargs) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:
        """Compute regime performance attribution table.

        Args:
            prices: Asset prices
            benchmark: Benchmark asset name
            perf_params: Performance parameters
            drop_benchmark: Exclude benchmark from results

        Returns:
            Tuple of (performance table, regime data dictionary)
        """
        regime_id_func_kwargs = dict(
            prices=prices,
            benchmark=benchmark,
            include_start_date=True,
            include_end_date=True
        )

        return super().compute_regimes_pa_perf_table(
            regime_id_func_kwargs=regime_id_func_kwargs,
            prices=prices,
            benchmark=benchmark,
            perf_params=perf_params,
            freq=self.freq,
            is_report_pa_returns=True,
            is_use_benchmark_means=False,
            regime_ids=self.get_regime_ids(),
            drop_benchmark=drop_benchmark
        )


class BenchmarkReturnsPositiveNegativeRegime(RegimeClassifier):
    """Regime classifier based on positive vs negative benchmark returns.

    Classifies periods into two simple regimes based on benchmark return sign,
    useful for up/down market analysis.
    """

    def __init__(self,
                 freq: str = 'QE',
                 return_type: ReturnTypes = ReturnTypes.RELATIVE,
                 regime_ids_colors: Dict[str, str] = None):
        """Initialize positive/negative regime classifier.

        Args:
            freq: Sampling frequency (default: 'QE' for quarter-end)
            return_type: Type of returns to compute
            regime_ids_colors: Mapping of regime names to colors
        """
        super().__init__()
        self.freq = freq
        self.return_type = return_type
        self.regime_ids_colors = regime_ids_colors or {
            'Negative': mcolors['salmon'],
            'Positive': mcolors['darkgreen']
        }

    def compute_sampled_returns_with_regime_id(self,
                                               prices: Union[pd.DataFrame, pd.Series],
                                               benchmark: str,
                                               include_start_date: bool = True,
                                               include_end_date: bool = True,
                                               **kwargs) -> pd.DataFrame:
        """Classify periods by benchmark return sign.

        Args:
            prices: Asset prices
            benchmark: Benchmark column name
            include_start_date: Include first period
            include_end_date: Include last period

        Returns:
            DataFrame with returns and regime classification

        Raises:
            ValueError: If insufficient data for classification
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        sampled_returns_with_regime_id = ret.to_returns(
            prices=prices,
            freq=self.freq,
            return_type=self.return_type,
            include_start_date=include_start_date,
            include_end_date=include_end_date
        )

        if len(sampled_returns_with_regime_id.index) < 2:
            raise ValueError(
                f"Need at least 2 returns in time series: {sampled_returns_with_regime_id.index}\n"
                f"Decrease regime frequency"
            )

        benchmark_returns = sampled_returns_with_regime_id[benchmark]
        regime_ids = self.get_regime_ids()

        regime_classification = pd.Series(
            np.where(benchmark_returns < 0, regime_ids[0], regime_ids[1]),
            index=benchmark_returns.index,
            name=self.REGIME_COLUMN
        )

        sampled_returns_with_regime_id[self.REGIME_COLUMN] = regime_classification

        return sampled_returns_with_regime_id

    def compute_regimes_pa_perf_table(self,
                                      prices: pd.DataFrame,
                                      benchmark: str,
                                      perf_params: PerfParams,
                                      drop_benchmark: bool = False,
                                      **kwargs) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:
        """Compute regime performance attribution table.

        Args:
            prices: Asset prices
            benchmark: Benchmark asset name
            perf_params: Performance parameters
            drop_benchmark: Exclude benchmark from results

        Returns:
            Tuple of (performance table, regime data dictionary)
        """
        regime_id_func_kwargs = dict(
            prices=prices,
            benchmark=benchmark,
            include_start_date=True,
            include_end_date=True
        )

        return super().compute_regimes_pa_perf_table(
            regime_id_func_kwargs=regime_id_func_kwargs,
            prices=prices,
            benchmark=benchmark,
            perf_params=perf_params,
            freq=self.freq,
            is_report_pa_returns=True,
            is_use_benchmark_means=False,
            regime_ids=self.get_regime_ids(),
            drop_benchmark=drop_benchmark
        )


class BenchmarkVolsQuantilesRegime(RegimeClassifier):
    """Regime classifier based on benchmark volatility quantiles.

    Classifies periods based on realized volatility levels, enabling analysis
    of performance in different volatility environments.
    """

    def __init__(self,
                 freq: str = 'QE',
                 return_type: ReturnTypes = ReturnTypes.RELATIVE,
                 q: int = 4):
        """Initialize volatility quantiles regime classifier.

        Args:
            freq: Sampling frequency for volatility calculation
            return_type: Type of returns to compute
            q: Number of quantile buckets
        """
        super().__init__()
        self.freq = freq
        self.return_type = return_type
        self.q = q
        self.regime_colors: Dict[str, Tuple[float, ...]] = {}

    def compute_sampled_returns_with_regime_id(self,
                                               prices: pd.DataFrame,
                                               benchmark: str,
                                               include_start_date: bool = True,
                                               include_end_date: bool = True,
                                               **kwargs) -> pd.DataFrame:
        """Classify periods by benchmark volatility quantiles.

        Args:
            prices: Asset prices
            benchmark: Benchmark column name
            include_start_date: Include first period
            include_end_date: Include last period

        Returns:
            DataFrame with returns and regime classification
        """
        vols = ret.compute_sampled_vols(
            prices=prices[benchmark],
            freq_vol=self.freq,
            include_start_date=include_start_date,
            include_end_date=include_end_date
        )

        hue_name = f"{benchmark} vol"
        classificator, labels = dfc.add_quantile_classification(
            df=vols.to_frame(),
            x_column=benchmark,
            num_buckets=self.q,
            hue_name=hue_name,
            xvar_format='{:.0%}',
            bucket_prefix=hue_name
        )

        classificator = classificator.sort_index()

        sampled_returns_with_regime_id = ret.to_returns(
            prices=prices,
            freq=self.freq,
            return_type=self.return_type,
            include_start_date=include_start_date,
            include_end_date=include_end_date
        )

        sampled_returns_with_regime_id[self.REGIME_COLUMN] = classificator[hue_name]

        # Generate colors using colormap
        cmap = plt.get_cmap('RdYlGn', len(labels))
        colors = [cmap(n) for n in range(len(labels))]
        self.regime_colors = {k: v for k, v in zip(labels, colors)}

        return sampled_returns_with_regime_id

    def compute_regimes_pa_perf_table(self,
                                      prices: pd.DataFrame,
                                      benchmark: str,
                                      perf_params: PerfParams,
                                      **kwargs) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:
        """Compute regime performance attribution table.

        Args:
            prices: Asset prices
            benchmark: Benchmark asset name
            perf_params: Performance parameters

        Returns:
            Tuple of (performance table, regime data dictionary)
        """
        regime_id_func_kwargs = dict(
            prices=prices,
            benchmark=benchmark,
            include_start_date=True,
            include_end_date=True
        )

        return super().compute_regimes_pa_perf_table(
            regime_id_func_kwargs=regime_id_func_kwargs,
            prices=prices,
            benchmark=benchmark,
            perf_params=perf_params,
            freq=self.freq,
            is_report_pa_returns=True,
            is_use_benchmark_means=False,
            regime_ids=self.get_regime_ids()
        )

    def get_regime_colors(self) -> List[Tuple[float, ...]]:
        """Get list of regime colors in order."""
        return list(self.regime_colors.values())


# ============================================================================
# Convenience functions
# ============================================================================

def compute_bnb_regimes_pa_perf_table(prices: pd.DataFrame,
                                      benchmark: str = None,
                                      benchmark_price: pd.Series = None,
                                      freq: str = 'QE',
                                      return_type: ReturnTypes = ReturnTypes.RELATIVE,
                                      q: Union[np.ndarray, int] = None,
                                      regime_ids_colors: Dict[str, str] = None,
                                      perf_params: PerfParams = None,
                                      drop_benchmark: bool = False,
                                      **kwargs) -> pd.DataFrame:
    """Compute benchmark regime performance attribution table.

    Convenience function for computing regime-conditional performance using
    benchmark return quantiles classification.

    Args:
        prices: Asset price series
        benchmark: Benchmark column name in prices
        benchmark_price: Alternative benchmark price series to add to prices
        freq: Sampling frequency
        return_type: Type of returns to compute
        q: Quantile boundaries or number of quantiles
        regime_ids_colors: Mapping of regime names to colors
        perf_params: Performance parameters
        drop_benchmark: Exclude benchmark from results

    Returns:
        Regime-conditional performance table

    Raises:
        ValueError: If neither benchmark nor benchmark_price provided
    """
    if benchmark is None and benchmark_price is None:
        raise ValueError("Provide either benchmark name in prices or benchmark_price")

    if benchmark is not None and benchmark_price is None:
        if benchmark not in prices.columns:
            raise ValueError(f"{benchmark} is not in {prices.columns.to_list()}")
    elif benchmark_price is not None:
        if benchmark not in prices.columns:
            if not isinstance(benchmark_price, pd.Series):
                raise ValueError(f"benchmark_price must be pd.Series not {type(benchmark_price)}")
            benchmark_price = benchmark_price.reindex(index=prices.index, method='ffill').ffill()
            prices = pd.concat([benchmark_price, prices], axis=1)
            benchmark = benchmark_price.name

    regime_classifier = BenchmarkReturnsQuantilesRegime(
        freq=freq,
        return_type=return_type,
        q=q,
        regime_ids_colors=regime_ids_colors
    )

    regimes_pa_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(
        prices=prices,
        benchmark=benchmark,
        perf_params=perf_params,
        drop_benchmark=drop_benchmark
    )

    return regimes_pa_perf_table


class LocalTests(Enum):
    """Test cases for local development."""
    BNB_REGIME = 1
    BNB_PERF_TABLE = 2
    POS_NEG_REGIME = 3
    VOL_REGIME = 4
    TO_DICT = 5


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging.

    Integration tests that download real data and generate reports
    for quick verification during development.

    Args:
        local_test: Test case to run
    """
    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()
    perf_params = PerfParams()

    if local_test == LocalTests.BNB_REGIME:
        regime_classifier = BenchmarkReturnsQuantilesRegime(
            freq='QE',
            q=np.array([0.0, 0.17, 0.83, 1.0])
        )

        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark='SPY'
        )
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(
            prices=prices,
            benchmark='SPY',
            perf_params=perf_params
        )
        print(f"\nregime_means:\n{cond_perf_table}")
        print(f"\nregime_pa:\n{regime_datas}")

    elif local_test == LocalTests.BNB_PERF_TABLE:
        df = compute_bnb_regimes_pa_perf_table(
            prices=prices,
            benchmark='SPY',
            perf_params=PerfParams()
        )
        print(df)
        print(df.columns)

    elif local_test == LocalTests.POS_NEG_REGIME:
        regime_classifier = BenchmarkReturnsPositiveNegativeRegime(freq='QE')

        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark='SPY'
        )
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(
            prices=prices,
            benchmark='SPY',
            perf_params=perf_params
        )
        print(f"\ncond_perf_table:\n{cond_perf_table}")

    elif local_test == LocalTests.VOL_REGIME:
        regime_classifier = BenchmarkVolsQuantilesRegime(freq='QE', q=4)

        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark='SPY'
        )
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(
            prices=prices,
            benchmark='SPY',
            perf_params=perf_params
        )
        print(f"\ncond_perf_table:\n{cond_perf_table}")

    elif local_test == LocalTests.TO_DICT:
        print("Testing to_dict() method for all regime classifiers\n")

        # Test BenchmarkReturnsQuantilesRegime
        classifier1 = BenchmarkReturnsQuantilesRegime(
            freq='QE',
            return_type=ReturnTypes.RELATIVE,
            q=np.array([0.0, 0.25, 0.75, 1.0])
        )
        print("BenchmarkReturnsQuantilesRegime.to_dict():")
        print(classifier1.to_dict())
        print()

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.TO_DICT)