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
from qis.perfstats.config import ReturnTypes, RegimeData, PerfParams, PerfStat, SharpeConvention
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
        total_sum = regime_pa[regime_pa_columns].sum(axis=1)
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
        # pandas 3.0 CoW: chained assignment regime_pa1.loc[benchmark][regime_pa_columns] = ...
        # silently no-ops. Use single .loc[row, cols] indexer instead.
        regime_pa1.loc[benchmark, regime_pa_columns] = regime_avg.loc[benchmark]

    # Compute regime Sharpe ratios
    sharpe_convention = perf_params.sharpe_convention if perf_params is not None \
        and hasattr(perf_params, 'sharpe_convention') else SharpeConvention.PA
    if sharpe_convention in (SharpeConvention.ARITHMETIC, SharpeConvention.LOG):
        # additive regime Sharpe (sharpe_conventions.md sections 1 and 4): under ARITHMETIC,
        # sr_s = sqrt(af) * p_s * m_s / std(r) on the sampled simple returns; under LOG the same
        # construction on log(1+r). In both, sum_s sr_s equals the total Sharpe of the convention
        # exactly by linearity of the mean, so the pa additivity patch above does not apply, and
        # numerator and denominator are paired on the identical periodic return series.
        af_mult = get_annualization_factor(freq=freq)
        sampled_returns = sampled_returns_with_regime_id.drop(columns=[RegimeClassifier.REGIME_COLUMN])
        if sharpe_convention == SharpeConvention.LOG:
            # log-space decomposition (Sepp 2020): sr_s = sqrt(af) * p_s * mean(log(1+r) | s) / std(log(1+r)),
            # exactly additive to the log Sharpe, l = log(1+r) computed from the sampled simple returns
            log_returns_with_id = sampled_returns_with_regime_id.copy()
            log_returns_with_id[sampled_returns.columns] = np.log1p(sampled_returns)
            conditional_means, _ = compute_mean_freq_regimes(sampled_returns_with_regime_id=log_returns_with_id)
            conditional_means = conditional_means.T[given_columns]
            an_vol = np.sqrt(af_mult) * np.log1p(sampled_returns).std(ddof=1)
        else:
            # arithmetic decomposition: sr_s = sqrt(af) * p_s * m_s / std(r), exactly additive
            conditional_means = regime_avg.copy()
            conditional_means.columns = given_columns
            an_vol = np.sqrt(af_mult) * sampled_returns.std(ddof=1)
        # linear annualized regime contributions af * p_s * m_s from the conditional means
        regime_contrib = conditional_means.multiply(af_mult * norm_q[given_columns].to_numpy(), axis=1)
        regime_sharpe = regime_contrib.divide(an_vol, axis=0)
        regime_sharpe.columns = [f"{x}{RegimeData.REGIME_SHARPE.value}" for x in given_columns]
    else:  # SharpeConvention.PA, the default: unchanged
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
        # pandas 3.0 + Categorical input:
        #   - regime_data is typically Categorical (from pd.cut), so .map returns a Categorical
        #     whose categories are only the mapped color codes — assigning '#FFFFFF' to NaN
        #     positions raises "Cannot setitem on a Categorical with a new category".
        #   - Under the new `str` dtype, .astype(str) also preserves real NaN instead of
        #     coercing to the literal string 'nan', so the old .replace({'nan': ...}) no-ops.
        # Fix: map, coerce to plain object, then fillna with the neutral color.
        regime_id_color = regime_data.map(map_id_into_color)
        # astype(object) drops any Categorical / str-ExtensionArray wrapping so fillna accepts
        # an arbitrary string value.
        regime_id_color = regime_id_color.astype(object).fillna('#FFFFFF').astype(str)
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
            q: Quantile boundaries or number of quantiles (default: [0.0, 0.16, 0.84, 1.0],
                the one-sigma cut: P(Z < -1) = 15.87% rounds to 16%, central mass 68%
                against the normal's 68.27%. Changed from [0.0, 0.17, 0.83, 1.0] in 5.0.7)
            regime_ids_colors: Mapping of regime names to colors
        """
        super().__init__()
        self.freq = freq
        self.return_type = return_type
        self.q = q if q is not None else np.array([0.0, 0.16, 0.84, 1.0])  # one-sigma default, see compute_regime_sharpe_decomposition
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
            ValueError: If insufficient data for classification, or if the benchmark
                returns are degenerate (constant / too many ties) so that quantile
                bin edges are not unique.
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

        # Guard against a degenerate regime benchmark: pd.qcut requires strictly
        # increasing bin edges, but a constant / zero-return block (e.g. an overlay
        # nav with longer history than the principal, back-padded over the union
        # index) collapses interior quantiles onto the same value and raises a bare
        # "Bin edges must be unique". The condition below mirrors qcut exactly
        # (unique edges <= number of labels), so it fires iff qcut would have failed
        # and never on healthy data. Surface the cause instead of the pandas trace.
        labels = self.get_regime_ids()
        x_valid = x.dropna().to_numpy(dtype=float)
        probs = (np.linspace(0.0, 1.0, int(self.q) + 1) if np.isscalar(self.q)
                 else np.asarray(self.q, dtype=float))
        edges = np.nanquantile(x_valid, probs) if x_valid.size > 0 else np.array([])
        if np.unique(edges).size <= len(labels):
            raise ValueError(
                f"Regime benchmark '{x.name}' is degenerate for q={self.q}: only "
                f"{max(np.unique(edges).size - 1, 0)} of {len(labels)} quantile bands "
                f"are non-empty (edges={np.unique(edges).tolist()}).\n"
                f"This usually means a constant or zero-return block from misaligned "
                f"navs — clip the inputs to their common live window before classifying."
            )

        quant0 = pd.qcut(x=x, q=self.q, labels=labels)
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

def compute_regime_sharpe_decomposition(returns: Union[pd.Series, pd.DataFrame],
                                        benchmark_returns: pd.Series,
                                        af: float,
                                        q: Union[np.ndarray, int] = None,
                                        regime_ids: List[str] = None,
                                        sharpe_convention: SharpeConvention = SharpeConvention.ARITHMETIC,
                                        ddof: int = 1,
                                        is_add_total: bool = True
                                        ) -> Union[pd.Series, pd.DataFrame]:
    """
    returns-level additive regime Sharpe decomposition, sr_s = sqrt(af) * p_s * m_s / std
    (sharpe_conventions.md sections 1 and 4)

    the standalone counterpart of the regime-Sharpe branch of
    compute_regimes_pa_perf_table_from_sampled_returns for callers that hold periodic
    returns rather than prices: no resampling, no annualization inference (af is explicit),
    and any index type is accepted (the index is never touched, so RangeIndex works)

    classification uses pd.qcut on the benchmark returns with the same labels as the
    regime classifier, so on an aligned panel without missing values the output equals the
    table branch to machine precision. All moments are computed per asset over the rows
    where both the asset and the benchmark are observed, which makes the decomposition
    exactly additive per asset for any missing-value pattern:
    sum_s sr_s = sqrt(af) * mean / std on that asset's sample

    q defaults to the one-sigma boundaries np.array([0.0, 0.16, 0.84, 1.0])
    (P(Z < -1) = 15.87% rounds to 16%, central mass 68% against the normal's 68.27%),
    the single library-wide default shared with BenchmarkReturnsQuantilesRegime since 5.0.7

    supported conventions: ARITHMETIC (on simple returns) and LOG (the same construction
    on log(1 + r)), both exactly additive. PA does not decompose additively without the
    c-adjustment: use compute_regimes_pa_perf_table_from_sampled_returns for the adjusted
    p.a. regime bars
    """
    if not isinstance(benchmark_returns, pd.Series):
        raise ValueError(f"benchmark_returns must be pd.Series, got {type(benchmark_returns)!r}")
    if sharpe_convention == SharpeConvention.PA:
        raise ValueError("PA regime Sharpe requires the c-adjusted table path: "
                         "use compute_regimes_pa_perf_table_from_sampled_returns")
    is_series = isinstance(returns, pd.Series)
    returns_df = returns.to_frame() if is_series else returns
    if not isinstance(returns_df, pd.DataFrame):
        raise ValueError(f"returns must be pd.Series or pd.DataFrame, got {type(returns)!r}")
    if q is None:
        q = np.array([0.0, 0.16, 0.84, 1.0])

    benchmark_valid = benchmark_returns.dropna()
    if len(benchmark_valid) < 3:
        raise ValueError(f"need at least 3 benchmark returns to classify, got {len(benchmark_valid)!r}")
    n_buckets = int(q) if np.isscalar(q) else len(np.asarray(q)) - 1
    if regime_ids is None:
        regime_ids = ['Bear', 'Normal', 'Bull'] if n_buckets == 3 else [f"Q{n + 1}" for n in range(n_buckets)]
    if len(regime_ids) != n_buckets:
        raise ValueError(f"regime_ids must have {n_buckets} labels, got {regime_ids!r}")
    regime_id = pd.qcut(x=benchmark_valid, q=q, labels=regime_ids)

    if sharpe_convention == SharpeConvention.LOG:
        returns_df = np.log1p(returns_df)

    out = {}
    for asset in returns_df.columns:
        r = returns_df[asset].reindex(benchmark_valid.index).dropna()
        regimes = regime_id.reindex(r.index)
        sigma = r.std(ddof=ddof)
        n = len(r)
        row = {f"{regime}{RegimeData.REGIME_SHARPE.value}":
               float(np.sqrt(af) * ((regimes == regime).sum() / n) * (r[regimes == regime].mean() if (regimes == regime).any() else 0.0) / sigma)
               for regime in regime_ids}
        if is_add_total:
            row[f"Total{RegimeData.REGIME_SHARPE.value}"] = float(np.sqrt(af) * r.mean() / sigma)
        out[asset] = row
    table = pd.DataFrame.from_dict(out, orient='index')
    return table.iloc[0] if is_series else table
