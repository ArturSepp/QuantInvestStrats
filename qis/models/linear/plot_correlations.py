# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from typing import Optional, Union

import qis
# qis
import qis.utils.dates as da
import qis.utils.np_ops as npo
import qis.utils.df_str as dfs
import qis.perfstats.returns as ret
from qis.perfstats.config import PerfParams, ReturnTypes
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs

import qis.plots.time_series as pts
import qis.plots.utils as put
import qis.plots.heatmap as phe
from qis.plots.derived.regime_data import add_bnb_regime_shadows

import qis.models.linear.corr_cov_matrix as ccm
import qis.models.linear.ewm as ewm
from qis.models.linear.corr_cov_matrix import CorrMatrixOutput
from qis.models.linear.ewm import InitType


def plot_returns_corr_table(prices: pd.DataFrame,
                            var_format: str = '{:.0%}',
                            freq: Optional[str] = None,
                            cmap: str = 'PiYG',
                            return_type: ReturnTypes = ReturnTypes.RELATIVE,
                            is_fig_out: bool = True,
                            ax: plt.Subplot = None,
                            **kwargs
                            ) -> Union[Optional[plt.Figure], pd.DataFrame]:
    """
    plot corr table of prices
    """
    returns = ret.to_returns(prices=prices, return_type=return_type, freq=freq)
    corr = ccm.compute_masked_covar_corr(data=returns, is_covar=False)
    if is_fig_out:
        return phe.plot_heatmap(df=corr,
                                var_format=var_format,
                                cmap=cmap,
                                ax=ax,
                                **kwargs)
    else:
        return corr


def plot_returns_ewm_corr_table(prices: pd.DataFrame,
                                span: Union[int, np.ndarray] = None,
                                ewm_lambda: float = 0.94,
                                var_format: str = '{:.0%}',
                                return_type: ReturnTypes = ReturnTypes.LOG,
                                init_type: InitType = InitType.ZERO,
                                freq: Optional[str] = None,
                                cmap: str = 'PiYG',
                                is_last: bool = True,
                                ax: plt.Subplot = None,
                                **kwargs
                                ) -> plt.Figure:
    """
     plot corr table of rices
    """
    returns = ret.to_returns(prices=prices, return_type=return_type, freq=freq)
    init_value = ewm.set_init_dim2(data=returns.to_numpy(), init_type=init_type)
    corr = ewm.compute_ewm_covar_tensor(a=returns.to_numpy(),
                                        span=span,
                                        ewm_lambda=ewm_lambda,
                                        covar0=init_value,
                                        is_corr=True)
    if is_last:
        ar = corr[-1]
    else:
        ar = npo.tensor_mean(corr)

    df = pd.DataFrame(ar, index=prices.columns, columns=prices.columns)
    fig = phe.plot_heatmap(df=df,
                           var_format=var_format,
                           cmap=cmap,
                           ax=ax,
                           **kwargs)
    return fig


def plot_returns_corr_matrix_time_series(prices: pd.DataFrame,
                                         corr_matrix_output: CorrMatrixOutput = CorrMatrixOutput.FULL,
                                         return_type: ReturnTypes = ReturnTypes.LOG,
                                         time_period: da.TimePeriod = None,
                                         freq: Optional[str] = None,
                                         span: Union[int, np.ndarray] = None,
                                         ewm_lambda: float = 0.97,
                                         init_type: InitType = InitType.X0,
                                         init_value: np.ndarray = None,
                                         var_format: str = '{:.0%}',
                                         legend_stats: pts.LegendStats = pts.LegendStats.AVG_LAST,
                                         trend_line: put.TrendLine = put.TrendLine.AVERAGE,
                                         regime_benchmark: str = None,
                                         regime_params: BenchmarkReturnsQuantileRegimeSpecs = None,
                                         perf_params: PerfParams = None,
                                         ax: plt.Subplot = None,
                                         **kwargs
                                         ) -> None:
    """
    plot time series of returns correlations infered from prices
    """
    returns = ret.to_returns(prices=prices, return_type=return_type, freq=freq)

    corr_pandas = ccm.compute_ewm_corr_df(df=returns,
                                          corr_matrix_output=corr_matrix_output,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=init_type,
                                          init_value=init_value)
    if time_period is not None:
        corr_pandas = time_period.locate(corr_pandas)

    pts.plot_time_series(df=corr_pandas,
                         legend_stats=legend_stats,
                         trend_line=trend_line,
                         var_format=var_format,
                         ax=ax,
                         **kwargs)
    if regime_benchmark is not None:
        if regime_benchmark in prices.columns:
            pivot_prices = prices[regime_benchmark].reindex(index=corr_pandas.index, method='ffill')
        else:
            raise KeyError(f"{regime_benchmark} not in {prices.columns}")

        if regime_benchmark is not None and regime_params is not None:
            add_bnb_regime_shadows(ax=ax,
                                   pivot_prices=pivot_prices,
                                   benchmark=regime_benchmark,
                                   regime_params=regime_params,
                                   perf_params=perf_params)


def plot_corr_matrix_from_covar(covar: pd.DataFrame,
                                corr_format: str = '{:.2f}',
                                vol_format: str = '{:.1%}',
                                cmap: str = 'PiYG',
                                title: Optional[Union[str, bool]] = True,
                                ax: plt.Subplot = None,
                                **kwargs
                                ) -> Optional[plt.Figure]:

    """Plot a correlation matrix heatmap with volatilities on the diagonal.

    Creates a lower-triangular correlation matrix heatmap where the diagonal
    displays volatilities (standard deviations) and the lower triangle shows
    correlations between variables. The upper triangle is masked (empty).

    Args:
        covar (pd.DataFrame): Covariance matrix with variables as both index
            and columns. Must be a square symmetric matrix.
        corr_format (str, optional): Format string for correlation values
            in the lower triangle. Defaults to '{:.2f}'.
        vol_format (str, optional): Format string for volatility values
            on the diagonal. Defaults to '{:.1%}'.
        cmap (str, optional): Colormap name for the heatmap. Defaults to 'PiYG'.
        title (Optional[Union[str, bool]], optional): Title for the plot.
            If True, uses default title. If False or None, no title is shown.
            If string, uses the provided title. Defaults to True.
        ax (plt.Subplot, optional): Matplotlib axes object to plot on.
            If None, creates a new figure. Defaults to None.
        **kwargs: Additional keyword arguments passed to the underlying
            heatmap plotting function.

    Returns:
        Optional[plt.Figure]: Figure object containing the heatmap if ax is None,
            otherwise None when plotting on provided axes.

    Note:
        The function converts the covariance matrix to correlations using
        npo.covar_to_corr() and extracts volatilities as the square root
        of diagonal covariance elements. Grid lines are added around each
        cell for better visual separation.
    """
    corr = npo.covar_to_corr(covar)
    # add nans to upper diagonal
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr_flt = pd.DataFrame(index=covar.index, columns=covar.columns, dtype=float)
    corr_flt[mask == False] = corr
    # replace diagonal with vol
    mask = np.eye(covar.shape[0], dtype=bool)
    vols = np.sqrt(np.diag(covar))
    corr_flt[mask] = np.diag(vols)

    # Create a custom annotation matrix: empty strings on the diagonal, values off-diagonal
    annot = np.full(corr_flt.shape, "", dtype=object)
    for i in range(corr_flt.shape[0]):
        for j in range(corr_flt.shape[1]):
            if i > j:
                annot[i, j] = corr_format.format(corr_flt.iloc[i, j])  # format as string with 2 decimals
            elif i == j:
                annot[i, j] = vol_format.format(corr_flt.iloc[i, j])  # format as string with 2 decimals

    if title is not None:
        if not isinstance(title, str):
            if title is True:
                title = 'Diagonal: volatilities; Lower diagonal: correlations'
            else:
                title = None

    # Plot heatmap with custom annotations
    fig = phe.plot_heatmap(df=corr_flt,
                           annot=annot,
                           cmap=cmap,
                           var_format=None,
                           vmin=-1, vmax=1, center=0,
                           title=title,
                           ax=ax,
                           **kwargs)
    if ax is None:
        ax = fig.axes[0]

    for vline_column in np.arange(1, len(covar.columns)+1):
        for hline_row in np.arange(1, len(covar.columns)+1):
            ax.vlines([hline_row-1], hline_row-1, hline_row, lw=1)
            ax.hlines([vline_column-1], vline_column-1, vline_column, lw=1)
            ax.vlines([hline_row], hline_row-1, hline_row, lw=1)
            ax.hlines([vline_column], vline_column-1, vline_column, lw=1)
    return fig


class UnitTests(Enum):
    CORR_TABLE = 1
    CORR_MATRIX = 2
    EWMA_CORR = 3
    PLOT_CORR_FROM_COVAR = 4


def run_unit_test(unit_test: UnitTests):
    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.CORR_TABLE:
        plot_returns_corr_table(prices=prices)

    elif unit_test == UnitTests.CORR_MATRIX:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
        plot_returns_corr_matrix_time_series(prices=prices, regime_benchmark='SPY', ax=ax)

    elif unit_test == UnitTests.EWMA_CORR:
        plot_returns_ewm_corr_table(prices=prices.iloc[:, :5])

    elif unit_test == UnitTests.PLOT_CORR_FROM_COVAR:
        returns = ret.to_returns(prices=prices, freq='ME')
        covar = 12.0 * ccm.compute_masked_covar_corr(data=returns, is_covar=True)
        plot_corr_matrix_from_covar(covar)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_CORR_FROM_COVAR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
