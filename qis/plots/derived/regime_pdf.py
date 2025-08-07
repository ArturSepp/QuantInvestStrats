"""
plot returns heatmap table by monthly and annual
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from matplotlib.ticker import FuncFormatter

# qis
import qis.plots.utils as put
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs, BenchmarkReturnsQuantilesRegime


def plot_regime_pdf(prices: pd.DataFrame,
                    benchmark: str,
                    regime_params: BenchmarkReturnsQuantileRegimeSpecs = BenchmarkReturnsQuantileRegimeSpecs(),
                    ax: plt.Subplot = None,
                    var_format: str = '{:.0%}',
                    is_histogram: bool = False,
                    is_multiple_stack: bool = False,
                    title: str = None,
                    fontsize: int = 10,
                    bins: int = 30,
                    legend_loc: str = None,
                    **kwargs
                    ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    regime_classifier = BenchmarkReturnsQuantilesRegime(regime_params=regime_params)
    sampled_returns_with_regime_id = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices,
                                                                                              benchmark=benchmark,
                                                                                              **regime_params._asdict())

    if is_histogram:
        sns.histplot(data=sampled_returns_with_regime_id,
                     x=benchmark,
                     hue=regime_classifier.REGIME_COLUMN,
                     hue_order=regime_classifier.get_regime_ids_colors().keys(),
                     multiple='stack' if is_multiple_stack else 'layer',
                     bins=bins,
                     palette=regime_classifier.get_regime_ids_colors().values(),
                     ax=ax)


    else:
        sns.kdeplot(data=sampled_returns_with_regime_id,
                    x=benchmark,
                    hue=regime_classifier.REGIME_COLUMN,
                    hue_order=regime_classifier.get_regime_ids_colors().keys(),
                    palette=regime_classifier.get_regime_ids_colors().values(),
                    ax=ax)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: var_format.format(x)))

    put.set_legend(ax=ax, legend_loc=legend_loc, fontsize=fontsize, **kwargs)

    ax.get_yaxis().set_visible(False)
    put.set_spines(ax=ax, **kwargs)
    if title is not None:
        ax.set_title(label=title, **kwargs)

    return fig


class LocalTests(Enum):
    REGIME_PDF = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.test_data import load_etf_data
    prices = load_etf_data()[['SPY', 'TLT']].dropna()

    if local_test == LocalTests.REGIME_PDF:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True)
            plot_regime_pdf(prices=prices, benchmark='SPY', is_histogram=False, ax=axs[0])
            plot_regime_pdf(prices=prices, benchmark='SPY', is_histogram=True, ax=axs[1])

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.REGIME_PDF)
