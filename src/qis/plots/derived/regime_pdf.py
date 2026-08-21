"""
the distribution of the benchmark's own returns, split by the regimes its quantiles define, drawn
as overlaid densities in the classifier's colours. ``plot_regime_pdf`` is the only entry point:
``is_histogram`` switches from the kernel density to bars, and ``is_multiple_stack`` stacks the
regimes rather than overlaying them - it applies to the histogram branch only.
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# qis
import qis.plots.utils as put
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantilesRegime


def plot_regime_pdf(prices: pd.DataFrame,
                    benchmark: str,
                    regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
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

    sampled_returns_with_regime_id = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices,
                                                                                              benchmark=benchmark)

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
