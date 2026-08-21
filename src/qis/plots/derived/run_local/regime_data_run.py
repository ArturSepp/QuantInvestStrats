"""Development runner extracted from ``qis.plots.derived.regime_data``."""

import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from qis.perfstats.config import PerfParams
from qis.perfstats.regime_classifier import (BenchmarkReturnsQuantilesRegime, BenchmarkVolsQuantilesRegime, compute_bnb_regimes_pa_perf_table)

from qis.plots.derived.regime_data import (
    add_bnb_regime_shadows,
    plot_regime_boxplot,
    plot_regime_data,
)

class Locals(Enum):
    BNB_REGIME = 1
    VOL_REGIME = 2
    BNB_REGIME_SHADOWS = 3
    BNB_PERF_TABLE = 4
    AVG_PLOT = 5

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()

    kwargs = dict(var_format='{:.1f}')

    perf_params = PerfParams()

    if local == Locals.BNB_REGIME:
        regime_classifier = BenchmarkReturnsQuantilesRegime()
        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices, benchmark='SPY')
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(prices=prices,
                                                                                        benchmark='SPY',
                                                                                        perf_params=perf_params)
        print(f"regime_means:\n{cond_perf_table}")
        print(f"regime_pa:\n{regime_datas}")

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), tight_layout=True)

        plot_regime_data(regime_classifier=regime_classifier,
                         drop_sharpe_from_labels=True,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         title='(A) Weekly roll',
                         is_use_vbar=True,
                         is_add_totals=False,
                         add_bar_values=False,
                         fontsize=8,
                         ncols=3,
                         bbox_to_anchor=(0.5, 1.12),
                         pad=15,
                         ax=ax,
                         **kwargs)

        plot_regime_data(regime_classifier=regime_classifier,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         is_use_vbar=False,
                         bbox_to_anchor=None,
                         **kwargs)

    elif local == Locals.VOL_REGIME:
        perf_params = PerfParams()
        regime_classifier = BenchmarkVolsQuantilesRegime()
        regime_ids = regime_classifier.compute_sampled_returns_with_regime_id(prices=prices, benchmark='SPY')
        print(f"regime_ids:\n{regime_ids}")

        cond_perf_table, regime_datas = regime_classifier.compute_regimes_pa_perf_table(prices=prices,
                                                                                   benchmark='SPY',
                                                                                   perf_params=perf_params)
        print(f"regime_means:\n{cond_perf_table}")
        print(f"regime_pa:\n{regime_datas}")

        plot_regime_data(regime_classifier=regime_classifier,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         is_use_vbar=True,
                         **kwargs)

        plot_regime_data(regime_classifier=regime_classifier,
                         prices=prices,
                         benchmark='SPY',
                         perf_params=perf_params,
                         is_use_vbar=False,
                         **kwargs)

    elif local == Locals.BNB_REGIME_SHADOWS:
        import qis.plots.time_series as pts
        with sns.axes_style('white'):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
            pts.plot_time_series(df=prices, ax=ax)

            add_bnb_regime_shadows(ax=ax,
                                   data_df=prices,
                                   benchmark='SPY',
                                   regime_classifier=BenchmarkReturnsQuantilesRegime(),
                                   perf_params=PerfParams())

    elif local == Locals.BNB_PERF_TABLE:
        df = compute_bnb_regimes_pa_perf_table(prices=prices,
                                               benchmark='SPY',
                                               regime_classifier=BenchmarkReturnsQuantilesRegime(),
                                               perf_params=PerfParams())
        print(df)

    elif local == Locals.AVG_PLOT:
        perf_params = PerfParams()
        regime_classifier = BenchmarkVolsQuantilesRegime()

        with sns.axes_style('white'):
            fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
            plot_regime_boxplot(regime_classifier=regime_classifier,
                                prices=prices,
                                benchmark='SPY',
                                perf_params=perf_params,
                                ax=ax,
                                **kwargs)

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.BNB_REGIME)
