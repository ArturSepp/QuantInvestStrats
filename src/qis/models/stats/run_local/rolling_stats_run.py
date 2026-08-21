
import matplotlib.pyplot as plt
from enum import Enum
from qis.models.stats.rolling_stats import RollingPerfStat, compute_rolling_perf_stat


class Locals(Enum):
    ROLLING_STATS = 1


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import qis.plots.time_series as pts
    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()

    if local == Locals.ROLLING_STATS:

        for rolling_perf_stat in RollingPerfStat:
            stats, title = compute_rolling_perf_stat(prices=prices,
                                              rolling_perf_stat=rolling_perf_stat,
                                              roll_freq='W-WED',
                                              roll_periods=5*52)
            pts.plot_time_series(df=stats,
                                 var_format='{:.2f}',
                                 title=f"{title}")

    plt.show()


if __name__ == '__main__':

    run_local(local=Locals.ROLLING_STATS)
