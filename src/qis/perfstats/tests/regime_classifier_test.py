

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from qis.perfstats.config import ReturnTypes, PerfParams
from qis.perfstats.regime_classifier import (BenchmarkReturnsQuantilesRegime,
                                             compute_bnb_regimes_pa_perf_table,
                                             BenchmarkReturnsPositiveNegativeRegime,
                                             BenchmarkVolsQuantilesRegime)


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
    from qis.tests.price_data_test import load_etf_data
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