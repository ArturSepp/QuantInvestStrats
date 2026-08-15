
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.contour import plot_contour

class LocalTests(Enum):
    SHARPE_VOL = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.SHARPE_VOL:

        global_kwargs = {'fontsize': 12}

        n = 41
        vol_ps = np.linspace(0.04, 0.20, n)
        vol_xys = np.linspace(0.00, 0.20, n)
        sharpes = np.zeros((n, n))
        for n1, vol_p in enumerate(vol_ps):
            for n2, vol_xy in enumerate(vol_xys):
                sharpes[n1, n2] = (2.0*vol_xy*vol_xy-0.25*vol_p*vol_p)/vol_p

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        plot_contour(x=vol_ps,
                     y=vol_xys,
                     z=sharpes,
                     fig=fig,
                     **global_kwargs)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SHARPE_VOL)
