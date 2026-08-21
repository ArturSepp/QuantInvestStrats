

from enum import Enum
import qis.perfstats.returns as ret
from qis.models.linear.pca import compute_data_pca_r2

class Locals(Enum):
    PCA_R2 = 1


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()
    print(prices)
    returns = ret.to_returns(prices=prices)

    if local == Locals.PCA_R2:
        pca_r2 = compute_data_pca_r2(data=returns,
                                     freq='YE',
                                     ewm_lambda=0.97)
        print(pca_r2)


if __name__ == '__main__':

    run_local(local=Locals.PCA_R2)
