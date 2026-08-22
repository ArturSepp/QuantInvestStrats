"""Rolling PCA variance shares for a seeded multi-asset return panel.

``qis.compute_data_pca_r2`` applies an exponentially weighted covariance estimate through time
and reports the share explained by each principal component. This offline example visualises
quarterly observations with ``qis.plot_stack``.

Run with ``python -m examples.models.pca_variance_explained``.
"""

import matplotlib.pyplot as plt
import pandas as pd

import qis
from qis.datasets.synthetic import generate_synthetic_prices


TICKERS = ['SEQ_US', 'SEQ_EU', 'SBD_TSY', 'SBD_IG', 'SCM_GLD', 'SAL_HF']


def compute_rolling_pca_shares() -> pd.DataFrame:
    """Return quarterly PCA variance shares from weekly log returns."""
    prices = generate_synthetic_prices(
        start='2005-01-03',
        end='2025-12-31',
        apply_quirks=False,
    )[TICKERS]
    returns = qis.to_returns(
        prices=prices,
        freq='W-WED',
        is_log_returns=True,
        drop_first=True,
    )
    return qis.compute_data_pca_r2(
        data=returns,
        freq='QE',
        ewm_lambda=0.97,
        is_corr=False,
    )


def run_example(show: bool = True) -> pd.DataFrame:
    """Run the PCA example and optionally display its chart."""
    pca_shares = compute_rolling_pca_shares()
    print(pca_shares.tail().round(3).to_string())
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    qis.plot_stack(
        df=pca_shares,
        add_cum_levels=True,
        is_yaxis_limit_01=True,
        legend_loc=None,
        x_date_freq='YE',
        date_format='%Y',
        title='Rolling PCA shares of the EWMA covariance matrix',
        ax=ax,
    )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return pca_shares


if __name__ == '__main__':
    run_example()
