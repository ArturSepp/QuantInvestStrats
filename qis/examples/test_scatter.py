import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import qis.plots as qp

np.random.seed(2)


def get_random_data(is_random_beta: bool = True,
                    n: int = 10000
                    ) -> pd.DataFrame:

    x = np.random.normal(0.0, 1.0, n)
    eps = np.random.normal(0.0, 1.0, n)
    if is_random_beta:
        beta = np.random.normal(1.0, 1.0, n)*np.abs(x)
    else:
        beta = np.ones(n)
    y = beta*x + eps
    df = pd.concat([pd.Series(x, name='x'), pd.Series(y, name='y')], axis=1)
    df = df.sort_values(by='x', axis=0)
    return df


df = get_random_data(n=100000)

kwargs = dict(xvar_format='{:.1f}', yvar_format='{:.1f}', order=1)

with sns.axes_style('darkgrid'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    qp.plot_scatter(df=df,
                    full_sample_order=1,
                    fit_intercept=False,
                    title='Linear regression',
                    ax=axs[0],
                    **kwargs)
    qp.plot_classification_scatter(df=df,
                                   full_sample_order=None,
                                   fit_intercept=False,
                                   title='Localized sextile regression',
                                   ax=axs[1],
                                   **kwargs)
    qp.align_y_limits_ax12(ax1=axs[0], ax2=axs[1])
    qp.set_suptitle(fig, 'Estimation of noisy regression: y=beta*abs(x)*x + noise, beta=Normal(1, 1), x=Normal(0, 1), noise=Normal(0, 1)')


plt.show()