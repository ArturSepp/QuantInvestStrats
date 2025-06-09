import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import qis as qis


def get_test_data(n_days: int = 1000) -> Tuple[pd.Series, pd.Series]:
    """
    generate randpm nav and volume
    """
    index = pd.date_range(start='1Jan2020', periods=n_days, freq='D')
    returns = pd.Series(np.random.normal(0.0, 0.2/np.sqrt(260), n_days), index=index, name='nav')
    nav = qis.returns_to_nav(returns=returns)
    volume = returns.abs().cumsum().rename('volume')
    return nav, volume


def generate_figure_2x(num_assets: int = 5, n_days: int = 1000) -> List[plt.Figure]:
    """
    plot figure for each asst
    """
    figs = []
    for n in np.arange(num_assets):
        nav, volume = get_test_data(n_days=n_days)
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(16, 12), tight_layout=True)
            figs.append(fig)
            title = f"asset num {n+1}"
            qis.plot_time_series_2ax(df1=nav, df2=volume,
                                     var_format='{:,.2f}',
                                     var_format_yax2='{:,.2f}',
                                     title=title, ax=ax)
    return figs


figs = generate_figure_2x(num_assets=5, n_days=1000)
qis.save_figs_to_pdf(figs=figs, file_name='test_price_report', local_path=qis.get_output_path())
