"""
plot descriptive table
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional
from enum import Enum

# qis
from qis.plots.table import plot_df_table
from qis.perfstats.desc_table import compute_desc_table, DescTableType


def plot_desc_table(df: Union[pd.DataFrame, pd.Series],
                    desc_table_type: DescTableType = DescTableType.SHORT,
                    var_format: str = '{:.2f}',
                    annualize_vol: bool = False,
                    is_add_tstat: bool = False,
                    norm_variable_display_type: str = '{:.1f}',  # for t-stsat
                    ax: plt.Subplot = None,
                    **kwargs
                    ) -> Optional[plt.Figure]:
    """
    data corresponds to matrix of returns with index = time and columns = tickers
    data can contain nan in columns
    output is index = tickers, columns = descriptive data
    data converted to str
    """
    table_data = compute_desc_table(df=df,
                                    desc_table_type=desc_table_type,
                                    var_format=var_format,
                                    annualize_vol=annualize_vol,
                                    is_add_tstat=is_add_tstat,
                                    norm_variable_display_type=norm_variable_display_type,
                                    **kwargs)

    fig = plot_df_table(df=table_data,
                        ax=ax,
                        **kwargs)
    return fig


class UnitTests(Enum):
    TABLE = 1


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    returns = load_etf_data().dropna().asfreq('QE').pct_change()

    if unit_test == UnitTests.TABLE:
        plot_desc_table(df=returns,
                        desc_table_type=DescTableType.EXTENSIVE,
                        var_format='{:.2f}',
                        annualize_vol=True,
                        is_add_tstat=False)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.TABLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
