
import pandas as pd
from enum import Enum
from qis.utils.df_groups import get_group_dict


class LocalTests(Enum):
    GROUP = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.GROUP:

        group_data = pd.Series(dict(SPY='Equities', QQQ='Equities', EEM='Equities', TLT='Bonds',
                                    IEF='Bonds', SHY='Bonds', LQD='Credit', HYG='HighYield', GLD='Gold'))

        group_dict = get_group_dict(group_data=group_data)
        print(f"group_dict=\n{group_dict}")

        group_dict_ordered = get_group_dict(group_data=group_data, group_order=list(group_data.unique()))
        print(f"group_dict_ordered=\n{group_dict_ordered}")

        group_dict_subset = get_group_dict(group_data=group_data,
                                           index_data=group_data.index[:5].to_list(),
                                           group_order=list(group_data.unique()))
        print(f"group_dict_subset=\n{group_dict_subset}")


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.GROUP)
