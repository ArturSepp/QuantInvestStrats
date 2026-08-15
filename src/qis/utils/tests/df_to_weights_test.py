
import numpy as np
import pandas as pd
from enum import Enum
from qis.utils.df_to_weights import (mult_df_columns_with_vector,
                                     mult_df_columns_with_vector_group,
                                     df_to_equal_weight_allocation,
                                     df_to_weight_allocation_sum1)


class LocalTests(Enum):
    CONV = 1
    EW_ALLOC = 2
    AC_EQUAL_WEIGHT_ALLOCATION = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import qis.utils.df_ops as dfo

    if local_test == LocalTests.CONV:
        constituent_dict = {'f1': np.array((0, np.nan, 1, 1)),
                            'f2': np.array((0, np.nan, np.nan, 1)),
                            'f3': np.array((0, np.nan, 1, 1)),
                            'f4': np.array((0, np.nan, 1, 1)),
                            'f5': np.array((0, np.nan, np.nan, 1)),
                            'f6': np.array((0, np.nan, 1, 1))}

        constituent_prices = pd.DataFrame.from_dict(constituent_dict, orient='columns')
        constituent_data = dfo.df_indicator_like(constituent_prices)
        print(constituent_data)

        desc_dict = {'f1': ('eq', 1.0),
                     'f2': ('fi', 0.5),
                     'f3': ('fi', 1.5),
                     'f4': ('fx', 1.0),
                     'f5': ('fx', 2.0),
                     'f6': ('fx', 3.0)}

        desc_data = pd.DataFrame.from_dict(desc_dict, orient='index', columns=['ac', 'acw'])
        print(desc_data)

        conv = mult_df_columns_with_vector(df=constituent_data, vector=desc_data['acw'])
        print(f"conv:\n{conv}")

        conv_norm = mult_df_columns_with_vector(df=constituent_data, vector=desc_data['acw'], is_norm=True)
        print(f"conv_norm:\n{conv_norm}")

        ac_conv = mult_df_columns_with_vector_group(df=constituent_data, vector=desc_data['acw'],
                                                    group_data=desc_data['ac'])
        for ac, data in ac_conv.items():
            print(f"{ac}:\n{data}")

        ac_conv = mult_df_columns_with_vector_group(df=constituent_data, vector=desc_data['acw'],
                                                    group_data=desc_data['ac'],
                                                    is_norm=True)
        for ac, data in ac_conv.items():
            print(f"norm {ac}:\n{data}")

        ac_conv_pd = mult_df_columns_with_vector_group(df=constituent_data, vector=desc_data['acw'],
                                                       group_data=desc_data['ac'],
                                                       is_norm=True,
                                                       return_df=True)
        print(f"ac_conv_pd:\n{ac_conv_pd}")

    elif local_test == LocalTests.EW_ALLOC:
        constituent_dict = {'f1': np.array((0, np.nan, 2, 3)),
                            'f2': np.array((0, np.nan, np.nan, 4)),
                            'f3': np.array((0, np.nan, 1, 1)),
                            'f4': np.array((0, np.nan, 1, 1)),
                            'f5': np.array((0, np.nan, np.nan, 1)),
                            'f6': np.array((0, np.nan, 1, 1))}

        constituent_prices = pd.DataFrame.from_dict(constituent_dict, orient='columns')
        print(constituent_prices)
        weights = df_to_equal_weight_allocation(constituent_prices)
        print(weights)

    elif local_test == LocalTests.AC_EQUAL_WEIGHT_ALLOCATION:
        universe_data = dict(SPY='Equities',
                             QQQ='Equities',
                             EEM='Equities',
                             TLT='Bonds',
                             IEF='Bonds',
                             LQD='Credit',
                             HYG='Credit',
                             GLD='Gold')
        # each asset class has equal weight allocation
        group_data = pd.Series(universe_data)
        constituents = pd.DataFrame(1.0, columns=group_data.index, index=['date1', 'date2'])
        constituents.loc['date1', 'EEM'] = np.nan
        constituents.loc['date2', 'IEF'] = np.nan
        within_ac_weight = pd.Series(1.0, index=group_data.index)
        within_ac_weight['HYG'] = 0.5
        rel_ac_weight_by_ints = mult_df_columns_with_vector_group(df=constituents,
                                                                  vector=within_ac_weight,
                                                                  group_data=group_data,
                                                                  is_norm=True, nan_fill_zero=True,
                                                                  return_df=True)
        rel_ac_weight_by_ints = df_to_weight_allocation_sum1(df=rel_ac_weight_by_ints)
        print(rel_ac_weight_by_ints)
        print(rel_ac_weight_by_ints.sum(axis=1))


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.AC_EQUAL_WEIGHT_ALLOCATION)
