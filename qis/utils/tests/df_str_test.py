
import pandas as pd
from enum import Enum
from qis.utils.df_str import df_all_to_str, tabulate_df


class LocalTests(Enum):
    DF_TO_STR = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.DF_TO_STR:
        df = pd.DataFrame({
            "c1": ("a", "bb", "ccc", "dddd", "eeeeee"),
            "c2": (11, 22, 33, 44, 55),
            "a3235235235": [1, 2, 3, 4, 5]
        })
        print(df)

        fmts = df_all_to_str(df)
        print(fmts)

        stats_str = tabulate_df(df, showindex=True, floatfmt='.2f', headers=df.columns)
        print(stats_str)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.DF_TO_STR)
