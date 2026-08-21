

from enum import Enum
from qis.perfstats.desc_table import compute_desc_table, DescTableType


class Locals(Enum):
    TABLE = 1


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    returns = load_etf_data().dropna().asfreq('QE').pct_change()

    if local == Locals.TABLE:
        df = compute_desc_table(df=returns,
                                desc_table_type=DescTableType.EXTENSIVE,
                                var_format='{:.2f}',
                                annualize_vol=True,
                                is_add_tstat=False)
        print(df)


if __name__ == '__main__':

    run_local(local=Locals.TABLE)
