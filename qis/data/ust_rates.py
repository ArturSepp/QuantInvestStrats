"""
load ust rates data
"""
import numpy as np
from enum import Enum

import pandas as pd

import qis.file_utils as fu

LOCAL_RESOURCE_PATH = fu.get_local_paths()['LOCAL_RESOURCE_PATH']
UST_FILE_NAME = 'ust_rates'


COLUMN_MAP = {"BC_1MONTH": '1m',
              "BC_3MONTH": '3m',
              "BC_6MONTH": '6m',
              "BC_1YEAR": '1y',
              "BC_2YEAR": '2y',
              "BC_3YEAR": '3y',
              "BC_5YEAR": '5y',
              "BC_7YEAR": '7y',
              "BC_10YEAR": '10y',
              "BC_20YEAR": '20y',
              "BC_30YEAR": '30y'}


def generate_ust_data(folder: str = f"{LOCAL_RESOURCE_PATH}xml"):

    from ust import save_xml, read_rates, available_years, year_now

    # save UST yield rates to local folder for selected years
    for year in available_years():
        save_xml(year, folder=folder)
    save_xml(year_now(), folder=folder, overwrite=True)
    df = read_rates(start_year=1990, end_year=2022, folder=folder)
    df = df.drop('BC_30YEARDISPLAY', axis=1).rename(COLUMN_MAP, axis=1)
    df = df.replace({0.0: np.nan})
    fu.save_df_to_csv(df=df, file_name=UST_FILE_NAME, local_path=LOCAL_RESOURCE_PATH)


def load_ust_rates() -> pd.DataFrame:
    df = fu.load_df_from_csv(file_name=UST_FILE_NAME, local_path=LOCAL_RESOURCE_PATH)
    df = df.fillna(method='ffill')
    return df


def load_ust_3m_rate() -> pd.Series:
    df = fu.load_df_from_csv(file_name=UST_FILE_NAME, local_path=LOCAL_RESOURCE_PATH)
    df = df.fillna(method='ffill')
    return df['3m'] / 100.0


class UnitTests(Enum):
    GENERATE_DATA = 1
    READ_DATA = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.GENERATE_DATA:
        generate_ust_data()

    elif unit_test == UnitTests.READ_DATA:
        df = load_ust_rates()
        print(df)


if __name__ == '__main__':

    unit_test = UnitTests.GENERATE_DATA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
