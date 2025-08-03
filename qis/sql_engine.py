"""
file utils to work with sql_engine
requires SQLAlchemy>=2.0.0
"""


import pandas as pd
import yaml
from sqlalchemy import create_engine, Engine
from pathlib import Path
from qis.file_utils import timer, INDEX_COLUMN
from enum import Enum
from typing import Dict, Union, NamedTuple, Optional, Literal, List


def get_engine(path: str = 'AWS_POSTGRES') -> Engine:
    full_file_path = Path(__file__).parent.joinpath('settings.yaml')
    with open(full_file_path) as settings:
        settings_data = yaml.load(settings, Loader=yaml.Loader)
    path = settings_data[path]
    engine = create_engine(path)
    return engine


@timer
def save_df_dict_to_sql(engine: Engine,
                        table_name: str,
                        dfs: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                        schema: Optional[str] = None,
                        index_col: Optional[str] = INDEX_COLUMN,
                        if_exists: Literal["fail", "replace", "append"] | Literal["truncate-append"] = "fail",
                        ) -> None:
    """
    save pandas dict to sql engine
    """
    for key, df in dfs.items():
        if df is not None and isinstance(df, pd.DataFrame):
            if index_col is not None:
                df = df.reset_index(names=index_col)
            if if_exists == "truncate-append":
                schema_str = f"{schema}." if schema else ""
                with engine.connect() as con:
                    statement = f"TRUNCATE TABLE {schema_str}{table_name}_{key}"
                    con.execute(statement)
                    con.commit()
                df.to_sql(
                    f"{table_name}_{key}",
                    engine,
                    schema=schema,
                    if_exists='append',
                    method='multi',
                    chunksize=1000
                )
            else:
                df.to_sql(f"{table_name}_{key}", engine, schema=schema, if_exists=if_exists)


@timer
def load_df_dict_from_sql(engine: Engine,
                          table_name: str,
                          dataset_keys: List[Union[str, Enum, NamedTuple]],
                          schema: Optional[str] = None,
                          index_col: Optional[str] = INDEX_COLUMN,
                          columns: Optional[List[str]] = None,
                          drop_sql_index: bool = True
                          ) -> Dict[str, pd.DataFrame]:
    """
    pandas dict from csv files
    """
    pandas_dict = {}
    for key in dataset_keys:
        # df will have index set by index_col with added column 'index' from sql
        df = pd.read_sql_table(table_name=f"{table_name}_{key}", con=engine, schema=schema,
                               index_col=index_col,
                               columns=columns)
        if drop_sql_index:
            df = df.drop('index', axis=1)
            #  df[index_col] = pd.to_datetime(df[index_col])
            #  df = df.set_index(index_col, drop=True)
        pandas_dict[key] = df
    return pandas_dict
