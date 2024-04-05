"""
path management for saving and reading from local drive
use settings.yaml to set local directories

Content of settings.yaml:
RESOURCE_PATH:
  'C:\\your_folder\\'
UNIVERSE_PATH:
  'C:\\your_folder\\'
OUTPUT_PATH:
  'C:\\your_folder\\'
# optional
POSTGRES:
  "postgresql://user:password@database:port"
"""
import os
import functools
import platform
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from typing import Dict, List, NamedTuple, Optional, Union, Literal
from matplotlib.backends.backend_pdf import PdfPages
from sqlalchemy.engine.base import Engine
from enum import Enum

from qis.local_path import get_paths


""""
Path specifications
"""
LOCAL_PATHS = get_paths()
RESOURCE_PATH = LOCAL_PATHS['RESOURCE_PATH']
UNIVERSE_PATH = LOCAL_PATHS['UNIVERSE_PATH']
OUTPUT_PATH = LOCAL_PATHS['OUTPUT_PATH']


DATE_FORMAT = '%Y%m%d_%H%M'
INDEX_COLUMN = 'timestamp'  # for Postresql


class FileData(NamedTuple):
    extension: str
    folder: Union[str, None]


class FileTypes(FileData, Enum):
    FIGURE = FileData(extension='.png', folder='figures')
    PNG = FileData(extension='.PNG', folder='figures')
    EPS = FileData(extension='.eps', folder='figures')
    SVG = FileData(extension='.svg', folder='figures')
    CSV = FileData(extension='.csv', folder='csv')
    FEATHER = FileData(extension='.feather', folder='feather')
    EXCEL = FileData(extension='.xlsx', folder='excel')
    PPTX = FileData(extension='.pptx', folder='pptx')
    WORD = FileData(extension='.docx', folder=None)
    PDF = FileData(extension='.pdf', folder=None)
    TXT = FileData(extension='.txt', folder='txt')
    PARQUET = FileData(extension='.parquet', folder='parquet')
    ZIP = FileData(extension='.zip', folder=None)


class PathData(NamedTuple):
    platform: str
    path: str


class ResourcePath(PathData, Enum):
    """
    specify window and lynox local paths to read resourse files
    """
    WINDOWS = PathData(platform='Windows', path=RESOURCE_PATH)
    LINUX = PathData(platform='Linux', path=RESOURCE_PATH)


class OutputPath(PathData, Enum):
    """
    specify window and lynox local paths for outputs
    """
    WINDOWS = PathData(platform='Windows', path=OUTPUT_PATH)
    LINUX = PathData(platform='Linux', path=OUTPUT_PATH)


def get_output_path() -> str:
    """
    platform dependent output path
    """
    run_platform = platform.system()
    path_data = next((item for item in OutputPath if item.value.platform == run_platform), None)
    if path_data is None:
        raise TypeError(f"unknown platform {run_platform}")
    return path_data.path


def get_resource_path() -> str:
    """
    platform dependent resourse path
    """
    run_platform = platform.system()
    path_data = next((item for item in ResourcePath if item.value.platform == run_platform), None)
    if path_data is None:
        raise TypeError(f"unknown platform {run_platform}")
    return path_data.path


def join_file_name_parts(parts: List[str]) -> str:
    """
    set standard for joining parts of file name
    """
    return '_'.join(parts)


def get_local_file_path(file_name: Optional[str],
                        file_type: Optional[FileTypes] = None,
                        local_path: Optional[str] = None,
                        folder_name: str = None,
                        subfolder_name: str = None,
                        key: str = None,
                        is_output_file: bool = False
                        ) -> str:
    """
    file data management is organised as:
    file_path = RESOURCE_PATH/folder_name/subfolder_name/file_name+file_type.value
    default value without optional arguments will be:
    file_path = RESOURCE_PATH/file_name.file_type.value

    for datasets, we can define datasets keys so the file paths are:
    file_path = RESOURCE_PATH/folder_name/subfolder_name/file_name+_key+file_type.value
    or if file_name is None:
    file_path = RESOURCE_PATH/folder_name/subfolder_name/key+file_type.value

    if local_path is not None: file_path=local_path
    if local_path in not None and file_name and file_type is passed: file_path=local_path//file_name+file_type.value
    if local_path in not None and file_name is None and key is not None and file_type is passed: file_path=local_path//key+file_type.value
    if local_path in not None and file_name and key and file_type is passed: file_path=local_path//file_name_key+file_type.value
    """
    if local_path is None:

        if is_output_file:
            local_path = OUTPUT_PATH
        else:
            local_path = RESOURCE_PATH

        if folder_name is not None:
            if subfolder_name is not None:
                local_path = join(local_path, folder_name, subfolder_name)
            else:
                local_path = join(local_path, folder_name)

    if file_name is not None:
        if key is not None:
            if file_name is None:
                file_name = key
            else:
                file_name = join_file_name_parts([file_name, key])
        if file_type is not None:  # by convention no file folder
            file_name = file_name + file_type.extension
        file_path = join(local_path, file_name)

    elif file_name is None and key is not None:
        file_name = key
        if file_type is not None:
            file_name = file_name + file_type.extension
        file_path = join(local_path, file_name)

    else:
        file_path = local_path

    return file_path


def timer(func):
    """
    Print the runtime of the decorated function
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        if run_time < 60.0:
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        else:
            minuts = np.floor(run_time/60.0)
            secs = run_time - 60.0*minuts
            print(f"Finished {func.__name__!r} in {minuts:.0f}m {secs:.0f}secs")
        return value
    return wrapper_timer


"""
Pandas to/from Excel core
"""


def delocalize_df(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.tz_localize(None)
    return data


def save_df_to_excel(data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
                     file_name: str,
                     local_path: Optional[str] = None,
                     folder_name: str = None,
                     subfolder_name: str = None,
                     key: str = None,
                     add_current_date: bool = False,
                     sheet_names: List[str] = None,
                     transpose: bool = False,
                     ) -> str:
    """
    pandas or list of pandas to one excel
    """
    if add_current_date:
        file_name = f"{file_name}_{pd.Timestamp.now().strftime(DATE_FORMAT)}"

    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)

    excel_writer = pd.ExcelWriter(file_path)
    if isinstance(data, list):  # publish with sheet names
        if sheet_names is None:
            sheet_names = [f"Sheet {n+1}" for n, _ in enumerate(data)]
        for df, name in zip(data, sheet_names):
            df = delocalize_df(df)
            if transpose:
                df = df.T
            df.to_excel(excel_writer=excel_writer, sheet_name=name)
    elif isinstance(data, dict):  # publish with sheet names
        for key, df in data.items():
            df = delocalize_df(df)
            if transpose:
                df = df.T
            df.to_excel(excel_writer=excel_writer, sheet_name=key)
    else:
        if transpose:
            data = data.T
        data = delocalize_df(data)
        data.to_excel(excel_writer=excel_writer)

    excel_writer.close()

    return file_path


def load_df_from_excel(file_name: str,
                       sheet_name: str = 'Sheet1',
                       local_path: Optional[str] = None,
                       folder_name: str = None,
                       subfolder_name: str = None,
                       key: str = None,
                       is_index: bool = True,
                       delocalize: bool = False  # excel data may have local time which are unwanted
                       ) -> pd.DataFrame:
    """
    one file, one sheet to pandas
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    if os.path.isfile(file_path):
        excel_reader = pd.ExcelFile(file_path, engine='openpyxl')
    else:
        raise FileNotFoundError(f"file data {file_path} nor found")

    index_col = 0 if is_index else None
    df = excel_reader.parse(sheet_name=sheet_name, index_col=index_col)

    if is_index and delocalize:
        df = delocalize_df(df)

    return df


def save_df_dict_to_excel(datasets: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                          file_name: str,
                          local_path: Optional[str] = None,
                          folder_name: str = None,
                          subfolder_name: str = None,
                          key: str = None,
                          add_current_date: bool = False,
                          delocalize: bool = False
                          ) -> str:
    """
    dictionary of pandas to same Excel
    """
    if add_current_date:
        file_name = f"{file_name}_{pd.Timestamp.now().strftime(DATE_FORMAT)}"

    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)

    excel_writer = pd.ExcelWriter(file_path)
    for key, data in datasets.items():
        if delocalize:
            data = delocalize_df(data)
        data.to_excel(excel_writer=excel_writer, sheet_name=key)
    excel_writer.close()
    return file_path


def load_df_dict_from_excel(file_name: str,
                            dataset_keys: Optional[List[Union[str, Enum, NamedTuple]]],
                            local_path: Optional[str] = None,
                            folder_name: str = None,
                            subfolder_name: str = None,
                            key: str = None,
                            is_index: bool = True,
                            delocalize: bool = False,
                            tz: str = None
                            ) -> Dict[str, pd.DataFrame]:
    """
    loag Excel sheets to pandas dict
    dataset_keys = None, it will read all sheets
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)

    if os.path.isfile(file_path):
        excel_reader = pd.ExcelFile(file_path, engine='openpyxl')
    else:
        raise FileNotFoundError(f"file data {file_path} not found")
    if dataset_keys is None:
        dataset_keys = excel_reader.sheet_names
    index_col = 0 if is_index else None
    pandas_dict = {}
    for key in dataset_keys:
        try:
            df = excel_reader.parse(sheet_name=key, index_col=index_col)
        except:
            raise TypeError(f"sheet_name data {key} nor found")
        if delocalize:
            df = delocalize_df(df)
        if tz is not None:
            df.index = df.index.tz_localize(tz)
        pandas_dict[key] = df

    return pandas_dict


"""
Pandas to/from CSV core
"""


def save_df_to_csv(df: pd.DataFrame,
                   file_name: str = None,
                   folder_name: str = None,
                   subfolder_name: str = None,
                   key: str = None,
                   add_current_date: bool = False,
                   local_path: Optional[str] = None
                   ) -> None:
    """
    pandas to csv
    """
    if add_current_date:
        file_name = f"{file_name}_{pd.Timestamp.now().strftime(DATE_FORMAT)}"

    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.CSV,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    df.to_csv(path_or_buf=file_path)


def load_df_from_csv(file_name: Optional[str] = None,
                     local_path: Optional[str] = None,
                     folder_name: str = None,
                     subfolder_name: str = None,
                     key: str = None,
                     is_index: bool = True,
                     parse_dates: bool = True,
                     dayfirst: Optional[bool] = None,
                     tz: str = None,
                     drop_duplicated: bool = False
                     ) -> pd.DataFrame:
    """
    pandas from csv
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.CSV,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)

    if is_index:
        index_col = 0
        parse_dates = parse_dates
    else:
        index_col = None
        parse_dates = None

    if os.path.isfile(file_path):
        try:
            df = pd.read_csv(filepath_or_buffer=file_path,
                             index_col=index_col,
                             parse_dates=parse_dates,
                             dayfirst=dayfirst)

        except UnicodeDecodeError:  # try without index
            df = pd.read_csv(filepath_or_buffer=file_path)
    else:
        raise FileNotFoundError(f"not found {file_name} with file_path={file_path}")

    if drop_duplicated:
        df = df.loc[~df.index.duplicated(keep='first')]

    if not df.empty:
        if tz is not None:
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert(tz=tz)
            else:
                df.index = df.index.tz_localize(tz=tz)

    return df


def append_df_to_csv(df: pd.DataFrame,
                     file_name: str = None,
                     folder_name: str = None,
                     subfolder_name: str = None,
                     key: str = None,
                     local_path: Optional[str] = None,
                     keep: Optional[Literal['first', 'last']] = None
                     ) -> None:
    """
    append csv file
    """
    # check if file exist
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.CSV,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    if os.path.isfile(file_path):  # append using format of old file
        old_df = load_df_from_csv(file_name=file_name,
                                  local_path=local_path,
                                  folder_name=folder_name,
                                  subfolder_name=subfolder_name,
                                  key=key)
        df = pd.concat([old_df, df], axis=0)
        if keep is not None:
            df = df.loc[~df.index.duplicated(keep=keep)]

    else:
        pass

    save_df_to_csv(df=df,
                   file_name=file_name,
                   folder_name=folder_name,
                   subfolder_name=subfolder_name,
                   key=key,
                   local_path=local_path)


def save_df_dict_to_csv(datasets: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                        file_name: Optional[str] = None,
                        local_path: Optional[str] = None,
                        folder_name: str = None,
                        subfolder_name: str = None,
                        add_current_date: bool = False
                        ) -> None:
    """
    pandas dict to csv files
    """
    if add_current_date:
        file_name = f"{file_name}_{pd.Timestamp.now().strftime(DATE_FORMAT)}"

    for key, data in datasets.items():
        if data is not None:
            file_path = get_local_file_path(file_name=file_name,
                                            file_type=FileTypes.CSV,
                                            local_path=local_path,
                                            folder_name=folder_name,
                                            subfolder_name=subfolder_name,
                                            key=key)
            data.to_csv(path_or_buf=file_path)


def load_df_dict_from_csv(dataset_keys: List[Union[str, Enum, NamedTuple]],
                          file_name: Optional[str],
                          local_path: Optional[str] = None,
                          folder_name: str = None,
                          subfolder_name: str = None,
                          is_index: bool = True,
                          dayfirst: Optional[bool] = None,  # will give priority to formats where day come first
                          force_not_found_error: bool = False,
                          ) -> Dict[str, pd.DataFrame]:
    """
    pandas dict from csv files
    """
    index_col = 0 if is_index else None
    pandas_dict = {}
    for key in dataset_keys:
        file_path = get_local_file_path(file_name=file_name,
                                        file_type=FileTypes.CSV,
                                        local_path=local_path,
                                        folder_name=folder_name,
                                        subfolder_name=subfolder_name,
                                        key=key)
        if os.path.isfile(file_path):
            data = pd.read_csv(filepath_or_buffer=file_path,
                               index_col=index_col,
                               parse_dates=True,
                               dayfirst=dayfirst)
            pandas_dict[key] = data
        else:
            message = f"file data {file_path}, {key} not found"
            if force_not_found_error:
                raise FileNotFoundError(message)
            else:
                print(message)

    return pandas_dict


#############################################################
#  Pandas to/from feather
#############################################################

@timer
def save_df_dict_to_sql(engine: Engine,
                        table_name: str,
                        dfs: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                        schema: Optional[str] = None,
                        index_col: Optional[str] = INDEX_COLUMN
                        ) -> None:
    """
    save pandas dict to sql engine
    """
    for key, df in dfs.items():
        if df is not None:
            if index_col is not None:
                df = df.reset_index(names=index_col)
            df.to_sql(f"{table_name}_{key}", engine, schema=schema, if_exists='replace')


@timer
def load_df_dict_from_sql(engine: Engine,
                          table_name: str,
                          dataset_keys: List[Union[str, Enum, NamedTuple]],
                          schema: Optional[str] = None,
                          index_col: Optional[str] = INDEX_COLUMN,
                          columns: Optional[List[str]] = None
                          ) -> Dict[str, pd.DataFrame]:
    """
    pandas dict from csv files
    """
    pandas_dict = {}
    for key in dataset_keys:
        df = pd.read_sql_table(table_name=f"{table_name}_{key}", con=engine, schema=schema,
                               index_col=index_col,
                               columns=columns)
        if index_col is not None and index_col in df.columns:
            df[index_col] = pd.to_datetime(df[index_col])
            df = df.set_index(index_col)
        pandas_dict[key] = df
    return pandas_dict


#############################################################
#  Dataframe to/from feather
#############################################################

def save_df_to_feather(df: pd.DataFrame,
                       file_name: Optional[str] = None,
                       local_path: Optional[str] = None,
                       folder_name: str = None,
                       subfolder_name: str = None,
                       key: str = None,
                       index_col: Optional[str] = INDEX_COLUMN
                       ) -> None:
    """
    save df to feather files
    index_col stands for the index: needs to be reset when saving and put back when loading
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.FEATHER,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    if index_col is not None and index_col not in df.columns:  # index is unique and preserved
        df = df.reset_index(names=index_col)
    else:  # drop to avoid entering extra field when loading back
        df = df.reset_index(drop=True)
    df.to_feather(path=file_path)


def append_df_to_feather(df: pd.DataFrame,
                         file_name: str = None,
                         folder_name: str = None,
                         subfolder_name: str = None,
                         key: str = None,
                         local_path: Optional[str] = None,
                         keep: Optional[Literal['first', 'last']] = None,
                         index_col: Optional[str] = INDEX_COLUMN
                         ) -> pd.DataFrame:
    """
    append csv file
    """
    # check if file exist
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.FEATHER,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    if os.path.isfile(file_path):  # append using format of old file
        old_df = load_df_from_feather(file_name=file_name,
                                      local_path=local_path,
                                      folder_name=folder_name,
                                      subfolder_name=subfolder_name,
                                      key=key,
                                      index_col=index_col)
        df = pd.concat([old_df, df], axis=0)
        if keep is not None:
            df = df.loc[~df.index.duplicated(keep=keep)]

    else:
        pass

    save_df_to_feather(df=df,
                       file_name=file_name,
                       folder_name=folder_name,
                       subfolder_name=subfolder_name,
                       key=key,
                       local_path=local_path,
                       index_col=index_col)
    return df


def load_df_from_feather(file_name: Optional[str] = None,
                         local_path: Optional[str] = None,
                         folder_name: str = None,
                         subfolder_name: str = None,
                         key: str = None,
                         index_col: Optional[str] = INDEX_COLUMN
                         ) -> pd.DataFrame:
    """
    load dfs from feather files
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.FEATHER,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    if os.path.isfile(file_path):  # append using format of old file
        df = pd.read_feather(file_path)
    else:
        raise FileNotFoundError(f"not found {file_name} with file_path={file_path}")

    if index_col is not None and index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col)
    return df


@timer
def save_df_dict_to_feather(dfs: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                            file_name: Optional[str] = None,
                            local_path: Optional[str] = None,
                            folder_name: str = None,
                            subfolder_name: str = None,
                            index_col: Optional[str] = INDEX_COLUMN
                            ) -> None:
    """
    pandas dict to csv files
    """
    for key, df in dfs.items():
        if df is not None:
            file_path = get_local_file_path(file_name=file_name,
                                            file_type=FileTypes.FEATHER,
                                            local_path=local_path,
                                            folder_name=folder_name,
                                            subfolder_name=subfolder_name,
                                            key=key)
            if index_col not in df.columns:
                df = df.reset_index(names=index_col)
            df.to_feather(path=file_path)


def load_df_dict_from_feather(dataset_keys: List[Union[str, Enum, NamedTuple]],
                              file_name: Optional[str],
                              local_path: Optional[str] = None,
                              folder_name: str = None,
                              subfolder_name: str = None,
                              force_not_found_error: bool = False,
                              index_col: Optional[str] = INDEX_COLUMN
                              ) -> Dict[str, pd.DataFrame]:
    """
    pandas dict from csv files
    """
    pandas_dict = {}
    for key in dataset_keys:
        file_path = get_local_file_path(file_name=file_name,
                                        file_type=FileTypes.FEATHER,
                                        local_path=local_path,
                                        folder_name=folder_name,
                                        subfolder_name=subfolder_name,
                                        key=key)
        if os.path.isfile(file_path):
            df = pd.read_feather(file_path)
            if index_col is not None and index_col in df.columns:
                df.index = pd.to_datetime(df[index_col])
                df = df.set_index(index_col)
            pandas_dict[key] = df
        else:
            message = f"file data {file_path}, {key} not found"
            if force_not_found_error:
                raise FileNotFoundError(message)
            else:
                print(message)

    return pandas_dict


"""
Pandas to/from parquet core
"""


def save_df_to_parquet(df: pd.DataFrame,
                       file_name: str,
                       folder_name: Optional[str] = None,
                       subfolder_name: str = None,
                       key: Optional[str] = None,
                       local_path: Optional[str] = None,
                       delocalize: bool = False
                       ) -> None:
    """
    pandas to parquet
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.PARQUET,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    if delocalize:
        df = delocalize_df(df)
    df.to_parquet(path=file_path)


def load_df_from_parquet(file_name: Optional[str],
                         folder_name: str = None,
                         subfolder_name: str = None,
                         key: Optional[str] = None,
                         local_path: Optional[str] = None,
                         delocalize: bool = False
                         ) -> pd.DataFrame:
    """
    pandas from parquet
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.PARQUET,
                                    local_path=local_path,
                                    folder_name=folder_name,
                                    subfolder_name=subfolder_name,
                                    key=key)
    if os.path.isfile(file_path):
        df = pd.read_parquet(path=file_path)
    else:
        raise FileNotFoundError(f"not found {file_name} with path={file_path}")
    if delocalize:
        df = delocalize_df(df)

    return df


def save_df_dict_to_parquet(datasets: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                            file_name: Optional[str] = None,
                            folder_name: str = None,
                            subfolder_name: str = None,
                            local_path: Optional[str] = None,
                            delocalize: bool = False
                            ) -> None:
    """
    pandas dict to parquet files
    """
    for key, data in datasets.items():
        if data is not None:
            file_path = get_local_file_path(file_name=file_name,
                                            file_type=FileTypes.PARQUET,
                                            local_path=local_path,
                                            folder_name=folder_name,
                                            subfolder_name=subfolder_name,
                                            key=key)
            if delocalize:
                data = delocalize_df(data)
            data.to_parquet(path=file_path)


def load_df_dict_from_parquet(dataset_keys: List[Union[str, Enum, NamedTuple]],
                              file_name: Optional[str],
                              folder_name: str = None,
                              subfolder_name: str = None,
                              local_path: Optional[str] = None,
                              force_not_found_error: bool = False
                              ) -> Dict[str, pd.DataFrame]:
    """
    pandas dict from parquet files
    """
    pandas_dict = {}
    for key in dataset_keys:
        file_path = get_local_file_path(file_name=file_name,
                                        file_type=FileTypes.PARQUET,
                                        local_path=local_path,
                                        folder_name=folder_name,
                                        subfolder_name=subfolder_name,
                                        key=key)
        if os.path.isfile(file_path):
            pandas_dict[key] = pd.read_parquet(path=file_path)
        else:
            message = f"file data {file_name}, {key} not found"
            if force_not_found_error:
                raise FileNotFoundError(message)
            else:
                print(message)

    return pandas_dict


"""
For pdfs
"""


def get_pdf_path(file_name: str,
                 local_path: Union[None, str] = None,
                 add_current_date: bool = True
                 ) -> str:

    if add_current_date:
        file_name = join_file_name_parts([file_name, pd.Timestamp.now().strftime(DATE_FORMAT)])

    file_path = get_local_file_path(file_name=file_name, file_type=FileTypes.PDF, local_path=local_path, is_output_file=True)

    return file_path


def get_all_folder_files(folder_path: str):
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    print(files)



"""
Figure files
"""


def save_fig(fig: plt.Figure,
             file_name: str,
             local_path: Optional[str] = None,
             dpi: int = 300,
             file_type=FileTypes.PNG,
             add_current_date: bool = False,
             **kwargs
             ) -> str:
    """
    save matplotlib figure
    """
    if add_current_date:
        file_name = join_file_name_parts([file_name, pd.Timestamp.now().strftime(DATE_FORMAT)])
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=file_type,
                                    local_path=local_path,
                                    is_output_file=True)
    if file_type == FileTypes.PNG:
        fig.savefig(file_path, dpi=dpi)
    elif file_type == FileTypes.EPS:
        fig.savefig(file_path, dpi=dpi, format='eps')
    elif file_type == FileTypes.SVG:
        fig.savefig(file_path, dpi=dpi, format='svg')
    elif file_type == FileTypes.PDF:
        fig.savefig(file_path, dpi=dpi, format='pdf')
    else:
        raise NotImplementedError(f"{file_type}")
    return file_path


def save_figs(figs: Dict[str, plt.Figure],
              file_name: str = '',
              local_path: Optional[str] = None,
              dpi: int = 300,
              file_type=FileTypes.PNG,
              add_current_date: bool = False,
              **kwargs
              ) -> None:
    """
    save matplotlib figures dict
    """
    for key, fig in figs.items():
        file_path = save_fig(fig=fig,
                             file_name=f"{file_name}_{key}",
                             local_path=local_path,
                             dpi=dpi,
                             file_type=file_type,
                             add_current_date=add_current_date,
                             **kwargs)
        print(file_path)


def save_figs_to_pdf(figs: Union[List[plt.Figure], Dict[str, plt.Figure]],
                     file_name: str,
                     orientation: Literal['portrait', 'landscape'] = 'portrait',
                     local_path: Optional[str] = None,
                     add_current_date: bool = True
                     ) -> str:
    """
    create PDF of list of plf figures
    """
    if add_current_date:
        file_name = join_file_name_parts([file_name, pd.Timestamp.now().strftime(DATE_FORMAT)])

    file_path = get_local_file_path(file_name=file_name, file_type=FileTypes.PDF, is_output_file=True, local_path=local_path)

    with PdfPages(file_path) as pdf:
        if isinstance(figs, Dict):
            for _, fig in figs.items():
                if fig is not None:
                    pdf.savefig(fig, orientation=orientation)
        else:
            for fig in figs:
                if fig is not None:
                    pdf.savefig(fig, orientation=orientation)

    print(f"created PDF doc: {file_path}")
    return file_path


class UnitTests(Enum):
    LOCAL_PATHS = 1
    FOLDER_FILES = 2
    NAMES = 3
    DATA_FILE = 4
    UNIVERSE = 5


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.LOCAL_PATHS:
        print(get_paths())
        print(platform.system())
        print(OUTPUT_PATH)

    elif unit_test == UnitTests.FOLDER_FILES:
        get_all_folder_files(folder_path="C://")

    elif unit_test == UnitTests.NAMES:
        file_name = join_file_name_parts(['head', 'tails'])
        print(file_name)

    elif unit_test == UnitTests.DATA_FILE:
        file_path = get_local_file_path(file_name='ETH', file_type=FileTypes.CSV)
        print(file_path)

    elif unit_test == UnitTests.UNIVERSE:
        file_path = get_local_file_path(file_name='ETH',
                                        folder_name='bitmex',
                                        key='1d',
                                        file_type=FileTypes.CSV)
        print(file_path)


if __name__ == '__main__':

    unit_test = UnitTests.UNIVERSE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
