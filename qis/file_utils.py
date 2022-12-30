"""
path management for saving and reading from local drive
use local_paths.yaml to set local directories

Content of local_paths.yaml:
RESOURCE_PATH:
  'C:/your_folder'
UNIVERSE_PATH:
  'C:/your_folder'
OUTPUT_PATH:
  'C:/your_folder'
"""
import datetime
import platform
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir
from os.path import isfile, join
from typing import Dict, List, NamedTuple, Optional, Union
from matplotlib.backends.backend_pdf import PdfPages
from enum import Enum


DATE_FORMAT = '%Y%m%d_%H%M'


def get_local_paths() -> Dict[str, str]:
    """
    read path specs in local_paths.yaml
    """
    full_file_path = Path(__file__).parent.joinpath('local_paths.yaml')
    with open(full_file_path) as settings:
        settings_data = yaml.load(settings, Loader=yaml.Loader)
    return settings_data


LOCAL_PATHS = get_local_paths()
RESOURCE_PATH = LOCAL_PATHS['RESOURCE_PATH']
UNIVERSE_PATH = LOCAL_PATHS['UNIVERSE_PATH']
OUTPUT_PATH = LOCAL_PATHS['OUTPUT_PATH']
LOCAL_RESOURCE_PATH = LOCAL_PATHS['LOCAL_RESOURCE_PATH']


""""
Path specifications
"""


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
    PARAMS = FileData(extension='.xlsx', folder='data\\params')
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


def get_resource_file_path(file_name: Optional[str],
                           subfolder_name: Optional[str] = None,
                           subsubfolder_name: Optional[str] = None,
                           key: Optional[str] = None,
                           local_path: str = None,
                           file_type: Optional[FileTypes] = FileTypes.CSV
                           ) -> str:
    """
    file data management is organised as:
    RESOURCE_PATH/subfolder_name/subsubfolder_name/file_name+key.file_type.value
    default value without optional arguments will be:
    RESOURCE_PATH/file_name.file_type.value
    """

    if local_path is None:
        if file_name is None and key is None and subfolder_name is None and subsubfolder_name is None:
            raise ValueError(f"file_name or key or subfolder_name must be given")

        local_path = RESOURCE_PATH

        if key is not None:
            if file_name is None:
                file_name = key
            else:
                file_name = join_file_name_parts([file_name, key])
        else:
            if file_name is None:  # will return directory path
                file_name = ''

        if subfolder_name is not None:
            if subsubfolder_name is not None:
                local_path = join(local_path, subfolder_name, subsubfolder_name)
            else:
                local_path = join(local_path, subfolder_name)

    if file_name is not None:
        local_path = join(local_path, file_name)

    if file_type is not None:  # by convention no file folder
        file_path = local_path + file_type.extension
    else:
        file_path = local_path

    return file_path


def get_output_file_path(file_name: Optional[str],
                         file_type: Optional[FileTypes] = None,
                         local_path: Optional[str] = None,
                         subfolder_name: Optional[str] = None
                         ) -> str:

    if local_path is not None:
        path_data = local_path
    else:
        path_data = OUTPUT_PATH
        if file_type is not None and file_type.folder is not None:  # append folder
            path_data = join(path_data, file_type.folder)

    if subfolder_name is not None:
        path_data = join(path_data, subfolder_name)

    if file_name is not None:
        if file_type is not None:
            file_path = join(path_data, file_name + file_type.extension)
        else:
            file_path = join(path_data, file_name)
    else:
        file_path = path_data

    return file_path


def get_local_file_path(file_name: Optional[str],
                        file_type: Optional[FileTypes] = FileTypes.CSV,
                        local_path: Optional[str] = None,
                        subfolder_name: str = 'data',
                        subsubfolder_name: str = None,
                        key: str = None,
                        is_output_file: bool = False
                        ) -> str:
    """
    default path is resource_file_path/data/
    """
    if is_output_file:
        file_path = get_output_file_path(file_name=file_name, file_type=file_type, subfolder_name=subfolder_name,
                                         local_path=local_path)
    else:
        file_path = get_resource_file_path(file_name=file_name,
                                           subfolder_name=subfolder_name,
                                           subsubfolder_name=subsubfolder_name,
                                           key=key,
                                           file_type=file_type,
                                           local_path=local_path)
    return file_path


def get_param_file_path(file_name: Optional[str] = None,
                        file_type: FileTypes = FileTypes.PARAMS
                        ) -> str:
    path_data = RESOURCE_PATH
    if file_name is not None:
        file_path = join(path_data, file_type.folder, file_name + file_type.extension)
    else:
        file_path = join(path_data, file_type.folder)
    return file_path





"""
Pandas to/from Excel core
"""


def delocalize_df(data: pd.DataFrame, is_delocalize: bool) -> pd.DataFrame:
    if is_delocalize and isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.tz_localize(None)
    return data


def save_df_to_excel(data: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
                     file_name: str,
                     local_path: Optional[str] = None,
                     subfolder_name: str = None,
                     subsubfolder_name: str = None,
                     key: str = None,
                     is_add_current_date: bool = True,
                     sheet_names: List[str] = None,
                     is_transpose: bool = False,
                     is_output_file: bool = False
                     ) -> str:
    """
    pandas or list of pandas to one excel
    """
    if is_output_file and is_add_current_date:
        file_name = f"{file_name}_{datetime.datetime.now().strftime(DATE_FORMAT)}"

    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key,
                                    is_output_file=is_output_file)

    excel_writer = pd.ExcelWriter(file_path)
    if isinstance(data, list):  # publish with sheet names
        if sheet_names is None:
            sheet_names = [f"Sheet {n+1}" for n, _ in enumerate(data)]
        for df, name in zip(data, sheet_names):
            df = delocalize_df(df, is_delocalize=True)
            if is_transpose:
                df = df.T
            df.to_excel(excel_writer=excel_writer, sheet_name=name)
    elif isinstance(data, dict):  # publish with sheet names
        for key, df in data.items():
            df = delocalize_df(df, is_delocalize=True)
            if is_transpose:
                df = df.T
            df.to_excel(excel_writer=excel_writer, sheet_name=key)
    else:
        if is_transpose:
            data = data.T
        data = delocalize_df(data, is_delocalize=True)
        data.to_excel(excel_writer=excel_writer)

    excel_writer.close()

    return file_path


def load_df_from_excel(file_name: str,
                       sheet_name: str = 'Sheet1',
                       local_path: Optional[str] = None,
                       subfolder_name: str = None,
                       subsubfolder_name: str = None,
                       key: str = None,
                       is_index: bool = True,
                       is_delocalize: bool = False  # excel data may have local time which are unwanted
                       ) -> pd.DataFrame:
    """
    one file, one sheet to pandas
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key)
    try:
        excel_reader = pd.ExcelFile(file_path, engine='openpyxl')
    except FileNotFoundError:
        raise TypeError(f"file data {file_path} nor found")

    index_col = 0 if is_index else None
    df = excel_reader.parse(sheet_name=sheet_name, index_col=index_col)

    if is_index:
        df = delocalize_df(df, is_delocalize=is_delocalize)

    return df


def save_df_dict_to_excel(datasets: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                          file_name: str,
                          local_path: Optional[str] = None,
                          subfolder_name: str = None,
                          subsubfolder_name: str = None,
                          key: str = None,
                          is_add_current_date: bool = False,
                          is_output_file: bool = False,
                          is_delocalize: bool = False
                          ) -> str:
    """
    dictionary of pandas to same Excel
    """
    if is_output_file and is_add_current_date:
        file_name = f"{file_name}_{datetime.datetime.now().strftime(DATE_FORMAT)}"

    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key,
                                    is_output_file=is_output_file)

    excel_writer = pd.ExcelWriter(file_path)
    for key, data in datasets.items():
        data = delocalize_df(data, is_delocalize=is_delocalize)
        data.to_excel(excel_writer=excel_writer, sheet_name=key)
    excel_writer.close()
    return file_path


def load_df_dict_from_excel(file_name: str,
                            dataset_keys: Optional[List[Union[str, Enum, NamedTuple]]],
                            local_path: Optional[str] = None,
                            subfolder_name: str = None,
                            subsubfolder_name: str = None,
                            key: str = None,
                            is_index: bool = True,
                            is_delocalize: bool = False,
                            tz: str = None
                            ) -> Dict[str, pd.DataFrame]:
    """
    loag Excel sheets to pandas dict
    dataset_keys = None, it will read all sheets
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.EXCEL,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key)
    try:
        excel_reader = pd.ExcelFile(file_path, engine='openpyxl')
    except FileNotFoundError:
        raise TypeError(f"file data {file_path} not found")
    if dataset_keys is None:
        dataset_keys = excel_reader.sheet_names
    index_col = 0 if is_index else None
    pandas_dict = {}
    for key in dataset_keys:
        try:
            df = excel_reader.parse(sheet_name=key, index_col=index_col)
        except:
            raise TypeError(f"sheet_name data {key} nor found")
        df = delocalize_df(df, is_delocalize=is_delocalize)
        if tz is not None:
            df.index = df.index.tz_localize(tz)
        pandas_dict[key] = df

    return pandas_dict


"""
Pandas to/from CSV core
"""


def save_df_to_csv(df: pd.DataFrame,
                   file_name: str = None,
                   subfolder_name: str = None,
                   subsubfolder_name: str = None,
                   key: str = None,
                   local_path: Optional[str] = None,
                   is_delocalize: bool = False,
                   is_output_file: bool = False
                   ) -> None:
    """
    pandas to csv
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.CSV,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key,
                                    is_output_file=is_output_file)
    df = delocalize_df(df, is_delocalize=is_delocalize)
    df.to_csv(path_or_buf=file_path)


def load_df_from_csv(file_name: Optional[str] = None,
                     local_path: Optional[str] = None,
                     subfolder_name: str = None,
                     subsubfolder_name: str = None,
                     key: str = None,
                     is_index: bool = True,
                     parse_dates: bool = True,
                     is_delocalize: bool = False,  # excel data may have local time which are unwanted
                     dayfirst: Optional[bool] = None,
                     tz: str = None,
                     is_remove_dubpicated: bool = False
                     ) -> pd.DataFrame:
    """
    pandas from csv
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.CSV,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key)

    if is_index:
        index_col = 0
        parse_dates = parse_dates
    else:
        index_col = None
        parse_dates = None

    try:
        df = pd.read_csv(filepath_or_buffer=file_path,
                         index_col=index_col,
                         parse_dates=parse_dates,
                         dayfirst=dayfirst)
    except:
        raise FileNotFoundError(f"not found {file_name} with file_path={file_path}")

    if is_remove_dubpicated:
        df = df.loc[~df.index.duplicated(keep='first')]

    if not df.empty:
        if tz is not None:
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert(tz=tz)
            else:
                df.index = df.index.tz_localize(tz=tz)
        elif is_delocalize:
            df = delocalize_df(df, is_delocalize=is_delocalize)

    return df


def save_df_dict_to_csv(datasets: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                        file_name: Optional[str] = None,
                        local_path: Optional[str] = None,
                        subfolder_name: str = None,
                        subsubfolder_name: str = None,
                        is_delocalize: bool = False,
                        is_output_file: bool = False
                        ) -> None:
    """
    pandas dict to csv files
    """
    for key, data in datasets.items():
        if data is not None:
            file_path = get_local_file_path(file_name=file_name,
                                            file_type=FileTypes.CSV,
                                            local_path=local_path,
                                            subfolder_name=subfolder_name,
                                            subsubfolder_name=subsubfolder_name,
                                            key=key,
                                            is_output_file=is_output_file)
            data = delocalize_df(data, is_delocalize=is_delocalize)
            data.to_csv(path_or_buf=file_path)


def load_df_dict_from_csv(dataset_keys: List[Union[str, Enum, NamedTuple]],
                          file_name: Optional[str],
                          local_path: Optional[str] = None,
                          subfolder_name: str = None,
                          subsubfolder_name: str = None,
                          is_index: bool = True,
                          is_force_not_found_error: bool = False,
                          is_delocalize: bool = False
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
                                        subfolder_name=subfolder_name,
                                        subsubfolder_name=subsubfolder_name,
                                        key=key)
        try:
            data = pd.read_csv(filepath_or_buffer=file_path,
                               index_col=index_col,
                               parse_dates=True)
            data = delocalize_df(data, is_delocalize=is_delocalize)
            pandas_dict[key] = data
        except:
            message = f"file data {file_path}, {key} not found"
            if is_force_not_found_error:
                raise FileNotFoundError(message)
            else:
                print(message)

    return pandas_dict


"""
Pandas to/from feather
"""


def save_df_to_feather(df: pd.DataFrame,
                       file_name: Optional[str] = None,
                       local_path: Optional[str] = None,
                       subfolder_name: str = None,
                       subsubfolder_name: str = None,
                       is_output_file: bool = False
                       ) -> None:
    """
    pandas dict to csv files
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.FEATHER,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    is_output_file=is_output_file)
    df = df.reset_index(names='index')
    df.to_feather(path=file_path)


def load_df_from_feather(file_name: Optional[str] = None,
                         local_path: Optional[str] = None,
                         subfolder_name: str = None,
                         subsubfolder_name: str = None,
                         key: str = None
                         ) -> pd.DataFrame:
    """
    pandas from csv
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.FEATHER,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key)
    try:
        df = pd.read_feather(file_path)
    except:
        raise FileNotFoundError(f"not found {file_name} with file_path={file_path}")

    df.index = pd.to_datetime(df['index'])
    df = df.set_index('index')

    return df


def save_df_dict_to_feather(datasets: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                            file_name: Optional[str] = None,
                            local_path: Optional[str] = None,
                            subfolder_name: str = None,
                            subsubfolder_name: str = None,
                            is_output_file: bool = False
                            ) -> None:
    """
    pandas dict to csv files
    """
    for key, df in datasets.items():
        if df is not None:
            file_path = get_local_file_path(file_name=file_name,
                                            file_type=FileTypes.FEATHER,
                                            local_path=local_path,
                                            subfolder_name=subfolder_name,
                                            subsubfolder_name=subsubfolder_name,
                                            key=key,
                                            is_output_file=is_output_file)
            df = df.reset_index(names='index')
            df.to_feather(path=file_path)


def load_df_dict_from_feather(dataset_keys: List[Union[str, Enum, NamedTuple]],
                              file_name: Optional[str],
                              local_path: Optional[str] = None,
                              subfolder_name: str = None,
                              subsubfolder_name: str = None,
                              is_index: bool = True,
                              is_force_not_found_error: bool = False,
                              ) -> Dict[str, pd.DataFrame]:
    """
    pandas dict from csv files
    """
    pandas_dict = {}
    for key in dataset_keys:
        file_path = get_local_file_path(file_name=file_name,
                                        file_type=FileTypes.FEATHER,
                                        local_path=local_path,
                                        subfolder_name=subfolder_name,
                                        subsubfolder_name=subsubfolder_name,
                                        key=key)
        try:
            data = pd.read_feather(file_path)
            if is_index:
                data.index = pd.to_datetime(data['index'])
                data = data.set_index('index')
            pandas_dict[key] = data
        except:
            message = f"file data {file_path}, {key} not found"
            if is_force_not_found_error:
                raise FileNotFoundError(message)
            else:
                print(message)

    return pandas_dict


"""
Pandas to/from parquet core
"""


def save_df_to_parquet(df: pd.DataFrame,
                       file_name: str,
                       subfolder_name: Optional[str] = None,
                       subsubfolder_name: str = None,
                       key: Optional[str] = None,
                       local_path: Optional[str] = None,
                       is_output_file: bool = False,
                       is_delocalize: bool = False
                       ) -> None:
    """
    pandas to parquet
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.PARQUET,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key,
                                    is_output_file=is_output_file)
    df = delocalize_df(df, is_delocalize=is_delocalize)
    df.to_parquet(path=file_path)


def load_df_from_parquet(file_name: Optional[str],
                         subfolder_name: str = None,
                         subsubfolder_name: str = None,
                         key: Optional[str] = None,
                         local_path: Optional[str] = None,
                         is_delocalize: bool = False
                         ) -> pd.DataFrame:
    """
    pandas from parquet
    """
    file_path = get_local_file_path(file_name=file_name,
                                    file_type=FileTypes.PARQUET,
                                    local_path=local_path,
                                    subfolder_name=subfolder_name,
                                    subsubfolder_name=subsubfolder_name,
                                    key=key)
    try:
        df = pd.read_parquet(path=file_path)
    except:
        raise FileNotFoundError(f"not found {file_name} with path={file_path}")

    df = delocalize_df(df, is_delocalize=is_delocalize)

    return df


def save_df_dict_to_parquet(datasets: Dict[Union[str, Enum, NamedTuple], pd.DataFrame],
                            file_name: Optional[str] = None,
                            subfolder_name: str = None,
                            subsubfolder_name: str = None,
                            local_path: Optional[str] = None,
                            is_delocalize: bool = False,
                            is_output_file: bool = False
                            ) -> None:
    """
    pandas dict to parquet files
    """
    for key, data in datasets.items():
        if data is not None:
            file_path = get_local_file_path(file_name=file_name,
                                            file_type=FileTypes.PARQUET,
                                            local_path=local_path,
                                            subfolder_name=subfolder_name,
                                            subsubfolder_name=subsubfolder_name,
                                            key=key,
                                            is_output_file=is_output_file)
            data = delocalize_df(data, is_delocalize=is_delocalize)
            data.to_parquet(path=file_path)


def load_df_dict_from_parquet(dataset_keys: List[Union[str, Enum, NamedTuple]],
                              file_name: Optional[str],
                              subfolder_name: str = None,
                              subsubfolder_name: str = None,
                              local_path: Optional[str] = None,
                              is_force_not_found_error: bool = False
                              ) -> Dict[str, pd.DataFrame]:
    """
    pandas dict from parquet files
    """
    pandas_dict = {}
    for key in dataset_keys:
        file_path = get_local_file_path(file_name=file_name,
                                        file_type=FileTypes.PARQUET,
                                        local_path=local_path,
                                        subfolder_name=subfolder_name,
                                        subsubfolder_name=subsubfolder_name,
                                        key=key)
        try:
            pandas_dict[key] = pd.read_parquet(path=file_path)

        except:
            message = f"file data {file_name}, {key} not found"
            if is_force_not_found_error:
                raise FileNotFoundError(message)
            else:
                print(message)

    return pandas_dict


"""
For pdfs
"""


def get_pdf_path(file_name: str,
                 local_path: Union[None, str] = None,
                 is_add_current_date: bool = True
                 ) -> str:

    if is_add_current_date:
        file_name = join_file_name_parts([file_name, datetime.datetime.now().strftime(DATE_FORMAT)])

    file_path = get_output_file_path(file_name=file_name, file_type=FileTypes.PDF, local_path=local_path)

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
             is_add_current_date: bool = False,
             **kwargs
             ) -> str:
    """
    save matplotlib figure
    """
    if is_add_current_date:
        file_name = join_file_name_parts([file_name, datetime.datetime.now().strftime(DATE_FORMAT)])
    file_path = get_output_file_path(file_name=file_name,
                                     file_type=file_type,
                                     local_path=local_path)
    if file_type == FileTypes.PNG:
        fig.savefig(file_path, dpi=dpi)  # , bbox_inches=bbox_inches
    elif file_type == FileTypes.EPS:
        fig.savefig(file_path, dpi=dpi, format='eps')  # , bbox_inches=bbox_inches
    elif file_type == FileTypes.SVG:
        fig.savefig(file_path, dpi=dpi, format='svg')  # , bbox_inches=bbox_inches
    elif file_type == FileTypes.PDF:
        fig.savefig(file_path, dpi=dpi, format='pdf')  # , bbox_inches=bbox_inches
    else:
        raise NotImplementedError(f"{file_type}")
    return file_path


def save_figs(figs: Dict[str, plt.Figure],
              local_path: Optional[str] = None,
              dpi: int = 300,
              file_type=FileTypes.PNG,
              is_add_current_date: bool = False,
              **kwargs
              ) -> None:
    """
    save matplotlib figures dict
    """
    for key, fig in figs.items():
        file_path = save_fig(fig=fig,
                             file_name=key,
                             local_path=local_path,
                             dpi=dpi,
                             file_type=file_type,
                             is_add_current_date=is_add_current_date,
                             **kwargs)
        print(file_path)


def figs_to_pdf(figs: Union[List[plt.Figure], Dict[str, plt.Figure]],
                file_name: str,
                orientation: str = 'portrait',
                local_path: Optional[str] = None,
                is_add_current_date: bool = True
                ) -> str:
    """
    create PDF of list of plf figures
    """
    if is_add_current_date:
        file_name = join_file_name_parts([file_name, datetime.datetime.now().strftime(DATE_FORMAT)])

    file_path = get_output_file_path(file_name=file_name, file_type=FileTypes.PDF, local_path=local_path)

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
        print(get_local_paths())
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
        file_path = get_resource_file_path(file_name='ETH',
                                           subfolder_name='bitmex',
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
