"""
provide generic for data appending set data savings
"""
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict
import qis.file_utils as fu


@dataclass
class DfOutDict:
    df_out_dict: Dict[str, pd.DataFrame] = None
    last_data: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.df_out_dict = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        # for df_out_dict[key]
        return self.df_out_dict[key]

    def append(self, df: pd.DataFrame, name: str) -> None:
        if name not in self.df_out_dict.keys():
            self.df_out_dict[name] = df
        else:
            raise ValueError(f"{name} exist in {self.df_out_dict.keys()}")

    def set_last_df(self, data: pd.DataFrame) -> None:
        # way to first make a placement and then append it
        self.last_data = data

    def append_last_df(self, name: str) -> None:
        # way to first make a placement and then append it
        if self.last_data is None:
            raise ValueError(f"self.last_data is None")
        self.append(df=self.last_data, name=name)
        self.last_data = None

    def print(self):
        for key, df in self.df_out_dict.items():
            print(f"{key}")
            print(df)

    def save(self, file_name: str) -> None:
        if len(self.df_out_dict.keys()) > 0:
            file_path = fu.save_df_to_excel(data=self.df_out_dict, file_name=file_name)
            print(f"saved output data to excel:\n {file_path}")
