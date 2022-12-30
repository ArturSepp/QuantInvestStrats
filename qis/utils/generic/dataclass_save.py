"""
implementation of dataclass with save method
"""
import pandas as pd
from dataclasses import dataclass
from typing import get_type_hints
import qis.file_utils as fu


@dataclass
class DataClassSave:
    pass

    def save_to_excel(self,
                      file_name: str,
                      is_transpose: bool = False,
                      indedate_format: str = None): #'%d%b%Y'
        data = {}
        for field_name, field_type in get_type_hints(self).items():
            if issubclass(field_type, pd.DataFrame):
                df = getattr(self, field_name)
                if df is not None:
                    if indedate_format is not None:
                        df.index = [x.strftime(indedate_format) if isinstance(x, pd.Timestamp) else x for x in df.index]
                    if is_transpose:
                        df = df.transpose()
                    data[field_name] = df
        fu.save_df_to_excel(data=data, file_name=file_name)
