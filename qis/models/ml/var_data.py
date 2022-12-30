import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# qis
import qis.models.linear.ewm as ewm


class DataType(Enum):
    PRICE = 1  # price data
    RETURN = 2  # precomputed log or price return
    LEVEL = 3  # level type data


class TransformType(Enum):
    NONE = 1
    LOG_RETURN = 2  # t1 = ln(S1/S2)
    CHANGE = 3   # t1 = S1 - S0
    LAG = 4  # t1 = S0


class NormalizationType(Enum):
    NONE = 1
    INVERSE_VOL = 2


class SmoothingType(Enum):
    NONE = 1
    EWMA = 2


@dataclass
class VarData:
    """
    to enter data for model variables
    keep data for either price, return data, or level data
    """
    data: pd.DataFrame
    data_type: DataType = DataType.RETURN
    transform_type: TransformType = TransformType.NONE
    normalization_type: NormalizationType = NormalizationType.NONE
    smoothing_type: SmoothingType = SmoothingType.NONE
    vol_lambda: float = 0.94
    smoothing_lambda: float = 0.94

    def to_var_data(self) -> pd.DataFrame:

        row_data = self.data

        # 1. get data transform
        data = self.get_transform(row_data=self.data)

        # 2. get data normalization
        data = self.get_normalization(row_data=data)

        # 3. get data smoothing
        data = self.get_smoothing(row_data=data)

        return data

    def get_transform(self, row_data: pd.DataFrame) -> pd.DataFrame:
        """
        compute transform of data dependent on data_type
        """
        if self.transform_type == TransformType.NONE:
            var_data = row_data
        else:
            if self.data_type == DataType.PRICE:
                if self.transform_type == TransformType.LOG_RETURN:
                    var_data = np.log(row_data.divide(row_data.shift(1)))
                elif self.transform_type == TransformType.CHANGE:
                    var_data = row_data.subtract(row_data.shift(1))
                elif self.transform_type == TransformType.LAG:
                    var_data = row_data.shift(1)
                else:
                    raise TypeError(f"transform {self.transform_type} of {type(self.transform_type)} not implemented")

            elif self.data_type == DataType.RETURN:
                if self.transform_type == TransformType.LOG_RETURN:
                    raise TypeError(f"transform {self.transform_type} not supported for {DataType.RETURN}")
                elif self.transform_type == TransformType.CHANGE:
                    raise TypeError(f"transform {self.transform_type} not supported for {DataType.RETURN}")
                elif self.transform_type == TransformType.LAG:
                    var_data = row_data.shift(1)
                else:
                    raise TypeError(f"transform {self.transform_type} of {type(self.transform_type)} not implemented")

            elif self.data_type == DataType.LEVEL:
                if self.transform_type == TransformType.LOG_RETURN:
                    var_data = np.log(row_data.divide(row_data.shift(1)))
                elif self.transform_type == TransformType.CHANGE:
                    var_data = row_data.subtract(row_data.shift(1))
                elif self.transform_type == TransformType.LAG:
                    var_data = row_data.shift(1)
                else:
                    raise TypeError(f"transform {self.transform_type} of {type(self.transform_type)} not implemented")

            else:
                raise TypeError(f"data_type {self.data_type} of {type(self.data_type)} not implemented")

        return var_data

    def get_normalization(self, row_data: pd.DataFrame) -> pd.DataFrame:

        if self.normalization_type == NormalizationType.INVERSE_VOL:
            vols_f = ewm.compute_ewm_vol(data=row_data,
                                           ewm_lambda=self.vol_lambda,
                                           mean_adj_type=ewm.MeanAdjType.EWMA,
                                           apply_sqrt=True,
                                           annualize=False)

            var_data = row_data.divide(vols_f)

        else:
            var_data = row_data

        return var_data

    def get_smoothing(self, row_data: pd.DataFrame) -> pd.DataFrame:

        if self.smoothing_type == SmoothingType.EWMA:
            var_data = ewm.compute_ewm(data=row_data,
                                         ewm_lambda=self.smoothing_lambda,
                                         is_unit_vol_scaling=True)

        else:
            var_data = row_data

        return var_data

    def get_dim(self):
        return len(self.data.columns)


def validate_vardata(data: VarData) -> None:
    if not isinstance(data, VarData):
        raise TypeError(f"data type {type(data)} must be fm.VarData")


def validate_asset_returns(factor_returns: pd.DataFrame, asset_returns: pd.DataFrame) -> None:

    if not pd.infer_freq(asset_returns.index) == pd.infer_freq(factor_returns.index):
        raise ValueError(f"freq of asset returns={pd.infer_freq(asset_returns.index)}"
                         f" must equal to freq of factor return={pd.infer_freq(factor_returns.index)}")

    if not asset_returns.index.equals(factor_returns.index):
        # can be relaxed in future
        raise ValueError(f"indices must be equal")
