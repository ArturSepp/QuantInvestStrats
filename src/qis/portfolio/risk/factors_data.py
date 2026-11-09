"""
Tradable factors data container.

Holds a panel of tradable risk-factor prices and exposes typed, time-sliced
access. The container is factor-set agnostic: the concrete factor universe
(its enum and any vols) is injected by the caller via ``factors``, so a
specific model such as MATF defines ``RiskFactors`` in its own layer while this
container stays generic. Instances load from CSV; ``from_sql`` is a stub that raises, kept as
the place a project wires its own connection. The factor prices are built upstream in the
production layer.
"""
from __future__ import annotations

import pandas as pd
import qis as qis
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Type, Union

FILE_NAME = 'futures_risk_factors'


@dataclass
class FactorsData:
    """
    tradable factor price levels, optionally validated against a factor-set descriptor.

    Passing ``factors`` turns the column names into a checked contract: construction fails if an
    expected factor is missing, which catches a renamed or dropped column at load time rather than
    in the middle of an estimation. Leaving it None keeps the container generic.

    Attributes:
        factors_prices: factor price levels, one column per factor, indexed by date
        factors: str-valued Enum whose values are the expected column names. None skips validation

    Raises:
        ValueError: if ``factors`` is given and a factor it names is absent from the columns
    """
    factors_prices: pd.DataFrame
    factors: Optional[Type[Enum]] = None

    def __post_init__(self):
        if self.factors is not None:
            expected = [e.value for e in self.factors]
            missing = [c for c in expected if c not in self.factors_prices.columns]
            if missing:
                raise ValueError(f"factors_prices is missing expected factor columns: {missing!r}")

    @classmethod
    def load(cls,
             local_path: str,
             file_name: str = FILE_NAME,
             factors: Optional[Type[Enum]] = None,
             time_period: Optional[qis.TimePeriod] = None
             ) -> FactorsData:
        """Load factor prices from a CSV produced by the production layer."""
        factors_prices = qis.load_df_from_csv(file_name=file_name, local_path=local_path)
        if time_period is not None:
            factors_prices = time_period.locate(factors_prices)
        return cls(factors_prices=factors_prices, factors=factors)

    @classmethod
    def from_sql(cls,
                 query: str,
                 connection=None,
                 factors: Optional[Type[Enum]] = None,
                 time_period: Optional[qis.TimePeriod] = None
                 ) -> FactorsData:
        """Load factor prices from a SQL/Ramen source.

        Forward hook for database-backed instantiation using ``qis.sql_engine``.
        Wire the project connection here so cohorts can be sourced from the
        Ramen database without changing any consumer. Not yet implemented.
        """
        raise NotImplementedError("from_sql: wire to qis.sql_engine with the project connection")

    @property
    def factor_names(self) -> List[str]:
        """Factor column names in panel order."""
        return list(self.factors_prices.columns)

    def get_factor_prices(self,
                          factor: Union[str, Enum],
                          time_period: Optional[qis.TimePeriod] = None
                          ) -> pd.Series:
        """Price series for one factor, by name or enum member."""
        name = factor.value if isinstance(factor, Enum) else factor
        if name not in self.factors_prices.columns:
            raise ValueError(f"factor {name!r} not in {self.factor_names!r}")
        prices = self.factors_prices[name]
        if time_period is not None:
            prices = time_period.locate(prices)
        return prices

    def get_prices(self, time_period: Optional[qis.TimePeriod] = None) -> pd.DataFrame:
        """Full factor price panel, optionally time-sliced."""
        if time_period is not None:
            return time_period.locate(self.factors_prices)
        return self.factors_prices
