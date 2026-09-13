"""Labelled independent and conditional shocks, including factor-family splits."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from qis.portfolio.risk.risk_model import RiskModel
from qis.portfolio.risk.stress_testing import conditional_factor_shock


class ScenarioMode(Enum):
    """Policy for factors which a request leaves unspecified.

    Attributes:
        INDEPENDENT: Set unspecified factors to zero.
        CONDITIONAL: Complete them jointly using the supplied covariance.
    """

    INDEPENDENT = "independent"
    CONDITIONAL = "conditional"


class ShockConvention(Enum):
    """Explicit units of requested factor or family bumps.

    Attributes:
        LOG: Log-return bumps, with family weights applied in log units.
        SIMPLE: Simple-return bumps, split before applying log1p.
    """

    LOG = "log"
    SIMPLE = "simple"


@dataclass(frozen=True)
class StressScenarios:
    """Requested factor/family anchors with explicit completion and return units.

    NaN cells denote unspecified anchors. Fully supplied factor rows are retained
    under either completion policy. A family key expands through RiskModel
    metadata before conditional completion; member/family conflicts fail.

    Attributes:
        anchors: Scenario-by-factor-or-group requested bumps.
        mode: Default missing-factor completion policy.
        convention: Units for every supplied bump in this batch.
        descriptions: Optional plain descriptions keyed by scenario ID.
        scenario_modes: Optional row-specific completion policies which remain
            fixed even when resolve supplies a comparison-mode override.
    """

    anchors: pd.DataFrame
    mode: ScenarioMode = ScenarioMode.INDEPENDENT
    convention: ShockConvention = ShockConvention.LOG
    descriptions: Mapping[str, str] = field(default_factory=dict)
    scenario_modes: Mapping[str, ScenarioMode] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the request shape while preserving explicit zero anchors."""
        if not isinstance(self.mode, ScenarioMode) or not isinstance(
            self.convention, ShockConvention
        ):
            raise ValueError("mode and convention must use their public enums")
        anchors = self.anchors.copy(deep=True)
        if anchors.empty or anchors.index.has_duplicates or anchors.columns.has_duplicates:
            raise ValueError("scenario anchors must be nonempty with unique labels")
        if anchors.index.isna().any() or anchors.columns.isna().any():
            raise ValueError("scenario and anchor labels must not be missing")
        anchors = anchors.astype(float)
        if np.isinf(anchors.to_numpy()).any() or anchors.isna().all(axis=1).any():
            raise ValueError("every scenario needs finite supplied anchors")
        if not set(self.descriptions).issubset(anchors.index):
            raise ValueError("descriptions must refer to supplied scenario IDs")
        if not set(self.scenario_modes).issubset(anchors.index) or any(
            not isinstance(mode, ScenarioMode) for mode in self.scenario_modes.values()
        ):
            raise ValueError("scenario_modes must map scenario IDs to ScenarioMode")
        object.__setattr__(self, "scenario_modes", MappingProxyType(dict(self.scenario_modes)))
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "descriptions", MappingProxyType(dict(self.descriptions)))

    def expanded_anchors(self, risk_model: RiskModel, risk_date: pd.Timestamp) -> pd.DataFrame:
        """Expand families, splitting simple bumps before converting to log returns.

        Args:
            risk_model: Shared fitted response model with optional group metadata.
            risk_date: Exact model date used for factor identity.

        Returns:
            Factor log anchors, with NaN only for unspecified factors.
        """
        risk_date = pd.Timestamp(risk_date)
        if risk_model.factor_loadings is None or risk_date not in risk_model.factor_loadings:
            raise ValueError("scenario resolution requires an exact factor-model date")
        factors = risk_model.factor_loadings[risk_date].columns
        groups = risk_model.factor_groups or {}
        unknown = set(self.anchors.columns) - set(factors) - set(groups)
        if unknown:
            raise ValueError(f"unknown scenario factors/groups: {sorted(unknown)}")
        result = pd.DataFrame(np.nan, index=self.anchors.index, columns=factors)
        for scenario, row in self.anchors.iterrows():
            supplied = {}
            for key, value in row.dropna().items():
                if key in groups:
                    group = groups[key]
                    bumps = {
                        member: value * weight
                        for member, weight in zip(group.members, group.weights)
                    }
                else:
                    bumps = {key: value}
                overlap = set(supplied).intersection(bumps)
                if overlap:
                    raise ValueError(f"conflicting family/member anchors: {sorted(overlap)}")
                supplied.update(bumps)
            values = pd.Series(supplied)
            if self.convention is ShockConvention.SIMPLE:
                if (values <= -1.0).any():
                    raise ValueError("each expanded simple-return bump must exceed -1")
                values = np.log1p(values)
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError("expanded factor anchors must be finite")
            result.loc[scenario, values.index] = values
        return result

    def resolve(
        self, risk_model: RiskModel, risk_date: pd.Timestamp, mode: ScenarioMode | None = None
    ) -> pd.DataFrame:
        """Return complete log-shock vectors in the model factor order.

        Args:
            risk_model: Shared fitted response model.
            risk_date: Exact covariance/model date.
            mode: Optional explicit completion override for a comparison exhibit.

        Returns:
            Complete scenario-by-factor log shocks, ready for portfolio evaluation.
        """
        effective_mode = self.mode if mode is None else mode
        if not isinstance(effective_mode, ScenarioMode):
            raise ValueError("mode must be a ScenarioMode")
        expanded = self.expanded_anchors(risk_model, risk_date)
        rows = {}
        for label, row in expanded.iterrows():
            row_mode = self.scenario_modes.get(label, effective_mode)
            if row_mode is ScenarioMode.INDEPENDENT:
                rows[label] = row.fillna(0.0)
            elif row.notna().all():
                # A supplied full vector needs no solve, even for a singular fitted block.
                rows[label] = row
            else:
                if risk_model.factor_covar is None:
                    raise ValueError("conditional scenarios require factor covariance")
                covariance = risk_model.factor_covar[pd.Timestamp(risk_date)]
                rows[label] = conditional_factor_shock(covariance, row.dropna().to_dict())
        completed = pd.DataFrame.from_dict(rows, orient="index").reindex(
            index=expanded.index, columns=expanded.columns
        )
        # Preserve supplied anchors exactly, including zero, after floating-point solves.
        return completed.where(expanded.isna(), expanded)
