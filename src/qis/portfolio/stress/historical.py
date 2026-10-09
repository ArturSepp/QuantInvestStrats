"""Historical scenario windows and factor-based selection independent of fitting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HistoricalScenarioSelection:
    """Select the worst monthly returns of a named factor and replay full vectors.

    Missing ranking history or missing co-factors on a selected date raise an
    error rather than substituting a less severe month. The fitting history is
    owned by the caller and does not determine this scenario window.

    Attributes:
        factor: Factor whose log return determines selection, not portfolio P&L.
        start_date: Inclusive first scenario month; no prior opening return is used.
        count: Requested number of months, ordered by factor loss.
        end_date: Optional upper date, capped by the portfolio risk date.
    """

    factor: str = "Equity"
    start_date: str = "2006-01-01"
    count: int = 10
    end_date: str | None = None

    def __post_init__(self):
        """Validate named-factor selection, finite dates and a positive count."""
        if not isinstance(self.factor, str) or not self.factor.strip():
            raise ValueError("historical ranking factor must be a nonempty name")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("historical count must be a positive integer")
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date) if self.end_date is not None else None
        if pd.isna(start) or start.tz is not None or (end is not None and (
                pd.isna(end) or end.tz is not None or end < start)):
            raise ValueError("historical selection dates must be valid and ordered")
        object.__setattr__(self, "start_date", str(start.date()))
        if end is not None:
            object.__setattr__(self, "end_date", str(end.date()))

    def window(self, history: pd.DataFrame, cutoff) -> pd.DataFrame:
        """Return completed monthly vectors on the requested calendar window.

        Args:
            history: Unique month-end log-return panel for the complete model.
            cutoff: Latest allowed model date.

        Returns:
            Calendar-reindexed vectors with complete ranking-factor observations.
        """
        if (not isinstance(history.index, pd.DatetimeIndex) or history.index.has_duplicates
                or history.index.hasnans or history.index.tz is not None
                or not history.index.is_month_end.all() or history.columns.has_duplicates
                or self.factor not in history):
            raise ValueError("historical selection requires unique month-end factor vectors")
        end = pd.Timestamp(cutoff)
        if self.end_date is not None:
            end = min(end, pd.Timestamp(self.end_date))
        start = pd.Timestamp(self.start_date)
        months = pd.date_range(start.to_period("M").end_time.normalize(),
                               pd.offsets.MonthEnd().rollback(end), freq="ME")
        window = history.reindex(months)
        if len(window) < self.count:
            raise ValueError("insufficient months for the requested historical scenario count")
        if not np.isfinite(window[self.factor]).all():
            missing = window.index[~np.isfinite(window[self.factor])].strftime("%Y-%m-%d")
            raise ValueError(f"missing ranking-factor history: {self.factor}: {missing.tolist()}")
        return window

    def select(self, history: pd.DataFrame, cutoff) -> pd.DataFrame:
        """Return complete factor vectors in loss order, with chronological ties.

        Args:
            history: Month-end factor log returns including the requested window.
            cutoff: Latest allowed model date.

        Returns:
            Selected vectors ordered by the ranking factor's ascending return.
        """
        window = self.window(history, cutoff)
        dates = window[self.factor].sort_values(kind="stable").head(self.count).index
        selected = window.loc[dates]
        if not np.isfinite(selected.to_numpy()).all():
            missing = selected.columns[(~np.isfinite(selected)).any()].tolist()
            raise ValueError(f"selected historical months lack complete factor shocks: {missing}")
        return selected

    def metadata(self, window: pd.DataFrame) -> dict:
        """Describe the actual selection window for reports and provenance.

        Args:
            window: Validated calendar window returned by window().

        Returns:
            Portable ranking policy, dates and observation count.
        """
        return dict(basis="factor_return", factor=self.factor, count=self.count,
                    requested_start=self.start_date, requested_end=self.end_date,
                    history_start=str(window.index.min().date()),
                    history_end=str(window.index.max().date()), available_months=len(window),
                    order="ascending factor log return; chronological ties")


def historical_captions(policy):
    """Describe factor selection consistently in historical pages and the guide."""
    year = policy["requested_start"][:4]
    title = f"{policy['count']} worst {policy['factor']} months since {year}"
    subtitle = (f"Ranked by {policy['factor']} factor return; complete monthly vectors applied "
                f"to current holdings. History: {policy['history_start']} to "
                f"{policy['history_end']}. Historical replay, not realised portfolio performance.")
    return title, subtitle
