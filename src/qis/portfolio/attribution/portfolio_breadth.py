"""Point-in-time diagnostics for how a portfolio uses its investable universe.

The module keeps three concepts separate.  *Universe breadth* asks how many assets are available
and how many independent correlation directions they represent.  *Capital breadth* asks how many
positions the absolute target weights represent after concentration.  *Risk breadth* applies the
same inverse-Herfindahl construction to absolute Euler risk contributions.  The latter uses
``compute_portfolio_risk_contributions`` from the canonical QIS risk layer; this module does not
introduce another risk decomposition.

Every row is evaluated on a target-weight date.  When ``covar_dict`` is supplied, the latest
covariance dated at or before that row is used and strictly positive finite diagonal variances
define the investable universe.  Otherwise QIS estimates an exponentially weighted second-moment
covariance from the supplied return observations up to that date.  ``span`` is consequently in
rows of ``returns``: pass monthly returns for a 36-month span.  No resampling is performed and no
observation after the evaluation date is read.  In the estimated case, an asset becomes investable
after its first finite return; a zero estimated variance can therefore be investable but not yet
risk-measurable.

For absolute weight shares p_i and absolute Euler-risk shares q_i, effective counts are

    N_eff_weight = 1 / sum_i p_i**2,
    N_eff_risk   = 1 / sum_i q_i**2.

The effective independent-universe count is the same participation ratio applied to the
eigenvalue shares of the eligible-assets correlation matrix.  Selection coverage times sizing
evenness reconciles exactly to capital utilisation.  A long-short portfolio is supported by
normalising absolute rather than net weights.  Cash is not inferred from names or zero variance;
only columns explicitly listed in ``cash_columns`` are excluded.

These are breadth and concentration diagnostics, not performance attribution.  Active breadth,
benchmark-relative weights and claims that breadth causes alpha are deliberately outside this
module.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qis.models.linear.ewm import NanBackfill, compute_ewm_covar_tensor
from qis.portfolio.risk.contributions import compute_portfolio_risk_contributions


INVESTABLE_COUNT = "Investable assets"
INVESTED_COUNT = "Invested assets"
EFFECTIVE_UNIVERSE_COUNT = "Effective independent assets"
EFFECTIVE_CAPITAL_COUNT = "Effective capital assets"
EFFECTIVE_RISK_COUNT = "Effective risk contributors"
SELECTION_COVERAGE = "Selection coverage"
SIZING_EVENNESS = "Sizing evenness"
CAPITAL_UTILISATION = "Capital utilisation"
RISK_BREADTH_EFFICIENCY = "Risk breadth efficiency"
RISK_MEASURABLE_COUNT = "Risk-measurable assets"
GROSS_WEIGHT = "Gross target weight"
INVESTABLE_GROSS_WEIGHT = "Investable gross target weight"
UNAVAILABLE_INVESTED_COUNT = "Unavailable invested assets"
UNAVAILABLE_GROSS_WEIGHT = "Unavailable gross target weight"

COUNT_COLUMNS = (
    INVESTABLE_COUNT,
    INVESTED_COUNT,
    EFFECTIVE_UNIVERSE_COUNT,
    EFFECTIVE_CAPITAL_COUNT,
    EFFECTIVE_RISK_COUNT,
)
EFFICIENCY_COLUMNS = (
    SELECTION_COVERAGE,
    SIZING_EVENNESS,
    CAPITAL_UTILISATION,
    RISK_BREADTH_EFFICIENCY,
)
AUDIT_COLUMNS = (
    RISK_MEASURABLE_COUNT,
    GROSS_WEIGHT,
    INVESTABLE_GROSS_WEIGHT,
    UNAVAILABLE_INVESTED_COUNT,
    UNAVAILABLE_GROSS_WEIGHT,
)

_PSD_RELATIVE_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class PortfolioBreadthResult:
    """Container for portfolio-breadth statistics and their numerical audit trails.

    Attributes:
        metrics: Breadth counts, efficiency ratios and audit statistics.  Rows are the sorted
            target-weight dates and columns include ``COUNT_COLUMNS``, ``EFFICIENCY_COLUMNS`` and
            ``AUDIT_COLUMNS``.
        availability: Derived investability mask on the evaluation dates and retained assets.
        absolute_weight_shares: Absolute material target weights normalised over investable
            positions.  A row is zero when no investable position is held.
        absolute_risk_contribution_shares: Absolute Euler risk contributions normalised over the
            risk-measurable positions.  A row is zero when portfolio risk cannot be attributed.
        covariance_dates: Covariance observation used at each evaluation date.  Values are
            ``NaT`` before the first point-in-time covariance or return observation.
        span: EWM covariance span in return observations.  It is recorded even when an externally
            supplied ``covar_dict`` makes it inactive.
        position_threshold: Strict absolute-weight cutoff used to identify material positions.
        covariance_source: ``"provided"`` or ``"returns-ewma"``.
    """

    metrics: pd.DataFrame
    availability: pd.DataFrame
    absolute_weight_shares: pd.DataFrame
    absolute_risk_contribution_shares: pd.DataFrame
    covariance_dates: pd.Series
    span: int
    position_threshold: float
    covariance_source: str

    @property
    def counts(self) -> pd.DataFrame:
        """Return the five count and effective-count series."""
        return self.metrics.loc[:, list(COUNT_COLUMNS)]

    @property
    def efficiency(self) -> pd.DataFrame:
        """Return the four dimensionless breadth-efficiency series."""
        return self.metrics.loc[:, list(EFFICIENCY_COLUMNS)]

    @property
    def audit(self) -> pd.DataFrame:
        """Return data-quality and investability reconciliation statistics."""
        return self.metrics.loc[:, list(AUDIT_COLUMNS)]


def compute_portfolio_breadth(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        *,
        covar_dict: Optional[Mapping[pd.Timestamp, pd.DataFrame]] = None,
        span: int = 36,
        position_threshold: float = 1.0e-4,
        cash_columns: Optional[Sequence[str]] = None,
        ) -> PortfolioBreadthResult:
    """Compute point-in-time universe, capital and risk breadth on target-weight dates.

    The function does not forward-fill target weights: every row in ``weights`` is an evaluation
    point and is measured as supplied.  It does use the latest covariance or return observation at
    or before that date.  Provided covariance matrices take precedence over return-based risk
    estimation.  Their strictly positive finite diagonal variances define investability; returns
    are used only to establish the common asset universe in that case.

    Without ``covar_dict``, the native rows of ``returns`` feed QIS's unadjusted EWM covariance
    recursion using a second moment about zero and ``NanBackfill.ZERO_FILL``.  ``span`` counts those
    rows, so the caller controls the economic horizon by supplying returns at the intended
    frequency.  Availability is cumulative and point-in-time: an asset is available from its
    first finite return, never before it.

    Args:
        returns: Asset return observations.  Columns define the complete investment universe.
        weights: Target portfolio weights.  Its dates define the evaluation dates and its columns
            must be contained in ``returns``.  NaN targets are treated as zero; infinite targets
            are rejected.
        covar_dict: Optional dated covariance matrices.  At each evaluation date the latest matrix
            at or before the date is used.  Matrices may contain NaN rows for unavailable assets,
            but the positive-variance available submatrix must be finite, symmetric and positive
            semidefinite up to numerical tolerance.
        span: Positive EWM span in native return observations, used only when ``covar_dict`` is
            absent.  For example, ``span=36`` means 36 months only when ``returns`` is monthly.
        position_threshold: Non-negative absolute target weight below which a position is treated
            as immaterial.  The comparison is strict: ``abs(weight) > position_threshold``.
        cash_columns: Optional explicit columns to remove before all counts and calculations.  QIS
            never infers cash from an asset name or covariance.

    Returns:
        A ``PortfolioBreadthResult`` with metrics and aligned audit panels.

    Raises:
        TypeError: If returns, weights, covariance matrices or parameters have invalid types.
        ValueError: If labelled axes, dates, asset coverage or numerical inputs are invalid.
    """
    clean_returns, clean_weights, cash = _validate_and_align_inputs(
        returns=returns,
        weights=weights,
        span=span,
        position_threshold=position_threshold,
        cash_columns=cash_columns,
    )
    assets = clean_returns.columns
    evaluation_dates = clean_weights.index
    aligned_weights = clean_weights.reindex(columns=assets).fillna(0.0)

    if covar_dict is None:
        covariance_source = "returns-ewma"
        covariance_provider = _EstimatedCovarianceProvider(
            returns=clean_returns,
            span=span,
        )
    else:
        covariance_source = "provided"
        covariance_provider = _ProvidedCovarianceProvider(
            covar_dict=covar_dict,
            assets=assets,
            cash_columns=cash,
        )

    metric_rows = {}
    availability_rows = {}
    weight_share_rows = {}
    risk_share_rows = {}
    covariance_dates = {}

    for date, weight_row in aligned_weights.iterrows():
        covariance_date, covariance, available = covariance_provider.at(date)
        available = available.reindex(assets, fill_value=False).astype(bool)
        metrics, weight_shares, risk_shares = _compute_date_breadth(
            date=date,
            weights=weight_row,
            availability=available,
            covariance=covariance,
            position_threshold=position_threshold,
        )
        metric_rows[date] = metrics
        availability_rows[date] = available
        weight_share_rows[date] = weight_shares
        risk_share_rows[date] = risk_shares
        covariance_dates[date] = covariance_date

    metrics = pd.DataFrame.from_dict(metric_rows, orient="index")
    metrics = metrics.loc[:, list(COUNT_COLUMNS + EFFICIENCY_COLUMNS + AUDIT_COLUMNS)]
    availability = pd.DataFrame.from_dict(availability_rows, orient="index").astype(bool)
    weight_shares = pd.DataFrame.from_dict(weight_share_rows, orient="index").fillna(0.0)
    risk_shares = pd.DataFrame.from_dict(risk_share_rows, orient="index").fillna(0.0)
    covariance_date_series = pd.Series(covariance_dates, name="Covariance date")

    for output in (metrics, availability, weight_shares, risk_shares, covariance_date_series):
        output.index.name = evaluation_dates.name

    return PortfolioBreadthResult(
        metrics=metrics,
        availability=availability,
        absolute_weight_shares=weight_shares,
        absolute_risk_contribution_shares=risk_shares,
        covariance_dates=covariance_date_series,
        span=span,
        position_threshold=float(position_threshold),
        covariance_source=covariance_source,
    )


class _EstimatedCovarianceProvider:
    """Serve causal EWM covariance states and return-derived availability."""

    def __init__(self, returns: pd.DataFrame, span: int) -> None:
        """Precompute the causal covariance tensor once over the return panel."""
        self._returns = returns
        values = returns.to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        self._finite_cumulative = np.logical_or.accumulate(np.isfinite(values), axis=0)
        self._covariance_tensor = compute_ewm_covar_tensor(
            a=values,
            span=span,
            nan_backfill=NanBackfill.ZERO_FILL,
        )

    def at(self, date: pd.Timestamp) -> Tuple[pd.Timestamp, Optional[pd.DataFrame], pd.Series]:
        """Return the last risk state whose return date is not later than ``date``."""
        position = _latest_position(index=self._returns.index, date=date, source="returns")
        if position is None:
            return pd.NaT, None, pd.Series(False, index=self._returns.columns, dtype=bool)
        covariance = pd.DataFrame(
            self._covariance_tensor[position],
            index=self._returns.columns,
            columns=self._returns.columns,
        )
        availability = pd.Series(
            self._finite_cumulative[position],
            index=self._returns.columns,
            dtype=bool,
        )
        return self._returns.index[position], covariance, availability


class _ProvidedCovarianceProvider:
    """Serve validated caller-supplied covariance matrices without forward-looking lookup."""

    def __init__(
            self,
            covar_dict: Mapping[pd.Timestamp, pd.DataFrame],
            assets: pd.Index,
            cash_columns: Sequence[str],
            ) -> None:
        """Validate, sort and align the dated covariance mapping."""
        if not isinstance(covar_dict, Mapping):
            raise TypeError("covar_dict must be a mapping of dates to pandas DataFrames")
        if not covar_dict:
            raise ValueError("covar_dict must not be empty")
        dated_covariances = []
        for raw_date, covariance in covar_dict.items():
            try:
                date = pd.Timestamp(raw_date)
            except Exception as exception:
                raise TypeError(f"invalid covariance date {raw_date!r}") from exception
            clean = _validate_covariance(covariance=covariance, date=date)
            if cash_columns:
                retained = clean.index.difference(pd.Index(cash_columns), sort=False)
                clean = clean.loc[retained, retained]
            dated_covariances.append((date, clean))
        dated_covariances.sort(key=lambda item: item[0])
        self._dates = pd.DatetimeIndex([item[0] for item in dated_covariances])
        if self._dates.has_duplicates:
            raise ValueError("covar_dict dates must be unique after timestamp conversion")
        self._covariances = [item[1] for item in dated_covariances]
        self._assets = assets

    def at(self, date: pd.Timestamp) -> Tuple[pd.Timestamp, Optional[pd.DataFrame], pd.Series]:
        """Return the latest covariance at or before ``date`` and its positive-variance mask."""
        position = _latest_position(index=self._dates, date=date, source="covar_dict")
        if position is None:
            return pd.NaT, None, pd.Series(False, index=self._assets, dtype=bool)
        covariance = self._covariances[position].reindex(index=self._assets, columns=self._assets)
        diagonal = np.diag(covariance.to_numpy(dtype=float))
        availability = pd.Series(
            np.isfinite(diagonal) & (diagonal > 0.0),
            index=self._assets,
            dtype=bool,
        )
        return self._dates[position], covariance, availability


def _validate_and_align_inputs(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        span: int,
        position_threshold: float,
        cash_columns: Optional[Sequence[str]],
        ) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[str, ...]]:
    """Validate labelled inputs, remove explicit cash and return chronological copies."""
    _validate_frame(frame=returns, name="returns", allow_nan=True)
    _validate_frame(frame=weights, name="weights", allow_nan=True)
    if isinstance(span, bool) or not isinstance(span, (int, np.integer)) or span <= 0:
        raise ValueError("span must be a positive integer")
    if isinstance(position_threshold, bool) or not isinstance(
            position_threshold, (int, float, np.integer, np.floating)
            ):
        raise TypeError("position_threshold must be a finite non-negative number")
    if not np.isfinite(position_threshold) or position_threshold < 0.0:
        raise ValueError("position_threshold must be a finite non-negative number")

    if cash_columns is None:
        cash = ()
    elif isinstance(cash_columns, str):
        cash = (cash_columns,)
    else:
        try:
            cash = tuple(cash_columns)
        except TypeError as exception:
            raise TypeError("cash_columns must be a sequence of asset labels") from exception
    if len(set(cash)) != len(cash):
        raise ValueError("cash_columns must not contain duplicates")
    known_assets = set(returns.columns).union(weights.columns)
    unknown_cash = [asset for asset in cash if asset not in known_assets]
    if unknown_cash:
        raise ValueError(f"cash_columns missing from returns and weights: {unknown_cash[:5]}")

    clean_returns = returns.drop(columns=list(cash), errors="ignore").copy().sort_index()
    clean_weights = weights.drop(columns=list(cash), errors="ignore").copy().sort_index()
    if clean_returns.shape[1] == 0:
        raise ValueError("returns must contain at least one non-cash asset")
    missing_weight_assets = clean_weights.columns.difference(clean_returns.columns, sort=False)
    if not missing_weight_assets.empty:
        raise ValueError(
            f"weights columns missing from returns: {missing_weight_assets.tolist()[:5]}"
        )

    try:
        clean_returns = clean_returns.astype(float)
        clean_weights = clean_weights.astype(float)
    except (TypeError, ValueError) as exception:
        raise TypeError("returns and weights must contain numeric values") from exception
    if np.isinf(clean_returns.to_numpy()).any():
        raise ValueError("returns must not contain infinite values")
    if np.isinf(clean_weights.to_numpy()).any():
        raise ValueError("weights must not contain infinite values")
    return clean_returns, clean_weights, cash


def _validate_frame(frame: pd.DataFrame, name: str, allow_nan: bool) -> None:
    """Validate the labelled axes shared by return and target-weight frames."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be a pandas DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError(f"{name} index must be unique")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    if frame.index.hasnans:
        raise ValueError(f"{name} index must not contain NaT")
    if not allow_nan and frame.isna().any(axis=None):
        raise ValueError(f"{name} must not contain NaN")


def _validate_covariance(covariance: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Validate a labelled covariance while allowing NaNs for unavailable assets."""
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError(f"covariance at {date} must be a pandas DataFrame")
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance at {date} must be non-empty and square")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise ValueError(f"covariance at {date} must have unique asset labels")
    if not covariance.index.equals(covariance.columns):
        raise ValueError(f"covariance at {date} index and columns must be identical")
    try:
        clean = covariance.astype(float).copy()
    except (TypeError, ValueError) as exception:
        raise TypeError(f"covariance at {date} must contain numeric values") from exception
    if np.isinf(clean.to_numpy()).any():
        raise ValueError(f"covariance at {date} must not contain infinite values")
    return clean


def _latest_position(index: pd.DatetimeIndex, date: pd.Timestamp, source: str) -> Optional[int]:
    """Find the last timestamp not later than ``date`` and reject incompatible time zones."""
    try:
        position = int(index.searchsorted(date, side="right")) - 1
    except TypeError as exception:
        message = f"{source} and weights dates must have compatible time zones"
        raise ValueError(message) from exception
    return None if position < 0 else position


def _compute_date_breadth(
        date: pd.Timestamp,
        weights: pd.Series,
        availability: pd.Series,
        covariance: Optional[pd.DataFrame],
        position_threshold: float,
        ) -> Tuple[dict, pd.Series, pd.Series]:
    """Compute breadth metrics and normalized audit shares for one evaluation date."""
    absolute_weights = weights.abs()
    material = absolute_weights > position_threshold
    invested = availability & material
    unavailable_invested = (~availability) & material

    weight_shares = pd.Series(0.0, index=weights.index)
    investable_gross_weight = float(absolute_weights.loc[availability].sum())
    material_gross_weight = float(absolute_weights.loc[invested].sum())
    if material_gross_weight > 0.0:
        weight_shares.loc[invested] = absolute_weights.loc[invested] / material_gross_weight
    effective_capital = _inverse_hhi(weight_shares.to_numpy())

    risk_shares = pd.Series(0.0, index=weights.index)
    risk_measurable = pd.Series(False, index=weights.index, dtype=bool)
    effective_universe = 0.0
    effective_risk = 0.0
    if covariance is not None and bool(availability.any()):
        covariance = covariance.reindex(index=weights.index, columns=weights.index)
        diagonal = np.diag(covariance.to_numpy(dtype=float))
        risk_measurable = availability & np.isfinite(diagonal) & (diagonal > 0.0)
        risk_assets = risk_measurable.index[risk_measurable]
        if len(risk_assets) > 0:
            risk_covar = _validated_risk_submatrix(
                covariance=covariance.loc[risk_assets, risk_assets],
                date=date,
            )
            effective_universe = _effective_independent_count(risk_covar)
            risk_weights = weights.reindex(risk_assets).where(invested.reindex(risk_assets), 0.0)
            contributions = compute_portfolio_risk_contributions(
                w=risk_weights,
                covar=risk_covar,
            )
            absolute_contributions = contributions.abs()
            contribution_total = float(absolute_contributions.sum())
            if contribution_total > 0.0:
                risk_shares.loc[risk_assets] = absolute_contributions / contribution_total
                effective_risk = _inverse_hhi(risk_shares.to_numpy())

    investable_count = float(availability.sum())
    invested_count = float(invested.sum())
    selection_coverage = _safe_ratio(invested_count, investable_count)
    sizing_evenness = _safe_ratio(effective_capital, invested_count)
    capital_utilisation = _safe_ratio(effective_capital, investable_count)
    risk_breadth_efficiency = _safe_ratio(effective_risk, invested_count)

    metrics = {
        INVESTABLE_COUNT: investable_count,
        INVESTED_COUNT: invested_count,
        EFFECTIVE_UNIVERSE_COUNT: effective_universe,
        EFFECTIVE_CAPITAL_COUNT: effective_capital,
        EFFECTIVE_RISK_COUNT: effective_risk,
        SELECTION_COVERAGE: selection_coverage,
        SIZING_EVENNESS: sizing_evenness,
        CAPITAL_UTILISATION: capital_utilisation,
        RISK_BREADTH_EFFICIENCY: risk_breadth_efficiency,
        RISK_MEASURABLE_COUNT: float(risk_measurable.sum()),
        GROSS_WEIGHT: float(absolute_weights.sum()),
        INVESTABLE_GROSS_WEIGHT: investable_gross_weight,
        UNAVAILABLE_INVESTED_COUNT: float(unavailable_invested.sum()),
        UNAVAILABLE_GROSS_WEIGHT: float(absolute_weights.loc[~availability].sum()),
    }
    return metrics, weight_shares, risk_shares


def _validated_risk_submatrix(covariance: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Return a symmetric PSD covariance or fail rather than hide a broken risk state."""
    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"positive-variance covariance submatrix at {date} must be fully finite"
        )
    if not np.allclose(values, values.T, atol=1.0e-12, rtol=1.0e-10):
        raise ValueError(f"positive-variance covariance submatrix at {date} must be symmetric")
    symmetric = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -_PSD_RELATIVE_TOLERANCE * scale:
        raise ValueError(
            f"positive-variance covariance submatrix at {date} must be positive semidefinite"
        )
    return pd.DataFrame(symmetric, index=covariance.index, columns=covariance.columns)


def _effective_independent_count(covariance: pd.DataFrame) -> float:
    """Compute inverse-HHI participation ratio of correlation eigenvalue shares."""
    values = covariance.to_numpy(dtype=float)
    volatility = np.sqrt(np.diag(values))
    correlation = values / np.outer(volatility, volatility)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues = np.linalg.eigvalsh(correlation)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    eigenvalues[np.abs(eigenvalues) <= _PSD_RELATIVE_TOLERANCE * scale] = 0.0
    if float(np.min(eigenvalues)) < 0.0:
        raise ValueError("eligible-assets correlation matrix must be positive semidefinite")
    return _inverse_hhi(eigenvalues)


def _inverse_hhi(non_negative_values: np.ndarray) -> float:
    """Return the participation ratio of non-negative values, or zero for no mass."""
    values = np.asarray(non_negative_values, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    shares = values / total
    return float(1.0 / np.sum(np.square(shares)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide breadth measures with a zero convention that preserves decomposition."""
    if denominator > 0.0:
        return float(numerator / denominator)
    return 0.0
