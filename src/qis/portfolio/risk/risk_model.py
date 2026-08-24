"""
Point-in-time risk computations on dated covariance matrices.

``RiskModel`` is the canonical computation layer for ex-ante tracking error,
factor exposures, benchmark beta, systematic/residual tracking-error
decomposition, and Euler marginal tracking-error contributions. Covariance
matrices define the authoritative asset universe at each date. Dated weights
are selected as-of that grid without look-ahead, and model data are never
interpolated.

All results inherit the covariance matrix's units. The stack convention is
annualised fractional covariance, but this module performs no annualisation
and does not assert a frequency. Factor loadings have shape assets x factors.

The similarly named ``LinearModel`` is a returns-attribution container used by
the tracking-error factsheet. It is intentionally separate from this
weights-and-covariance container.

Ex-post tracking error from NAVs and return differences lives in the adjacent
``ex_post_tracking_error.py`` module. This module contains no ex-post quantity,
covariance estimation, or plotting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Union


WEIGHT_TOL: float = 1e-10
COVAR_SYMMETRY_TOL: float = 1e-12
UNASSIGNED_GROUP: str = 'Unassigned'


WeightInput = Union[pd.Series, pd.DataFrame]


@dataclass
class RiskModel:
    """Dated covariance matrices with an optional point-in-time factor block.

    The covariance at each date is authoritative for both the asset universe
    and total-risk computations. The optional factor block enables factor
    exposures and systematic/residual decompositions; it never replaces or
    rebuilds the supplied covariance.

    Attributes:
        covar: Covariance matrices by date, with matching asset index and columns.
        factor_loadings: Optional asset-by-factor loading matrices by date.
        factor_covar: Optional factor covariance matrices by date.
        residual_vars: Optional asset residual variances by date.

    Raises:
        ValueError: If the covariance or factor data are incomplete, non-finite,
            misaligned, non-unique, non-square, or non-symmetric.
    """

    covar: Dict[pd.Timestamp, pd.DataFrame]
    factor_loadings: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    factor_covar: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    residual_vars: Optional[Dict[pd.Timestamp, pd.Series]] = None

    def __post_init__(self) -> None:
        """Validate and normalise the dated model inputs."""
        self.covar = self._normalise_date_mapping(self.covar, field_name='covar')
        if not self.covar:
            raise ValueError("covar must not be empty")

        if self.factor_loadings is not None:
            self.factor_loadings = self._normalise_date_mapping(
                self.factor_loadings, field_name='factor_loadings')
        if self.factor_covar is not None:
            self.factor_covar = self._normalise_date_mapping(
                self.factor_covar, field_name='factor_covar')
        if self.residual_vars is not None:
            self.residual_vars = self._normalise_date_mapping(
                self.residual_vars, field_name='residual_vars')

        if self.factor_loadings is None and (
                self.factor_covar is not None or self.residual_vars is not None):
            raise ValueError("factor_loadings is required when factor_covar or residual_vars "
                             "is supplied")
        if self.factor_covar is None and self.residual_vars is not None:
            raise ValueError("factor_covar is required when residual_vars is supplied")
        if self.factor_covar is not None and self.residual_vars is None:
            raise ValueError("residual_vars is required when factor_covar is supplied")

        for date, covar in self.covar.items():
            self._validate_square_matrix(covar, field_name='covar', date=date)

        if self.factor_loadings is not None:
            self._validate_factor_loadings()
        if self.factor_covar is not None and self.residual_vars is not None:
            self._validate_factor_block()

    @staticmethod
    def _normalise_date_mapping(data: Dict[pd.Timestamp, object],
                                field_name: str,
                                ) -> Dict[pd.Timestamp, object]:
        """Return a mapping with Timestamp keys, rejecting invalid or duplicate dates."""
        if not isinstance(data, dict):
            raise ValueError(f"{field_name} must be a dict keyed by date, got "
                             f"{type(data).__name__}")
        normalised: Dict[pd.Timestamp, object] = {}
        for raw_date, value in data.items():
            try:
                date = pd.Timestamp(raw_date)
            except Exception as exc:
                raise ValueError(f"{field_name} has invalid date key {raw_date!r}") from exc
            if date in normalised:
                raise ValueError(f"{field_name} has duplicate date {date}")
            normalised[date] = value
        try:
            return dict(sorted(normalised.items()))
        except TypeError as exc:
            raise ValueError(f"{field_name} dates must use compatible time zones") from exc

    @staticmethod
    def _validate_square_matrix(matrix: pd.DataFrame,
                                field_name: str,
                                date: pd.Timestamp,
                                ) -> None:
        """Validate a labelled covariance matrix."""
        if not isinstance(matrix, pd.DataFrame):
            raise ValueError(f"{field_name}[{date}] must be a pd.DataFrame, got "
                             f"{type(matrix).__name__}")
        if matrix.index.has_duplicates or matrix.columns.has_duplicates:
            duplicates = list(matrix.index[matrix.index.duplicated()])
            duplicates.extend(list(matrix.columns[matrix.columns.duplicated()]))
            raise ValueError(f"{field_name}[{date}] has duplicate labels {duplicates}")
        if not matrix.index.equals(matrix.columns):
            missing_rows = matrix.columns.difference(matrix.index).tolist()
            missing_columns = matrix.index.difference(matrix.columns).tolist()
            raise ValueError(f"{field_name}[{date}] index must equal columns; "
                             f"missing rows {missing_rows}, missing columns {missing_columns}")
        try:
            values = matrix.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{date}] must contain numeric values") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{field_name}[{date}] contains non-finite values")
        if not np.allclose(values, values.T, rtol=0.0, atol=COVAR_SYMMETRY_TOL):
            max_asymmetry = float(np.max(np.abs(values - values.T)))
            raise ValueError(f"{field_name}[{date}] is not symmetric within "
                             f"{COVAR_SYMMETRY_TOL}; max asymmetry={max_asymmetry}")

    def _validate_factor_loadings(self) -> None:
        """Validate the loadings date grid, asset universe, and finite values."""
        assert self.factor_loadings is not None
        self._validate_date_grid(self.factor_loadings, field_name='factor_loadings')
        for date, loadings in self.factor_loadings.items():
            if not isinstance(loadings, pd.DataFrame):
                raise ValueError(f"factor_loadings[{date}] must be a pd.DataFrame, got "
                                 f"{type(loadings).__name__}")
            if loadings.index.has_duplicates or loadings.columns.has_duplicates:
                raise ValueError(f"factor_loadings[{date}] has duplicate asset or factor labels")
            covar_assets = self.covar[date].index
            if set(loadings.index) != set(covar_assets):
                missing = covar_assets.difference(loadings.index).tolist()
                extra = loadings.index.difference(covar_assets).tolist()
                raise ValueError(f"factor_loadings[{date}] asset mismatch; missing {missing}, "
                                 f"extra {extra}")
            try:
                values = loadings.to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"factor_loadings[{date}] must contain numeric values") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"factor_loadings[{date}] contains non-finite values")
            self.factor_loadings[date] = loadings.reindex(index=covar_assets)

    def _validate_factor_block(self) -> None:
        """Validate factor covariance and residual variance alignment."""
        assert self.factor_loadings is not None
        assert self.factor_covar is not None
        assert self.residual_vars is not None
        self._validate_date_grid(self.factor_covar, field_name='factor_covar')
        self._validate_date_grid(self.residual_vars, field_name='residual_vars')

        for date, loadings in self.factor_loadings.items():
            factor_covar = self.factor_covar[date]
            residual_vars = self.residual_vars[date]
            self._validate_square_matrix(factor_covar, field_name='factor_covar', date=date)
            if set(factor_covar.index) != set(loadings.columns):
                missing = loadings.columns.difference(factor_covar.index).tolist()
                extra = factor_covar.index.difference(loadings.columns).tolist()
                raise ValueError(f"factor_covar[{date}] factor mismatch; missing {missing}, "
                                 f"extra {extra}")
            self.factor_covar[date] = factor_covar.reindex(
                index=loadings.columns, columns=loadings.columns)

            if not isinstance(residual_vars, pd.Series):
                raise ValueError(f"residual_vars[{date}] must be a pd.Series, got "
                                 f"{type(residual_vars).__name__}")
            if residual_vars.index.has_duplicates:
                duplicates = residual_vars.index[residual_vars.index.duplicated()].tolist()
                raise ValueError(f"residual_vars[{date}] has duplicate assets {duplicates}")
            covar_assets = self.covar[date].index
            if set(residual_vars.index) != set(covar_assets):
                missing = covar_assets.difference(residual_vars.index).tolist()
                extra = residual_vars.index.difference(covar_assets).tolist()
                raise ValueError(f"residual_vars[{date}] asset mismatch; missing {missing}, "
                                 f"extra {extra}")
            try:
                values = residual_vars.to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"residual_vars[{date}] must contain numeric values") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"residual_vars[{date}] contains non-finite values")
            self.residual_vars[date] = residual_vars.reindex(index=covar_assets)

    def _validate_date_grid(self, data: Dict[pd.Timestamp, object], field_name: str) -> None:
        """Require an optional model field on exactly the covariance date grid."""
        covar_dates = set(self.covar)
        data_dates = set(data)
        if data_dates != covar_dates:
            missing = sorted(covar_dates - data_dates)
            extra = sorted(data_dates - covar_dates)
            raise ValueError(f"{field_name} date grid mismatch; missing {missing}, extra {extra}")

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Sorted covariance date grid.

        Returns:
            Covariance dates in increasing order.
        """
        return pd.DatetimeIndex(self.covar.keys())

    def _covar_at_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """Return the exact-date covariance or raise with the nearest earlier date."""
        date = pd.Timestamp(date)
        if date in self.covar:
            return self.covar[date]
        earlier_dates = self.dates[self.dates < date]
        nearest = earlier_dates[-1] if len(earlier_dates) > 0 else None
        nearest_message = str(nearest) if nearest is not None else 'none'
        raise KeyError(f"date {date} is not on the covar grid; nearest earlier grid date: "
                       f"{nearest_message}")

    def _align_weights(self,
                       weights: pd.Series,
                       date: pd.Timestamp,
                       role: str,
                       strict: bool = True,
                       outside_universe_remedy: Optional[str] = None,
                       ) -> pd.Series:
        """Align one weight vector to the authoritative covariance universe."""
        if not isinstance(weights, pd.Series):
            raise ValueError(f"{role} must be a pd.Series, got {type(weights).__name__}")
        if weights.index.has_duplicates:
            duplicates = weights.index[weights.index.duplicated()].tolist()
            raise ValueError(f"{role} has duplicate assets {duplicates}")
        covar = self._covar_at_date(date)
        try:
            numeric_weights = weights.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role} must contain numeric values") from exc

        extra = numeric_weights.index.difference(covar.index)
        extra_values = numeric_weights.reindex(extra)
        material = extra_values.fillna(0.0).abs() > WEIGHT_TOL
        if strict and bool(material.any()):
            offenders = extra_values.loc[material].to_dict()
            message = f"{role} has weighted assets outside covar[{pd.Timestamp(date)}]: {offenders}"
            if outside_universe_remedy is not None:
                message = f"{message}. {outside_universe_remedy}"
            raise ValueError(message)
        return numeric_weights.reindex(covar.index).fillna(0.0)

    def _aligned_weight_history(self,
                                weights: WeightInput,
                                role: str,
                                strict: bool = True,
                                outside_universe_remedy: Optional[str] = None,
                                ) -> pd.DataFrame:
        """Align static or dated weights as-of each covariance date."""
        if isinstance(weights, pd.Series):
            aligned = {
                date: self._align_weights(
                    weights=weights,
                    date=date,
                    role=role,
                    strict=strict,
                    outside_universe_remedy=outside_universe_remedy)
                for date in self.dates
            }
            return pd.DataFrame.from_dict(aligned, orient='index')
        if not isinstance(weights, pd.DataFrame):
            raise ValueError(f"{role} must be a pd.Series or pd.DataFrame, got "
                             f"{type(weights).__name__}")
        if not weights.index.is_unique:
            raise ValueError(f"{role} index must be unique")
        if weights.columns.has_duplicates:
            duplicates = weights.columns[weights.columns.duplicated()].tolist()
            raise ValueError(f"{role} has duplicate assets {duplicates}")

        dated_weights = weights.copy()
        try:
            dated_weights.index = pd.DatetimeIndex(dated_weights.index)
        except Exception as exc:
            raise ValueError(f"{role} index must contain dates") from exc
        if not dated_weights.index.is_unique:
            raise ValueError(f"{role} index has duplicate dates after Timestamp conversion")
        dated_weights = dated_weights.sort_index()
        asof_weights = dated_weights.reindex(index=self.dates, method='ffill')
        aligned = {
            date: self._align_weights(
                weights=asof_weights.loc[date],
                date=date,
                role=role,
                strict=strict,
                outside_universe_remedy=outside_universe_remedy)
            for date in self.dates
        }
        return pd.DataFrame.from_dict(aligned, orient='index')

    def _align_groups(self, group_data: pd.Series, date: pd.Timestamp) -> pd.Series:
        """Align ticker-to-group labels and expose missing labels as Unassigned."""
        if not isinstance(group_data, pd.Series):
            raise ValueError(f"group_data must be a pd.Series, got {type(group_data).__name__}")
        if group_data.index.has_duplicates:
            duplicates = group_data.index[group_data.index.duplicated()].tolist()
            raise ValueError(f"group_data has duplicate assets {duplicates}")
        assets = self._covar_at_date(date).index
        return group_data.reindex(assets).fillna(UNASSIGNED_GROUP)

    def _align_group_loadings(self,
                              group_loadings: pd.DataFrame,
                              date: pd.Timestamp,
                              ) -> pd.DataFrame:
        """Validate and align an asset-by-group loading matrix to covariance assets."""
        if not isinstance(group_loadings, pd.DataFrame):
            raise ValueError(
                f"group_loadings must be a pd.DataFrame, got "
                f"{type(group_loadings).__name__}")
        if group_loadings.index.has_duplicates:
            duplicates = group_loadings.index[
                group_loadings.index.duplicated()].tolist()
            raise ValueError(f"group_loadings has duplicate assets {duplicates}")
        if group_loadings.columns.has_duplicates:
            duplicates = group_loadings.columns[
                group_loadings.columns.duplicated()].tolist()
            raise ValueError(f"group_loadings has duplicate groups {duplicates}")
        try:
            numeric_loadings = group_loadings.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("group_loadings must contain numeric values") from exc
        if np.isinf(numeric_loadings.to_numpy(dtype=float)).any():
            raise ValueError("group_loadings contains infinite values")
        assets = self._covar_at_date(date).index
        return numeric_loadings.reindex(index=assets).fillna(0.0)

    @staticmethod
    def _compute_tracking_error(active_weights: pd.Series,
                                covar: pd.DataFrame,
                                ) -> float:
        """Compute the square root of the active-weight covariance quadratic form."""
        variance = float(active_weights @ covar @ active_weights)
        return float(np.sqrt(variance))

    def compute_tre_at_date(self,
                            benchmark_weights: pd.Series,
                            portfolio_weights: pd.Series,
                            date: pd.Timestamp,
                            group_data: Optional[pd.Series] = None,
                            total_column: str = 'Total',
                            strict: bool = True,
                            ) -> Union[float, pd.Series]:
        """Compute ex-ante tracking error on one covariance date.

        For active weights ``d = w_p - w_b``, tracking error is
        ``sqrt(d' Sigma d)``. When groups are requested, each group result
        zeroes active weights outside that group before applying the full
        covariance. These standalone sleeve risks are not additive and are
        not forced to sum to total tracking error. No annualisation is applied.

        Args:
            benchmark_weights: Benchmark weights indexed by asset.
            portfolio_weights: Portfolio weights indexed by asset.
            date: Exact covariance-grid date.
            group_data: Optional group label per asset. Missing labels become
                ``'Unassigned'``.
            total_column: Label for total tracking error when groups are returned.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Total tracking error as a float, or total and standalone group
            tracking errors as a Series.

        Raises:
            KeyError: If ``date`` is not an exact covariance-grid date.
            ValueError: If weights or model data violate the alignment policy.
        """
        covar = self._covar_at_date(date)
        benchmark = self._align_weights(
            weights=benchmark_weights,
            date=date,
            role='benchmark_weights',
            strict=strict)
        portfolio = self._align_weights(
            weights=portfolio_weights,
            date=date,
            role='portfolio_weights',
            strict=strict)
        active_weights = portfolio - benchmark
        total = self._compute_tracking_error(active_weights=active_weights, covar=covar)
        if group_data is None:
            return total

        groups = self._align_groups(group_data=group_data, date=date)
        results = {total_column: total}
        for group in pd.unique(groups):
            group_active_weights = active_weights.where(groups == group, other=0.0)
            results[group] = self._compute_tracking_error(
                active_weights=group_active_weights, covar=covar)
        return pd.Series(results, dtype=float)

    def compute_tre_by_group_loadings_at_date(
            self,
            benchmark_weights: pd.Series,
            portfolio_weights: pd.Series,
            date: pd.Timestamp,
            group_loadings: pd.DataFrame,
            total_column: str = 'Total',
            strict: bool = True,
    ) -> pd.Series:
        """Compute ex-ante tracking error for fractional or overlapping group loadings.

        For active weights ``d = w_p - w_b`` and an asset-by-group loading
        matrix ``L``, group ``g`` has standalone tracking error
        ``sqrt((d * L_g)' Sigma (d * L_g))``. Loadings may be fractional,
        overlapping, or signed and need not sum to one. Group tracking errors
        are not additive and are not forced to sum to total tracking error.
        Missing covariance-universe rows and loading NaNs become zero; extra
        loading rows are ignored. No annualisation is applied.

        Args:
            benchmark_weights: Benchmark weights indexed by asset.
            portfolio_weights: Portfolio weights indexed by asset.
            date: Exact covariance-grid date.
            group_loadings: Asset-by-group loading matrix. Column order defines
                result order.
            total_column: Label for total tracking error.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Series containing total and standalone group tracking errors.

        Raises:
            KeyError: If ``date`` is not an exact covariance-grid date.
            ValueError: If weights or group loadings violate the alignment policy,
                loading labels are duplicated, values are nonnumeric or infinite,
                or ``total_column`` duplicates a group label.
        """
        covar = self._covar_at_date(date)
        benchmark = self._align_weights(
            weights=benchmark_weights,
            date=date,
            role='benchmark_weights',
            strict=strict)
        portfolio = self._align_weights(
            weights=portfolio_weights,
            date=date,
            role='portfolio_weights',
            strict=strict)
        loadings = self._align_group_loadings(
            group_loadings=group_loadings,
            date=date)
        if total_column in loadings.columns:
            raise ValueError(
                f"total_column {total_column!r} duplicates a group_loadings column")

        active_weights = portfolio - benchmark
        results = {
            total_column: self._compute_tracking_error(
                active_weights=active_weights,
                covar=covar),
        }
        for group in loadings.columns:
            results[group] = self._compute_tracking_error(
                active_weights=active_weights * loadings[group],
                covar=covar)
        return pd.Series(results, dtype=float)

    def compute_tre_history(self,
                            benchmark_weights: WeightInput,
                            portfolio_weights: WeightInput,
                            group_data: Optional[pd.Series] = None,
                            total_column: str = 'Total',
                            strict: bool = True,
                            ) -> Union[pd.Series, pd.DataFrame]:
        """Compute ex-ante tracking error on the covariance date grid.

        Dated weights are selected as-of each covariance date using forward
        fill; dates before the first weight observation receive zero weights.
        The mathematics and group convention are those of
        :meth:`compute_tre_at_date`. No annualisation is applied.

        Args:
            benchmark_weights: Static Series or dated DataFrame of benchmark weights.
            portfolio_weights: Static Series or dated DataFrame of portfolio weights.
            group_data: Optional group label per asset. Missing labels become
                ``'Unassigned'``.
            total_column: Label for total tracking error when groups are returned.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            A Series named ``'Tracking error'``, or a DataFrame containing total
            and standalone group tracking errors.

        Raises:
            ValueError: If weights or model data violate the alignment policy.
        """
        benchmark_history = self._aligned_weight_history(
            weights=benchmark_weights,
            role='benchmark_weights',
            strict=strict)
        portfolio_history = self._aligned_weight_history(
            weights=portfolio_weights,
            role='portfolio_weights',
            strict=strict)
        results = {
            date: self.compute_tre_at_date(
                benchmark_weights=benchmark_history.loc[date],
                portfolio_weights=portfolio_history.loc[date],
                date=date,
                group_data=group_data,
                total_column=total_column,
                strict=strict)
            for date in self.dates
        }
        if group_data is None:
            return pd.Series(results, name='Tracking error', dtype=float)
        return pd.DataFrame.from_dict(results, orient='index')

    def compute_exposures_at_date(self,
                                  portfolio_weights: pd.Series,
                                  date: pd.Timestamp,
                                  strict: bool = True,
                                  ) -> pd.Series:
        """Compute factor exposures on one covariance date.

        Loadings ``B`` have shape assets x factors and exposure is ``B' w``.
        The construction-time alignment guarantee is retained: the method does
        not reindex or relax the loadings matrix.

        Args:
            portfolio_weights: Portfolio weights indexed by asset.
            date: Exact covariance-grid date.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Factor exposure Series indexed by factor.

        Raises:
            KeyError: If ``date`` is not an exact covariance-grid date.
            ValueError: If ``factor_loadings`` is missing or weights violate the
                alignment policy.
        """
        self._require_factor_loadings()
        assert self.factor_loadings is not None
        weights = self._align_weights(
            weights=portfolio_weights,
            date=date,
            role='portfolio_weights',
            strict=strict)
        return self.factor_loadings[pd.Timestamp(date)].T @ weights

    def compute_exposures_history(self,
                                  portfolio_weights: WeightInput,
                                  strict: bool = True,
                                  ) -> pd.DataFrame:
        """Compute factor exposures on the covariance date grid.

        Dated weights are selected as-of each covariance date using forward
        fill; dates before the first weight observation receive zero weights.
        At every date the exposure is ``B_t' w_t`` for asset-by-factor
        loadings ``B_t``.

        Args:
            portfolio_weights: Static Series or dated DataFrame of portfolio weights.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            DataFrame with covariance dates as rows and factors as columns.

        Raises:
            ValueError: If ``factor_loadings`` is missing or weights violate the
                alignment policy.
        """
        self._require_factor_loadings()
        weight_history = self._aligned_weight_history(
            weights=portfolio_weights,
            role='portfolio_weights',
            strict=strict)
        results = {
            date: self.compute_exposures_at_date(
                portfolio_weights=weight_history.loc[date],
                date=date,
                strict=strict)
            for date in self.dates
        }
        return pd.DataFrame.from_dict(results, orient='index')

    @staticmethod
    def _validate_benchmark_variance(benchmark_variance: float) -> None:
        """Reject a benchmark that cannot define covariance beta."""
        if benchmark_variance <= 0.0:
            raise ValueError(f"benchmark variance must be positive, got {benchmark_variance}")

    def compute_benchmark_beta_at_date(self,
                                       benchmark_weights: pd.Series,
                                       portfolio_weights: pd.Series,
                                       date: pd.Timestamp,
                                       strict: bool = True,
                                       ) -> float:
        """Compute ex-ante portfolio beta to a weighted benchmark.

        For portfolio weights ``w_p``, benchmark weights ``w_b``, and
        covariance ``Sigma``, beta is
        ``(w_p' Sigma w_b) / (w_b' Sigma w_b)``. The factor block is not
        required.

        Args:
            benchmark_weights: Benchmark weights indexed by asset.
            portfolio_weights: Portfolio weights indexed by asset.
            date: Exact covariance-grid date.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Scalar ex-ante portfolio beta to the benchmark.

        Raises:
            KeyError: If ``date`` is not an exact covariance-grid date.
            ValueError: If weights violate the alignment policy or benchmark
                variance is nonpositive.
        """
        covar = self._covar_at_date(date)
        benchmark = self._align_weights(
            weights=benchmark_weights,
            date=date,
            role='benchmark_weights',
            strict=strict,
            outside_universe_remedy=(
                "estimate the covariance on the joint universe of assets and "
                "benchmark constituents upstream"))
        portfolio = self._align_weights(
            weights=portfolio_weights,
            date=date,
            role='portfolio_weights',
            strict=strict)
        benchmark_variance = float(benchmark @ covar @ benchmark)
        self._validate_benchmark_variance(benchmark_variance)
        return float(portfolio @ covar @ benchmark) / benchmark_variance

    def compute_benchmark_beta_history(self,
                                       benchmark_weights: WeightInput,
                                       portfolio_weights: WeightInput,
                                       strict: bool = True,
                                       ) -> pd.Series:
        """Compute ex-ante benchmark beta on the covariance date grid.

        Dated weights are selected as-of each covariance date using forward
        fill; dates before the first weight observation receive zero weights.
        At every date beta is
        ``(w_p' Sigma w_b) / (w_b' Sigma w_b)``.

        Args:
            benchmark_weights: Static Series or dated DataFrame of benchmark weights.
            portfolio_weights: Static Series or dated DataFrame of portfolio weights.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Series named ``'Benchmark beta'`` on the covariance date grid.

        Raises:
            ValueError: If weights violate the alignment policy or benchmark
                variance is nonpositive.
        """
        benchmark_history = self._aligned_weight_history(
            weights=benchmark_weights,
            role='benchmark_weights',
            strict=strict,
            outside_universe_remedy=(
                "estimate the covariance on the joint universe of assets and "
                "benchmark constituents upstream"))
        portfolio_history = self._aligned_weight_history(
            weights=portfolio_weights,
            role='portfolio_weights',
            strict=strict)
        results = {
            date: self.compute_benchmark_beta_at_date(
                benchmark_weights=benchmark_history.loc[date],
                portfolio_weights=portfolio_history.loc[date],
                date=date,
                strict=strict)
            for date in self.dates
        }
        return pd.Series(results, name='Benchmark beta', dtype=float)

    def compute_benchmark_beta_loadings_at_date(self,
                                                benchmark_weights: pd.Series,
                                                date: pd.Timestamp,
                                                strict: bool = True,
                                                ) -> pd.Series:
        """Compute per-asset loadings to a weighted benchmark.

        The loading vector is ``c = Sigma w_b / (w_b' Sigma w_b)``, so the
        beta of any aligned portfolio is the linear form ``c' w_p``. The
        factor block is not required.

        Args:
            benchmark_weights: Benchmark weights indexed by asset.
            date: Exact covariance-grid date.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Benchmark-beta loading Series indexed by covariance asset.

        Raises:
            KeyError: If ``date`` is not an exact covariance-grid date.
            ValueError: If weights violate the alignment policy or benchmark
                variance is nonpositive.
        """
        covar = self._covar_at_date(date)
        benchmark = self._align_weights(
            weights=benchmark_weights,
            date=date,
            role='benchmark_weights',
            strict=strict,
            outside_universe_remedy=(
                "estimate the covariance on the joint universe of assets and "
                "benchmark constituents upstream"))
        benchmark_variance = float(benchmark @ covar @ benchmark)
        self._validate_benchmark_variance(benchmark_variance)
        return (covar @ benchmark) / benchmark_variance

    def compute_tre_decomposition_at_date(self,
                                          benchmark_weights: pd.Series,
                                          portfolio_weights: pd.Series,
                                          date: pd.Timestamp,
                                          strict: bool = True,
                                          ) -> pd.Series:
        """Decompose tracking error into factor and residual components.

        For active weights ``d = w_p - w_b``, asset-by-factor loadings ``B``,
        factor covariance ``Sigma_x``, and diagonal residual variance ``D``,
        the components are ``TE_f^2 = (B' d)' Sigma_x (B' d)`` and
        ``TE_r^2 = d' D d``. The returned tracking error is
        ``sqrt(TE_f^2 + TE_r^2)``. It equals covariance tracking error only
        when the supplied covariance satisfies ``Sigma = B Sigma_x B' + D``;
        inconsistent views are not reconciled silently.

        Args:
            benchmark_weights: Benchmark weights indexed by asset.
            portfolio_weights: Portfolio weights indexed by asset.
            date: Exact covariance-grid date.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Series with ``tracking_error``, ``factor_te``, and ``residual_te``.

        Raises:
            KeyError: If ``date`` is not an exact covariance-grid date.
            ValueError: If the factor block is missing or weights violate the
                alignment policy.
        """
        self._require_factor_block()
        assert self.factor_loadings is not None
        assert self.factor_covar is not None
        assert self.residual_vars is not None
        benchmark = self._align_weights(
            weights=benchmark_weights,
            date=date,
            role='benchmark_weights',
            strict=strict)
        portfolio = self._align_weights(
            weights=portfolio_weights,
            date=date,
            role='portfolio_weights',
            strict=strict)
        date = pd.Timestamp(date)
        active_weights = portfolio - benchmark
        factor_exposures = self.factor_loadings[date].T @ active_weights
        factor_variance = float(
            factor_exposures @ self.factor_covar[date] @ factor_exposures)
        residual_variance = float(
            active_weights.pow(2.0) @ self.residual_vars[date])
        return pd.Series({
            'tracking_error': np.sqrt(factor_variance + residual_variance),
            'factor_te': np.sqrt(factor_variance),
            'residual_te': np.sqrt(residual_variance),
        }, dtype=float)

    def compute_tre_decomposition_history(self,
                                          benchmark_weights: WeightInput,
                                          portfolio_weights: WeightInput,
                                          strict: bool = True,
                                          ) -> pd.DataFrame:
        """Compute factor/residual tracking-error decomposition through time.

        Dated weights are selected as-of each covariance date using forward
        fill. Each row uses the factor-block formulas documented by
        :meth:`compute_tre_decomposition_at_date`; no annualisation is applied.

        Args:
            benchmark_weights: Static Series or dated DataFrame of benchmark weights.
            portfolio_weights: Static Series or dated DataFrame of portfolio weights.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            DataFrame with ``tracking_error``, ``factor_te``, and ``residual_te`` columns.

        Raises:
            ValueError: If the factor block is missing or weights violate the
                alignment policy.
        """
        self._require_factor_block()
        benchmark_history = self._aligned_weight_history(
            weights=benchmark_weights,
            role='benchmark_weights',
            strict=strict)
        portfolio_history = self._aligned_weight_history(
            weights=portfolio_weights,
            role='portfolio_weights',
            strict=strict)
        results = {
            date: self.compute_tre_decomposition_at_date(
                benchmark_weights=benchmark_history.loc[date],
                portfolio_weights=portfolio_history.loc[date],
                date=date,
                strict=strict)
            for date in self.dates
        }
        return pd.DataFrame.from_dict(results, orient='index')

    def compute_marginal_tre_at_date(self,
                                     benchmark_weights: pd.Series,
                                     portfolio_weights: pd.Series,
                                     date: pd.Timestamp,
                                     group_data: Optional[pd.Series] = None,
                                     strict: bool = True,
                                     ) -> pd.DataFrame:
        """Compute canonical Euler marginal tracking-error contributions.

        With active weights ``d = w_p - w_b`` and
        ``TE = sqrt(d' Sigma d)``, each asset contribution is
        ``mcte_i = d_i (Sigma d)_i / TE`` and the contributions sum to TE.
        At TE equal to zero, all contributions are defined as zero.

        When a complete factor block is present, systematic and residual
        columns use ``B Sigma_x B' d`` and ``D d`` respectively. They sum to
        total per asset when ``Sigma = B Sigma_x B' + D``; inconsistent
        supplied views are shown without silent reconciliation. Group rows
        aggregate asset contributions additively and therefore differ from
        the standalone, non-additive group risks returned by ``compute_tre_*``.

        Args:
            benchmark_weights: Benchmark weights indexed by asset.
            portfolio_weights: Portfolio weights indexed by asset.
            date: Exact covariance-grid date.
            group_data: Optional group label per asset. Missing labels become
                ``'Unassigned'``.
            strict: If True, reject material weights outside the covariance universe.

        Returns:
            Per-asset DataFrame, or a DataFrame with ``Total`` and group rows.
            Column ``mcte`` is always present; a complete factor block adds
            ``mcte_systematic`` and ``mcte_residual``.

        Raises:
            KeyError: If ``date`` is not an exact covariance-grid date.
            ValueError: If weights or model data violate the alignment policy.
        """
        covar = self._covar_at_date(date)
        benchmark = self._align_weights(
            weights=benchmark_weights,
            date=date,
            role='benchmark_weights',
            strict=strict)
        portfolio = self._align_weights(
            weights=portfolio_weights,
            date=date,
            role='portfolio_weights',
            strict=strict)
        active_weights = portfolio - benchmark
        tracking_error = self._compute_tracking_error(
            active_weights=active_weights, covar=covar)
        if tracking_error == 0.0:
            mcte = pd.Series(0.0, index=covar.index)
        else:
            mcte = active_weights * (covar @ active_weights) / tracking_error
        contributions = pd.DataFrame({'mcte': mcte}, index=covar.index)

        if self.factor_covar is not None and self.residual_vars is not None:
            self._require_factor_block()
            assert self.factor_loadings is not None
            date = pd.Timestamp(date)
            if tracking_error == 0.0:
                systematic_mcte = pd.Series(0.0, index=covar.index)
                residual_mcte = pd.Series(0.0, index=covar.index)
            else:
                systematic_gradient = (
                    self.factor_loadings[date]
                    @ self.factor_covar[date]
                    @ (self.factor_loadings[date].T @ active_weights))
                residual_gradient = self.residual_vars[date] * active_weights
                systematic_mcte = active_weights * systematic_gradient / tracking_error
                residual_mcte = active_weights * residual_gradient / tracking_error
            contributions['mcte_systematic'] = systematic_mcte
            contributions['mcte_residual'] = residual_mcte

        if group_data is None:
            return contributions
        groups = self._align_groups(group_data=group_data, date=date)
        grouped = {'Total': contributions.sum(axis=0)}
        for group in pd.unique(groups):
            grouped[group] = contributions.loc[groups == group].sum(axis=0)
        return pd.DataFrame.from_dict(grouped, orient='index')

    def _require_factor_loadings(self) -> None:
        """Raise when a factor-exposure method lacks its required field."""
        if self.factor_loadings is None:
            raise ValueError("factor_loadings is required for factor exposure computations")

    def _require_factor_block(self) -> None:
        """Raise when a decomposition method lacks factor covariance data."""
        self._require_factor_loadings()
        if self.factor_covar is None:
            raise ValueError("factor_covar is required for systematic/residual decomposition")
        if self.residual_vars is None:
            raise ValueError("residual_vars is required for systematic/residual decomposition")
