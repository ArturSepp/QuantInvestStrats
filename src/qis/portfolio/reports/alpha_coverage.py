"""Explicit, opt-in coverage policy for report-only holdings alpha."""

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HoldingsAlphaCoverage:
    """Allow bounded missing-beta dust and explicitly supplied proxy histories.

    Attributes:
        weight_tolerance: Maximum absolute missing-beta instrument exposure, in NAV units.
            The default 1e-6 is 0.01 bp of NAV; known betas are never omitted.
        total_weight_tolerance: Maximum sum of absolute omitted exposures, in NAV units.
            The default 1e-5 is 0.1 bp of NAV. Long/short positions never offset this cap.
        proxy_map: Instrument identifier to proxy-price column. No automatic mapping, chaining
            or portfolio NAV regression. Select mappings independently of inference outcomes.
        proxy_prices: Positive reference-currency levels for the mapped proxies, with ordered,
            unique dates. NaNs are allowed, not filled. The caller must ensure matching currency,
            hedging and risk conventions and point-in-time availability of the supplied history.

    Passing None to a factsheet keeps strict coverage. This object enables a reporting-only
    approximation, without changing holdings, NAV, instrument beta warm-up or Sharpe inference.
    A proxy uses the same EWMA estimator/warm-up and opening-date lag as an instrument.
    Own eligible betas always win. Both tolerances may be zero for proxy-only assistance.
    """

    weight_tolerance: float = 1e-6
    total_weight_tolerance: float = 1e-5
    proxy_map: Mapping[str, str] = field(default_factory=dict)
    proxy_prices: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for name in ("weight_tolerance", "total_weight_tolerance"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 <= value < 1):
                raise ValueError(f"{name} must be finite and in [0, 1) NAV units")
        if not isinstance(self.proxy_map, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.proxy_map.items()
        ):
            raise ValueError("proxy_map must map nonempty instrument names to proxy names")
        object.__setattr__(self, "proxy_map", dict(self.proxy_map))
        prices = self.proxy_prices
        if prices is None:
            if self.proxy_map:
                raise ValueError("proxy_map requires explicit proxy_prices")
            return
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            raise ValueError("proxy_prices must be a nonempty DataFrame")
        if (not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique
                or not prices.index.is_monotonic_increasing or prices.index.hasnans
                or not prices.columns.is_unique):
            raise ValueError("proxy_prices need unique increasing dates and unique columns")
        if not set(self.proxy_map.values()).issubset(prices.columns):
            raise ValueError("all proxy_map values must name supplied proxy_prices columns")
        values = prices.to_numpy(dtype=float)
        if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
            raise ValueError("proxy_prices must be positive finite levels or NaN")
        object.__setattr__(self, "proxy_prices", prices.copy(deep=True))

    def for_instruments(self, instruments):
        """Restrict an already validated universe policy for a paired or constituent report.

        Args:
            instruments: Actual instrument identifiers in that constituent portfolio.

        Returns:
            Policy with only relevant mappings, retaining supplied proxy history and tolerances.
        """
        return replace(self, proxy_map={key: value for key, value in self.proxy_map.items()
                                       if key in instruments})


def validate_alpha_coverage(coverage, instruments=None):
    """Fail on configuration errors before report fallbacks can hide a misspelled mapping."""
    if coverage is None:
        return
    if not isinstance(coverage, HoldingsAlphaCoverage):
        raise ValueError("alpha_coverage must be HoldingsAlphaCoverage or None")
    # Revalidate in case a caller mutated the contained mapping or DataFrame after construction.
    coverage.__post_init__()
    if instruments is not None:
        unknown = set(coverage.proxy_map).difference(instruments)
        if unknown:
            raise ValueError(f"proxy_map instruments absent from the report: {sorted(unknown)}")
