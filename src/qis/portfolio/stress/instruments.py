"""Public instrument terms and the read-only payoff extension context."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


class InstrumentType(Enum):
    """Supported primitive payoffs.

    Attributes:
        DELTA_1: Funded exposure with an exponential response.
        CALL: Signed intrinsic call payoff.
        PUT: Signed intrinsic put payoff.
        FUTURE: Change in futures quote times signed units and multiplier.
    """

    DELTA_1 = "delta_1"
    CALL = "call"
    PUT = "put"
    FUTURE = "future"


class ResponseBasis(Enum):
    """Currency basis of the fitted response.

    Attributes:
        REFERENCE: Response includes conversion into the portfolio reference currency.
        LOCAL: Response describes only the quote in its own currency.
    """

    REFERENCE = "reference"
    LOCAL = "local"


class KinkPolicy(Enum):
    """One-sided quote derivative used consistently across a holding's legs.

    Attributes:
        LEFT: Derivative approached from lower quotes.
        RIGHT: Derivative approached from higher quotes.
        MIDPOINT: Average of left and right derivatives.
    """

    LEFT = "left"
    RIGHT = "right"
    MIDPOINT = "midpoint"


@dataclass(frozen=True)
class Underlying:
    """An actual quote with a possibly shared fitted response.

    Attributes:
        quote_id: Unique actual quote identifier, independent of the response proxy.
        spot0: Positive baseline quote in quote-currency units.
        currency: Quote currency.
        response_id: RiskModel response row, or None for explicitly deterministic cash.
        response_basis: Whether the fitted response includes reference-currency FX.
    """

    quote_id: str
    spot0: float
    currency: str
    response_id: str | None
    response_basis: ResponseBasis

    def __post_init__(self):
        """Reject ambiguous or unsupported quote terms."""
        if not self.quote_id or not self.currency:
            raise ValueError("quote_id and currency must be nonempty")
        if not np.isfinite(self.spot0) or self.spot0 <= 0:
            raise ValueError("spot0 must be positive for multiplicative quote shocks")
        if self.response_id is not None and not self.response_id:
            raise ValueError("response_id must be nonempty or explicitly None")
        if not isinstance(self.response_basis, ResponseBasis):
            raise ValueError("response_basis must be a ResponseBasis")


@dataclass(frozen=True)
class InstrumentLeg:
    """A signed primitive valued in its underlying's quote currency.

    Attributes:
        instrument_type: Primitive enum.
        underlying_id: Actual quote ID in the portfolio registry.
        quantity: Signed number of units; negative means short.
        multiplier: Positive contract multiplier.
        strike: Nonnegative option strike; absent for funded assets and futures.
    """

    instrument_type: InstrumentType
    underlying_id: str
    quantity: float
    multiplier: float = 1.0
    strike: float | None = None

    def __post_init__(self):
        """Validate contract terms before any scenario is evaluated."""
        if not isinstance(self.instrument_type, InstrumentType):
            raise ValueError("instrument_type must be an InstrumentType")
        if not self.underlying_id or not np.isfinite(self.quantity):
            raise ValueError("underlying_id and finite signed quantity are required")
        if not np.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("multiplier must be positive and finite")
        option = self.instrument_type in (InstrumentType.CALL, InstrumentType.PUT)
        if option:
            if self.strike is None or not np.isfinite(self.strike) or self.strike < 0:
                raise ValueError("options require a nonnegative finite strike")
        elif self.strike is not None:
            raise ValueError("strike is only valid for calls and puts")

    def get_payoff(self, quotes: pd.Series, spot0: float) -> pd.Series:
        """Evaluate signed model value locally, before currency conversion.

        Args:
            quotes: Scenario quotes indexed by scenario ID.
            spot0: Baseline quote, used as the futures reference level.

        Returns:
            Local-currency model values with the supplied index.
        """
        if not np.isfinite(quotes.to_numpy(dtype=float)).all() or (quotes <= 0).any():
            raise ValueError("scenario quotes must be positive and finite")
        if self.instrument_type is InstrumentType.CALL:
            value = (quotes - self.strike).clip(lower=0)
        elif self.instrument_type is InstrumentType.PUT:
            value = (self.strike - quotes).clip(lower=0)
        elif self.instrument_type is InstrumentType.FUTURE:
            value = quotes - spot0
        else:
            value = quotes
        return self.quantity * self.multiplier * value

    def get_quote_delta(self, spot0: float, kink_policy: KinkPolicy) -> float:
        """Return current local derivative with respect to the actual quote.

        Args:
            spot0: Baseline quote.
            kink_policy: Common one-sided derivative policy for this holding.

        Returns:
            Signed units including multiplier and intrinsic exercise state.
        """
        if not isinstance(kink_policy, KinkPolicy):
            raise ValueError("kink_policy must be a KinkPolicy")
        slope = 1.0
        if self.instrument_type in (InstrumentType.CALL, InstrumentType.PUT):
            call_slope = float(spot0 > self.strike)
            if spot0 == self.strike:
                call_slope = {
                    KinkPolicy.LEFT: 0.0,
                    KinkPolicy.RIGHT: 1.0,
                    KinkPolicy.MIDPOINT: 0.5,
                }[kink_policy]
            slope = call_slope if self.instrument_type is InstrumentType.CALL else call_slope - 1.0
        return float(self.quantity * self.multiplier * slope)


class PayoffContext:
    """Read-only labelled market view for public composite payoff implementations.

    DataFrame/Series properties return defensive copies. Quotes and FX include only
    the supplied scenario rows; baseline values are supplied separately.

    Attributes:
        quote_currencies: Quote-currency code for each actual quote.
        response_log_shocks: Scenario-by-shared-response log returns.
        quotes: Scenario-by-actual-quote local prices.
        baseline_quotes: Actual quote baselines.
        fx_rates: Scenario FX, in reference currency per unit of local currency.
        baseline_fx_rates: Baseline reference/local currency conversions.
        factor_log_shocks: Fully resolved scenario-by-factor log shocks.
        quote_response_jacobian: Derivative of each local log quote by shared response.
        fx_response_jacobian: Derivative of each log FX conversion by shared response.
        reference_currency: Portfolio value currency.
    """

    def __init__(
        self,
        quotes: pd.DataFrame,
        baseline_quotes: pd.Series,
        fx_rates: pd.DataFrame,
        baseline_fx_rates: pd.Series,
        factor_log_shocks: pd.DataFrame,
        quote_response_jacobian: pd.DataFrame,
        fx_response_jacobian: pd.DataFrame,
        reference_currency: str,
        quote_currencies: pd.Series,
        response_log_shocks: pd.DataFrame,
    ):
        """Copy the market view so custom payoffs cannot alter shared evaluations."""
        self._quote_currencies = quote_currencies.copy(deep=True)
        self._response_log_shocks = response_log_shocks.copy(deep=True)
        self._quotes = quotes.copy(deep=True)
        self._baseline_quotes = baseline_quotes.copy(deep=True)
        self._fx_rates = fx_rates.copy(deep=True)
        self._baseline_fx_rates = baseline_fx_rates.copy(deep=True)
        self._factor_log_shocks = factor_log_shocks.copy(deep=True)
        self._quote_response_jacobian = quote_response_jacobian.copy(deep=True)
        self._fx_response_jacobian = fx_response_jacobian.copy(deep=True)
        self._reference_currency = reference_currency

    @property
    def quote_currencies(self) -> pd.Series:
        """Return each actual quote's currency."""
        return self._quote_currencies.copy(deep=True)

    @property
    def response_log_shocks(self) -> pd.DataFrame:
        """Return shared fitted response log returns."""
        return self._response_log_shocks.copy(deep=True)

    @property
    def quotes(self) -> pd.DataFrame:
        """Return local scenario quotes."""
        return self._quotes.copy(deep=True)

    @property
    def baseline_quotes(self) -> pd.Series:
        """Return current local quotes."""
        return self._baseline_quotes.copy(deep=True)

    @property
    def fx_rates(self) -> pd.DataFrame:
        """Return scenario reference-per-local FX rates."""
        return self._fx_rates.copy(deep=True)

    @property
    def baseline_fx_rates(self) -> pd.Series:
        """Return baseline reference-per-local FX rates."""
        return self._baseline_fx_rates.copy(deep=True)

    @property
    def factor_log_shocks(self) -> pd.DataFrame:
        """Return resolved factor log shocks."""
        return self._factor_log_shocks.copy(deep=True)

    @property
    def quote_response_jacobian(self) -> pd.DataFrame:
        """Return local log-quote sensitivities to shared log responses."""
        return self._quote_response_jacobian.copy(deep=True)

    @property
    def fx_response_jacobian(self) -> pd.DataFrame:
        """Return log-FX sensitivities to shared log responses."""
        return self._fx_response_jacobian.copy(deep=True)

    @property
    def reference_currency(self) -> str:
        """Return the portfolio reference currency."""
        return self._reference_currency


@runtime_checkable
class HoldingPayoff(Protocol):
    """Public extension for a composite valued in reference currency.

    Attributes:
        implementation_id: Stable implementation/version identifier for the audit.
        coverage: Plain-language payoff approximation and omitted contract states.
        boundary_policy: Declared current derivative convention at discontinuities.
    """

    implementation_id: str
    coverage: str
    boundary_policy: str

    def evaluate(self, context: PayoffContext) -> pd.Series:
        """Return scenario model payoff values in reference currency.

        Args:
            context: Public quotes, FX and mappings; no private engine imports.

        Returns:
            Finite Series with exactly the scenario index.
        """
        ...

    def response_jacobian(self, context: PayoffContext) -> pd.Series:
        """Return current dollar sensitivities indexed by shared response ID.

        Args:
            context: Zero-shock market view, with the shared response mappings.

        Returns:
            Finite labelled derivative, including zero for every unused response.
            If unavailable, raise NotImplementedError; full risk analysis fails
            explicitly while direct payoff valuation remains usable.
        """
        ...
