"""Dated holdings, anchored payoff valuation and shared-response risk mapping."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from qis.portfolio.risk.risk_model import RiskModel
from qis.portfolio.stress.instruments import (
    HoldingPayoff,
    InstrumentLeg,
    InstrumentType,
    KinkPolicy,
    PayoffContext,
    ResponseBasis,
    Underlying,
)
from qis.portfolio.stress._valuation import build_context


@dataclass(frozen=True)
class PortfolioHolding:
    """Original position identity with an observed reference-currency mark.

    Attributes:
        holding_id: Unique source position ID retained in every contributor table.
        name: Display name.
        observed_mtm: Signed source value in portfolio reference currency.
        legs: Vanilla decomposition; a funded holding has one DELTA_1 leg.
        payoff: Optional public composite, mutually exclusive with legs.
        kink_policy: Common quote-derivative side for all vanilla legs.
        metadata: Plain string annotations such as account or liquidity tier.
    """

    holding_id: str
    name: str
    observed_mtm: float
    legs: tuple[InstrumentLeg, ...] = ()
    payoff: HoldingPayoff | None = None
    kink_policy: KinkPolicy = KinkPolicy.MIDPOINT
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate payoff ownership and preserve the supplied source identity."""
        if not self.holding_id or not self.name or not np.isfinite(self.observed_mtm):
            raise ValueError("holding_id, name and finite observed_mtm are required")
        if not isinstance(self.kink_policy, KinkPolicy):
            raise ValueError("kink_policy must be a KinkPolicy")
        legs = tuple(self.legs)
        if bool(legs) == (self.payoff is not None):
            raise ValueError("supply either nonempty legs or a composite payoff")
        if any(not isinstance(leg, InstrumentLeg) for leg in legs):
            raise ValueError("every leg must be an InstrumentLeg")
        funded = [leg for leg in legs if leg.instrument_type is InstrumentType.DELTA_1]
        if funded:
            if len(legs) != 1 or funded[0].quantity == 0:
                raise ValueError("a funded holding requires one nonzero DELTA_1 leg")
            if self.observed_mtm * funded[0].quantity < 0:
                raise ValueError("funded quantity and observed_mtm must have the same sign")
        if self.payoff is not None:
            if not isinstance(self.payoff, HoldingPayoff):
                raise ValueError("composite must implement the public HoldingPayoff protocol")
            for attr in ("implementation_id", "coverage", "boundary_policy"):
                if not isinstance(getattr(self.payoff, attr), str) or not getattr(
                    self.payoff, attr
                ):
                    raise ValueError(f"composite must declare {attr}")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.metadata.items()):
            raise ValueError("holding metadata must contain string keys and values")
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_delta_one(self) -> bool:
        """Return whether this is a single funded holding."""
        return bool(self.legs and self.legs[0].instrument_type is InstrumentType.DELTA_1)

    def model_payoff(self, context: PayoffContext) -> pd.Series:
        """Evaluate the payoff model before adding a derivative's mark offset.

        Args:
            context: Shared public scenario market view.

        Returns:
            Scenario model values in reference currency.
        """
        quotes, fx = context.quotes, context.fx_rates
        quote0, fx0 = context.baseline_quotes, context.baseline_fx_rates
        currencies = context.quote_currencies
        if self.payoff is not None:
            value = self.payoff.evaluate(context)
        elif self.is_delta_one:
            key = self.legs[0].underlying_id
            currency = currencies.loc[key]
            value = (
                self.observed_mtm
                * (quotes[key] / quote0.loc[key])
                * (fx[currency] / fx0.loc[currency])
            )
        else:
            value = pd.Series(0.0, index=quotes.index)
            for leg in self.legs:
                key = leg.underlying_id
                value += leg.get_payoff(quotes[key], quote0.loc[key]) * fx[currencies.loc[key]]
        if (
            not isinstance(value, pd.Series)
            or not value.index.equals(quotes.index)
            or not np.isfinite(value.to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"invalid payoff values for holding {self.holding_id}")
        return value.astype(float).rename(self.holding_id)

    def get_pnl(self, context: PayoffContext, baseline_context: PayoffContext) -> pd.Series:
        """Return change in modeled value from the zero-shock anchor.

        Args:
            context: Scenario market view.
            baseline_context: Single-row zero-shock market view.

        Returns:
            Scenario P&L in reference currency.
        """
        if self.is_delta_one:
            key = self.legs[0].underlying_id
            currency = context.quote_currencies.loc[key]
            direction = (
                context.quote_response_jacobian.loc[key]
                + context.fx_response_jacobian.loc[currency]
            )
            g = context.response_log_shocks @ direction
            return (self.observed_mtm * np.expm1(g)).rename(self.holding_id)
        baseline = self.model_payoff(baseline_context).iloc[0]
        return self.model_payoff(context) - baseline

    def get_mtm(self, context: PayoffContext, baseline_context: PayoffContext) -> pd.Series:
        """Return observed mark plus modeled payoff change.

        Args:
            context: Scenario market view.
            baseline_context: Single-row zero-shock market view.

        Returns:
            Scenario values in reference currency.
        """
        return self.observed_mtm + self.get_pnl(context, baseline_context)

    def response_jacobian(self, context: PayoffContext) -> pd.Series:
        """Return local dollar derivative by shared underlying response at one scenario.

        Args:
            context: A single-scenario market view retaining original quote/FX baselines.

        Returns:
            Dollar sensitivities indexed by every fitted response ID.
        """
        quote_map, fx_map = context.quote_response_jacobian, context.fx_response_jacobian
        result = pd.Series(0.0, index=quote_map.columns)
        if len(context.quotes) != 1:
            raise ValueError("response Jacobian requires exactly one scenario")
        if self.payoff is not None:
            if np.any(context.response_log_shocks.to_numpy() != 0.0):
                method = getattr(self.payoff, "scenario_response_jacobian", None)
                if method is None:
                    raise NotImplementedError(
                        "Composite must implement scenario_response_jacobian for stressed "
                        "local risk; disable ordinary_asset_bands to retain valuation only."
                    )
                result = method(context)
            else:
                result = self.payoff.response_jacobian(context)
        elif self.is_delta_one:
            key = self.legs[0].underlying_id
            currency = context.quote_currencies.loc[key]
            value = (self.observed_mtm * (context.quotes.iloc[0][key]
                     / context.baseline_quotes[key]) * (context.fx_rates.iloc[0][currency]
                     / context.baseline_fx_rates[currency]))
            result = value * (quote_map.loc[key] + fx_map.loc[currency])
        else:
            quotes, fx = context.quotes.iloc[0], context.fx_rates.iloc[0]
            for leg in self.legs:
                key = leg.underlying_id
                currency = context.quote_currencies.loc[key]
                spot = quotes.loc[key]
                local_payoff = leg.get_payoff(
                    pd.Series([spot]), context.baseline_quotes[key]
                ).iloc[0]
                dollar_delta = leg.get_quote_delta(spot, self.kink_policy) * spot
                result += fx.loc[currency] * (
                    dollar_delta * quote_map.loc[key] + local_payoff * fx_map.loc[currency]
                )
        if (
            not isinstance(result, pd.Series)
            or result.index.has_duplicates
            or set(result.index) != set(quote_map.columns)
            or not np.isfinite(result.to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"invalid response Jacobian for holding {self.holding_id}")
        return result.reindex(quote_map.columns).astype(float).rename(self.holding_id)


@dataclass(frozen=True)
class PortfolioValuationResult:
    """Evaluated scenario values with explicit source-mark and model anchors.

    Attributes:
        factor_log_shocks: Complete labelled factor shocks.
        mtm: Scenario-by-original-holding stressed values.
        pnl: Scenario-by-original-holding changes from observed values.
        audit: Current observed value, model baseline and constant basis offset.
    """

    factor_log_shocks: pd.DataFrame
    mtm: pd.DataFrame
    pnl: pd.DataFrame
    audit: pd.DataFrame

    @property
    def portfolio_mtm(self) -> pd.Series:
        """Return total stressed portfolio value per scenario."""
        return self.mtm.sum(axis=1).rename("portfolio_mtm")

    @property
    def portfolio_pnl(self) -> pd.Series:
        """Return total portfolio P&L per scenario."""
        return self.pnl.sum(axis=1).rename("portfolio_pnl")


@dataclass(frozen=True)
class InstrumentPortfolio:
    """An absolute-position snapshot referencing the canonical QIS risk model.

    Covariances/residual variances must be annualised; response betas map log
    returns. The reporting denominator affects percentages only.

    Attributes:
        holdings: Original holdings with signed observed reference-currency marks.
        underlyings: Actual quote registry, keyed by quote ID.
        risk_model: Shared response model, independent of instrument marks.
        risk_date: Exact available model date, no later than valuation_date.
        valuation_date: Position snapshot date.
        reference_currency: Currency for values, P&L and dollar sensitivities.
        reporting_denominator: Positive amount used only for ratios.
        denominator_label: Factual denominator description, e.g. Gross assets.
        fx_rates: Currency to quote in reference-currency units per local unit.
            Each FX Underlying is quoted in reference_currency and uses a
            REFERENCE response; omit the reference currency itself.
    """

    holdings: tuple[PortfolioHolding, ...]
    underlyings: Mapping[str, Underlying]
    risk_model: RiskModel
    risk_date: pd.Timestamp
    valuation_date: pd.Timestamp
    reference_currency: str
    reporting_denominator: float
    denominator_label: str
    fx_rates: Mapping[str, Underlying] = field(default_factory=dict)

    def __post_init__(self):
        """Validate identities, exact dates, currencies and the complete factor block."""
        holdings = tuple(self.holdings)
        ids = [holding.holding_id for holding in holdings]
        if not holdings or len(ids) != len(set(ids)):
            raise ValueError("holdings must be nonempty with unique holding_id values")
        risk_date, valuation_date = pd.Timestamp(self.risk_date), pd.Timestamp(self.valuation_date)
        if pd.isna(risk_date) or pd.isna(valuation_date):
            raise ValueError("portfolio dates must be finite")
        if risk_date > valuation_date or risk_date not in self.risk_model.covar:
            raise ValueError("risk_date must be exact and no later than valuation_date")
        if (
            not self.reference_currency
            or not self.denominator_label
            or not np.isfinite(self.reporting_denominator)
            or self.reporting_denominator <= 0
        ):
            raise ValueError(
                "positive reporting_denominator and currency/denominator labels required"
            )
        model = self.risk_model
        if any(
            item is None
            for item in (model.factor_loadings, model.factor_covar, model.residual_vars)
        ):
            raise ValueError("factor stress requires the complete RiskModel factor/residual block")
        betas = model.factor_loadings[risk_date]
        if betas.empty:
            raise ValueError("factor stress requires nonempty responses and factors")
        if (model.residual_vars[risk_date] < 0).any():
            raise ValueError("residual variances must be nonnegative")
        for covariance in (model.covar[risk_date], model.factor_covar[risk_date]):
            if np.linalg.eigvalsh(covariance.to_numpy()).min() < -1e-12:
                raise ValueError("risk covariances must be positive semidefinite")
        underlyings, fx_rates = dict(self.underlyings), dict(self.fx_rates)
        if not underlyings:
            raise ValueError("underlyings must be nonempty")
        if self.reference_currency in fx_rates:
            raise ValueError("reference currency FX is deterministically one; do not supply it")
        for currency, fx in fx_rates.items():
            if (
                not currency
                or fx.currency != self.reference_currency
                or fx.response_basis is not ResponseBasis.REFERENCE
            ):
                raise ValueError("FX quotes must be reference currency per local currency")
        for key, quote in underlyings.items():
            if key != quote.quote_id:
                raise ValueError("underlying registry keys must match actual quote_id values")
            if quote.currency != self.reference_currency and quote.currency not in fx_rates:
                raise ValueError(f"missing reference-per-local FX quote for {quote.currency}")
        for quote in (*underlyings.values(), *fx_rates.values()):
            if quote.response_id is not None and quote.response_id not in betas.index:
                raise ValueError(f"missing fitted response {quote.response_id}")
        for holding in holdings:
            for leg in holding.legs:
                if leg.underlying_id not in underlyings:
                    raise ValueError(f"unknown actual quote {leg.underlying_id}")
        object.__setattr__(self, "holdings", holdings)
        object.__setattr__(self, "underlyings", MappingProxyType(underlyings))
        object.__setattr__(self, "fx_rates", MappingProxyType(fx_rates))
        object.__setattr__(self, "risk_date", risk_date)
        object.__setattr__(self, "valuation_date", valuation_date)

    def _baseline_context(self) -> PayoffContext:
        """Construct a single zero-shock view without colliding with scenario labels."""
        factors = self.risk_model.factor_loadings[self.risk_date].columns
        return build_context(self, pd.DataFrame(0.0, index=["baseline"], columns=factors))

    def evaluate(self, factor_log_shocks: pd.DataFrame) -> PortfolioValuationResult:
        """Evaluate a batch through the same anchored holding payoff functions.

        Args:
            factor_log_shocks: Complete scenario-by-factor log shocks.

        Returns:
            Holding and aggregate values/P&L plus baseline audit.
        """
        context = build_context(self, factor_log_shocks)
        baseline_context = self._baseline_context()
        baselines = pd.Series(
            {h.holding_id: h.model_payoff(baseline_context).iloc[0] for h in self.holdings},
            name="model_baseline",
        )
        observed = pd.Series(
            {h.holding_id: h.observed_mtm for h in self.holdings}, name="observed_mtm"
        )
        pnl = pd.concat(
            [h.get_pnl(context, baseline_context) for h in self.holdings], axis=1, sort=False
        )
        mtm = pnl.add(observed, axis=1)
        audit = pd.concat(
            [observed, baselines, (observed - baselines).rename("basis_offset")],
            axis=1, sort=False
        )
        audit["name"] = [h.name for h in self.holdings]
        audit["implementation"] = [
            h.payoff.implementation_id if h.payoff is not None else "qis.intrinsic.v1"
            for h in self.holdings
        ]
        audit["coverage"] = [
            h.payoff.coverage
            if h.payoff is not None
            else ("funded exponential response" if h.is_delta_one else "intrinsic; no time value")
            for h in self.holdings
        ]
        audit["boundary_policy"] = [
            h.payoff.boundary_policy if h.payoff is not None else h.kink_policy.value
            for h in self.holdings
        ]
        for column in sorted({key for h in self.holdings for key in h.metadata}):
            audit[f"metadata:{column}"] = [h.metadata.get(column, "") for h in self.holdings]
        if not np.isfinite(mtm.to_numpy()).all():
            raise ValueError("nonfinite portfolio values")
        return PortfolioValuationResult(context.factor_log_shocks, mtm, pnl, audit)

    def get_mtm(self, delta_f: pd.Series) -> pd.Series:
        """Evaluate one labelled factor vector, returning values by holding.

        Args:
            delta_f: Complete factor log-shock vector.

        Returns:
            Reference-currency holding values.
        """
        if not isinstance(delta_f, pd.Series):
            raise ValueError("delta_f must be a labelled Series")
        return self.evaluate(delta_f.to_frame().T).mtm.iloc[0]

    def get_pnl(self, delta_f: pd.Series) -> pd.Series:
        """Evaluate one labelled factor vector, returning P&L by holding.

        Args:
            delta_f: Complete factor log-shock vector.

        Returns:
            Reference-currency holding P&L.
        """
        if not isinstance(delta_f, pd.Series):
            raise ValueError("delta_f must be a labelled Series")
        return self.evaluate(delta_f.to_frame().T).pnl.iloc[0]

    def response_jacobian(self, delta_f: pd.Series | None = None) -> pd.DataFrame:
        """Return original-holding-by-shared-response local dollar sensitivities.

        Args:
            delta_f: Complete labelled factor log shocks at which to differentiate.
                None retains the original current-exposure calculation. Original marks,
                quote baselines and futures settlement references are never rebased.

        Returns:
            Dollar sensitivities at the supplied scenario, before denominator scaling.
            Composite payoffs require a scenario_response_jacobian(context) method
            at nonzero response shocks; declared boundary policies remain in force.
        """
        if delta_f is not None and not isinstance(delta_f, pd.Series):
            raise ValueError("delta_f must be a labelled Series or None")
        context = (self._baseline_context() if delta_f is None
                   else build_context(self, delta_f.to_frame().T))
        return pd.DataFrame([h.response_jacobian(context) for h in self.holdings])
