"""Explicit account accounting and provenance for portfolio stress reports."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd


class PositionRole(Enum):
    """Declared accounting role; negative marks alone do not identify borrowing.

    Attributes:
        ASSET: Funded investment with a nonnegative mark.
        CASH: Cash or deposit, including an explicitly declared overdraft.
        FINANCING: Drawn borrowing with a nonpositive signed mark.
        SHORT: Short investment liability, distinct from financing.
        DERIVATIVE: Derivative mark of either sign; not its risk notional.
        OTHER: Other dated account asset or liability.
    """

    ASSET = "asset"
    CASH = "cash"
    FINANCING = "financing"
    SHORT = "short"
    DERIVATIVE = "derivative"
    OTHER = "other"


class ReportingBasis(Enum):
    """Accounting meaning of the denominator used for report ratios.

    Attributes:
        NET_EQUITY: Complete signed account equity after liabilities.
        GROSS_ASSETS: Sum of positive accounting marks, not derivative notionals.
        FIXED_NOTIONAL: Explicit hypothetical comparison amount.
    """

    NET_EQUITY = "net_equity"
    GROSS_ASSETS = "gross_assets"
    FIXED_NOTIONAL = "fixed_notional"


class AccountAssetClass(Enum):
    """Caller-assigned account allocation buckets, separate from risk factors.

    Attributes:
        LIQUIDITY: Cash and liquid deposits.
        BORROWING: Signed financing liabilities displayed relative to gross assets.
        CREDIT: Compatibility alias for BORROWING; displayed as Borrowing.
        FIXED_INCOME: Bonds and other caller-declared fixed-income products.
        EQUITY: Equity investments.
        ALTERNATIVES: Other explicitly classified investments.
    """

    LIQUIDITY = "Liquidity"
    BORROWING = "Borrowing"
    CREDIT = "Borrowing"
    FIXED_INCOME = "Fixed Income"
    EQUITY = "Equity"
    ALTERNATIVES = "Alternatives"


@dataclass(frozen=True)
class PortfolioAccounting:
    """Reconciled reference-currency account ledger, independent of fitted coverage.

    The ledger is complete even when some positions have no risk model. All marks
    include accrued amounts exactly once. Short and derivative liabilities remain
    separate from declared borrowing. Values describe accounting marks, not gross
    derivative risk exposure. Scenario equity includes P&L from valued holdings;
    missing risk coverage remains disclosed.

    Attributes:
        positions: Unique holding-ID index; columns mtm and role (PositionRole).
            Optional asset_class (AccountAssetClass or its display value) and currency
            columns map each position to its allocation category and currency. Both
            columns must be supplied together. Marks remain in reference currency;
            allocation currency is independent of fitted-return currency.
        net_equity: Explicit positive account equity, reconciled to signed marks.
        reporting_basis: Meaning of the portfolio's reporting denominator.
        fixed_notional: Positive explicit denominator when using FIXED_NOTIONAL.
        tolerance: Absolute reference-currency reconciliation tolerance.
    """

    positions: pd.DataFrame
    net_equity: float
    reporting_basis: ReportingBasis = ReportingBasis.NET_EQUITY
    fixed_notional: float | None = None
    tolerance: float = 0.01

    def __post_init__(self):
        """Snapshot the ledger and reject ambiguous roles or unreconciled equity."""
        p = self.positions.copy(deep=True)
        if (p.empty or not p.index.is_unique or p.index.hasnans
                or not p.columns.is_unique or not {"mtm", "role"}.issubset(p)):
            raise ValueError("accounting requires unique position IDs and mtm/role columns")
        if not np.isfinite(p.mtm).all() or not p.role.map(
                lambda value: isinstance(value, PositionRole)).all():
            raise ValueError("finite marks and explicit PositionRole values required")
        if (not isinstance(self.reporting_basis, ReportingBasis)
                or not np.isfinite(self.net_equity) or self.net_equity <= 0
                or not np.isfinite(self.tolerance) or self.tolerance < 0):
            raise ValueError("positive equity, ReportingBasis and nonnegative tolerance required")
        if p.loc[p.role.eq(PositionRole.ASSET), "mtm"].lt(0).any():
            raise ValueError("ASSET marks must be nonnegative; declare short/other liabilities")
        if p.loc[p.role.isin([PositionRole.FINANCING, PositionRole.SHORT]), "mtm"].gt(0).any():
            raise ValueError("FINANCING and SHORT marks must be nonpositive")
        if abs(float(p.mtm.sum()) - self.net_equity) > self.tolerance:
            raise ValueError("signed accounting ledger does not reconcile to net equity")
        if self.reporting_basis is ReportingBasis.FIXED_NOTIONAL:
            if (self.fixed_notional is None or not np.isfinite(self.fixed_notional)
                    or self.fixed_notional <= 0):
                raise ValueError("FIXED_NOTIONAL requires a positive explicit fixed_notional")
        elif self.fixed_notional is not None:
            raise ValueError("fixed_notional applies only to FIXED_NOTIONAL reporting")
        allocation_columns = {"asset_class", "currency"}
        present = allocation_columns.intersection(p.columns)
        if present and present != allocation_columns:
            raise ValueError("allocation requires both asset_class and currency columns")
        if present:
            p["asset_class"] = p.asset_class.map(
                lambda value: value.value if isinstance(value, AccountAssetClass) else value)
            p["asset_class"] = p.asset_class.replace("Credit", AccountAssetClass.BORROWING.value)
            if not p.asset_class.isin([item.value for item in AccountAssetClass]).all():
                raise ValueError("every allocation needs a declared AccountAssetClass")
            financing = p.role.eq(PositionRole.FINANCING)
            p.loc[financing, "asset_class"] = AccountAssetClass.BORROWING.value
            if p.loc[~financing, "asset_class"].eq(AccountAssetClass.BORROWING.value).any():
                raise ValueError("Borrowing allocation is reserved for FINANCING positions")
            valid_currencies = p.currency.map(
                lambda value: isinstance(value, str) and bool(value.strip()))
            if not valid_currencies.all():
                raise ValueError("every allocation needs a nonempty currency classification")
            p["currency"] = p.currency.str.strip().str.upper()
        object.__setattr__(self, "positions", p)

    def gross_asset_allocation(self) -> pd.DataFrame:
        """Aggregate positive marks by declared asset class and allocation currency.

        Returns:
            Class-by-currency reference-currency amounts in the five standard buckets.
            An empty table indicates that the caller supplied no allocation mapping.
            Liabilities remain in the reconciled balance sheet, outside gross allocation.
        """
        p = self.positions
        if not {"asset_class", "currency"}.issubset(p.columns):
            return pd.DataFrame()
        positive = p.loc[p.mtm.gt(0), ["mtm", "asset_class", "currency"]]
        allocation = positive.pivot_table(index="asset_class", columns="currency",
            values="mtm", aggfunc="sum", fill_value=0.)
        return allocation.reindex([item.value for item in AccountAssetClass], fill_value=0.)

    def allocation_with_borrowing(self) -> pd.DataFrame:
        """Add signed financing to gross invested-asset allocations by currency.

        Returns:
            Class-by-currency amounts. Positive asset classes sum to gross assets;
            Borrowing is negative. Dividing each row by gross assets shows funding
            leverage without changing the 100% total invested-asset allocation.
        """
        allocation = self.gross_asset_allocation()
        if allocation.empty:
            return allocation
        financing = self.positions.loc[self.positions.role.eq(PositionRole.FINANCING)]
        by_currency = financing.groupby("currency").mtm.sum()
        currencies = allocation.columns.union(by_currency.index, sort=False)
        allocation = allocation.reindex(columns=currencies, fill_value=0.)
        allocation.loc[AccountAssetClass.BORROWING.value] = by_currency.reindex(
            currencies, fill_value=0.)
        return allocation

    @property
    def gross_assets(self) -> float:
        """Return positive accounting marks including cash and accrued amounts."""
        return float(self.positions.mtm.clip(lower=0).sum())

    @property
    def borrowing(self) -> float:
        """Return positive drawn borrowing from declared financing rows."""
        return -float(self.positions.loc[
            self.positions.role.eq(PositionRole.FINANCING), "mtm"].sum())

    @property
    def denominator(self) -> float:
        """Return the amount specified by the declared reporting basis."""
        if self.reporting_basis is ReportingBasis.NET_EQUITY:
            return float(self.net_equity)
        if self.reporting_basis is ReportingBasis.GROSS_ASSETS:
            return self.gross_assets
        return float(self.fixed_notional)

    @property
    def denominator_label(self) -> str:
        """Return the factual caption shared by report headers and exports."""
        return {ReportingBasis.NET_EQUITY: "Net equity / NAV",
                ReportingBasis.GROSS_ASSETS: "Gross assets",
                ReportingBasis.FIXED_NOTIONAL: "Hypothetical notional"}[self.reporting_basis]

    def reconcile_holdings(self, holdings):
        """Check that every valued holding belongs to the complete dated ledger.

        Args:
            holdings: PortfolioHolding sequence with signed reference-currency marks.
        """
        for holding in holdings:
            if holding.holding_id not in self.positions.index:
                raise ValueError(f"valued holding missing from accounting: {holding.holding_id}")
            difference = self.positions.loc[holding.holding_id, "mtm"] - holding.observed_mtm
            if abs(difference) > self.tolerance:
                raise ValueError(f"holding mark differs from accounting: {holding.holding_id}")

    def summary(self) -> pd.Series:
        """Return reconciled balance-sheet levels and funding leverage ratios."""
        liabilities = -float(self.positions.mtm.clip(upper=0).sum())
        return pd.Series({"gross_assets": self.gross_assets, "borrowing": self.borrowing,
                          "other_liabilities": liabilities - self.borrowing,
                          "net_equity": self.net_equity,
                          "assets_to_equity": self.gross_assets / self.net_equity,
                          "debt_to_equity": self.borrowing / self.net_equity,
                          "reporting_denominator": self.denominator})


class HistorySource(Enum):
    """Declared origin of a fitted response history.

    Attributes:
        OBSERVED: Observed product price or NAV history.
        SYNTHETIC: Explicitly constructed product NAV history.
        PROXY: Another instrument or basket used as the response.
    """

    OBSERVED = "observed"
    SYNTHETIC = "synthetic"
    PROXY = "proxy"


class ReturnBasis(Enum):
    """Return convention of a supplied fitted response.

    Attributes:
        TOTAL: Complete product return including its accrued income.
        EXCESS: Return after the explicitly declared funding adjustment.
        PRICE: Price change excluding separately paid income.
    """

    TOTAL = "total"
    EXCESS = "excess"
    PRICE = "price"


@dataclass(frozen=True)
class ResponseProvenance:
    """Caller-owned response construction facts; reporting never alters the series.

    Attributes:
        source: Observed, synthetic or proxy history origin.
        currency: Currency of the fitted return series.
        return_basis: Total or excess convention used in the captured fit.
        construction: Plain description of how the response was produced.
        assumptions: Explicit rates, day counts, coefficients or funding rules.
        limitations: Payoff, history or coverage limitations.
        first_return: Optional first fitted return date.
        last_return: Optional last fitted return date.
    """

    source: HistorySource
    currency: str
    return_basis: ReturnBasis
    construction: str
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    first_return: str | None = None
    last_return: str | None = None

    def __post_init__(self):
        """Validate portable provenance without inventing acquisition facts."""
        if (not isinstance(self.source, HistorySource) or not self.currency
                or not isinstance(self.return_basis, ReturnBasis) or not self.construction):
            raise ValueError("provenance requires source, currency, basis and construction")
        for name in ("assumptions", "limitations"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("response assumptions and limitations must be nonempty strings")
            object.__setattr__(self, name, values)
        dates = [pd.Timestamp(d) for d in (self.first_return, self.last_return) if d is not None]
        if any(pd.isna(d) or d.tz is not None for d in dates) or (
                len(dates) == 2 and dates[0] > dates[1]):
            raise ValueError("response provenance dates must be finite and ordered")


def provenance_frame(provenance: Mapping[str, ResponseProvenance]) -> pd.DataFrame:
    """Snapshot supplied response provenance into an exportable table.

    Args:
        provenance: Response ID to validated construction record.

    Returns:
        Response-indexed source, convention, construction and qualification table.
    """
    rows = {key: {"source": value.source.value, "currency": value.currency,
                  "return_basis": value.return_basis.value, "construction": value.construction,
                  "assumptions": " | ".join(value.assumptions),
                  "limitations": " | ".join(value.limitations),
                  "first_return": value.first_return, "last_return": value.last_return}
            for key, value in provenance.items()}
    return pd.DataFrame.from_dict(rows, orient="index")
