"""Offline EUR/CHF return coordinates, exact notional payoffs and settled FX accounting.

Run with QIS 5.47.0 or later (including the accounting API). Frozen synthetic
asset paths are assigned illustrative quote currencies; spots/rates are teaching inputs.
No downloads, file outputs, optional dependencies or production allocations are used.
"""

from enum import Enum

import numpy as np
import pandas as pd
import qis
from qis.datasets.synthetic import generate_synthetic_prices


def run_example(verbose: bool = True) -> dict:
    """Compute all four daily/monthly simple/log views and one explicit funded account.

    Args:
        verbose: Print the last completed return per mode and account cash flows.

    Returns:
        Typed-result comparisons, one cash-relative result and a self-financed ledger.
    """
    prices = generate_synthetic_prices(
        start="2024-01-31",
        end="2024-04-30",
        seed=20260725,
        apply_quirks=False,
    )[["SEQ_EU", "SEQ_US"]].rename(columns={"SEQ_EU": "SYN_EUR", "SEQ_US": "SYN_CHF"})
    dates = prices.index
    ccys = pd.Series({"SYN_EUR": "EUR", "SYN_CHF": "CHF"})
    step = np.arange(len(dates))
    spots = pd.DataFrame(
        {"USD": 1.0, "EUR": 1.1 * np.exp(0.0003 * step), "CHF": 1.15 * np.exp(-0.0002 * step)},
        index=dates,
    )
    rates = pd.DataFrame({"USD": 0.05, "EUR": 0.03, "CHF": 0.01}, index=dates)
    fx = qis.FxRatesData(spots, rates)
    comparisons = {}
    for freq in ("B", "ME"):
        for coordinate in (qis.FxReturnCoordinate.SIMPLE, qis.FxReturnCoordinate.LOG):
            results = {}
            for kind in (
                qis.FxReturnType.LOCAL_TOTAL,
                qis.FxReturnType.LOCAL_EXCESS,
                qis.FxReturnType.REFERENCE_TOTAL,
                qis.FxReturnType.REFERENCE_EXCESS,
            ):
                reference = "USD" if kind.value.startswith("reference") else None
                results[kind.value] = qis.compute_fx_returns(
                    fx,
                    qis.FxAssetReturnsRequest(
                        prices,
                        ccys,
                        kind,
                        coordinate,
                        reference_ccy=reference,
                        frequency=freq,
                        cash_policy=qis.FxCashPolicy(quote_lag=1),
                    ),
                )[freq].values
            translated = qis.compute_fx_returns(
                fx,
                qis.FxAssetReturnsRequest(
                    prices,
                    ccys,
                    qis.FxReturnType.LOCAL_EXCESS_TRANSLATED,
                    qis.FxReturnCoordinate.SIMPLE,
                    reference_ccy="USD",
                    frequency=freq,
                ),
            )[freq]
            forward = qis.compute_fx_returns(
                fx,
                qis.FxCurrencyReturnsRequest(
                    ("EUR", "CHF"),
                    "USD",
                    qis.FxReturnType.FX_FORWARD,
                    frequency=freq,
                ),
            )[freq]
            forward_by_asset = pd.DataFrame({a: forward.values[c] for a, c in ccys.items()})
            results["translated_local_excess_pnl"] = translated.values
            results["fx_forward_pnl"] = forward_by_asset
            assert translated.synthetic_index is None and forward.synthetic_index is None

            # Independent endpoint prices, terminal spot and opening deposit cash flows.
            grid = translated.values.index
            p = prices.reindex(grid, method="ffill")
            s = pd.DataFrame({a: spots[c] for a, c in ccys.items()}).reindex(grid, method="ffill")
            local = p / p.shift(1) - 1
            spot_change = s / s.shift(1) - 1
            dt = grid.to_series().diff().dt.days / 365.0
            local_cash = pd.DataFrame({a: rates[c] for a, c in ccys.items()})
            local_cash = local_cash.reindex(grid, method="ffill").shift(1).mul(dt, axis=0)
            usd_cash = rates.USD.reindex(grid, method="ffill").shift(1) * dt
            usd = (1 + local) * (1 + spot_change) - 1
            if coordinate is qis.FxReturnCoordinate.SIMPLE:
                expected = [local, local - local_cash, usd, usd.sub(usd_cash, axis=0)]
            else:
                expected = [
                    np.log1p(local),
                    np.log1p(local) - np.log1p(local_cash),
                    np.log1p(usd),
                    np.log1p(usd).sub(np.log1p(usd_cash), axis=0),
                ]
            for actual, reference in zip(list(results.values())[:4], expected):
                np.testing.assert_allclose(actual, reference, atol=1e-12, equal_nan=True)
            np.testing.assert_allclose(
                translated.values,
                (local - local_cash) * (1 + spot_change),
                atol=1e-12,
                equal_nan=True,
            )
            forward_ratio = (1 + local_cash).rdiv(1 + usd_cash, axis=0)
            np.testing.assert_allclose(
                forward_by_asset, 1 + spot_change - forward_ratio, atol=1e-12, equal_nan=True
            )
            comparison = pd.concat(results, axis=1)
            comparisons[freq, coordinate.value] = comparison
            if verbose:
                print(
                    f"\n{freq} / {coordinate.value}: four returns; last two kinds ALWAYS SIMPLE P&L"
                )
                print(comparison.iloc[-1].unstack(0).round(6).to_string())

    cash_relative = qis.compute_fx_returns(
        fx,
        qis.FxAssetReturnsRequest(
            prices,
            ccys,
            qis.FxReturnType.REFERENCE_EXCESS,
            qis.FxReturnCoordinate.CASH_RELATIVE_SIMPLE,
            reference_ccy="USD",
            frequency="ME",
        ),
    )["ME"]
    log_excess = comparisons["ME", "log"]["reference_excess"]
    np.testing.assert_allclose(
        cash_relative.values, np.expm1(log_excess), atol=1e-12, equal_nan=True
    )
    normalization = qis.FxIndexNormalization(
        100.0, 1.0, "capital_proportional", "ME", "each_interval", "as_returned", "stop"
    )
    normalized = forward.to_index(normalization)
    assert normalized.iloc[0].eq(100.0).all()  # Synthetic forward index; no collateral added.

    trade_dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-04-30"])
    targets = pd.DataFrame(
        [[0.7, 0.3], [0.4, 0.6], [0.0, 0.0]], index=trade_dates, columns=prices.columns
    )
    contracts = pd.DataFrame(
        [
            [
                trade_dates[0],
                trade_dates[1],
                pd.Timestamp("2024-03-04"),
                "EUR",
                -1.0,
                "asset_currency_value",
            ],
            [trade_dates[1], trade_dates[2], trade_dates[2], "EUR", -1.0, "asset_currency_value"],
        ],
        columns=[
            "trade_date",
            "fixing_date",
            "settlement_date",
            "currency",
            "notional_weight",
            "notional_basis",
        ],
        index=["initial_hedge", "reset_hedge"],
    )
    account = qis.backtest_fx_account(
        prices,
        ccys,
        targets,
        fx,
        spec=qis.FxAccountSpec("USD", 100.0, "reference", "fixed", "reference"),
        local_settlement_dates=pd.DatetimeIndex([]),
        spot_is_observed=spots.notna(),
        forwards=contracts,
    )
    assert account.contracts.paid.all() and account.asset_units.iloc[-1].eq(0).all()
    assert account.accounts.forward_value.iloc[-1] == 0
    np.testing.assert_allclose(
        account.nav.iloc[-1] - 100.0, account.pnl.to_numpy().sum(), atol=1e-10
    )
    np.testing.assert_allclose(
        account.accounts.cash, 100.0 + account.cash_flows.sum(axis=1).cumsum(), atol=1e-10
    )
    if verbose:
        print(
            "\nAccount: funded units, explicit EUR hedge reset, fixing then delayed cash payment."
        )
        print(
            account.accounts.loc[pd.to_datetime(["2024-02-29", "2024-03-04", "2024-04-30"])]
            .round(6)
            .to_string()
        )
        print(
            "Reporting resampling preserves this ledger; changing request frequency changes resets."
        )
    return {
        "comparisons": comparisons,
        "cash_relative": cash_relative,
        "normalized_forward_index": normalized,
        "account": account,
    }


class Locals(Enum):
    """Offline example selection."""

    RETURNS_AND_ACCOUNT = "returns_and_account"


def run_local(local: Locals):
    """Run the selected offline teaching workflow."""
    if local is not Locals.RETURNS_AND_ACCOUNT:
        raise ValueError(local)
    return run_example()


if __name__ == "__main__":
    run_local(local=Locals.RETURNS_AND_ACCOUNT)
