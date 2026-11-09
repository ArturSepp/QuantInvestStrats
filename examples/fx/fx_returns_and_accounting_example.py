"""Offline EUR/CHF return coordinates, exact notional payoffs and settled FX accounting.

Uses the typed FX, monthly cash and accounting APIs. Frozen synthetic
asset paths are assigned illustrative quote currencies; spots/rates are teaching inputs.
No downloads, file outputs, optional dependencies or production allocations are used.
"""

from enum import Enum

import numpy as np
import pandas as pd
import qis
from qis.datasets.synthetic import generate_synthetic_prices


def run_example(verbose: bool = True) -> dict:
    """Compute four return views, common monthly cash and explicit funded/signed accounts.

    Args:
        verbose: Print the last completed return per mode and account cash flows.

    Returns:
        Typed comparisons, cash-relative/index results and reference/local-financed ledgers.
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
                        prices=prices,
                        local_ccys=ccys,
                        return_type=kind,
                        coordinate=coordinate,
                        reference_ccy=reference,
                        frequency=freq,
                        cash_policy=qis.FxCashPolicy(quote_lag=1),
                    ),
                )[freq].values
            translated = qis.compute_fx_returns(
                fx,
                qis.FxAssetReturnsRequest(
                    prices=prices,
                    local_ccys=ccys,
                    return_type=qis.FxReturnType.LOCAL_EXCESS_TRANSLATED,
                    coordinate=qis.FxReturnCoordinate.SIMPLE,
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
            prices=prices,
            local_ccys=ccys,
            return_type=qis.FxReturnType.REFERENCE_EXCESS,
            coordinate=qis.FxReturnCoordinate.CASH_RELATIVE_SIMPLE,
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
            "Reporting resampling preserves this ledger; "
            "return frequency sets analytical hedge resets."
        )
    monthly_cash = run_monthly_cash_example(verbose=verbose)
    local_accounts = run_local_financing_example(verbose=verbose)
    return {
        "monthly_cash": monthly_cash,
        "local_accounts": local_accounts,
        "comparisons": comparisons,
        "cash_relative": cash_relative,
        "normalized_forward_index": normalized,
        "account": account,
    }


def run_monthly_cash_example(verbose: bool = True) -> dict:
    """Measure one unhedged wealth path against the same monthly-reset cash account."""
    dates = pd.date_range("2023-12-31", "2024-03-31", freq="D")
    prices = generate_synthetic_prices(
        start="2023-12-29", end="2024-03-31", apply_quirks=False)[["SEQ_EU"]]
    prices = prices.reindex(dates, method="ffill")
    rates = pd.DataFrame({"USD": .02, "EUR": .05}, index=dates)
    rates.loc["2024-01-31":, "USD"] = .08
    rates.loc["2024-02-29":, "USD"] = .03
    spots = pd.DataFrame({"USD": 1., "EUR": 1.1 * np.exp(.0002 * np.arange(len(dates)))},
                         index=dates)
    fx = qis.FxRatesData(spots, rates)
    policy = qis.FxCashPolicy(accrual=qis.FxCashAccrual.MONTHLY_INDEX)
    measured = {}
    for frequency in ("D", "ME", "QE"):
        request = qis.FxAssetReturnsRequest(
            prices=prices, local_ccys="EUR", reference_ccy="USD",
            return_type=qis.FxReturnType.REFERENCE_EXCESS,
            coordinate=qis.FxReturnCoordinate.LOG, frequency=frequency, cash_policy=policy)
        measured[frequency] = qis.compute_fx_returns(fx, request)[frequency].values
    cash = qis.compute_monthly_cash_index(fx, "USD", dates)
    # Independent opening deposits: 31 January days, 29 February days, 31 March days.
    expected_cash = (1 + .02 * 31/365) * (1 + .08 * 29/365) * (1 + .03 * 31/365)
    np.testing.assert_allclose(cash.iloc[-1], expected_cash, rtol=0, atol=1e-14)
    expected_excess = np.log(
        (prices.iloc[-1, 0] * spots.EUR.iloc[-1]) /
        (prices.iloc[0, 0] * spots.EUR.iloc[0])) - np.log(expected_cash)
    np.testing.assert_allclose(
        [measured["D"].iloc[1:, 0].sum(), measured["ME"].iloc[1:, 0].sum(),
         measured["QE"].iloc[-1, 0]], expected_excess, rtol=0, atol=1e-14)
    opening_request = qis.FxAssetReturnsRequest(
        prices=prices, local_ccys="EUR", reference_ccy="USD",
        return_type=qis.FxReturnType.REFERENCE_EXCESS,
        coordinate=qis.FxReturnCoordinate.LOG, frequency="QE", cash_policy=qis.FxCashPolicy())
    opening = qis.compute_fx_returns(fx, opening_request)["QE"].values
    comparison = pd.Series({
        "daily log excess sum": measured["D"].iloc[1:, 0].sum(),
        "monthly log excess sum": measured["ME"].iloc[1:, 0].sum(),
        "quarterly monthly-index log excess": measured["QE"].iloc[-1, 0],
        "quarterly period-opening log excess": opening.iloc[-1, 0],
    })
    assert abs(comparison.iloc[-1] - comparison.iloc[-2]) > 1e-4
    if verbose:
        print("\nSame unhedged wealth path, changing rates; decimal log excess:")
        print(comparison.round(8).to_string())
    return {"returns": measured, "cash_index": cash, "comparison": comparison}


def run_local_financing_example(verbose: bool = True) -> dict:
    """Check zero/nonzero-net books against two independent local debt settlements."""
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-04-30"])
    prices = generate_synthetic_prices(
        start="2024-01-31", end="2024-04-30", apply_quirks=False)[["SEQ_EU", "SEQ_US"]]
    prices = prices.reindex(dates)
    ccys = pd.Series({"SEQ_EU": "EUR", "SEQ_US": "CHF"})
    spots = pd.DataFrame({"USD": 1., "EUR": [1.1, 1.15, 1.12],
                          "CHF": [1.15, 1.12, 1.14]}, index=dates)
    rates = pd.DataFrame({"USD": .05, "EUR": .03, "CHF": .01}, index=dates)
    fx = qis.FxRatesData(spots, rates)
    results = {}
    for label, weights in (("zero_net", [.5, -.5]), ("nonzero_net", [.7, -.3])):
        targets = pd.DataFrame([weights, [0., 0.]], index=dates[[0, 2]], columns=prices.columns)
        account = qis.backtest_fx_account(
            prices, ccys, targets, fx,
            spec=qis.FxAccountSpec("USD", 100., "local", "fixed", "reference"),
            local_settlement_dates=dates[1:], spot_is_observed=spots.notna())
        # Independent two-period cash flows, not another general ledger/backtester.
        s = spots[list(ccys)].to_numpy()
        p = prices.to_numpy()
        r = rates[list(ccys)].iloc[0].to_numpy()
        dt1, dt2 = np.diff(dates).astype("timedelta64[D]").astype(float) / 365.
        units = 100. * np.asarray(weights) / (p[0] * s[0])
        debt0 = -units * p[0]
        sweep1 = ((units * p[1] + debt0 * (1 + r * dt1)) * s[1]).sum()
        debt1 = -units * p[1]
        sweep2 = ((units * p[2] + debt1 * (1 + r * dt2)) * s[2]).sum()
        cash1 = 100. * (1 + .05 * dt1) + sweep1
        cash2 = cash1 * (1 + .05 * dt2) + sweep2
        np.testing.assert_allclose(account.nav, [100., cash1, cash2], rtol=0, atol=1e-10)
        np.testing.assert_allclose(account.local_balances.iloc[-1], 0., rtol=0, atol=1e-10)
        np.testing.assert_allclose(account.pnl.to_numpy().sum(), cash2 - 100., atol=1e-10)
        assert account.asset_units.iloc[-1].eq(0.).all()
        results[label] = account
    if verbose:
        print("\nLocal financing, fixed USD capital 100, explicit intermediate/final sweeps:")
        print(pd.DataFrame({k: v.nav for k, v in results.items()}).round(6).to_string())
    return results


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
