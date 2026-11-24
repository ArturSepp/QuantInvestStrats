"""Two synthetic FX liquidity reconciliations. Run python -m examples.fx.fx_collateral_liquidity."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from qis.fx.fx_rates_data import FxRatesData
from qis.portfolio.fx_accounting import FxAccountSpec, backtest_fx_account
from qis.portfolio.fx_liquidity import compute_fx_liquidity


def _account(capital, spot, targets, notional_basis):
    """Value a short EUR forward at zero rates on five daily observations."""
    dates = targets.index.union(spot.index)
    assets = pd.DataFrame(100.0, index=dates, columns=targets.columns)
    ccys = pd.Series("EUR", index=assets.columns, dtype=object)
    spots = pd.DataFrame({"EUR": spot, "USD": 1.0}, index=dates)
    fx = FxRatesData(spots, pd.DataFrame(0.0, index=dates, columns=spots.columns))
    forwards = pd.DataFrame(
        [
            dict(
                trade_date=dates[0],
                fixing_date=dates[2],
                settlement_date=dates[-1],
                currency="EUR",
                notional_weight=-1.0,
                notional_basis=notional_basis,
            )
        ],
        index=["EUR hedge"],
    )
    return backtest_fx_account(
        assets,
        ccys,
        targets,
        fx,
        spec=FxAccountSpec("USD", capital, "reference", "fixed", "zero"),
        local_settlement_dates=pd.DatetimeIndex([]),
        spot_is_observed=spots.notna(),
        forwards=forwards,
    )


def _daily(result):
    """Display USD daily closing cash and NAV beside the largest intraday shortfall."""
    rows = result.reconciliation.xs(("USD", "Treasury"), level=["currency", "agreement"])
    days = rows.groupby(rows.index.normalize())
    frame = days.agg(
        opening_free_cash=("opening_free_cash", "first"),
        equity_trades=("equity_trades", "sum"),
        initial_margin=("initial_margin", "sum"),
        settled_vm=("settled_vm", "sum"),
        gross_settlement=("gross_settlement", "sum"),
        net_settlement=("net_settlement", "sum"),
        restricted_cash=("restricted_cash", "last"),
        pending_ledger_cash=("pending_ledger_cash", "last"),
        closing_free_cash=("closing_free_cash", "last"),
        shortfall=("shortfall", "max"),
    )
    # An opening deficit persists until the first flow, even if that flow cures it.
    frame["shortfall"] = frame.shortfall.where(
        frame.shortfall >= -frame.opening_free_cash, -frame.opening_free_cash
    )
    return frame.join(result.valuation[["forward_value", "nav"]])


def run_example(verbose=True):
    """Return the independently reproducible daily illustrations and cutoff analytics."""
    dates = pd.date_range("2025-02-03", periods=5)
    empty = pd.DataFrame(index=dates[:1])
    margin_book = _account(50_000.0, pd.Series(1.0, index=dates), empty, "capital")
    collateral = 0.05 * abs(margin_book.contracts.loc["EUR hedge", "reference_notional"])
    flows = pd.DataFrame(
        [
            (dates[1] + pd.Timedelta(hours=10), "USD", "Treasury", "initial_margin", collateral),
            (dates[-1] + pd.Timedelta(hours=16), "USD", "Treasury", "initial_margin", -collateral),
        ],
        columns=["at", "currency", "agreement", "kind", "amount"],
    )
    margin = compute_fx_liquidity(margin_book, flows)

    # Buy USD 8m of EUR equity, keeping USD 750k cash. Sell the EUR principal forward.
    # Sell on the call date. The cash settles two days later.
    capital = 8_750_000.0
    targets = pd.DataFrame({"EUR equity": 8_000_000.0 / capital}, index=dates[[0, 1]])
    book = _account(
        capital,
        pd.Series([1.0, 1.15, 1.15, 1.15, 1.15], index=dates),
        targets,
        "asset_currency_value",
    )
    vm = pd.DataFrame(
        [
            dict(
                at=dates[1] + pd.Timedelta(hours=10),
                currency="USD",
                agreement="Treasury",
                kind="settled_vm",
                amount=-1_200_000.0,
                contract_id="EUR hedge",
            )
        ]
    )
    timings = pd.DataFrame(
        {"equity_trades": dates[[0, 3]] + pd.Timedelta(hours=15)}, index=dates[[0, 1]]
    )
    hedge = compute_fx_liquidity(book, vm, ledger_flow_times=timings)
    cases = {"initial_margin": (_daily(margin), margin), "hedge_cash_call": (_daily(hedge), hedge)}
    if verbose:
        for name, (daily, result) in cases.items():
            print(f"\n{name} | USD | amounts in currency units")
            print(daily.round(2).to_string())
            print(result.summary.to_string())
    return cases


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Write the daily, cutoff and summary tables."
    )
    directory = parser.parse_args().output_dir
    examples = run_example()
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        for case, (daily, result) in examples.items():
            daily.to_csv(directory / f"{case}_daily.csv")
            result.reconciliation.to_csv(directory / f"{case}_cutoffs.csv")
            result.summary.to_csv(directory / f"{case}_summary.csv")
