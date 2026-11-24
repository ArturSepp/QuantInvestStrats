---
myst:
  html_meta:
    description: >-
      Explicit self-financed FX accounting for signed equities, local debt, collateral cash,
      principal hedge resets, contract fixing and delayed settlement.
---

# Self-financed FX accounting

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-11-08](https://github.com/ArturSepp/QuantInvestStrats/commit/fae104880a01c21f3882fe8af3a6796cd669b96d)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A self-financed account pays for every trade and settlement through its own balances.
Zero net target exposure does not mean zero capital, zero financing balances or zero
marked exposure between rebalances. FX contracts additionally distinguish their marked
value, fixing payoff and eventual cash payment.

## Overview

`backtest_fx_account` extends QIS accounting for local-currency borrowing/deposits and
zero-cost FX contracts. It keeps equity units, local financing, spendable reference cash
and unsettled forward values separately. `FxAccountSpec` declares the financing and
notional policy; `FxAccountData` retains the complete monetary ledger.

The ordinary `backtest_model_portfolio` remains the funded unit engine. Its instrument
carry convention is not an exact general replacement for an unfunded payoff plus collateral:
carry is based on closing instrument value, whereas account interest uses opening cash.
The new account does not change that engine's behaviour or defaults.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Funded local total-return levels; signed monetary P&L, not logged zero-cost returns |
| Sampling grid | Explicit daily boundaries; each interval is a cash reset; event dates are supplied |
| Annualisation | Annual decimal rates accrue actual calendar days divided by 365 |
| Mean adjustment | None |
| Timing | Opening balance interest, marks/fixing/payment, local sweep, equity trades, new forwards |
| Output units | Reference-currency capital/P&L; equity units and local-currency financing balances |
| qis default | Financing, notional and cash-yield policies are mandatory; missing quotes raise |

Let u be held equity units, P local funded price, S reference currency per local unit,
L the signed local financing balance, C spendable reference cash and M forward/receivable
value. Initial capital is positive. Target equity weights specify signed opening notionals
relative to fixed initial capital or current marked NAV. Units then drift until a target date.
A locally financed long equity has negative L; a short has a positive local deposit.

The local/reference benchmarks apply symmetrically to borrowing and lending. Reference
cash is unrestricted and can be negative; the cash-yield policy selects its benchmark or
zero interest explicitly. There are no external contributions, fees, transaction costs,
short-borrow charges, cross-currency basis, segregated margin, haircuts or credit limits.
These are declared modelling assumptions, not estimates of execution costs or dealer terms.
Local total-return levels already include the supplied distributions and embedded hedges.
Never supply an excess-return index and subtract cash from it again.

Quote currencies are static explicit IDs. Resolve subunits, ADRs and currency spells upstream.
The mandatory `spot_is_observed` mask supplies original quote availability because
`FxRatesData` causally fills spots during construction. Optional carry is for valuation only;
it cannot authorise a trade, fixing or financing sweep. Rate quotes align as of the opening
boundary without future filling. A source with no availability provenance cannot establish
that its filled marks were executable.

## Methodology

### Cash and local financing

**Identity.** At each marked boundary the account value is:

$$
V_t=C_t+\sum_i (u_{i,t}P_{i,t}+L_{i,t})S_{i,t}+\sum_j M_{j,t}.
$$

**Proof.** Each term is a separately owned asset or liability expressed in reference currency.
Their sum is equity capital; a cash transfer paired with an opposite asset/liability change
leaves it unchanged. $\square$

Opening balances earn the opening annual rate times elapsed calendar days/365; the resulting
balance is carried into the next interval. Under local financing, an equity trade changes L by
minus the local purchase amount. It creates no external reference cash contribution. On a
specified local settlement date, transfer the marked asset-plus-local-balance value into C
and reset L to minus the current held local asset value. Keep units u unchanged until an equity
trade is explicitly scheduled. Under reference financing, equity purchases instead debit C.

Local settlement is a strategy decision. With unit equity prices 100, 110, 100 and FX 1, 2, 1,
zero rates and one held unit, settling each increment gives $10 \times 2 - 10 \times 1 = 10$ reference
P&L; endpoint settlement gives zero. Resampling a completed account must not silently replace
its financing/settlement calendar.

### Forward strike, fixing and payment

Each schedule row names trade date, fixing date, settlement date, currency, signed notional
weight and sizing basis. A positive quantity buys local currency forward; a negative quantity
hedges it. Principal-hedge sizing uses net marked equity exposure in that currency after
same-date equity trades. Capital sizing uses the declared fixed/current-NAV rule. Hold the
local quantity q and strike K unchanged until fixing. A roll is another explicit row.

Define a flat simple discount D for a supplied current annual rate and remaining ACT/365F
year fraction. The synthetic fair forward to fixing T is:

$$
F_{t,T}=S_t\frac{D_L(t,T)}{D_R(t,T)}.
$$

At trade set K to this value, giving zero initial contract value. Before fixing, mark the
contract at q times the difference between the current fair forward and K, discounted in
reference currency to the separately specified payment date. This is a synthetic flat-curve
contract: no observed forward curve or basis is inferred from short-rate quotes.

At fixing, crystallise the nominal reference payoff:

$$
\Pi_T=q(S_T-K).
$$

Keep that amount as a discounted reference receivable until payment. It receives no
spendable-cash interest before settlement; changing discount value remains forward P&L.
At payment add the nominal amount to C and remove the receivable exactly once. Contracts
may remain open beyond the final reporting date; the account then retains their marked value.
This explicitly defined cash-settled contract is not a claim to replicate every dealer NDF,
exchange futures or collateral agreement.

### Reconciliation and capital

**Identity.** NAV change equals translated asset-price P&L, local financing interest and FX
translation, forward value change plus payments, and opening reference-cash interest.

**Proof.** Difference the marked balance identity. Equity purchases and financing sweeps
cancel between balances. Forward payment cancels the removed receivable. The remaining
terms are the reported P&L legs. $\square$

The implementation checks this identity and event-time conservation at every date. Monetary
losses can exceed initial capital. Such a result remains a valid signed ledger; a current-NAV
reset cannot size a trade from nonpositive capital, and funded return reporting then raises.
Zero net exposure never creates a return on zero capital by dividing through a net notional.

## Worked example

A pure forward account has initial USD capital 100, a 100 USD opening EUR notional, EURUSD
moving from 1 to 1.1 over 29 days, and identical 6% local/reference rates. Its strike is 1,
forward P&L is 10, and collateral earns 100 x 0.06 x 29/365. Final capital is 110.476712.
No dummy funded asset is needed:

```python
import numpy as np
import pandas as pd
import qis

dates = pd.to_datetime(['2024-01-31', '2024-02-29'])
raw_spots = pd.DataFrame({'USD': 1., 'EUR': [1., 1.1]}, index=dates)
fx = qis.FxRatesData(raw_spots, pd.DataFrame({'USD': .06, 'EUR': .06}, index=dates))
empty_assets = pd.DataFrame(index=dates)
contracts = pd.DataFrame([[dates[0], dates[1], dates[1], 'EUR', 1., 'capital']],
    columns=['trade_date', 'fixing_date', 'settlement_date', 'currency',
             'notional_weight', 'notional_basis'])
policy = qis.FxAccountSpec('USD', 100., 'local', 'fixed', 'reference')
result = qis.backtest_fx_account(
    empty_assets, pd.Series(dtype=object), empty_assets.iloc[:1], fx,
    spec=policy, local_settlement_dates=pd.DatetimeIndex([]),
    spot_is_observed=raw_spots.notna(), forwards=contracts,
)
expected = 110. + 100.*.06*29/365
assert np.isclose(result.nav.iloc[-1], expected, rtol=0, atol=1e-12)
assert np.isclose(result.pnl.to_numpy().sum(), expected-100., rtol=0, atol=1e-12)
assert result.contracts.paid.all()
assert result.accounts.forward_value.iloc[-1] == 0
```

The old funded-overlay carry recipe gives 110.524384 because it applies the carry to closing
value 110. The excess is 4.7671 bp of initial capital, even with constant rates. Independent
cash-flow tests also cover signed equity books, full and partial principal hedges, delayed
payment, capital resizing, negative capital, stale observations and non-USD reference accounts.

### Cash available at a hedge payment cutoff

Use `compute_fx_liquidity` from `qis.portfolio.fx_liquidity` to ask whether the
hedge can pay in the required currency at the required time. Start with the
marked `FxAccountData` and supply the operational schedule. The output has one
row per cutoff, currency and netting agreement. A negative closing free-cash
balance gives the funding needed to meet the obligations due at that cutoff.

| USD illustration | Opening resources | At the cutoff | Subsequent cash |
|---|---|---|---|
| Initial margin | USD 50,000 cash against USD 50,000 forward notional | Posting 5% moves USD 2,500 into restricted cash. Free cash is USD 47,500 and marked NAV is USD 50,000. | Release restores free cash to USD 50,000. Trading P&L is zero throughout the flat-market example. |
| EUR equity with a short EUR forward | USD 8m invested at EURUSD 1.00 and USD 750,000 free cash | EURUSD rises to 1.15. USD 1.20m of settled variation margin is due at 10:00 on 4 February. The USD funding gap is **450,000**. | The equity sale on 4 February delivers USD 1.20m at 15:00 on 6 February. The 7 February forward maturity has zero remaining payment. |

The second illustration keeps daily marked NAV at USD 8.75m. The USD 1.20m
equity gain offsets the hedge loss in NAV, but the sale receivable cannot pay
the call before its value date. Comparing the exact cutoff with the subsequent
value date makes that distinction observable.

```python
from examples.fx.fx_collateral_liquidity import run_example

cases = run_example(verbose=False)
daily_im, initial = cases['initial_margin']
daily_fx, hedge = cases['hedge_cash_call']
np.testing.assert_allclose(daily_im.restricted_cash, [0, 2500, 2500, 2500, 0])
np.testing.assert_allclose(daily_im.closing_free_cash, [50000, 47500, 47500, 47500, 50000])
np.testing.assert_allclose(daily_im.nav, 50000)
np.testing.assert_allclose(daily_fx.closing_free_cash, [750000, -450000, -450000, 750000, 750000])
np.testing.assert_allclose(daily_fx.pending_ledger_cash, [0, 1200000, 1200000, 0, 0])
np.testing.assert_allclose(daily_fx.net_settlement, 0, atol=1e-8)
np.testing.assert_allclose(daily_fx.nav, 8750000)
assert hedge.summary.loc[('USD', 'Treasury'), 'binding_cutoff'] == pd.Timestamp('2025-02-04 10:00')
```

Run `python -m examples.fx.fx_collateral_liquidity` to display the daily cash
reconciliation beside the original forward mark and NAV. The optional
`--output-dir <directory>` writes the daily, cutoff and agreement-summary tables.
[Runnable example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/fx/fx_collateral_liquidity.py).

#### Supplying operational terms

`compute_fx_liquidity(account, flows, ...)` takes a long table of movements with
`at`, `currency`, `agreement`, `kind` and `amount`. Use the same timezone-naive
clock for all payment locations. A flow becomes usable at its timestamp.
Simultaneous amounts net within their currency and agreement. Use
`payment_cutoffs` to inspect additional times with no movement.

| Flow kind | Sign and treatment |
|---|---|
| `initial_margin`, `collateralised_vm` | Positive posts, negative releases. Each balance must cover its own releases. Posting increases pledged securities or restricted cash. The derivative still pays its full contractual settlement. |
| `settled_vm` | Positive is cash received, negative is paid. Supply the ledger `contract_id`. The gross maturity payoff less the signed VM already settled is the remaining payment for that contract. |
| `received_vm` | Positive is cash collateral received from the counterparty, negative returns it. The receipt stays segregated and unavailable for outgoing cash payments. |
| `funding`, `cash_flow` | Signed external cash, including funding actually drawn and repayments. Enter only movements additional to those already imported from the ledger. |

`opening_cash` allocates starting balances by currency and agreement. Its
reference-currency balances must sum to ledger initial capital. Other currency
balances supplement the liquidity analysis. `contract_terms` assigns each
executed contract to an agreement and optionally gives its `settlement_at`
time on the contractual settlement date. By default reference cash and forward
receipts belong to `Treasury`, and ledger cash movements arrive at day-end.
`ledger_flow_times` can supply usable timestamps for the ledger's daily equity
trades, local settlements and cash interest, including a later value date.
A positive `pending_ledger_cash` is booked in ledger cash and still unavailable
operationally. Contract marks appear only at valuation closes.

Supply `collateral` as dedicated inventory snapshots with time, currency,
agreement, market `value` expressed in the payment currency, `haircut` and
boolean `eligible`. Each snapshot replaces the full inventory for that bucket.
Credit equals the sum of eligible values times one minus their haircuts.
The model applies available credit to posted collateral first and takes the
remainder from free cash. A change of capacity assumes substitution at the
specified cutoff. Supply a zero-value snapshot to withdraw the inventory.
Unused credit can meet margin but cannot pay a cash settlement. Assign each lot
to one bucket and exclude collateral already pledged elsewhere.

Read `reconciliation` for opening and closing free cash, signed margin movements,
restricted cash, pledged credit, the gross and net forward payments and shortfall.
`summary` reports minimum free cash, peak shortfall and its first binding cutoff
by currency and agreement. `currency_summary` sums simultaneous agreement
deficits before finding each currency's peak. Surplus in a separate agreement
does not offset another agreement's call. `valuation` carries the underlying
ledger cash, forward mark and NAV in reference units. The overlay preserves the
ledger's interest basis. Supply incremental funding costs or collateral
remuneration as dated cash flows under the actual agreement.
[Calculation and input contract](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/fx_liquidity.py).

## Implementation in qis

[Account API](api/generated/qis.backtest_fx_account.rst), [account policy](api/generated/qis.FxAccountSpec.rst) and [account result](api/generated/qis.FxAccountData.rst).

Use `FxAccountSpec`, `backtest_fx_account` and `FxAccountData`. The source is
[portfolio/fx_accounting.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/fx_accounting.py).
Input prices and targets have an exact labelled universe, with an explicit opening target.
A zero target closes units only on an observed execution date. `local_settlement_dates` is
required even when empty. Forward dates inside the sample must occur on its valuation grid;
future fixing/payment dates are allowed so historical prefixes need no future prices.

The result provides `accounts`, `pnl`, `cash_flows`, `asset_units`, `local_balances`,
`contracts`, `contract_values`, `diagnostics` and `provenance`, each returned as an owned copy.
`to_portfolio_data()` exports positive NAV for normal QIS reporting. It adds no cash or
funding and does not disguise the monetary contract ledger as funded instrument prices.
The three entry points are additive; the existing backtester and typed FX return API retain
their meanings. Consumer applications provide data and schedules, not another P&L recursion.

## Interpretation and limitations

> **Insight.** A paid payoff is cash; an unpaid payoff is a receivable. Both contribute to
> NAV, but only the former changes the balance used for cash interest.

> **Pitfall.** A zero-sum target can develop nonzero marked exposure as assets and currencies
> move. Resetting units or debt at every observation changes the strategy unless scheduled.

The valuation grid also controls cash compounding. Daily accrual and one monthly simple
accrual can differ even under the same annual quotes. Separate reporting frequency from the
account's reset calendar; do not demand equality between different contracts.

Current-NAV sizing may use marked receivables while cash remains negative because
the valuation account permits borrowing. The cutoff analysis measures the resulting
funding need. Execution constraints, credit approval and trading decisions remain
with the caller. The caller supplies margin terms, eligible assets and cash availability
under the actual agreement.

## See also

- [Portfolio backtesting](portfolio_backtesting.md)
- [Return conventions and FX adjustments](return_conventions_and_fx_adjustments.md)
- [Cash timing and FX adjustments](cash_rate_timing_and_fx_adjustments.md)
- [Reference-currency excess returns](reference_currency_excess_returns.md)

## References

1. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
