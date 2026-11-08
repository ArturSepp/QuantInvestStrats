---
myst:
  html_meta:
    description: >-
      Explicit self-financed FX accounting for signed equities, local debt, collateral cash,
      principal hedge resets, contract fixing and delayed settlement.
---

# Self-financed FX accounting

*Author: [Artur Sepp](https://github.com/ArturSepp)*

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
zero rates and one held unit, settling each increment gives 10 x 2 - 10 x 1 = 10 reference
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

## Implementation in qis

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

Current-NAV sizing may use marked receivables while cash remains negative: this model permits
borrowing. It is not a liquidity-constrained or segregated-collateral simulator. Exact internal
cash reconciliation does not verify instrument tradability, synthetic forward pricing against
market quotes or the predictive merit of any allocation. Trading costs and an account-specific
margin/credit contract require additional, explicitly validated capabilities.

## See also

- [Portfolio backtesting](portfolio_backtesting.md)
- [Return conventions and FX adjustments](return_conventions_and_fx_adjustments.md)
- [Cash timing and FX adjustments](cash_rate_timing_and_fx_adjustments.md)
- [Reference-currency excess returns](reference_currency_excess_returns.md)

## References

1. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
