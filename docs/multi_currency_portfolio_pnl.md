---
myst:
  html_meta:
    description: >-
      Compute local and USD total and excess returns, FX hedge payoffs and self-financed
      multi-currency portfolio P&L with the offline qis example.
---

# Multi-currency portfolio P&L in Python

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Use QIS to compare an asset's local and reference-currency returns, then account for a
portfolio's positions, cash and FX hedges. The offline workflow below shows local, excess
local, USD and excess USD returns together, followed by an account with an explicit hedge
reset and delayed settlement. All prices, spots and rates are synthetic teaching inputs.

## Choose the calculation

| Task | Public entry point | Result |
|---|---|---|
| Supply currency spots and domestic rates | `FxRatesData` | Provider-neutral input panels with currency IDs such as USD, EUR and CHF. |
| Convert asset returns or compute FX payoffs | `compute_fx_returns` | Typed panels selected by return kind, coordinate, reference currency and frequency. |
| Track positions, financing and settled P&L | `backtest_fx_account` | `FxAccountData`: monetary capital, cash flows, held units, financing balances and FX contracts. |

A converted return panel describes intervals. The account additionally records how trades
are funded, how many units remain held, and when a fixed hedge payoff becomes spendable cash.
The [return conventions](return_conventions_and_fx_adjustments.md) and
[self-financed accounting](self_financed_fx_accounting.md) chapters define these contracts.

## Run the existing offline example

Start with an [installed QIS core package](install.md). From a repository checkout, run:

```console
python examples/market_data/fx_returns_and_accounting_example.py
```

With only the installed package, save the
[complete example source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/market_data/fx_returns_and_accounting_example.py)
as `fx_returns_and_accounting_example.py` and run `python fx_returns_and_accounting_example.py`.
The same source is included at the end of this page. Execution requires no provider access,
credentials, optional extras or local data files, and writes no output files.

The example uses the frozen synthetic-price generator with seed 20260725 over
31 January–30 April 2024. Two generated asset paths are assigned the illustrative quote
currencies EUR and CHF; they do not represent observed European or Swiss investments.

| Input | Example convention |
|---|---|
| Asset prices | Local total-return levels, including reinvested distributions. |
| Currency IDs | `SYN_EUR` maps to EUR; `SYN_CHF` maps to CHF; the reporting currency is USD. |
| Spots | USD per currency unit; the USD column is one. |
| Rates | Annual decimal quotes: USD 0.05, EUR 0.03 and CHF 0.01. |
| Cash accrual | Opening-known rates, one-quote lag, actual calendar days divided by 365. |
| Analytical frequency | `B` for business-day intervals and `ME` for month-end intervals. |
| Additional hedging | The four asset-return views are unhedged; the later account has an explicit EUR hedge. |

Keep the same currency IDs when supplying spots and rates from another provider. Normalise
spot direction and annual rate units before constructing `FxRatesData`, and align asset/currency
metadata to the prices. Currency identifies the price quote, not the issuer's domicile.
The [FX data contract](fx_hedging_and_market_data.md) covers input alignment and availability.

## Read the four return views

`FxAssetReturnsRequest` selects the economic quantity through `FxReturnType` and its coordinate
through `FxReturnCoordinate`. `compute_fx_returns` returns one `FxReturnsData` per requested
frequency; `.values` is its return or payoff DataFrame.

| Return kind | Interpretation in this example |
|---|---|
| `LOCAL_TOTAL` | Total return of the asset in its quote currency. |
| `LOCAL_EXCESS` | Local return relative to the local cash leg, in the selected coordinate. |
| `REFERENCE_TOTAL` | Total return after translating the asset into USD. |
| `REFERENCE_EXCESS` | USD return relative to the USD cash leg, in the selected coordinate. |

In `SIMPLE`, excess means total return minus the interval cash return. In `LOG`, excess means
log asset growth minus log cash growth. Applying a logarithm to one plus SIMPLE excess does
not generally produce LOG excess. The separate `CASH_RELATIVE_SIMPLE` coordinate is the
cash-relative wealth return; the example checks its relation to LOG excess.

From a checkout, the existing function also returns its results for further analysis:

```python
from examples.market_data.fx_returns_and_accounting_example import run_example

workflow = run_example(verbose=False)
return_kinds = ["local_total", "local_excess", "reference_total", "reference_excess"]
monthly = workflow["comparisons"]["ME", "simple"]
last_month = monthly.iloc[-1].unstack(0)[return_kinds]
print((100 * last_month).round(4).to_string())

# Checked rounded values behind the table below; decimals, before percentage display.
assert last_month.loc["SYN_EUR"].round(6).tolist() == [0.019533, 0.017067, 0.026284, 0.022174]
assert last_month.loc["SYN_CHF"].round(6).tolist() == [-0.009996, -0.010818, -0.014343, -0.018452]
```

The last completed monthly interval ends on 30 April 2024. These are interval returns in
percent, without annualisation:

| Synthetic asset | Local | Excess local | USD | Excess USD |
|---|---:|---:|---:|---:|
| SYN_EUR | 1.9533% | 1.7067% | 2.6284% | 2.2174% |
| SYN_CHF | -0.9996% | -1.0818% | -1.4343% | -1.8452% |

Use `workflow["comparisons"]["B", "simple"]` for business-day returns, or replace `"simple"`
with `"log"` for the four LOG views. The extra columns `translated_local_excess_pnl` and
`fx_forward_pnl` remain **SIMPLE payoffs per opening reference notional** in every comparison.
They are not LOG returns even when displayed beside LOG asset returns. In the shared
notation, the former is the **translated self-financed payoff** $p^{L\to R}$ and the latter
is the **forward FX return** $\varphi$, long local currency through a forward.
The API names remain `LOCAL_EXCESS_TRANSLATED` and `FX_FORWARD`; see the
[notation and conventions](notation_and_conventions.md#fx-aliases-payoffs-and-account-capital).

> **Pitfall.** The analytical frequency sets the cash and additional-hedge reset intervals.
> Business-day cash includes intervening calendar days. A month-end request recomputes those
> intervals; it does not just sum daily excess returns. Resampling the reporting of an already
> completed account leaves that account's trading and settlement events intact.

## Follow the account through reset and settlement

The account starts with USD capital 100 and uses reference-currency financing. Targets on
31 January allocate 70% to SYN_EUR and 30% to SYN_CHF; 29 February changes them to 40% and 60%;
30 April closes both positions. Target notionals use the fixed initial capital. Held units
stay fixed between target dates while price and FX movements change their value. Resizing
can leave positive or negative USD cash balances.

The EUR hedge is sized from current EUR asset value after the corresponding equity trade.
Each contract keeps its quantity and strike until fixing. The first contract fixes on
29 February but pays on 4 March; a new contract opens on 29 February and fixes and pays on
30 April. Its marked value and its cash payment enter the ledger at their respective dates.

| Hedge input | Sign convention |
|---|---|
| Return request `hedge_ratios=1.0` | An additional full opening-principal short-local hedge. |
| Account contract `notional_weight=-1.0` with `asset_currency_value` | A short-local hedge sized to the current currency asset exposure. |

Those inputs belong to different APIs. Account contracts also require explicit trade,
fixing and settlement dates. Embedded instrument or share-class hedges remain in the
supplied prices and should not be added again.

Reuse the existing ledger rather than reconstructing P&L from return panels:

```python
account = workflow["account"]
print(account.accounts.tail(1).round(6).to_string())
assert account.contracts.paid.all()
assert account.asset_units.iloc[-1].eq(0).all()
assert round(float(account.nav.iloc[-1]), 6) == 91.520313
assert abs(float(account.nav.iloc[-1] - 100.0 - account.pnl.to_numpy().sum())) < 1e-10
```

The terminal capital is **USD 91.520313**, with positions closed and both contracts paid.
The account's additive monetary P&L reconciles to the capital change. Negative cash is a
financing balance; a fixed but unpaid forward payoff remains a marked claim until settlement.
Opening cash earns or pays interest once, under the specified financing policy.

| Object from `workflow` | What to reuse |
|---|---|
| `comparisons[(frequency, coordinate)]` | Four asset-return views and two explicitly SIMPLE payoff panels. |
| `cash_relative` | Typed monthly cash-relative result; read its `.values`. |
| `normalized_forward_index` | Optional synthetic forward-payoff index starting at 100 under the declared normalisation. |
| `account.accounts` and `account.nav` | Monetary balances and marked reference-currency capital. |
| `account.pnl` and `account.cash_flows` | Additive P&L components and actual cash movements. |
| `account.asset_units` and `account.contracts` | Held asset quantities and the FX contract lifecycle. |

An optional payoff index is not account capital and does not add collateral cash. A signed
portfolio can have zero net target exposure while requiring positive capital and financing
balances; net exposure is not a return denominator. See the
[account-capital rules](self_financed_fx_accounting.md#reconciliation-and-capital) for that case.

## Checks and scope

The canonical example independently checks endpoint price/spot conversions, opening cash
accrual, translated local-excess and FX-forward payoffs, and cash/P&L reconciliation. The
page's executable snippets check the displayed values. These establish implementation
correctness for the supplied inputs, not a forecast of future risk or investment performance.

The example uses synthetic covered-interest-parity forwards and the account's stated
financing rules. It does not model market forward basis, transaction costs, segregated
collateral, margin calls or credit constraints. Rate and reset schedules are explicit inputs.
Equity risk-model development is a separate research activity; this workflow does not activate
an equity model or change a production allocation.

## Complete runnable source

[Open the canonical source on GitHub](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/market_data/fx_returns_and_accounting_example.py).
The rendered site includes that file directly, so the page and example share one implementation.
In a Markdown viewer that displays the include directive as text, follow the source link.

```{literalinclude} ../examples/market_data/fx_returns_and_accounting_example.py
:language: python
:linenos:
```

## Further reading

- [Return conventions and FX adjustments](return_conventions_and_fx_adjustments.md): typed
  requests, cash rules, exact payoff identities and optional index normalisation.
- [Self-financed FX accounting](self_financed_fx_accounting.md): financing, resets, fixing and
  settlement, including signed positions and nonpositive capital.
- [FX hedging and market data](fx_hedging_and_market_data.md): provider-neutral currency inputs.
- [Reference-currency conversion and excess returns](reference_currency_excess_returns.md):
  partial hedges and consistent analytics coordinates.
