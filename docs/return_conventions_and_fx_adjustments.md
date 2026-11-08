---
myst:
  html_meta:
    description: >-
      Typed FX returns, explicit cash-relative coordinates, dated hedge metadata and optional
      normalized synthetic indices for statistical estimation and portfolio reporting.
---

# Return conventions and FX adjustments

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-11-08](https://github.com/ArturSepp/QuantInvestStrats/commit/fae104880a01c21f3882fe8af3a6796cd669b96d)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Overview

Use `compute_fx_returns` for current FX analytics. An `FxAssetReturnsRequest` describes a funded
asset panel or its locally financed payoff; an `FxCurrencyReturnsRequest` describes currency pairs directly. Both return a mapping
from cadence to `FxReturnsData`. Statistical consumers read `result.values`; a synthetic index is
absent unless the caller supplies an explicit normalization.

The existing `FxRatesData` storage remains two panels: USD per local currency unit and domestic
annual decimal rate quotes. Currency IDs such as USD, CHF and EUR are provider-independent.
Fetching, ticker choice and provider yield normalization belong to the data layer.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Explicit economic kind and SIMPLE, LOG or CASH_RELATIVE_SIMPLE coordinate |
| Sampling grid | Existing daily, monthly or per-asset cadence; end-labelled actual intervals |
| Annualisation | None in returned per-period values; statistical annualisation is separate |
| Mean adjustment | None |
| Timing | Cash and principal hedge use opening information; no future quote is read |
| Output units | Decimal funded return or payoff per opening reference notional; capital only when explicitly normalized |
| qis default | Kind and coordinate required for assets; no automatic NAV or zero filtering |

Write $S_t$ for reference currency per unit of local currency, $r_t^L$ for the funded local asset
return, $r^{FX}_{c,t}=S_t/S_{t-1}-1$ for spot return, and $h_{t-1}$ for the additional opening principal hedge.
Prices include any embedded instrument hedges; additional hedge ratios must not repeat them.

`FxCashPolicy` records ACT/365 Fixed, quote lag and annual-rate provenance. The default quote lag
is one reset boundary. Lag zero is closing-quote descriptive analytics, not earned opening cash.
Hedge pricing always uses opening quotes. The policy does not convert the provider's yield basis.

Dated `local_ccys` describe quote-currency spells, not domicile or economic currency exposure.
A change of currency is rejected unless `currency_changes='mask_boundary'` explicitly requests
missing crossing returns. No unit conversion or rebasing factor is inferred. Dated hedge matrices
are aligned as-of and shifted to the opening. Currency/hedge labels must cover every asset.

## Methodology

Let $\tau_t=\mathrm{calendar\ days}(t-1,t)/365$ denote the ACT/365 Fixed year fraction.

Opening annual decimal quotes $y^L_{t-1}$ and $y^R_{t-1}$ accrue over the actual interval:

$$
r^{f,L}_t=y^L_{t-1}\tau_t,\qquad
r^{f,R}_t=y^R_{t-1}\tau_t,\qquad
f_{t-1,t}=\frac{1+r^{f,L}_t}{1+r^{f,R}_t}-1,\qquad
\kappa_{t-1,t}=\frac{f_{t-1,t}}{1+f_{t-1,t}}.
$$

The reference-per-local forward has $F_{t-1,t}/S_{t-1}=1/(1+f_{t-1,t})$. The existing principal-hedge
kernel gives the funded reference asset return:

$$
r_t^R=r_t^L(1+r^{FX}_{c,t})+(1-h_{t-1})r^{FX}_{c,t}-h_{t-1}\kappa_{t-1,t}.
$$

The local gain remains exposed to terminal spot. A hedge of one does not remove that cross term.
For either local or reference cash $r^f_t$, the three excess coordinates are distinct:

$$
\tilde r_t=r_t-r^f_t,\qquad
q_t=\frac{1+r_t}{1+r^f_t}-1,\qquad
\tilde\ell_t=\log(1+r_t)-\log(1+r^f_t).
$$

`SIMPLE` excess is $\tilde r_t$, `CASH_RELATIVE_SIMPLE` is $q_t$, and `LOG` excess is $\tilde\ell_t$.
The identity $q_t=\exp(\tilde\ell_t)-1$ holds; $\tilde r_t=\exp(\tilde\ell_t)-1$ generally does not.
Cash-relative simple is rejected for non-excess kinds. Log output requires positive completed
asset and cash gross factors. Valid arithmetic losses remain available without a log or index.

> **Pitfall.** An arithmetic excess series is not a funded portfolio return. Compounding it
> defines a normalized synthetic index with the caller's declared capital and sizing policy;
> it does not establish collateral, borrowing or self-financing accounting.

### Exact payoffs per opening reference notional

**Definition.** `LOCAL_EXCESS_TRANSLATED` is the P&L of an asset financed with local cash, translated
at terminal spot. `FX_FORWARD` is the long-local forward's terminal settlement:

$$
p^{L\to R}_t=(r_t^L-r^{f,L}_t)(1+r^{FX}_{c,t}),\qquad
\varphi_{c,t}=\frac{S_t-F_{t-1,t}}{S_{t-1}}
=r^{FX}_{c,t}+\kappa_{t-1,t},\qquad \kappa_{t-1,t}=1-\frac{1+r^{f,R}_t}{1+r^{f,L}_t}.
$$

These are arithmetic payoffs, with no initial capital implied. They require opening cash
quotes and `SIMPLE`. A cash-relative local excess index would divide the financed gain by
local cash growth and produce a different payoff; do not use it as a translation input.

**Identity.** For the funded reference excess $\tilde r^R_t(h)=r_t^R(h)-r^{f,R}_t$:

$$
\tilde r^R_t(h)=(r_t^L-r^{f,L}_t)+(1-h_{t-1})\varphi_{c,t}
+r_t^L r^{FX}_{c,t}+\kappa_{t-1,t} r^{f,L}_t
=p^{L\to R}_t+(1+r^{f,L}_t-h_{t-1})\varphi_{c,t}.
$$

**Proof.** Expand the funded payoff, subtract reference cash and substitute
$\varphi_{c,t}=r^{FX}_{c,t}+\kappa_{t-1,t}$. The cash identity
$\kappa_{t-1,t}(1+r^{f,L}_t)=r^{f,L}_t-r^{f,R}_t$ gives the first decomposition.
Substituting $p^{L\to R}_t=(r_t^L-r^{f,L}_t)(1+r^{FX}_{c,t})$ gives the second. $\square$

The old currency-index approximation adds $f_{t-1,t}$ to spot. Its exact replacement adds
$\kappa_{t-1,t}$: the incremental payoff is $-f_{t-1,t}^2/(1+f_{t-1,t})$ when the forward gross factor is positive.
This comparison holds cash accrual, quotes and reset dates fixed. It is separate from
switching the cash day count, changing settlement cadence or revising portfolio funding.

Negative notional P&L, including a loss below minus one, remains valid. Raw notional payoffs
cannot request `LOG` or `CASH_RELATIVE_SIMPLE`. To analyse log changes of a synthetic index,
first explicitly normalize the completed payoff into capital, then take log index returns.
Never sum log spot and log carry to approximate a forward settlement.

## Worked example

This offline example compares EUR and CHF equities against USD cash at two actual month intervals.
All inputs are synthetic. It verifies the unhedged translation independently before constructing
an explicitly requested index.

```python
import numpy as np
import pandas as pd
import qis

frames = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-31'])
fx = qis.FxRatesData(
    pd.DataFrame({'USD': 1., 'EUR': [1.10, 1.12, 1.09],
                  'CHF': [1.15, 1.16, 1.14]}, index=frames),
    pd.DataFrame({'USD': .05, 'EUR': .03, 'CHF': .01}, index=frames))
prices = pd.DataFrame({'EU_equity': [100., 102., 101.],
                       'CH_equity': [100., 101., 103.]}, index=frames)
currencies = pd.Series({'EU_equity': 'EUR', 'CH_equity': 'CHF'})
results = {}
for kind in (qis.FxReturnType.LOCAL_TOTAL, qis.FxReturnType.LOCAL_EXCESS,
             qis.FxReturnType.REFERENCE_TOTAL, qis.FxReturnType.REFERENCE_EXCESS):
    reference = 'USD' if kind in (qis.FxReturnType.REFERENCE_TOTAL,
                                  qis.FxReturnType.REFERENCE_EXCESS) else None
    results[kind] = qis.compute_fx_returns(fx, qis.FxAssetReturnsRequest(
        prices=prices, local_ccys=currencies, return_type=kind,
        coordinate=qis.FxReturnCoordinate.SIMPLE,
        reference_ccy=reference, frequency='ME'))['ME']
reference = results[qis.FxReturnType.REFERENCE_TOTAL]
assert np.isclose(reference.values.EU_equity.iloc[1], 1.02*(1.12/1.10)-1)
excess = results[qis.FxReturnType.REFERENCE_EXCESS]
assert np.isclose(excess.values.EU_equity.iloc[1],
                  reference.values.EU_equity.iloc[1] - .05*29/365)
assert excess.synthetic_index is None
log_excess = qis.compute_fx_returns(fx, qis.FxAssetReturnsRequest(
    prices=prices, local_ccys=currencies, return_type=qis.FxReturnType.REFERENCE_EXCESS,
    coordinate=qis.FxReturnCoordinate.LOG, reference_ccy='USD'))['ME']
assert np.isclose(log_excess.values.EU_equity.iloc[1],
                  np.log1p(reference.values.EU_equity.iloc[1])-np.log1p(.05*29/365))
spot = qis.compute_fx_returns(fx, qis.FxCurrencyReturnsRequest(
    ('EUR', 'CHF'), reference_ccy='USD'))['ME']
assert spot.values.columns.tolist() == ['EUR', 'CHF']
normalization = qis.FxIndexNormalization(
    initial_value=100., notional_to_capital=1., notional_rule='capital_proportional',
    settlement_frequency='ME', reinvestment='each_interval',
    cash_treatment='as_returned', missing_returns='stop')
nav = reference.to_index(normalization)
assert nav.iloc[0].tolist() == [100., 100.]
assert np.isclose(nav.EU_equity.iloc[1], 100*1.02*(1.12/1.10))
comparison = pd.concat({kind.value: result.values for kind, result in results.items()}, axis=1)
```


The two notional payoffs use the same input data without creating indices:

```python
translated = qis.compute_fx_returns(fx, qis.FxAssetReturnsRequest(
    prices=prices, local_ccys=currencies, return_type=qis.FxReturnType.LOCAL_EXCESS_TRANSLATED,
    coordinate=qis.FxReturnCoordinate.SIMPLE, reference_ccy='USD', frequency='ME'))['ME']
forward = qis.compute_fx_returns(fx, qis.FxCurrencyReturnsRequest(
    ('EUR', 'CHF'), 'USD', return_type=qis.FxReturnType.FX_FORWARD,
    frequency='ME', diagnostics=True))['ME']
c_local, c_usd = .03*29/365, .05*29/365
assert np.isclose(translated.values.EU_equity.iloc[1], (.02-c_local)*1.12/1.10)
forward_quote = 1.10*(1+c_usd)/(1+c_local)
assert np.isclose(forward.values.EUR.iloc[1], (1.12-forward_quote)/1.10)
assert np.isclose(excess.values.EU_equity.iloc[1],
                  translated.values.EU_equity.iloc[1]
                  + (1+c_local)*forward.values.EUR.iloc[1])
assert translated.synthetic_index is None and forward.synthetic_index is None
forward_index = forward.to_index(normalization)
normalized_log = qis.to_returns(forward_index, freq='ME', is_log_returns=True)
assert np.isclose(normalized_log.EUR.iloc[1], np.log1p(forward.values.EUR.iloc[1]))
```

The normalization above starts at 100, uses one unit of opening notional per unit of
capital, resets monthly and reinvests each completed payoff. It adds no collateral cash.

The complete public example is also executed by the documentation gate:

```python
from examples.market_data.fx_returns_and_accounting_example import run_example
teaching = run_example(verbose=False)
assert set(teaching['comparisons']) == {('B', 'simple'), ('B', 'log'),
                                      ('ME', 'simple'), ('ME', 'log')}
assert teaching['account'].contracts.paid.all()
```

Change the frequency to `B` on a daily input panel for business-daily resets. Monthly endpoint
reset and compounded daily reset are different strategies; this API does not assert their equality.

## Implementation in qis

[FX return API](api/generated/qis.compute_fx_returns.rst), [asset request](api/generated/qis.FxAssetReturnsRequest.rst) and [currency request](api/generated/qis.FxCurrencyReturnsRequest.rst).

### Select the economic object before its coordinate

| Consumer / economic object | API selection | Units and interpretation |
|---|---|---|
| Local equity signal/risk input | LOCAL_EXCESS / LOG | Log asset wealth relative to local cash; any model-specific selection is explicit |
| Multi-fund signal/covariance in a reference book | REFERENCE_EXCESS / LOG | Same canonical cash-relative log panel for both consumers |
| Funded investor P&L and performance | REFERENCE_TOTAL / SIMPLE | Funded return; explicit index normalization or unit/cash accounting |
| Local financed asset payoff translated at terminal spot | LOCAL_EXCESS_TRANSLATED / SIMPLE | Signed reference P&L per opening reference notional |
| Long-local FX contract payoff | FX_FORWARD / SIMPLE | Signed settlement per opening reference notional; not a log return |
| FX spot diagnostic | FX_SPOT / SIMPLE or LOG | Spot movement only; no carry, principal or collateral cash |
| Optional synthetic capital | `FxReturnsData.to_index(FxIndexNormalization(...))` | Explicit sizing/reinvestment; not a financing ledger |
| Actual self-financed account | `backtest_fx_account(..., spec=FxAccountSpec(...))` | Reference monetary capital and P&L; cash, debt and unpaid contracts separated |

The equations in [exact payoffs](#exact-payoffs-per-opening-reference-notional) map to
LOCAL_EXCESS_TRANSLATED and FX_FORWARD respectively. The three excess-coordinate identities
map to SIMPLE, CASH_RELATIVE_SIMPLE and LOG. The account's balance and transfer identities
are defined in [cash and local financing](self_financed_fx_accounting.md#cash-and-local-financing).
These local Markdown anchors are the implementation map; no manuscript equation labels are
invented or inferred from an out-of-scope publication.

See the repository-only [offline returns and account example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/market_data/fx_returns_and_accounting_example.py)
for EUR/CHF equities at daily/monthly frequency in simple/log coordinates, both notional
payoffs, cash-relative checks, and an explicitly settled hedge account. No vendor service or
optional data package is needed. Different frequency requests change cash/hedge resets;
resampling a completed ledger only changes its presentation.


| Operation | Interface | Contract |
|---|---|---|
| Asset request | `FxAssetReturnsRequest` | Prices, local currency, economic kind, coordinate, optional reference hedge |
| Currency request | `FxCurrencyReturnsRequest` | Currency IDs and reference currency; no dummy asset |
| Economic kind | `FxReturnType` | Local/reference total/excess, FX_SPOT, LOCAL_EXCESS_TRANSLATED and FX_FORWARD |
| Coordinate | `FxReturnCoordinate` | SIMPLE, LOG, or excess-only CASH_RELATIVE_SIMPLE |
| Cash policy | `FxCashPolicy` | ACT/365F, quote lag, annual-rate provenance |
| Calculation | `compute_fx_returns` | Mapping from each cadence to the same typed result schema |
| Result | `FxReturnsData` | Owned copies of values, boundaries, currencies and opening hedges |
| Optional index | `FxIndexNormalization` | Explicit capital, notional, settlement, reinvestment, cash and gap handling |

`diagnostics=True` includes currency-level cash, spot and hedge-cost panels without replicating
all legs for every asset. Original dated currency spells retain off-grid changes. Property reads return copies; modifying them cannot alter a result.
The first valuation has no preceding return or opening boundary. Genuine zeros remain unless
`zero_return_to_nan=True` explicitly masks them.

A capital-proportional normalization compounds scaled simple increments and rejects nonpositive
gross factors. A fixed-notional normalization accumulates P&L against initial capital and is named
`accumulated_pnl_with_initial_capital`, not compounded wealth. `cash_treatment='as_returned'`
adds no extra funding: totals already contain their asset economics, while excess cash has already
been subtracted or used as numeraire. `missing_returns='skip'` explicitly preserves the historical
QIS gap policy; `stop` makes later capital unknown after an interior missing payoff. This is a
synthetic index operation. For funded units, debt, spendable cash, hedge rolls and delayed
payment use the separate [self-financed account](self_financed_fx_accounting.md).

### Immediate deprecation and direct migration

QIS 5.47.0 is the first public release of the unified FX API, exact notional payoffs,
and explicit accounting. Local validation introduced typed asset/spot analytics and the
first warning group in 5.45.0, and exact payoffs and currency-method warnings in 5.46.0.
The warning messages retain this development lineage; all seven deprecations first ship
on PyPI in 5.47.0, with removal in 6.0.0. Match examples to the installed QIS version
and check the built documentation version before relying on `latest` or `stable`.


The following methods emit `DeprecationWarning` from QIS 5.45.0 and are scheduled for removal in
6.0.0. Their existing implementations remain during this bounded retirement period. There is no
compatibility facade, old-shape adapter, or redirection that changes their economic meanings.

| Deprecated method on FxRatesData | Direct replacement |
|---|---|
| `compute_performance_of_local_ccy_asset_in_reference_ccy` | One-column asset request; the standalone low-level quote kernel is retained |
| `compute_returns_in_reference_ccy` | Reference-kind asset request; read `.values`, request `.to_index(...)` explicitly |
| `compute_fx_adjusted_returns` | Asset request with scalar or per-asset frequency; consume typed cadence results |
| `compute_returns_adjusted_by_local_rate` | Separate LOCAL_TOTAL and LOCAL_EXCESS requests; annual quotes stay on the data container |

The currency methods `get_daily_carry_local_return`, `get_fr^{FX}_{c,t}otal_return_nav` and
`get_carry_fx_return_nav` emit `DeprecationWarning` from QIS 5.46.0, with removal in 6.0.0.
Use an `FX_FORWARD` currency request and optional explicit normalization. Their old premium-based
implementations are retained only for the bounded retirement period; no compatibility facade
redirects them to new economics. Current examples and report currency-index panels use the exact
forward payoff. In particular, the old vol-matched carry diagnostic is replaced by a daily-reset
forward index with unit notional-to-capital, rather than being relabelled as exact carry.
Dated synthetic forward quote access and standalone low-level principal-hedge kernels remain
building blocks. In particular the standalone function and the deprecated method with the
same `compute_performance_of_local_ccy_asset_in_reference_ccy` name have different retirement
status. Use the typed API in new application examples; do not remove the retained kernel.

Migration is direct: select kind/coordinate/cash policy, replace tuple or DataFrame assumptions
with cadence-keyed `FxReturnsData.values`, and request normalization only when needed.
Replace old spot-plus-premium indices with exact FX_FORWARD under the same cash/reset policy;
that deliberate payoff delta is not a representation-only migration. Do not translate
`expm1(LOG excess)` into arithmetic excess: it produces cash-relative simple instead.
Run exercised FX call sites with `DeprecationWarning` visible, then as errors for the retired
method messages. Test warning behavior separately while legacy implementations remain.
Remove the seven container methods only in the named 6.0.0 removal release; no compatibility
facade is introduced and this documentation stage removes no API.

## Interpretation and limitations

The exact payoff types preserve annual input rates and the requested reset schedules. Existing
funded reference-return kernels are unchanged. Forward pricing is synthetic matched-period CIP:
market forward curves, cross-currency basis and collateral cash are not supplied by this model.
No equity risk-model augmentation,
residual-covariance policy or self-financed account engine is implied by this return container.
A currency switch needs a supplied rebasing convention or an explicit missing crossing return.
These results contain quote currencies, not a claim about a security's full economic FX exposure.

## See also

- [Notation and conventions](notation_and_conventions.md)
- [Reference-currency excess returns](reference_currency_excess_returns.md)
- [Cash rate timing and FX adjustments](cash_rat\tilde r_timing_and_fx_adjustments.md)
- [FX hedging and market data](fx_hedging_and_market_data.md)
- [Self-financed FX accounting](self_financed_fx_accounting.md)

- [qis bibliography](bibliography.md)
- [Canonical implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_returns.py)

## References

1. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
