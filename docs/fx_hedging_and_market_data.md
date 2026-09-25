---
myst:
  html_meta:
    description: >-
      Translate local-currency assets, apply covered-interest-parity FX hedges, and distinguish
      opening-principal hedges, cash payoffs and excess-return conventions in qis.
---

# FX hedging and market-data boundaries

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/QuantInvestStrats/commit/8a10dc6b72ed8db593e42d5eafe6d3e4b23419e6)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

FX translation expresses a local-currency asset in a reference currency. An FX hedge adds a
forward payoff to offset a stated amount of currency exposure. The result depends on quote
direction, hedge sizing, forward carry and timing; a hedge of opening principal does not also
hedge an unknown future investment gain.

## Overview

`qis.FxRatesData` accepts spot and annual short-rate panels and applies currency analytics to
supplied asset prices. Data acquisition is a separate step. Core calculations require pandas
inputs and no vendor connection.

[Covered interest parity (CIP)](https://www.bis.org/publications/qr-201609/covered-interest-parity-lost-understanding-cross-currency-basis)
relates spot, forward and matched cash-growth factors under a no-arbitrage funding model.
The qis getter derives its premium from short rates. Real forward quotes can include a
cross-currency basis and trading costs, so the model premium is not an executable quotation.

<a id="data-and-calculation-contract"></a>

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple returns; log output is $\log(1+R_t)$ of the complete payoff |
| Sampling grid | The `freq` grid of the pair calculation, default `ME` |
| Annualisation | Annual rates are divided by $\mathrm{AN}$; the CIP period is $\Delta=1/\mathrm{AN}$ |
| Mean adjustment | Not applicable |
| Timing | Hedge ratio and forward premium set at $t-1$ apply to the return at $t$ |
| Output units | Decimal returns in the reference currency |
| qis default | `hedge_ratio` has no default: 0 is unhedged, 1 hedges the opening principal; `freq='ME'` |

| Symbol or input | Meaning | Units or convention |
|---|---|---|
| `fx_spots` | Spot panel by currency | USD per one unit of each currency; USD column is 1 |
| $S_t$ | Local-to-reference cross | Reference currency per one unit of local currency |
| $P_t$ | Asset price in local currency | Positive level on a dated index |
| $r^{L}_t$, $r^{FX}_t$ | Local asset and cross-rate returns over $[t-1,t]$ | Simple periodic returns |
| $y^L_t$, $y^R_t$ | Local and reference short rates | Annualised decimal rates |
| $\mathrm{AN}$, $\Delta$ | Periods per year and CIP period | $\Delta=1/\mathrm{AN}$, selected from `freq` |
| $f_t$ | qis local/reference cash-growth premium | Simple fraction; inverse quote to $F_t/S_t-1$ |
| $F_t$ | Forward fixed at $t$ for the next period | Reference per local, same direction as $S_t$ |
| $h_t$ | Local opening principal sold forward | Fraction; 0 unhedged, 1 principal hedge |

The cross is `fx_spots[local] / fx_spots[reference]`. For EUR assets viewed in CHF, it is CHF per
EUR. Asset and currency labels must agree, and the panels must contain the necessary currencies.

Construction sorts and forward-fills spots chronologically. Rates are sorted and forward-filled on
the union of their source calendar and the spot calendar before being selected at the spot dates,
so off-grid rate updates remain available to later observations. Missing leading values remain
missing; future quotes are never carried backward.

The pair calculation aligns and forward-fills asset prices and cross rates. Hedge ratios and
forward premiums are as-of aligned to its return grid, then lagged one observation. Return $t$
uses the hedge and premium set at $t-1$. The first output return is initialized to zero so NAV
starts at one; it is not a realised hedge period.

## Methodology

### Cross-rate translation and covered interest parity

For the stated quote direction:

$$
S_t=\frac{\operatorname{spot}_t(\mathrm{local})}
          {\operatorname{spot}_t(\mathrm{reference})},
\qquad
1+r^{FX}_t=\frac{S_t}{S_{t-1}}.
$$

The getter's simple premium and the corresponding forward satisfy

$$
1+f_t=\frac{1+\Delta y^L_t}{1+\Delta y^R_t},
\qquad
\frac{F_t}{S_t}=\frac{1+\Delta y^R_t}{1+\Delta y^L_t}
              =\frac{1}{1+f_t}.
$$

Thus `get_forward_rate_for_local_ccy` does **not** return $F_t/S_t-1$ under the displayed spot
direction. With `is_log_returns=True` it returns $\log(1+f_t)$ instead. Supplied premiums must
match the convention selected for the hedge function.

The short-forward simple cost is $f_t/(1+f_t)$. A negative value is a gain to this hedge under
the model. The period uses the frequency factor, not the actual number of days between two
observations; state that convention when comparing with dated market forwards.

### Hedged, unhedged, cash, and futures exposures

For a cash asset, the reference-currency simple return is

$$
R_t
=r^L_t(1+r^{FX}_t)
 +(1-h_{t-1})r^{FX}_t
 -h_{t-1}\frac{f_{t-1}}{1+f_{t-1}}.
$$

Setting $h=0$ gives $(1+r^L_t)(1+r^{FX}_t)-1$. Setting $h=1$ removes the direct spot-return term,
but retains $r^L_t r^{FX}_t$: local investment gains remain exposed to terminal FX.
Log output is $\log(1+R_t)$ after computing the complete payoff. For an unhedged position only,
this decomposes into local log return plus FX log return.

An independent cash-flow derivation makes the hedge notional explicit. Let $V^L_{t-1}$ be local
opening asset value and sell $h_{t-1}V^L_{t-1}$ local currency forward:

$$
\begin{aligned}
W^R_t
&=V^L_{t-1}\frac{P_t}{P_{t-1}}S_t
  +h_{t-1}V^L_{t-1}(F_{t-1}-S_t),\\
R_t&=\frac{W^R_t}{V^L_{t-1}S_{t-1}}-1.
\end{aligned}
$$

For a *known local cash payoff* with periodic interest $c^L$, hedge the terminal amount by
setting $h=1+c^L$. Under the same CIP rates this locks terminal gross wealth at $1+c^R$,
independent of terminal FX. Hedging only opening principal leaves the interest exposed.
A risky asset's future gain is not known in advance, so this cash identity is not a general
equal-excess-return identity for hedged investments.

Futures require a separate exposure model: currency translation applies to P&L rather than
a fully funded cash-asset principal. Use `compute_futures_fx_adjusted_returns` for that convention,
and `compute_cash_fx_adjusted_returns` for funded cash exposures.

### Mean-variance optimal hedge ratio

`compute_fx_optimal_hedge` chooses the hedge ratio from a one-period mean-variance trade-off on
the log-return legs. Write the local return as a regression on the FX return,
$\ell^{L}_t=\beta\,\ell^{FX}_t+\varepsilon_t$, with $\varepsilon_t$ uncorrelated with
$\ell^{FX}_t$. Selling $h$ units of local currency forward earns $-h\,\ell^{FX}_t$ and costs the
annualised forward carry $c$, so to first order the hedged reference-currency return is

$$
R_t(h)\approx(1+\beta-h)\,\ell^{FX}_t+\varepsilon_t-h\,c .
$$

**Proposition (optimal hedge ratio).** Maximising
$\mathbb{E}[R(h)]-\lambda\operatorname{Var}(R(h))$ over $h$, with FX variance $\sigma^2_{FX}$ and
risk aversion $\lambda$, gives

$$
h^{*}=1+\beta-\frac{c}{2\lambda\,\sigma^{2}_{FX}} .
$$

**Proof.** The objective is $-h\,c-\lambda\big[(1+\beta-h)^2\sigma^2_{FX}+\sigma^2_\varepsilon\big]$
plus terms free of $h$. It is concave in $h$; setting its derivative
$-c+2\lambda(1+\beta-h)\sigma^2_{FX}$ to zero gives $h^{*}$. $\square$

The two limiting cases are returned alongside the optimum. `Beta Hedge` is $1+\beta$, which
removes the FX exposure implied by the regression. `Max Carry` is $1-c/(2\lambda\sigma^2_{FX})$,
which ignores the beta and tilts away from a full principal hedge when hedging is expensive.
All three ratios are clipped to `min_max_hedge`, $[0,1]$ by default.

The inputs are point-in-time EWMA estimates on the `freq` grid. `compute_fx_vol_beta`
estimates $\beta$ with `qis.compute_ewm_cross_xy` and the annualised $\sigma_{FX}$ with
`qis.compute_ewm_vol`, both with EWMA mean adjustment and span 36 by default. The variance
recursion is seeded with $0.08^2/12$, the monthly variance of an 8% annual volatility, on every
grid. The carry $c$ is the
short-forward cost $f/(1+f)$ divided by the period length $\Delta=1/\mathrm{AN}$. The default
$\lambda=4/3$ is a modelling choice, not an estimate.

> **Pitfall.** The optimum rises one-for-one with $\beta$. A positive local-on-FX beta, common
> for foreign equities viewed from a funding currency, can call for a hedge ratio above one
> before the clip; the clip then hides how far the unconstrained optimum sits from the bound.

### Total and excess returns

The panel helpers form excess returns using the reference currency's simple cash accrual $c_t$:

$$
R^{\mathrm{excess}}_t=R_t-c_t,
\qquad
\ell^{\mathrm{excess}}_t=\log(1+R_t)-\log(1+c_t).
$$

These are distinct conventions; `expm1` of log excess is not generally simple excess.
Simple and log **total** returns describe the same NAV when converted consistently.

For `compute_returns_in_reference_ccy` and `compute_returns_adjusted_by_local_rate`, raw annual
rate quotes are as-of aligned to the asset grid, divided by its annualisation factor, and shifted
one observation. The latter helper's third output is the annual rate quote panel for reporting.
Do not substitute that panel directly for periodic lagged cash accruals.

## Worked example

Consider one month with a EUR asset rising from 100 to 102, EUR/USD rising from 1.10 to 1.21,
annual EUR rates of 2.4% and annual USD rates of 4.8%. These are synthetic anchors.

| Quantity | Result |
|---|---:|
| Local asset return | 2% |
| FX return, USD per EUR | 10% |
| Unhedged USD return | 12.2% |
| Opening-principal-hedged USD return | Approximately 2.399601% |
| Known EUR cash payoff, with terminal amount hedged | 0.4%, the USD period cash rate |

The hedged asset return includes the 0.2 percentage-point local/FX cross-product and approximately
0.199601 percentage points of forward gain. The following compares qis with the independent
terminal-cash-flow calculation.

```python
from math import isclose
import pandas as pd
import qis

dates = pd.date_range('2024-01-31', periods=2, freq='ME')
spots = pd.DataFrame({'USD': [1.0, 1.0], 'EUR': [1.10, 1.21]}, index=dates)
rates = pd.DataFrame({'USD': 0.048, 'EUR': 0.024}, index=dates)
fx = qis.FxRatesData(fx_spots=spots, domestic_rates=rates)
asset = pd.Series([100.0, 102.0], index=dates, name='EUR asset')

_, unhedged = fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
    asset_price_local_ccy=asset, hedge_ratio=0.0,
    local_ccy='EUR', reference_ccy='USD', freq='ME', is_log_returns=False,
)
_, hedged = fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
    asset_price_local_ccy=asset, hedge_ratio=1.0,
    local_ccy='EUR', reference_ccy='USD', freq='ME', is_log_returns=False,
)
forward = 1.10 * 1.004 / 1.002
terminal_usd = 102.0 * 1.21 + 100.0 * (forward - 1.21)
cash_flow_return = terminal_usd / 110.0 - 1.0
assert isclose(unhedged.iloc[-1], 0.122, abs_tol=1e-12)
assert isclose(hedged.iloc[-1], cash_flow_return, abs_tol=1e-12)

local_cash = pd.Series([100.0, 100.2], index=dates, name='EUR cash')
_, covered_cash = fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
    asset_price_local_ccy=local_cash, hedge_ratio=1.002,
    local_ccy='EUR', reference_ccy='USD', freq='ME', is_log_returns=False,
)
assert isclose(covered_cash.iloc[-1], 0.004, abs_tol=1e-12)
```

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| FX and domestic-rate panel | $S_t$, $y^L_t$, $y^R_t$ | `qis.FxRatesData`, `qis.load_fx_rates_data` |
| Cross rates aligned to assets | $S_t$ per asset currency | `qis.get_aligned_fx_spots` |
| Local and FX legs | $r^{L}_t$, $r^{FX}_t$ or their logs | `qis.compute_local_and_fx_return` |
| Hedged reference-currency NAV | $R_t$ at hedge ratio $h$ | `qis.compute_performance_of_local_ccy_asset_in_reference_ccy` |
| Cash and futures translation | funded vs P&L-only exposure | `qis.compute_cash_fx_adjusted_returns`, `qis.compute_futures_fx_adjusted_returns` |
| FX volatility and beta | $\sigma_{FX}$, $eta$ | `qis.compute_fx_vol_beta` |
| Optimal, beta and carry hedges | $h^{*}$, $1+eta$, $1-c/(2\lambda\sigma^2_{FX})$ | `qis.compute_fx_optimal_hedge` |
| Panel of assets and its report | hedged NAVs and statistics | `qis.compute_multi_asset_fx_hedging`, `qis.run_asset_fx_hedging_report`, `qis.plot_multi_asset_fx_hedging_report` |
| Factor price levels | validated factor panel | `qis.FactorsData` |

### Minimal offline example

A longer deterministic example uses fixed 2023–2024 business-day panels. It needs no API key,
network or optional data package. These levels are a teaching path, not observed exchange rates.

```python
import pandas as pd
import qis

dates = pd.bdate_range('2023-01-02', '2024-12-31')
step = pd.Series(range(len(dates)), index=dates, dtype=float)
spots = pd.DataFrame(
    {'USD': 1.0, 'EUR': 1.05 * (1.0 + 0.0001 * step)}, index=dates,
)
annualised_rates = pd.DataFrame({'USD': 0.04, 'EUR': 0.03}, index=dates)
fx = qis.FxRatesData(fx_spots=spots, domestic_rates=annualised_rates)

eur_asset = pd.Series(100.0 * (1.0 + 0.0002) ** step, name='EUR asset')
eur_in_usd = fx.get_local_to_reference_fx_rate('EUR', 'USD')
monthly_forward = fx.get_forward_rate_for_local_ccy(
    local_ccy='EUR', reference_ccy='USD', freq='ME', is_log_returns=False,
)
unhedged_nav, unhedged_returns = (
    fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
        asset_price_local_ccy=eur_asset, hedge_ratio=0.0,
        local_ccy='EUR', reference_ccy='USD', freq='ME', is_log_returns=False,
    )
)
hedged_nav, hedged_returns = (
    fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
        asset_price_local_ccy=eur_asset, hedge_ratio=1.0,
        local_ccy='EUR', reference_ccy='USD', freq='ME', is_log_returns=False,
    )
)
```

The cross, premium and each NAV/return output are Series. EUR rates below USD rates make
`monthly_forward` negative under the getter's local/reference cash-growth convention.

For panels, `compute_returns_in_reference_ccy` returns NAV and return DataFrames at one frequency.
`compute_fx_adjusted_returns` groups per-asset frequencies and returns a dictionary of return
DataFrames. It replaces every exact zero return with NaN for estimation, including a genuine
zero; that policy is separate from currency valuation.

`compute_fx_optimal_hedge` estimates carry-tilted and beta-aware ratios using EWMA risk estimates,
the same exact forward cost, and clipping bounds. Supply enough history and distinguish any
full-sample normalization from a historical decision rule. Applying a ratio is separate from
estimating or executing it.

The implementation owners are the [FX container](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_rates_data.py)
and [hedge/payoff functions](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_hedging.py).
The [CIP and payoff tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/tests/fx_cip_identity_test.py)
check cash-flow identities; [alignment tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/tests/fx_spot_alignment_causality_test.py)
check that later quotes do not leak into earlier calculations.

### Data acquisition boundary

Free-data [Yahoo examples](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/market_data)
use the optional `data` extra (`qis[data]`) and require network access. They are separate from the
offline illustrations. Vendor-specific acquisition belongs in the data layer, including
`bbg-fetch` for Bloomberg. A Bloomberg account is not required to construct or use `FxRatesData`.
Local CSV loading reads supplied files; it does not itself update market observations.

<a id="constraints-and-failure-modes"></a>

## Interpretation and limitations

- Reversing a spot quote changes return and carry signs. Confirm reference-per-local units first.
- Nonpositive local/reference cash gross factors or simple forward gross factors raise an error.
  Log performance rejects a combined payoff with nonpositive terminal wealth. Losses are not
  clipped; remaining NaNs after the declared alignment policy stay missing.
- CIP omits basis, bid/ask spreads, collateral terms and trading costs unless provided elsewhere.
- A full opening-principal hedge retains local asset risk and its FX cross-product.
- Forward-filled prices, rates and spots may be stale. Calendar alignment does not certify
  tradability or freshness.
- Excess-return convention, hedge cadence, quote basis and one-period timing all affect results.
  Optimized hedge ratios are model outputs, not guaranteed cost savings.
- Translation changes the unit of account, not legal denomination, liquidity or underlying cash flows.

## See also

- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Performance analytics](performance_analytics_and_sharpe.md)
- [Reporting-frequency convention](_included/reporting_frequencies.md) and
  [packaged source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/reporting_frequencies.md)
- {doc}`FxRatesData API <api/generated/qis.FxRatesData>` and
  [source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_rates_data.py)
- {doc}`FX hedge API <api/generated/qis.compute_performance_of_local_ccy_asset_in_reference_ccy>` and
  [source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_hedging.py)

## References

1. Borio, C., McCauley, R., McGuire, P., and Sushko, V. (2016). Covered interest parity lost: understanding the cross-currency basis. *BIS Quarterly Review*, September. [Publisher page](https://www.bis.org/publications/qr-201609/covered-interest-parity-lost-understanding-cross-currency-basis). No-arbitrage cash/forward relation and limits of frictionless CIP.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
