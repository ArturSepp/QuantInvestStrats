---
myst:
  html_meta:
    description: >-
      Convert a multi-currency asset universe with per-asset FX hedge ratios, subtract
      reference-currency cash, and prepare excess-return panels for MATF factor estimation.
---

# Reference-currency conversion and excess returns

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A multi-asset portfolio needs a common currency of measurement. A local-currency asset return
is not the return earned by an investor in another currency. First translate each asset into
the portfolio's reference currency, applying its declared FX hedge ratio; then, for an
excess-return analysis, subtract cash earned in that **reference currency**. The same prepared
panel should feed the associated factor-risk and residual-alpha calculations.

## Overview

`qis.FxRatesData` is the common conversion layer. A variable named `fx_data` below holds this
container; it is not a separate estimator. It combines USD-quoted spot levels and annual
short-rate quotes with native-currency asset prices and per-asset hedge ratios:

1. Preserve the native-currency price/NAV panel and currency metadata.
2. Choose the portfolio reference currency and each asset's observation/hedge frequency.
3. Apply the FX hedge policy: `h=0` is unhedged, `h=0.5` hedges half the opening principal,
   and `h=1` hedges the full opening principal.
4. Produce reference-currency **total returns** for performance measurement, or subtract
   reference cash once to produce **excess returns** for estimation.
5. Supply the resulting frequency-keyed return panels to the factor-estimation layer.

This is the investor-currency risk-input workflow used by ROSAA-style universes. For
Multi-Asset Tradable Factors (MATF) estimation using
[FactorLasso](https://github.com/ArturSepp/FactorLasso), QIS owns the asset-return preparation;
the consuming model owns factor definitions, regression settings, constraints and covariance
assembly. The worked example needs neither ROSAA nor FactorLasso installed.

The [FX hedging chapter](fx_hedging_and_market_data.md) derives the hedge payoff and optimal
hedge choices. This chapter explains how to apply a **supplied** hedge policy to a universe
and prepare its estimation inputs. It does not choose an investment hedge policy or change
the analytics.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Native and converted simple total returns; arithmetic or log excess returns selected explicitly |
| Sampling grid | Per-asset `freq`; examples use `ME` and `QE`; conversion and hedge reset share that grid |
| Annualisation | Annual decimal cash quotes divided by $\mathrm{af}$: 12 for `ME`, 4 for `QE`; outputs remain periodic |
| Mean adjustment | None in conversion or cash subtraction; the beta example uses equal-weight OLS with an intercept, equivalent to sample demeaning |
| Timing | Return at $t$ covers $(t-1,t]$; opening hedge and CIP rates are lagged once; `cash_rate_lag=1` |
| Output units | Decimal periodic reference-currency returns; tables show percent per month; regression betas are dimensionless |
| qis default | `compute_fx_adjusted_returns`: simple total returns, `zero_return_to_nan=True`; explicitly request log excess returns and retain genuine zeros |

| Input or symbol | Meaning and required convention |
|---|---|
| `prices`, $P^L_{i,t}$ | Date-by-asset positive native-currency price or total-return NAV levels; no prior FX conversion or cash subtraction |
| `local_ccys` | Series indexed by the exact asset column labels, giving each supplied price's denomination |
| `hedge_ratios`, $h_{i,t}$ | Per-asset Series of constant hedge fractions; the single-frequency panel method also accepts a dated DataFrame |
| `reference_ccy`, $R$ | Portfolio's reporting and funding currency, such as USD, EUR, CHF or GBP |
| `fx_spots` | Date-by-currency panel in **USD per unit of currency**; the USD column is 1 |
| `domestic_rates`, $y^C_t$ | Annual decimal short-rate quotes for currency $C$, e.g. 0.04 means 4% per annum |
| $S^{L/R}_t$, $g^{L/R}_t$ | Reference currency per local unit and its simple return |
| $r^L_{i,t}$, $r^R_{i,t}(h)$ | Local total return and reference-currency total return under hedge policy $h$ |
| $r^{f,R}_t$ | Periodic reference-currency cash accrual; not an annual yield |
| $\tilde r^R_{i,t}$, $\tilde\ell^R_{i,t}$ | Arithmetic and log returns in excess of reference cash |
| $f^{L/R}_{t-1}$ | QIS local/reference cash-growth premium for the forthcoming period; not the forward-price premium |

Spots and rates must cover the local currencies and the reference currency, with enough
history before the first return being measured. The foreign-asset path obtains both currencies'
rates even for `h=0`; provide a complete `FxRatesData` rather than assuming rates can be omitted
for unhedged total returns. Same-currency excess returns still require reference cash rates.

Construction sorts and causally forward-fills the supplied observations. Leading missing data
remain missing; no future rate or spot is carried backward. Forward-filling is not a freshness
check. Validate stale quotes, duplicate dates, metadata coverage and the source observation
calendar before estimation.

**Denomination is not underlying economic exposure.** A USD-priced global equity index can
already contain exposure to non-USD businesses and currencies. Pass USD as its price currency;
the conversion does not reconstruct its holdings. Likewise, a currency-hedged share class is
already a distinct payoff: do not hedge its underlying basket a second time. An additional
conversion from that share class's denomination into another portfolio currency is a separate
investor overlay. Convert prices in pence to pounds consistently with their metadata.

## Methodology

### Translate native total returns before subtracting cash

**Definition.** The cross and its realised return are

$$
S^{L/R}_t=\frac{\operatorname{spot}_t(L)}{\operatorname{spot}_t(R)},
\qquad
g^{L/R}_t=\frac{S^{L/R}_t}{S^{L/R}_{t-1}}-1.
$$

For example, `fx_spots['EUR'] / fx_spots['CHF']` is CHF per EUR. USD is the storage
numeraire for the spot panel, not a restriction on the portfolio reference currency.

**Identity (unhedged funded asset).**

$$
1+r^R_{i,t}(0)=(1+r^L_{i,t})(1+g^{L/R}_t).
$$

**Proof.** Reference-currency wealth is local wealth multiplied by the contemporaneous spot
cross. Dividing terminal by opening wealth gives the product of the two growth factors.
$\square$

The local/FX cross-product is part of the payoff. Adding simple local and FX returns without
it is an approximation. Do not lag the realised FX return: it covers the same period as the
asset return. Only decisions and contracted inputs fixed at the period start are lagged.

### Apply the opening-principal hedge and forward carry

QIS uses a rate-implied forward under covered interest parity (CIP); it does not infer an
executable forward price from spot alone. With $\Delta=1/\mathrm{af}$,

$$
1+f^{L/R}_{t-1}
=\frac{1+\Delta y^L_{t-1}}{1+\Delta y^R_{t-1}}.
$$

The actual reference-per-local forward satisfies $F_{t-1}/S_{t-1}=1/(1+f^{L/R}_{t-1})$.
Consequently, the short-forward cost is $f/(1+f)$, **not** $f$. Rate-only CIP omits
cross-currency basis and execution costs; see the
[BIS discussion](https://www.bis.org/publications/qr-201609/covered-interest-parity-lost-understanding-cross-currency-basis).

**Identity (funded asset with hedge).** The simple total payoff is

$$
r^R_{i,t}(h)
=r^L_{i,t}(1+g^{L/R}_t)
 +(1-h_{i,t-1})g^{L/R}_t
 -h_{i,t-1}\frac{f^{L/R}_{t-1}}{1+f^{L/R}_{t-1}}.
$$

**Proof.** Hold local opening wealth $P^L_{i,t-1}$ and sell
$h_{i,t-1}P^L_{i,t-1}$ local units forward. Terminal reference wealth is
$P^L_{i,t}S_t+h_{i,t-1}P^L_{i,t-1}(F_{t-1}-S_t)$. Divide by opening wealth
$P^L_{i,t-1}S_{t-1}$, subtract one and substitute the CIP forward ratio. $\square$

At `h=1`, the direct FX return on opening principal disappears, but the local investment
gain remains exposed to terminal FX. Fully hedged does **not** mean simply dropping the
FX columns or returning native returns unchanged. It also does not prove equality between
native-currency and reference-currency excess returns. A hedge of a known terminal cash
payoff, rather than opening principal, is a different notional convention.

When local and reference currencies are identical, QIS bypasses the FX leg. The same-currency
total return is the native total return, regardless of the supplied hedge ratio.

### Deduct reference-currency cash once

Under the default one-observation lag, cash is as-of aligned to the **actual return grid**,
scaled to a periodic simple accrual, and shifted once:

$$
r^{f,R}_t=\frac{y^R_{t-1}}{\mathrm{af}}.
$$

Thus `cash_rate_lag=1` means one month on `ME` and one quarter on `QE`, not always one
calendar month. This is a frequency-factor approximation, not an actual-day-count deposit
or an observed cash-index return. Changing the lag affects cash subtraction only; hedge
carry remains fixed from the prior observation. The
[cash-timing chapter](cash_rate_timing_and_fx_adjustments.md) discusses these distinctions.

**Definition (two excess-return conventions).**

$$
\tilde r^R_{i,t}=r^R_{i,t}(h)-r^{f,R}_t,
\qquad
\tilde\ell^R_{i,t}
=\log(1+r^R_{i,t}(h))-\log(1+r^{f,R}_t).
$$

`is_log_returns=False, is_excess_returns=True` selects arithmetic excess;
`is_log_returns=True, is_excess_returns=True` selects log excess. For the latter, QIS first
computes the complete hedged simple payoff, takes its logarithm, then subtracts log cash.
It does not add hedged local and FX log legs as though that were an exact hedge payoff.

**Identity.** Exponentiating log excess gives a return relative to cash, not arithmetic excess:

$$
\exp(\tilde\ell^R_{i,t})-1
=\frac{r^R_{i,t}(h)-r^{f,R}_t}{1+r^{f,R}_t}.
$$

**Proof.** Exponentiate the difference of logarithms, subtract one and combine the numerator.
$\square$

> **Pitfall.** Do not subtract local cash first and then feed that excess-return NAV through
> the funded-asset conversion. It is a different payoff. The canonical investor-currency
> pipeline starts from native **total-return** levels, converts them, and subtracts **reference**
> cash once. Hedging carry is already in the converted total return; do not deduct it again.

`compute_returns_adjusted_by_local_rate` instead subtracts each asset's native cash rate
without any FX translation. That is useful for a deliberately native-currency analysis,
but it is not a replacement for investor-currency risk inputs. Its third output contains
annual rate quotes, not ready-to-subtract periodic accruals.

### Separate asset returns from MATF factor definitions

For an investor-currency factor fit, the asset response can be written as

$$
\tilde\ell^R_{i,t}
=\alpha_i+\sum_k\beta^R_{ik}x_{k,t}+\varepsilon_{i,t},
$$

where $x_{k,t}$ is the declared periodic return of factor $k$. Here $x$ denotes the supplied
factor-return series, not a price or an annual premium. The consuming model determines
whether the fit includes an intercept, uses EWMA demeaning or imposes constraints.

Changing reference currency or hedge ratio changes the asset response and can change its
estimated loadings, residuals and covariance. Do not reuse native-currency betas as though
they automatically describe an unhedged foreign investor's risk. A separate native-payoff
expected-return fit, where explicitly part of a CMA model, does not replace that risk fit.

FactorLasso receives numerical estimation inputs; it does not infer an asset's currency,
apply its hedge policy or know which cash has already been subtracted. Keep the distinction:

| Input | Preparation and interpretation |
|---|---|
| Asset panel | QIS reference-currency log excess returns with explicit hedge policy and cash lag |
| MATF factor panel | Preserve the model's declared tradable-factor return, financing, currency and scaling conventions |
| Funded total-return factor, if used | Convert/subtract cash according to its specified model contract; do not pretend it was already an excess-return factor |
| Already excess-return or futures P&L factor | Do not mechanically subtract reference cash again or apply a funded-principal FX overlay |
| Estimation cut | Factor and asset observations must refer to the same realised periods and use only information available by the fit date |

A fixed USD-defined MATF factor basis may be used to explain reference-currency asset excess
returns if that is the model specification; the betas are then sensitivities to that declared
basis. This is different from constructing a new set of investor-currency funded factor
portfolios. Currency-consistent asset measurement does not authorise silently redefining
the factors. Annual expected factor premia are also not historical return observations.

For mixed observation frequencies, retain separate `ME` and `QE` return panels. Compute
factor observations over matching periods in the consuming model; do not repeat a monthly
return across dates, forward-fill returns or multiply monthly returns by three. Moreover,
quarterly-reset hedging is not generally the same strategy as monthly-reset hedging whose
returns are compounded to quarter ends. In these QIS helpers, `freq` controls both return
measurement and the hedge-reset cadence.

## Worked example

### One-period arithmetic check

Reuse the [FX chapter's synthetic EUR/USD example](fx_hedging_and_market_data.md#worked-example):
a EUR asset rises 2%, USD per EUR rises 10%, and annual EUR/USD rates are respectively 2.4%
and 4.8%. Monthly USD cash is 0.4%. These are teaching inputs, not market observations.

| Opening-principal hedge | USD total return | USD arithmetic excess return |
|---|---:|---:|
| 0% | 12.200000% | 11.800000% |
| 50% | 7.299800% | 6.899800% |
| 100% | 2.399601% | 1.999601% |

Native EUR arithmetic excess is 1.8%; it is a different quantity from every USD column.
The following verifies the table from terminal cash flows, independently of QIS's payoff
implementation, and then checks the public method against those flows.

```python
import numpy as np
import pandas as pd
import qis

dates = pd.date_range('2024-01-31', periods=2, freq='ME')
fx_check = qis.FxRatesData(
    fx_spots=pd.DataFrame({'USD': 1.0, 'EUR': [1.10, 1.21]}, index=dates),
    domestic_rates=pd.DataFrame({'USD': 0.048, 'EUR': 0.024}, index=dates),
)
eur_asset = pd.Series([100.0, 102.0], index=dates, name='EUR asset')
forward = 1.10 * (1.0 + 0.048 / 12) / (1.0 + 0.024 / 12)
computed_rows = []
for hedge in (0.0, 0.5, 1.0):
    terminal_usd = 102.0 * 1.21 + hedge * 100.0 * (forward - 1.21)
    total = terminal_usd / (100.0 * 1.10) - 1.0
    cash = 0.048 / 12
    for log_returns in (False, True):
        _, excess = fx_check.compute_performance_of_local_ccy_asset_in_reference_ccy(
            asset_price_local_ccy=eur_asset, hedge_ratio=hedge,
            local_ccy='EUR', reference_ccy='USD', freq='ME',
            is_log_returns=log_returns, is_excess_returns=True, cash_rate_lag=1,
        )
        expected = np.log1p(total) - np.log1p(cash) if log_returns else total - cash
        np.testing.assert_allclose(excess.iloc[-1], expected, atol=1e-12)
        assert pd.isna(excess.iloc[0])  # No pre-inception holding period.
    computed_rows.append([100 * total, 100 * (total - cash)])
np.testing.assert_allclose(
    computed_rows, [[12.2, 11.8], [7.299800, 6.899800], [2.399601, 1.999601]],
    rtol=0, atol=0.0000005,
)
```

### A reproducible multi-currency universe

Use the frozen synthetic generator, seed 20260725, with clean paths from 31 December 2019
through 30 December 2022, sampled at calendar month ends. `SEQ_US`, `SEQ_EU` and `SBD_IG`
are assigned USD, EUR and CHF respectively **for this teaching example**. The generator
does not supply currency metadata. EUR and CHF spots below are synthetic paths derived from
other generated series; rates are illustrative annual quotes. There is no vendor download.

```python
from qis.datasets.synthetic import generate_synthetic_prices

daily = generate_synthetic_prices(
    start='2019-12-31', end='2022-12-30', seed=20260725, apply_quirks=False,
)
month_ends = pd.date_range('2019-12-31', '2022-12-31', freq='ME')
monthly = daily.reindex(month_ends, method='ffill')
prices = monthly[['SEQ_US', 'SEQ_EU', 'SBD_IG']].copy()
local_ccys = pd.Series({'SEQ_US': 'USD', 'SEQ_EU': 'EUR', 'SBD_IG': 'CHF'})
hedge_ratios = pd.Series({'SEQ_US': 0.0, 'SEQ_EU': 0.5, 'SBD_IG': 1.0})
reference_ccy = 'USD'

spots = pd.DataFrame({
    'USD': 1.0,
    'EUR': 1.10 * monthly['SCM_GLD'] / monthly['SCM_GLD'].iloc[0],
    'CHF': 1.05 * monthly['SBD_TSY'] / monthly['SBD_TSY'].iloc[0],
}, index=month_ends)
step = np.arange(len(month_ends))
rates = pd.DataFrame({
    'USD': 0.024 + 0.0002 * step,
    'EUR': 0.012 + 0.0001 * step,
    'CHF': 0.006 + 0.0001 * step,
}, index=month_ends)
fx_data = qis.FxRatesData(fx_spots=spots, domestic_rates=rates)

# Always select the return convention; QIS defaults are simple TOTAL returns.
asset_returns_dict = fx_data.compute_fx_adjusted_returns(
    prices=prices, local_ccys=local_ccys, hedge_ratios=hedge_ratios,
    reference_ccy=reference_ccy, freq='ME',
    is_log_returns=True, is_excess_returns=True,
    cash_rate_lag=1, zero_return_to_nan=False,
)
asset_log_excess = asset_returns_dict['ME']
assert asset_log_excess.columns.equals(prices.columns)
assert asset_log_excess.iloc[0].isna().all()

# Reporting uses total returns, with otherwise identical inputs and hedge policy.
total_navs, total_simple = fx_data.compute_returns_in_reference_ccy(
    asset_prices=prices, local_ccys=local_ccys, hedge_ratios=hedge_ratios,
    reference_ccy=reference_ccy, freq='ME',
    is_log_returns=False, is_excess_returns=False,
)
reference_cash = (rates[reference_ccy] / 12).shift(1)
np.testing.assert_allclose(
    asset_log_excess,
    np.log1p(total_simple).sub(np.log1p(reference_cash), axis=0),
    atol=1e-12, equal_nan=True,
)
```

The two panels describe different uses of the same assets: `total_navs` represents the
modelled investment payoff, whereas `asset_returns_dict` contains the log-excess estimation
inputs. Replacing the report NAV with a compounded excess-return series would remove cash
from the investor's reported performance.

### The same USD asset in USD and CHF: excess returns and betas

Continue with the **same** synthetic `SEQ_US` asset, spot paths and annual cash quotes
above. Compare USD and CHF investors, each with opening-principal hedge ratios 0, 0.5
and 1. All six cases use the same 36 monthly observations, January 2020 to December 2022,
and monthly hedge resets. The December 2022 label uses the last supplied business-day
observation, 30 December. Only the investor reference currency and hedge ratio vary;
the native asset history, factor series and estimation sample stay fixed.

To isolate the effect on a fitted loading, declare `SEQ_EM` to be a **USD-funded synthetic
equity factor proxy** for this example. Its monthly log total returns less lagged USD log
cash define one fixed excess-return factor. This extra teaching denomination is not supplied
by the generator. The factor is kept in that USD excess basis for every fit: it is neither
translated into CHF nor charged cash a second time. It is not an actual MATF factor or a
recommendation to use `SEQ_EM` as a benchmark.

Fit each asset's **log excess** returns to this factor using `qis.fit_multivariate_ols`,
with an intercept, equal observation weights and the whole common sample. This descriptive
OLS estimate has no rolling span or warm-up; its information cutoff is the end of the sample.
It is not a backtest or a reproduction of ROSAA's constrained FactorLasso estimator.
The slope is dimensionless and is not annualised. Each table mean below is a monthly sample
mean, not a compounded return or a forward CMA.

| Investor currency | Hedge ratio | Mean simple excess (%/month) | Mean log excess (%/month) | OLS beta |
|---|---:|---:|---:|---:|
| USD | 0 | 0.493614 | 0.343900 | 0.539930 |
| USD | 0.5 | 0.493614 | 0.343900 | 0.539930 |
| USD | 1 | 0.493614 | 0.343900 | 0.539930 |
| CHF | 0 | 0.448265 | 0.247902 | 0.565295 |
| CHF | 0.5 | 0.486670 | 0.317036 | 0.553099 |
| CHF | 1 | 0.525076 | 0.376469 | 0.540981 |

The following block reproduces the table, independently checks every converted payoff
against terminal cash flows, and checks the fitted coefficients against centred moments.

```python
# Continue the frozen monthly sample above; no additional random draws or downloads.
comparison_asset = prices['SEQ_US']
usd_cash = (rates['USD'] / 12).shift(1)
factor_log_excess = (
    qis.to_returns(monthly['SEQ_EM'], freq=None, is_log_returns=True)
    - np.log1p(usd_cash)
).rename('USD equity factor')

simple_by_view, log_by_view = {}, {}
for investor_ccy in ('USD', 'CHF'):
    for hedge in (0.0, 0.5, 1.0):
        key = (investor_ccy, hedge)
        for use_logs, outputs in ((False, simple_by_view), (True, log_by_view)):
            _, outputs[key] = (
                fx_data.compute_performance_of_local_ccy_asset_in_reference_ccy(
                    asset_price_local_ccy=comparison_asset, hedge_ratio=hedge,
                    local_ccy='USD', reference_ccy=investor_ccy, freq='ME',
                    is_log_returns=use_logs, is_excess_returns=True, cash_rate_lag=1,
                )
            )

        # Independent check: terminal asset value plus the contracted forward payoff.
        cross = spots['USD'] / spots[investor_ccy]  # Investor currency per USD.
        cash = (rates[investor_ccy] / 12).shift(1)
        forward = cross.shift(1) * (1 + cash) / (1 + usd_cash)
        terminal = (comparison_asset * cross
                    + hedge * comparison_asset.shift(1) * (forward - cross))
        gross = terminal / (comparison_asset.shift(1) * cross.shift(1))
        np.testing.assert_allclose(
            simple_by_view[key], gross - 1 - cash, atol=1e-12, equal_nan=True,
        )
        np.testing.assert_allclose(
            log_by_view[key], np.log(gross) - np.log1p(cash),
            atol=1e-12, equal_nan=True,
        )

log_comparison = pd.DataFrame(log_by_view)
simple_comparison = pd.DataFrame(simple_by_view)
common_dates = factor_log_excess.dropna().index.intersection(
    log_comparison.dropna(how='any').index,
)
assert common_dates.equals(month_ends[1:]) and len(common_dates) == 36
x = factor_log_excess.loc[common_dates].to_frame()
rows = []
for key in log_comparison:
    y = log_comparison.loc[common_dates, key]
    _, params, _ = qis.fit_multivariate_ols(x=x, y=y, fit_intercept=True, verbose=False)

    # Independent OLS check via centred cross-products, including the intercept.
    xc, yc = x.iloc[:, 0] - x.iloc[:, 0].mean(), y - y.mean()
    beta_check = float((xc * yc).sum() / (xc * xc).sum())
    alpha_check = y.mean() - beta_check * x.iloc[:, 0].mean()
    np.testing.assert_allclose(
        params.to_numpy(), [alpha_check, beta_check], rtol=0, atol=1e-12,
    )
    rows.append([
        100 * simple_comparison.loc[common_dates, key].mean(),
        100 * y.mean(), params['USD equity factor'],
    ])

comparison_summary = pd.DataFrame(
    rows, index=pd.MultiIndex.from_tuples(log_comparison.columns, names=['Currency', 'h']),
    columns=['Mean simple excess (%/month)', 'Mean log excess (%/month)', 'OLS beta'],
)
print(comparison_summary.to_string(float_format=lambda value: f'{value:.6f}'))
np.testing.assert_allclose(
    comparison_summary.to_numpy(),
    [
        [0.493614, 0.343900, 0.539930],
        [0.493614, 0.343900, 0.539930],
        [0.493614, 0.343900, 0.539930],
        [0.448265, 0.247902, 0.565295],
        [0.486670, 0.317036, 0.553099],
        [0.525076, 0.376469, 0.540981],
    ],
    rtol=0, atol=0.0000005,  # Match the six-decimal displayed values.
)

# A USD asset needs no USD overlay: all three hedge choices are identical controls.
for hedge in (0.5, 1.0):
    np.testing.assert_allclose(
        log_comparison[('USD', hedge)], log_comparison[('USD', 0.0)],
        rtol=0, atol=0, equal_nan=True,
    )
```

The USD rows coincide because the asset is already USD-denominated: there is no investor
FX overlay to hedge. For CHF, the unhedged beta is 0.565295 rather than 0.539930 even though
the factor has not changed. The additional currency return changes the asset's covariance
with that factor; half hedging moves the beta to 0.553099 in this particular sample.

#### Why full hedging does not reproduce the USD payoff exactly

Here, **fully hedged means 100% of opening principal, not the unknown end-of-month
asset value**. Suppose USD 100 grows to USD 110 during the month. The forward sells the
original USD 100; the USD 10 investment gain remains exposed to the closing exchange rate.
If the asset loses value, the opening hedge instead exceeds the remaining asset value.

For a simple illustration, assume zero USD and CHF interest rates, opening spot and forward
both equal to CHF 1 per USD, and closing spot of CHF 1.10 per USD. These are separate
teaching inputs, not the rates or returns in the 36-month table. The asset is worth CHF 121
at month end, and the forward loses CHF 10. Terminal wealth is therefore

$$
110\times1.10+100\times(1.00-1.10)=111\quad\text{CHF}.
$$

The fully hedged CHF return is 11%, versus the native USD return of 10%. The extra CHF 1
is the FX gain on the unhedged USD 10 profit: the local-return/FX cross term is
$0.10\times0.10=0.01$, or one percentage point of opening wealth. With non-zero rates,
exact forward carry also enters the payoff. The regression then uses log excess returns,
subtracting CHF log cash from the converted asset and USD log cash from the native asset.
Opening-principal hedging, cash normalisation and the logarithm do not imply identical
response series.

#### Why a different mean excess return alone does not change beta

With an intercept, OLS fits deviations from sample means. Adding the same constant to
every monthly log excess return changes the intercept by that constant but leaves beta
unchanged. Different means therefore do not, by themselves, explain different betas.

For the same asset, suppress its index and define the monthly response difference as
$\delta_t=\tilde\ell^{CHF}_t(1)-\tilde\ell^{USD}_t$, where the CHF response is fully
hedged. Let $x_t$ be the fixed USD factor used above. All moments below are sample moments
on the **same observations**, with non-zero factor variance.

**Identity (OLS with an intercept).**

$$
\widehat\beta^{CHF}(1)-\widehat\beta^{USD}
=\frac{\operatorname{Cov}(x,\delta)}{\operatorname{Var}(x)}.
$$

**Proof.** Each slope is the sample covariance of the factor and its response divided by
the factor variance. Subtract the two slopes and use covariance's linearity. Constants
vanish under sample demeaning. $\square$

Thus only the part of the response difference that covaries with the factor changes this
beta. Even a time-varying difference can leave beta unchanged if its sample covariance with
the factor is zero. In the table, the full-hedge difference is approximately
$0.540981-0.539930=0.001051$; it is not inferred from the difference in mean excess returns.
Neither equality of betas nor monotonic changes in beta or mean return are general hedging
identities. The constant-shift result assumes an intercept (or equivalent sample demeaning);
it need not hold for a regression forced through the origin. See
[regression conventions](regression_and_hac.md).

The following checks the principal-hedge illustration through QIS, verifies the beta-difference
identity, and confirms that adding 0.001 to every monthly log excess return shifts only the
intercept of the USD fit.

```python
principal_dates = pd.date_range('2024-01-31', periods=2, freq='ME')
principal_fx = qis.FxRatesData(
    fx_spots=pd.DataFrame({'USD': 1.0, 'CHF': [1.0, 1.0 / 1.10]}, index=principal_dates),
    domestic_rates=pd.DataFrame({'USD': 0.0, 'CHF': 0.0}, index=principal_dates),
)
_, principal_return = principal_fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
    asset_price_local_ccy=pd.Series([100.0, 110.0], index=principal_dates),
    hedge_ratio=1.0, local_ccy='USD', reference_ccy='CHF', freq='ME',
    is_log_returns=False, is_excess_returns=True, cash_rate_lag=1,
)
terminal_chf = 110.0 * 1.10 + 100.0 * (1.0 - 1.10)
np.testing.assert_allclose(terminal_chf, 111.0, rtol=0, atol=1e-12)
np.testing.assert_allclose(principal_return.iloc[-1], terminal_chf / 100 - 1, atol=1e-12)

usd_response = log_comparison.loc[common_dates, ('USD', 0.0)]
chf_response = log_comparison.loc[common_dates, ('CHF', 1.0)]
response_difference = chf_response - usd_response
factor = x.iloc[:, 0]
beta_difference = (comparison_summary.loc[('CHF', 1.0), 'OLS beta']
                   - comparison_summary.loc[('USD', 0.0), 'OLS beta'])
np.testing.assert_allclose(
    beta_difference, factor.cov(response_difference) / factor.var(), rtol=0, atol=1e-12,
)
_, original_params, _ = qis.fit_multivariate_ols(
    x=x, y=usd_response, fit_intercept=True, verbose=False,
)
_, shifted_params, _ = qis.fit_multivariate_ols(
    x=x, y=usd_response + 0.001, fit_intercept=True, verbose=False,
)
np.testing.assert_allclose(
    shifted_params - original_params, [0.001, 0.0], rtol=0, atol=1e-12,
)
```

> **Insight.** A shared factor basis does not imply a shared risk-beta matrix. These six
> fits keep the asset and factor histories fixed but measure the investor's actual currency
> payoff. The different coefficients are descriptive sensitivities to that fixed factor,
> not evidence that the underlying asset changed or that one hedge is universally preferable.

### Mixed frequencies and an independent panel check

For an illustration of dispatch only, assign the CHF bond a quarterly frequency. This is
not a claim that bonds should be estimated quarterly; the frequency is part of the supplied
instrument/estimator metadata. Each output frame is checked against terminal asset value
plus the contracted forward payoff, followed by reference-cash subtraction.

```python
asset_freq = pd.Series({'SEQ_US': 'ME', 'SEQ_EU': 'ME', 'SBD_IG': 'QE'})
mixed_returns = fx_data.compute_fx_adjusted_returns(
    prices=prices, local_ccys=local_ccys, hedge_ratios=hedge_ratios,
    reference_ccy=reference_ccy, freq=asset_freq,
    is_log_returns=True, is_excess_returns=True,
    cash_rate_lag=1, zero_return_to_nan=False,
)
assert set(mixed_returns) == {'ME', 'QE'}
assert mixed_returns['ME'].columns.tolist() == ['SEQ_US', 'SEQ_EU']
assert mixed_returns['QE'].columns.tolist() == ['SBD_IG']

for frequency, frame in mixed_returns.items():
    af = {'ME': 12, 'QE': 4}[frequency]
    reference_cash = (rates[reference_ccy].reindex(frame.index, method='ffill') / af).shift(1)
    for asset in frame:
        local_ccy = local_ccys[asset]
        price = prices[asset].reindex(frame.index, method='ffill')
        cross = (spots[local_ccy] / spots[reference_ccy]).reindex(frame.index)
        local_cash = (rates[local_ccy].reindex(frame.index, method='ffill') / af).shift(1)
        forward = cross.shift(1) * (1 + reference_cash) / (1 + local_cash)
        terminal = price * cross + hedge_ratios[asset] * price.shift(1) * (forward - cross)
        gross_payoff = terminal / (price.shift(1) * cross.shift(1))
        expected = np.log(gross_payoff) - np.log1p(reference_cash)
        np.testing.assert_allclose(frame[asset], expected, atol=1e-12, equal_nan=True)

# Remove dates with no returns at all, not individual missing asset observations.
# A downstream estimator must still apply its declared missing-data/warm-up policy.
estimation_inputs = {
    frequency: frame.dropna(how='all') for frequency, frame in mixed_returns.items()
}
```

These dictionaries are the asset side of the estimation handoff, not a fitted MATF model.
The preceding single-factor OLS comparison illustrates the currency effect only; it does
not manufacture MATF factors or replace the consuming model's estimator.

## Implementation in qis

### Select the entry point by the output you need

| Requirement | `FxRatesData` method | Output |
|---|---|---|
| One asset, fixed or time-varying hedge | `compute_performance_of_local_ccy_asset_in_reference_ccy` | `(nav_series, return_series)` |
| One common frequency, optional dated hedge matrix | `compute_returns_in_reference_ccy` | `(nav_dataframe, return_dataframe)` |
| Per-asset observation frequencies, estimation input | `compute_fx_adjusted_returns` | `{frequency: return_dataframe}`; a string frequency still returns a dictionary |
| Deliberately native-currency analysis, no conversion | `compute_returns_adjusted_by_local_rate` | `(native_returns, native_excess_returns, annual_rate_quotes)` |

For panel methods, a uniform hedge must be a Series indexed by asset, for example
`pd.Series(1.0, index=prices.columns)`, not a scalar. A constant currency string is supported
by the single-frequency panel method; an explicit indexed currency Series is clearer for
a multi-currency universe. Reindex metadata to the price columns and check for missing or
duplicated identifiers before calling. Do not invent currencies or hedge ratios for missing rows.

For time-varying hedges, pass a date-by-asset DataFrame to
`compute_returns_in_reference_ccy`. A hedge ratio dated $t$ is a decision for the next holding
period; the method as-of aligns and lags it internally. Do not shift an already decision-dated
hedge a second time. The standalone optimal-hedge estimator is separate from this application.

### Safe estimation and reporting contract

- **Estimation:** explicitly select `is_log_returns=True`, `is_excess_returns=True`,
  `cash_rate_lag=1`, and `zero_return_to_nan=False` for ROSAA-style log-excess inputs.
  These are caller choices, not a change to QIS's simple-total-return default.
- **Reporting:** use `is_log_returns=False`, `is_excess_returns=False`; apply the same
  declared currency and hedge treatment, while stating any different hedge-reset cadence.
- **Inception:** `FxRatesData` preserves an undefined first return as NaN, including the
  foreign-asset kernel's synthetic first-index zero. Never train on a fabricated inception
  return. Include earlier observations when a real return is required at the report start.
- **Valid zeros:** the mixed-frequency helper's default replaces exact zeros in the final
  return panel. With excess returns enabled, a flat total-return asset usually has a negative
  excess return, not zero; conversely, a zero excess return can be economically meaningful.
  Preserve genuine observations and choose the correct grid for appraisal-based assets.
- **Unsmoothing:** if required by the model, prepare the native-currency asset history first
  with the declared point-in-time unsmoothing policy, then convert it. This avoids treating
  liquid FX movements as stale private-asset valuations. Keep as-reported histories for
  investor-facing performance; see [private-asset unsmoothing](private_asset_unsmoothing.md).
- **No double preprocessing:** a return panel already expressed in reference-currency
  excess terms must not be passed through the same conversion and subtraction again.
- **Factor handoff:** record currency, hedge policy, grid, simple/log basis, cash lag, source
  cutoff, zero policy and unsmoothing policy with the inputs. Fit each snapshot only through
  its observation cutoff; FX conversion does not enforce downstream estimator warm-up.

For simple excess output, the returned NAV compounds `1 + arithmetic excess`: it is not
the ratio of investment wealth to cash wealth. For log excess output, compounding exponentiated
log excess is cash-relative wealth over valid periods. Neither replaces the total-return NAV
in the factsheet. Missing-data handling during NAV reconstruction also does not turn an
undefined return into a genuine estimation observation.

### Loading real observations

Construct `qis.FxRatesData(fx_spots=..., domestic_rates=...)` from supplied pandas panels,
or use `qis.FxRatesData.load(local_path=...)` with the existing CSV-pair format. The
[directory guide](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/README.md)
explains `qis.local_path.get_resource_path()` and private YAML configuration. Loading a
container does not download or refresh vendor data. The examples above are fully offline.

The implementation owner is
[fx_rates_data.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_rates_data.py),
using the funded-payoff kernel in
[fx_hedging.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_hedging.py).
The worked examples are executed by the existing
[documentation-example harness](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/documentation_examples_test.py)
and independently check terminal cash flows, reference-cash deduction, log/simple conventions,
inception, monthly/quarterly dispatch and the OLS coefficients against centred cross-products.
The regression uses the public
[fit_multivariate_ols](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/regression.py)
wrapper. This guide targets the checked 5.44.0 source contract.

## Interpretation and limitations

- Reference-currency measurement is necessary for a common portfolio risk/performance view;
  it does not prescribe which hedge policy maximises expected returns.
- Do not relabel already-converted prices as native prices. Original denomination, embedded
  share-class hedges and investor overlays must remain distinguishable.
- Forward carry is included in total returns and reference cash is subtracted only for
  excess returns. Short rates, realised cash returns and expected future cash anchors are
  different inputs; this workflow uses historical short-rate quotes.
- A full principal hedge can retain an asset-return/FX cross-product. Native-currency
  excess returns and hedged reference-currency excess returns are not exact substitutes.
- Missing or stale data can invalidate an estimate despite successful alignment. Exact-zero
  replacement cannot replace proper frequency selection or an observation mask.
- A quarterly cash approximation and quarterly hedge reset need not match compounded monthly
  cash and monthly hedge resets. State the investment cadence, not just the report cadence.
- This is the **funded-asset** conversion. For unfunded futures P&L, use the separate
  `qis.compute_futures_fx_adjusted_returns` convention and model collateral/funding explicitly.
  Applying a funded principal FX return to a futures factor creates exposure that is not there.
- Log output requires positive gross asset and cash payoffs. CIP ignores market basis,
  bid/ask spreads, collateral terms and additional hedge-implementation costs.
- A reference-currency risk fit and a native-payoff mean fit answer different questions.
  Their coexistence must be explicit in the consuming CMA model, not inferred from a
  currency label or an older convention note.

## See also

- [FX hedging and market-data boundaries](fx_hedging_and_market_data.md)
- [Cash rate timing and FX adjustments](cash_rate_timing_and_fx_adjustments.md)
- [Returns and NAVs](returns_and_navs.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Factor risk models](factor_risk_models.md)
- [Private-asset unsmoothing](private_asset_unsmoothing.md)
- [FactorLasso estimation documentation](https://github.com/ArturSepp/FactorLasso/tree/main/docs)

## References

1. Borio, C., McCauley, R., McGuire, P., and Sushko, V. (2016). Covered interest parity lost: understanding the cross-currency basis. *BIS Quarterly Review*, September. [Publisher page](https://www.bis.org/publications/qr-201609/covered-interest-parity-lost-understanding-cross-currency-basis). Rate-implied forward pricing and its market limitations.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
