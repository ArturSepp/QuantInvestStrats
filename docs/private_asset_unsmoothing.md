---
myst:
  html_meta:
    description: >-
      Explain AR return unsmoothing and debt-to-equity de-levering in qis, including
      coefficient timing, appraisal frequency, warm-up and full-sample limitations.
---

# Private-asset unsmoothing and de-levering

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/QuantInvestStrats/commit/8a10dc6b72ed8db593e42d5eafe6d3e4b23419e6)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Return unsmoothing estimates a less filtered return series from reported observations under an
explicit lag model. De-levering estimates an underlying asset return by removing a specified
financing structure. They address different effects: reporting delay and debt amplification.
Neither transformation creates a tradable price or recovers an unobserved return without assumptions.

## Overview

Appraisal-based returns can exhibit serial dependence and understate economic variability.
[Getmansky, Lo and Makarov (2004)](https://web.mit.edu/Alo/www/Papers/JFE2004Pub.pdf)
model reported returns as a finite moving average of latent economic returns. Their paper provides
the illiquidity and smoothing context. The qis methods below use **autoregressive filters of
observed returns**; the public name `unsmooth_returns_glm` does not mean that it fits the paper's
moving-average likelihood.

Choose the transformation from the economic question:

| Question | Method | Main information required |
|---|---|---|
| How does debt affect vehicle returns? | Simple-return de-levering | Debt/equity and period financing cost |
| How does a lagged reporting filter affect returns through time? | Rolling EWMA AR(q) unsmoothing | Sufficient history on the reporting grid |
| What does one fixed filter imply for a sample? | Static AR(q), estimated or supplied coefficients | Full sample or externally fixed coefficients |

<a id="data-and-calculation-contract"></a>

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple returns for de-levering; simple or log returns, retained throughout, for unsmoothing |
| Sampling grid | The reporting grid of each sleeve, `freq`, for example `QE` |
| Annualisation | Financing rates are divided by $\mathrm{AN}$ periods per year |
| Mean adjustment | EWMA mean with `InitType.X0` for rolling fits; full-sample fit for the static filter |
| Timing | Coefficients estimated through $t-1$ are applied to the return at $t$ |
| Output units | Decimal returns and reconstructed NAV levels |
| qis default | `compute_ar_unsmoothed_prices(ar_order=2, freq='QE', span=40, warmup_period=8)` on log returns |

| Symbol or input | Meaning | Units or convention |
|---|---|---|
| $x_t$ | Observed return supplied to an unsmoother | Simple or log; retain one basis throughout the filter |
| $r^{u}_t$ | Filter-adjusted return on that same basis | Decimal simple return or log return |
| $q$ | AR lag order | Number of observations, not calendar days |
| $b_{j,t}$ | Estimated coefficient on observed lag $j$ | Dimensionless; rolling estimate dated $t$ |
| $\Theta_t=\sum_{j=1}^{q}b_{j,t}$ | Coefficient sum | Controls the inversion denominator |
| $L$ | Debt divided by equity | Nonnegative scalar; $L=0.5$ corresponds to 1.5x assets/equity |
| $y_t$, $\mathrm{AN}$ | Annual financing rate and periods per year | Decimal annual rate; e.g. $\mathrm{AN}=12$ monthly |
| $c_t=y_t/\mathrm{AN}$ | Financing cost for the return period | Simple periodic rate under the helper's convention |
| $r^{V}_t$, $r^{A}_t$ | Vehicle and unlevered asset returns | **Simple** periodic returns |

Use date-indexed Series/DataFrames, with columns identifying assets. The price-level wrapper
requires positive NAVs and converts them on `freq`; a Series of per-asset frequencies allows
monthly and quarterly sleeves to use different grids. EWMA `span` counts observations on that
grid and is not a fixed-length window.

Annual financing quotes supplied as a Series are point in time in `delever_returns`: the period
ending on a return date is charged the latest quote dated on or before the previous return date,
so a quote first known at a period's end finances the next period, and the first return date takes
the latest quote dated before it. Missing leading quotes remain unavailable. The helper divides by $\mathrm{AN}$,
rather than applying an elapsed-day accrual.

## Methodology

### De-levering the financing identity

For constant debt/equity and a single financing rate:

$$
r^{V}_t=(1+L)r^{A}_t-Lc_t,
\qquad
r^{A}_t=\frac{r^{V}_t+Lc_t}{1+L}.
$$

The debt finances $L$ units of assets per unit of equity. `delever_returns` implements the inverse;
`lever_returns` implements the forward identity. Zero leverage returns an independent copy of the
input. Use simple returns explicitly: inserting log returns changes the meaning of this equation.

The calculation assumes a constant exposure over the represented return interval. It is not an
instrument-level reconstruction of interest expense, fees, discount-to-NAV movements or multiple
debt tranches. If both leverage and reporting delay are present, state which return series is
being filtered and financed; the transformations need not commute when rates or parameters vary.

### Rolling versus full-sample unsmoothing

The rolling engine estimates the observed return jointly on its first $q$ lags. For identified
coefficients, its inversion is

$$
r^{u}_t=
\frac{x_t-\sum_{j=1}^{q}b_{j,t-1}x_{t-j}}
     {1-\sum_{j=1}^{q}b_{j,t-1}}.
$$

Coefficients use information through $t-1$ when applied to return $t$. Mean adjustment affects
coefficient estimation; inversion uses the raw observed returns in the numerator.

`adjust_returns_with_ar` is the canonical engine for every lag order.
`unsmooth_returns_ar1_ewma` is its AR(1) wrapper. The default `MeanAdjType.EWMA` uses a
point-in-time `InitType.X0` mean seed. Appending later observations does not revise an existing
rolling prefix. `MeanAdjType.INSAMPLE` uses the full-sample mean and is descriptive, not suitable
for a historical decision path.

`adjust_returns_with_joint_unsmoothing` uses the same point-in-time seed while estimating its
own-lag and lagged-factor coefficients together. Its regression moments update only when the
target, own lag and factor lag are jointly observable, so factor history before a ragged asset's
inception cannot enter that asset's fit. A masked coefficient pair remains missing until it is
observable; the one-period application lag does not make future backward fill causal.

The price wrapper defaults to coefficient-sum bounds of -0.25 and 0.75. Clipping applies to the
**sum**, rescaling the coefficient vector, with optional further EWMA coefficient smoothing.
A positive denominator permits inversion; a cap does not establish that the smoothing model is
economically correct. Optional non-negativity and its tolerance are separate modelling choices.

Joint unsmoothing masks exactly `warmup_period` jointly observable coefficient dates per asset,
once, followed by the one-period application lag. Thus `warmup_period=8` makes the ninth jointly
observable coefficient available and the next observed return is the first corrected return.
Missing data or an unidentified fit can require more calendar rows. By default, entirely
unidentified columns stay NaN.
The policy enum `qis.models.unsmoothing.ar_lag.InsufficientData` supplies the alternatives:
`RAISE` reports those columns; explicitly choosing `PASSTHROUGH` returns them unchanged and must
be labelled as a skipped correction.

### Static AR(q) filter

`unsmooth_returns_glm` uses constant coefficients $\theta_j$ in the same observed-lag inversion:

$$
r^{u}_t=
\frac{x_t-\sum_{j=1}^{q}\theta_j x_{t-j}}
     {1-\sum_{j=1}^{q}\theta_j}.
$$

With `theta=None`, each column is fitted by full-sample `AutoReg`, including an intercept in
estimation. The inversion uses its lag coefficients, not the fitted intercept. This is a
descriptive estimate. Supplying `theta` skips estimation, sets the lag order from its length, and
applies the same supplied vector to every DataFrame column.

The static fit drops NaNs before estimation, so gaps compress the fitted lag sequence. Inversion
uses the original row order. Use a contiguous series on the declared reporting grid, or supply
a justified fixed filter; do not interpret compressed observations as regular calendar lags.

There are two current boundary behaviours to inspect:

- The first $q$ rows are retained unchanged as initial conditions. From row $q$ onward, a missing
  required lag produces NaN. Exclude the initial rows when evaluating the corrected sample.
- A supplied coefficient sum within $10^{-10}$ of one raises an error. An *estimated* sum within
  that tolerance instead returns the input unchanged with severe/infinite diagnostics. Always
  request diagnostics before accepting a static result.

For coefficient sum $\Theta<1$, the diagnostic `vol_inflation_factor` is $1/(1-\Theta)$: the multiplier on
the filter numerator. It is **not generally the ratio of output to input sample volatility**,
because subtracting correlated lagged returns also changes numerator variance.
`is_severe` flags $\lvert \Theta\rvert>0.95$; a negative sum can lack the intended smoothing
interpretation even when that flag is false.

## Worked example

A monthly vehicle return of 2.8%, debt/equity 0.5 and annual financing 4.8% imply a monthly cost
of 0.4%. The unlevered return is $(0.028+0.5\times0.004)/1.5=0.02$, or **2%**.

Separately, reported simple returns 1%, 3%, -1%, 2% with a fixed AR(1) coefficient 0.5 produce
corrected returns 5%, -5%, 5% after the initial row. These are fixed arithmetic illustrations,
not estimated private-market performance.

```python
import numpy as np
import pandas as pd
import qis

dates = pd.date_range('2024-01-31', periods=4, freq='ME')
vehicle = pd.Series([0.028], index=dates[:1], name='Vehicle')
asset = qis.delever_returns(
    returns=vehicle, leverage=0.50, financing_rate=0.048, periods_per_year=12,
)
np.testing.assert_allclose(asset, [0.02])

observed = pd.Series([0.01, 0.03, -0.01, 0.02], index=dates, name='Observed')
adjusted, diagnostics = qis.unsmooth_returns_glm(
    returns=observed, theta=0.5, return_diagnostics=True,
)
assert adjusted.iloc[0] == observed.iloc[0]  # Retained initial condition.
np.testing.assert_allclose(adjusted.iloc[1:], [0.05, -0.05, 0.05])
assert diagnostics.theta_sum == 0.5
assert diagnostics.vol_inflation_factor == 2.0
```

## Implementation in qis

### Minimal offline example

The frozen synthetic `SAL_HF` and `SAL_PE` paths include smoothing. `SAL_PE` reports monthly;
the example deliberately samples it quarterly to illustrate different estimator calendars,
while `SAL_HF` is sampled monthly. Dates and seed are fixed.

```python
import pandas as pd
import qis
from qis.datasets.synthetic import generate_synthetic_prices

prices = generate_synthetic_prices(
    start='2005-01-03', end='2025-12-31', seed=20260725, apply_quirks=True,
)[['SAL_HF', 'SAL_PE']]
frequencies = pd.Series({'SAL_HF': 'ME', 'SAL_PE': 'QE'})

unsmoothed_navs, unsmoothed_returns, betas, r_squared = (
    qis.compute_ar_unsmoothed_prices(
        prices=prices, ar_order=1, freq=frequencies, span=20, warmup_period=8,
        mean_adj_type=qis.MeanAdjType.EWMA, is_log_returns=True,
    )
)
monthly_vehicle_returns = qis.to_returns(
    prices=prices['SAL_HF'], freq='ME', is_log_returns=False, drop_first=True,
)
delevered_returns = qis.delever_returns(
    returns=monthly_vehicle_returns, leverage=0.50,
    financing_rate=0.04, periods_per_year=12,
)
```

The price wrapper returns four DataFrames: reconstructed NAVs, **simple** unsmoothed returns,
coefficient sums, and regression R-squared. With `is_log_returns=True`, filtering occurs in log
returns and the output return panel is converted with `expm1`. The mixed-frequency frames use the
union of their date indexes; non-observation rows remain missing in return/diagnostic panels.

The `betas` output records contemporaneous coefficient sums. The filter applies their
**one-observation-lagged** values within each frequency group. R-squared is a contemporaneous
fit diagnostic, clipped to [0, 1], not an out-of-sample score. Warm-up, observation counts and
valid histories differ by sleeve.

The price wrapper uses `to_returns` with its default forward-fill policy. To retain explicit gaps,
prepare a return panel with that policy stated and call `adjust_returns_with_ar` directly.
Inspect return availability when interpreting reconstructed NAVs: NAV initialization and missing
return handling do not certify that every displayed level came from an identified AR correction.

Implementation owners are [AR unsmoothing](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/unsmoothing/ar_lag.py)
and [financing transforms](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py).
[Prefix-invariance tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/unsmoothing/tests/unsmoothing_warmup_causality_test.py)
and [static-filter tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/unsmoothing/tests/test_ar_lag_glm.py)
provide executable contracts.

The [OCSL/GCF walkthrough](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/unsmoothing_and_delevering.py)
uses a bundled historical parquet panel and requires the optional `io` dependency. In a repository
checkout with `qis[io]` installed, run:

```console
python -m examples.perfstats.unsmoothing_and_delevering
```

It fetches no market data. Its historical observations and financing assumptions are example
inputs, not current quotes or an automatically updated assessment. The synthetic examples above
work on a core install.

<a id="constraints-and-failure-modes"></a>

## Interpretation and limitations

- Serial correlation can arise from economic dynamics as well as appraisal smoothing. An AR
  estimate alone cannot identify the cause.
- Repeated daily marks do not provide daily information about a quarterly appraisal process.
  Select the reporting grid before fitting lags.
- Full-sample coefficients and `INSAMPLE` means use later observations. Label them descriptive.
- Short samples, gaps, coefficient caps and large lag orders can dominate the result. Static
  estimation requires at least $4q$ nonmissing observations; that guard is not an adequacy test.
- A small inversion denominator amplifies noise. A sum above one can reverse signs in the static
  filter; severe diagnostics and retained initial rows require explicit handling.
- De-levering assumes constant debt/equity and one financing tier. Realised interest expense and
  leverage schedules require a more detailed model when economically material.
- Clipping adjusted returns changes their distribution. Neither de-levering nor unsmoothing
  creates liquidity, an executable price, or a historical valuation known at the time.

## See also

- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Frequency convention](frequency_convention_note.md)
- [Performance analytics](performance_analytics_and_sharpe.md)
- {doc}`Rolling unsmoother API <api/generated/qis.compute_ar_unsmoothed_prices>` and
  [source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/unsmoothing/ar_lag.py)
- {doc}`De-levering API <api/generated/qis.delever_returns>` and
  [source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py)

## References

1. Getmansky, M., Lo, A. W., and Makarov, I. (2004). An econometric model of serial correlation and illiquidity in hedge fund returns. *Journal of Financial Economics*, 74(3), 529–609. [DOI: 10.1016/j.jfineco.2004.04.001](https://doi.org/10.1016/j.jfineco.2004.04.001). [Author's copy](https://web.mit.edu/Alo/www/Papers/JFE2004Pub.pdf). Smoothing model and illiquidity interpretation; the AR implementation distinction is stated above.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
