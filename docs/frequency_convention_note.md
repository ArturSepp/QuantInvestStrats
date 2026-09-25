---
myst:
  html_meta:
    description: >-
      How reporting frequency affects volatility, Sharpe ratios and higher moments,
      and how qis calibrates factsheet windows, frequency guards and panel labels.
---

<a id="performance-statistics-are-frequency-relative-a-reporting-convention-for-internally-consistent-factsheets"></a>

# Performance statistics and reporting frequency

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-06-19](https://github.com/ArturSepp/QuantInvestStrats/commit/a39da9f97a11a25848e1d6ff8bc644f7501d53a8)*

Reporting frequency is the sampling grid used to estimate and label performance
statistics. This article explains its statistical consequences and the factsheet
convention implemented in [qis](https://github.com/ArturSepp/QuantInvestStrats).
Software reference: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Overview

<a id="abstract"></a>

Annualised volatility, Sharpe ratios, skewness, beta and correlation depend on the
return convention, observation frequency and estimation window. A reported
volatility of 12% is therefore incomplete without those choices. Frequency affects
both the information available to an estimator and, under serial dependence, the
population quantity it estimates.

The qis convention makes reporting frequency an explicit input, derives window
settings from that frequency and the report horizon, and labels panels that use a
different grid. It also rejects reporting frequencies finer than the observed data
and widens the multi-asset report's trailing correlation window at coarse frequencies.

<a id="2-the-silent-mixing-failure-mode"></a>

### The silent-mixing failure mode

A report may show native daily drawdowns, monthly Sharpe ratios and quarterly
regime statistics. Each can be useful, but their grids must be visible. In
particular, a native drawdown preserves observed intraperiod troughs that a
quarterly series can omit. Panel labels let the reader distinguish that intended
difference from an accidental mismatch between estimates.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns for the aggregation identities; tables follow `PerfParams.return_type` |
| Sampling grid | The reporting grid: `B`, `W-WED`, `ME` or `QE` |
| Annualisation | $\mathrm{AN}_f$ = 252, 52, 12 or 4; report windows count 260, 52, 12 or 4 per year |
| Mean adjustment | Population identities; the estimators are those of the performance chapter |
| Timing | Not applicable: reporting configuration |
| Output units | Annualised decimals and dimensionless ratios |
| qis default | Monthly reporting; the long preset for spans over `long_threshold_years=5` |

| Symbol or setting | Definition |
|---|---|
| $f$, $\mathrm{AN}_f$ | Sampling frequency and its annualisation factor from `qis.get_annualization_factor`: business daily 252, weekly 52, monthly 12 or quarterly 4. |
| $r_t$ | Simple return over one observation period, in decimal units. |
| $\ell_t = \log(1+r_t)$ | Log return, additive over adjacent periods when wealth is positive. |
| $\hat\sigma_f$ | Sample standard deviation at frequency $f$, for the stated return convention. |
| $k$, $S_k$ | Number of adjacent periods and their aggregate log return, $S_k = \sum_{i=1}^{k}\ell_i$. |
| $\rho_j$, $\mathrm{VR}(k)$ | Lag-$j$ autocorrelation of $\ell_t$ and its $k$-period variance ratio. |
| $\gamma_1$, $\gamma_2$ | Population skewness and excess kurtosis of a one-period log return. |
| $n_{\min}$ | Target minimum observation count for the trailing correlation window; default 12. |

The variance-ratio identity assumes covariance stationarity and finite variance.
The higher-moment scaling below additionally assumes independent, identically
distributed increments with finite third and fourth moments. These assumptions
describe theoretical log-return aggregation; they are not guarantees about a
sample of compounded simple returns. State the convention explicitly when using
`qis.to_returns(..., is_log_returns=...)`.

<a id="3-two-axes-reporting-frequency-and-horizon"></a>

### Two axes: reporting frequency and horizon

Reporting frequency selects the return grid and annualisation factor. The report
horizon selects window lengths and display settings. In
`fetch_default_report_kwargs`, an unspecified report period selects the long
preset; otherwise a period exceeding `long_threshold_years` (default five years)
selects long, and a shorter or equal period selects short. Explicit overrides can
change the presets and should be described in the report.

## Methodology

<a id="1-statistics-are-frequency-relative"></a>

### Statistics are frequency-relative

The usual annualised volatility estimate is

$$
\hat\sigma_{\mathrm{ann}}(f) = \hat\sigma_f\sqrt{\mathrm{AN}_f}.
$$

Under independent additive increments, population variance scales with elapsed
time. This does **not** make sample annualised volatilities identical across
frequencies: aggregation reduces the number of observations and changes sampling
error. Nor is unbiasedness of the sample standard deviation implied by an
unbiased sample variance. Exact additive scaling applies to log returns, whereas
multi-period simple returns compound.

With serial dependence, even the population scaling changes. For stationary log
returns, the variance ratio is

$$
\mathrm{VR}(k)
= \frac{\operatorname{Var}(S_k)}{k\,\operatorname{Var}(\ell_t)}
= 1 + 2\sum_{j=1}^{k-1}\left(1-\frac{j}{k}\right)\rho_j.
$$

When the coarse annualisation factor is $\mathrm{AN}_f/k$, its population annualised
log-return volatility equals the fine-frequency value times
$\sqrt{\mathrm{VR}(k)}$. Sample estimates need not satisfy that identity exactly.
A positive weighted sum of autocorrelations raises the variance ratio above one;
a negative weighted sum lowers it. Variance-ratio analysis is discussed by
[Lo and MacKinlay (1988)](https://doi.org/10.1093/rfs/1.1.41).

Illiquid or model-marked assets can exhibit return smoothing that depresses
observed short-horizon volatility; this is not necessarily resolved by changing
the reporting grid.
[Getmansky, Lo and Makarov (2004)](https://doi.org/10.1016/j.jfineco.2004.04.001)
model the serial correlation induced by illiquidity. Serial dependence also
invalidates the usual square-root-of-time scaling of a Sharpe ratio in general;
[Lo (2002)](https://doi.org/10.2469/faj.v58.n4.2453) treats its sampling and
annualisation consequences. A frequency label alone does not apply a statistical
correction for either effect.

<a id="4-the-convention-one-stated-frequency-applied-by-lookup"></a>

### The convention: one stated frequency, applied by lookup

The configuration derives the following presets from frequency and horizon.
Counts are periods of the indicated grid, shown as **long · short**.

| Reporting frequency | Sampling grid | Vol / Sharpe span or window | Beta span | Regime grid | Window periods/year |
|---|---|---|---|---|---|
| Daily | `B` | 260 · 260 | 780 · 260 | quarterly · monthly | 260 |
| Weekly | `W-WED` | 156 · 52 | 156 · 52 | quarterly · monthly | 52 |
| Monthly | `ME` | 36 · 12 | 36 · 12 | quarterly · monthly | 12 |
| Quarterly | `QE` | 12 · 4 | 12 · 4 | quarterly · monthly | 4 |

The last column sizes windows and spans only. It is not the annualisation factor $\mathrm{AN}_f$:
a daily preset uses 260-observation windows, while its volatility is annualised with 252.

The volatility and variance parameters are exponentially weighted spans; the
Sharpe parameter is a rolling window. Matching their counts does not make their
weighting kernels identical. Beta uses a longer span than volatility in the daily
long preset and the same count in the other presets shown here.

Regime classification follows the horizon axis independently: quarterly for long
reports, monthly for short reports. It is **not always coarser** than the reporting
grid. A quarterly short report still requests monthly regimes, so the underlying
data must support the intended regime analysis. The table is a configuration
contract, not evidence that every input contains enough information for every
panel.

<a id="5-the-up-sampling-guard-as-a-correctness-invariant"></a>

### The up-sampling guard as a correctness invariant

The report validator compares the requested grid with the input's inferred native
sampling frequency. It rejects a daily or weekly report from monthly observations;
equal or coarser reporting grids are accepted. Interpolating a monthly NAV cannot
recover its unobserved daily path.

This guard concerns observed timestamp frequency. It cannot establish economic
information frequency: a forward-filled daily series may still contain only
monthly marks. Irregular dates, stale prices and mixed-frequency columns require
the separate input checks described in
[incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md).

<a id="6-what-changes-with-frequency-and-what-does-not"></a>

### What changes with frequency, and what does not

Total return over fixed, retained endpoints is unchanged by regrouping simple
returns: compounding telescopes to the endpoint price ratio. Resampling that drops
an endpoint or changes missing-data treatment is a different comparison.
Volatility, Sharpe, beta, correlation and higher moments can change with frequency.

For independent, identically distributed log returns with the required finite
moments, additive aggregation gives

$$
\mathrm{skew}(S_k) = \frac{\gamma_1}{\sqrt{k}}, \qquad
\mathrm{excess\ kurtosis}(S_k) = \frac{\gamma_2}{k}.
$$

These are population identities, not exact relations between sample statistics.
The familiar division of daily skewness by $\sqrt{21}$ is therefore only an
illustration under those assumptions, not a conversion rule for observed monthly
simple-return skewness. Factsheets estimate skewness on the chosen return grid.

<a id="7-estimation-under-coarse-sampling-trailing-window-adaptation"></a>

### Estimation under coarse sampling: trailing-window adaptation

The multi-asset report widens its trailing correlation window using

$$
\mathrm{trailing\ years}
= \max\left(1,\left\lceil\frac{n_{\min}}{\mathrm{AN}_f}\right\rceil\right).
$$

With the default $n_{\min}=12$, the nominal window is one year at daily, weekly
and monthly frequency and three years at quarterly frequency. This rule applies
to the trailing correlation panel, not to every rolling statistic or recent
performance table. Missing data and a short history can still leave fewer than
12 usable paired returns. A longer window improves observation count at the cost
of responsiveness and does not guarantee a well-conditioned correlation matrix.

<a id="8-per-panel-labelling-discipline"></a>

### Per-panel labelling discipline

Rolling volatility, Sharpe, beta, correlations and return-scatter panels identify
their reporting grid and relevant window. Turnover and cost panels identify their
own sampling grid; regime panels identify their classification grid.

Running drawdown and time-under-water panels use the native observed NAV path and
label that grid. A resampled risk-table drawdown can differ because it omits
intraperiod troughs; the native panel and sampled table describe different paths.
Visible cumulative and annualised returns retain observed endpoints, while
frequency-based ratios use complete reporting boundaries on the asset's observed
support. The [packaged reporting note](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/reporting_frequencies.md)
records these implementation conventions.

## Worked example

The following offline example inspects the quarterly long-report preset. It
fetches configuration only and explicitly disables rate-data downloads.

```python
import qis

settings = qis.fetch_default_report_kwargs(
    time_period=None,
    reporting_frequency=qis.ReportingFrequency.QUARTERLY,
    add_rates_data=False,
)
assert settings["vol_rolling_window"] == 12
assert settings["sharpe_rolling_window"] == 12
assert settings["factor_beta_span"] == 12
assert settings["freq_regime"] == "QE"
```

Twelve quarterly periods represent three years. The same long preset at monthly
frequency uses 36 periods. Separately, the default trailing correlation rule gives
$\max(1,\lceil 12/4\rceil)=3$ years for quarterly data. These settings make the
calendar horizons comparable; they do not make their observation counts or
statistical uncertainty equal.

## Implementation in qis

<a id="9-implementation"></a>

`qis.ReportingFrequency` names the four frequencies and
`qis.fetch_default_report_kwargs` exposes the preset lookup. The
[`config.py` implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/config.py) contains
`make_factsheet_config`, the frequency validator and the underlying field schema.
`qis.factsheet` accepts a reporting frequency and generates the appropriate report
from prices, returns or stored portfolio histories.

The [reporting-frequency note](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/reporting_frequencies.md) documents
configuration details and usage. The
[multi-asset report implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/multi_assets_factsheet.py)
contains the trailing correlation rule. The
[reporting convention tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/test_reporting_conventions.py) cover
preset propagation, frequency guards and labels. For return and rate conventions,
including the three labelled Sharpe variants, see
[performance analytics and Sharpe](performance_analytics_and_sharpe.md).

## Interpretation and limitations

<a id="10-discussion-and-limitations"></a>

- The convention standardises reporting choices. It does not remove sampling
  uncertainty, serial correlation, illiquidity or differences in return/rate basis.
- Horizon thresholds and regime grids are configurable reporting choices. Disclose
  overrides when comparing mandates or report versions.
- More observations are not necessarily more independent information. Check
  stale prices and smoothing before interpreting a fine-grid statistic.
- The frequency guard and trailing-window rule have distinct scopes. Neither
  promises enough observations for every asset, estimator or regime.
- For strongly dependent returns, consider an explicitly specified dependence
  adjustment or several sampling frequencies. The ordinary factsheet annualisation
  factor is not such an adjustment.

## See also

- [Factsheets and reporting](factsheets_and_reporting.md).
- [Performance analytics and Sharpe](performance_analytics_and_sharpe.md).
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md).
- [Private-asset unsmoothing](private_asset_unsmoothing.md).

## References

1. Getmansky, M., Lo, A. W., and Makarov, I. (2004). An econometric model of serial correlation and illiquidity in hedge fund returns. *Journal of Financial Economics*, 74(3), 529–609. [DOI: 10.1016/j.jfineco.2004.04.001](https://doi.org/10.1016/j.jfineco.2004.04.001).
2. Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36–52. [DOI: 10.2469/faj.v58.n4.2453](https://doi.org/10.2469/faj.v58.n4.2453).
3. Lo, A. W., and MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test. *Review of Financial Studies*, 1(1), 41–66. [DOI: 10.1093/rfs/1.1.41](https://doi.org/10.1093/rfs/1.1.41).
4. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
