---
myst:
  html_meta:
    description: >-
      Factsheet inputs, report selection, return conventions, reporting grids, and
      reproducible offline examples for qis portfolio and multi-asset analytics.
---

# Factsheets and reporting

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/QuantInvestStrats/commit/b04f87a11327bf814cc38ea46934c5b993ad4ef8)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A factsheet combines performance, risk and portfolio diagnostics into a set of related panels.
Its interpretation depends on the input history, benchmark, return convention and observation
grid. A common page layout alone does not make two analyses comparable.

## Overview

`qis.factsheet` selects a report from its input and applies a reporting-frequency preset.
The four forms answer different questions:

- **Multi-asset:** how do instruments compare with a reference series?
- **Single strategy:** how do a portfolio's returns relate to its holdings, turnover and costs?
- **Strategy versus benchmark:** how do two portfolio books differ?
- **Multi-strategy:** how do several portfolio variants compare on shared tables and axes?

The [reference](factsheets.md) lists the API choices. The [gallery](gallery.md) illustrates
the page forms and links to scripts for regenerating their analytics.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple returns build levels; table volatility uses log returns |
| Sampling grid | The reporting preset grid; running drawdowns use the native path |
| Annualisation | $\mathrm{af}$ of the base grid: 252, 52, 12 or 4 |
| Mean adjustment | Sample moments, demeaned, as in the performance statistics |
| Timing | Units are held between rebalancings; weights at $t$ apply over $(t,t+1]$ |
| Output units | Decimal returns, dimensionless ratios and two-sided turnover |
| qis default | `factsheet(reporting_frequency='monthly')`; the long preset for spans over five years |

### Inputs and the four report archetypes

| Input | Selected report | Benchmark information |
|---|---|---|
| Price or return `Series` or `DataFrame` | Multi-asset universe | A reference column via `benchmark`, or a separate series/panel via `benchmark_prices`. With neither supplied, the first input column is used. |
| `PortfolioData` | Single strategy | Supply `benchmark_prices`. |
| `MultiPortfolioData` | Multiple strategies | Store the reference price panel on the object for regime and beta panels. |
| `MultiPortfolioData` with `kind='strategy_benchmark'` | Strategy versus benchmark portfolio | The first two portfolios are the default pair; the object's reference price panel supplies market regimes/betas. |

`qis.factsheet` dispatches each archetype to a generator that can also be called directly:
`qis.generate_multi_asset_factsheet` for a price universe, `qis.generate_strategy_factsheet`
for one `PortfolioData`, `qis.generate_multi_portfolio_factsheet` for several strategies, and
`qis.generate_strategy_benchmark_factsheet_plt` for a strategy against a benchmark portfolio,
whose exposure panel uses `qis.plot_exposures_strategy_vs_benchmark_stack`. Direct calls take
their windows and grids from a preset such as `qis.FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD`
or from `qis.fetch_default_report_kwargs`.

Use a sorted `DatetimeIndex` and meaningful, unique string column names. Prices and NAVs
are positive levels by default. On the **multi-asset path only**,
`data_is_returns=True` treats both the pandas input and any separately supplied
`benchmark_prices` as simple fractional returns: 0.01 means 1%. Do not combine return
input with an unconverted benchmark price panel under that switch. Portfolio objects already
contain their calculated NAVs and accounting records; the switch does not rebuild them.

Let $P_t$ denote a price/NAV level at observation $t$, and $r_t$ its simple return from the
previous observation. State whether the supplied history includes trading costs, fees and
cash flows. Raw price inputs do not imply any portfolio rebalancing or cost model.

## Methodology

### From returns to levels

For a complete positive history with an explicit starting level,

$$
r_t = \frac{P_t}{P_{t-1}} - 1,
\qquad
P_t = P_0 \prod_{j=1}^{t}(1+r_j).
$$

The multi-asset return switch compounds with `qis.returns_to_nav` and its simple-return
defaults, from $P_0 = 1$. When a column's first return $r_1$ is observed and non-zero, the
facade places $P_0$ one period of the inferred native frequency earlier (a business day for
exchange trading days), so $r_1$ enters every statistic. A missing or zero first return, as
from `pct_change()` or `pct_change().fillna(0)`, already marks the date of $P_0$. Called
directly, `qis.returns_to_nav` keeps the returns index, so its first level is $1 + r_1$ unless
the input has such a row. Inspect missing observations before conversion: the helper has its
own initialisation and forward-fill behaviour, and the switch is not a general missing-data
policy. See [incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md).

### Frequency, annualisation, and missing values

Three time scales matter: the native observation path, the base grid for sampled statistics,
and the regime/heatmap bins. The reporting preset coordinates their settings; it does not
resample every visible curve to one common grid.

| Reporting choice | Base grid | Window periods per year | Annualisation factor |
|---|---|---:|---:|
| Daily | Business day (`B`) | 260 | 252 |
| Weekly | Wednesday week-end (`W-WED`) | 52 | 52 |
| Monthly | Month-end (`ME`) | 12 | 12 |
| Quarterly | Quarter-end (`QE`) | 4 | 4 |

Window lengths and EWM spans are sized from the window column. Volatility and Sharpe ratios
are annualised with `qis.get_annualization_factor`, which gives 252 for business days.

The facade infers the full available reporting span when `time_period` is omitted.
A span **greater than** `long_threshold_years` (default 5.0) selects the long preset;
otherwise it selects the short preset. In a monthly long report, volatility/variance spans,
rolling Sharpe windows and beta spans are 36 observations. Turnover and cost totals roll over
12 observations, regimes use quarterly bins, and the annual-return heatmap uses yearly bins.
In a monthly short report, those risk windows/spans are 12 observations and regimes use
monthly bins. EWM spans are decay parameters; they are not fixed-length rolling windows.

The [reporting-frequency convention](_included/reporting_frequencies.md) records every preset.
Running drawdown and time-under-water panels use the native price path; risk-table drawdowns
use the configured sampling grid, extended by each asset's final observation so that a loss in
the current, unfinished period is included. Earlier intramonth losses can still appear in the
native drawdown panel without appearing in month-end risk statistics. A reporting grid finer than
the inferred input frequency is rejected. This guard cannot detect information hidden by
forward-filled monthly observations in a daily index.

### Performance, costs and turnover

Default performance tables use log returns for volatility, compound annual return as the
headline return, and zero-rate Sharpe statistics. These are separate conventions, described
in [performance analytics and Sharpe ratios](performance_analytics_and_sharpe.md).
`add_rates_data=False` is the offline default. Enabling rate downloads does not change
the input price/return convention; inspect the resulting rate data before interpreting an
excess-return statistic.

Portfolio reports inherit their NAV, costs and turnover from `PortfolioData`. A reporting
frequency changes how those records are summarised, not the underlying rebalancing schedule.
Between rebalancings, qis holds units and weights drift with prices.

For investor-capital turnover, use two-sided executed notional divided by NAV. Managed
futures require full contract notionals through `turnover_unit_notional`. Gross-normalised
book churn and volatility-normalised signal turnover answer different questions; neither
should silently replace realised investor turnover. The [turnover methodology](
turnover_conventions.md) specifies these conventions and their inputs.

## Worked example

### Compounding is not addition

A 10% gain followed by a 10% loss leaves 99% of the initial capital:

$$
\frac{P_2}{P_0} = 1.10 \times 0.90 = 0.99.
$$

This complete monthly example includes a zero-return baseline and performs no filling.

```python
import numpy as np
import pandas as pd

import qis

returns = pd.Series(
    [0.0, 0.10, -0.10],
    index=pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31']),
    name='Illustration',
)
nav = qis.returns_to_nav(
    returns, is_log_returns=False, ffill_between_nans=False
)
np.testing.assert_allclose(nav.to_numpy(), [1.0, 1.10, 0.99])
assert np.isclose(nav.iloc[-1] / nav.iloc[0] - 1.0, -0.01)
```

### Minimal offline example

This report uses a clean subset of the frozen synthetic fixture. It is a demonstration of
report construction, not historical market performance. The 2018–2025 sample and seed are fixed.

```python
import matplotlib.pyplot as plt

import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
figures = qis.factsheet(
    prices,
    benchmark='SEQ_US',
    reporting_frequency='monthly',
    add_rates_data=False,
    factsheet_name='Synthetic assets | monthly reporting',
)
assert figures and all(isinstance(figure, plt.Figure) for figure in figures)
for figure in figures:
    figure.canvas.draw()
    plt.close(figure)
```

The result is a multi-asset report. Selecting an input column as a reference does not create
a strategy backtest. The [gallery workflow](gallery.md#build-the-four-report-types) constructs
portfolio objects for the other three forms.

## Implementation in qis

Without `file_name`, the facade returns a list of Matplotlib figures and writes no report
file. With a filename, it saves the generated pages to PDF and returns the path string.
Pass an explicit, existing `local_path`; the default otherwise uses the package's configured
output directory. The PDF helper appends the current date by default. A PDF generation date
does not change the analytics sample cutoff.

Page count depends on the report type, selected appendices and available history.
Strategy factsheets include a separate **Performance confidence and stability** page by
default from the 5.47.2 development source. The order is summary, full-history monthly-return
appendix when needed, inference, then other requested appendices. Set
`add_performance_inference_sheet=False` to retain the previous page set; page one and its
calculations are unchanged. The inference page has two columns:

| Panel | Sharpe diagnosis | Holdings-weighted alpha diagnosis |
|---|---|---|
| Summary | Full history, recent five years and recent three years: arithmetic Sharpe, IID-normal and HAC intervals | The same named windows: annualised mean residual return, HAC interval and two-sided p-value |
| Expanding history | Arithmetic Sharpe and HAC band | Mean residual alpha and HAC band |
| Rolling window | Arithmetic Sharpe and HAC band | Mean residual alpha and HAC band |

The left column uses `perf_params.freq_vol`; its intervals do not apply to the compound
per-annum Sharpe shown elsewhere in the report. The right column uses `freq_beta` and
`factor_beta_span` from the report. Each column labels its frequency and annualisation,
and tables disclose actual dates and observation counts. Alpha eligibility does not shorten
the standalone Sharpe sample. The appendix selects **only the first supplied benchmark**,
before fitting a single-factor model; it does not select a benchmark by its result or take
one coefficient from a multi-factor regression. Other pages retain their existing benchmarks.

### Holdings-weighted alpha inference

For strategy simple return $r_{p,t}$, benchmark simple return $r_{b,t}$ and period cash return
$r_{f,t}$, the raw exposure-adjusted return is

$$
a_t = (r_{p,t}-r_{f,t})
      - \widehat{\beta}^{\mathrm{holdings}}_{t-1}(r_{b,t}-r_{f,t}).
$$

The portfolio beta is the sum of instrument betas times actual signed holdings exposures as
fractions of NAV, without renormalisation. Instrument betas use EWMA-demeaned log returns,
point-in-time `X0` initialisation and a warm-up of `factor_beta_span` paired observations.
Available pre-report history can warm the estimates; the realised alpha sample starts only
inside the performance measurement period and never includes padded starting zeros. The beta
and holdings known at the opening of each return interval are applied one period later.
Using log-return betas for this simple-return attribution is a second-order approximation;
the residual accounting identity itself is exact on the supplied simple-return basis.

This definition recognises changes in allocations, leverage and position direction at the
selected beta-grid dates, using the holdings observed at those dates. Exposure changes within
an interval, omitted systematic factors and costs remain in the realised residual; it is not
a pure measure of manager skill. It is **not** the intercept of a constant-beta regression of portfolio
NAV returns, which averages realised exposures over its estimation sample. See
[holdings exposure versus model-layer NAV regression](model_layer_attribution.md#holdings-exposure-versus-model-layer-nav-regression).
For a long-only strategy, separating the specified market exposure can reveal why a positive
Sharpe does not by itself establish positive residual alpha. One benchmark does not remove
every possible systematic exposure.

Default inference uses the existing internal `qis.utils.regression.estimate_hac_mean` on **unsmoothed**
$a_t$, with Bartlett HAC covariance, its one-parameter finite-sample correction and a normal
reference distribution. The null is zero mean exposure-adjusted return. The mean, standard
error and confidence limits are annualised linearly by $\mathrm{af}$; the p-value is unchanged.
The displayed expanding and rolling curves repeat that mean test on their stated samples,
not on an already EWMA-smoothed alpha curve. Both return legs use the same supplied cash
rates, or explicitly assume zero cash. Costs already present in NAV are not deducted again.

All non-zero held instruments need finite, warmed-up betas and known holdings at the opening
date. Missing coverage makes that period's alpha unavailable; the appendix neither sums a
partial set of betas nor substitutes beta one or a NAV-regression estimate. Missing holdings
or a missing reference therefore produce a warning and unavailable alpha diagnostics, not
fabricated inference. The Sharpe side remains independently available where eligible.

### Inference sample safeguards and interpretation

Completed sampling boundaries are used; missing interior observations are not filled or
dropped. Minimum eligibility is 24 returns and two observation-equivalent years, with an
additional HAC requirement of at least five observations per lag term. Automatic lag counts
use the existing Newey–West rule. These are display safeguards, not finite-sample coverage
guarantees. A five-year rolling window falls back to three or two complete calendar years
with a warning; unavailable or undefined estimates remain labelled as such rather than being
replaced by IID inference. The expanding chart normally warms up for three observation-equivalent
years, falling back to the eligible minimum with a warning. The two columns can therefore
have different actual sample dates, warm-ups and fallback windows.

Bands are pointwise, not simultaneous; overlapping windows are dependent and neither p-values
nor intervals adjust for strategy selection or repeated inspection. HAC inference remains
asymptotic and exploratory: it is not a calibrated certificate of skill, and it does not
propagate uncertainty about an unobserved true beta or a selected exposure model. The target
is the mean of the specified, realised holdings-beta-adjusted return process. The existing
[finite-sample qualification limits](factor_and_backtest_significance.md) remain in force:
Stage 5 is incomplete, and this reporting addition does not change legacy estimator defaults
or acceptance gates.

Clickable page footnotes link to [Sharpe inference](performance_analytics_and_sharpe.md#single-strategy-confidence-intervals),
the holdings/model-layer distinction and the significance guide. The canonical
[`VOLPARITY_PORTFOLIO` example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/strategy.py)
generates this appendix with SPY, the first benchmark, while retaining the other report
benchmarks. It uses the optional free-data download and the existing `run_local` convention;
generated report previews and supporting outputs remain outside the source checkout.

### Paired strategy-benchmark inference

From the 5.47.2 development source, `generate_strategy_benchmark_factsheet_plt` adds a
default-on **Incremental performance** page **before Brinson**, leaving page one unchanged.
Set `add_performance_inference_sheet=False` to omit inference both here and in any standalone
strategy factsheets appended by this call. The flag retains its previous forwarding behaviour.

The comparison portfolio is `portfolio_datas[benchmark_idx]`, subtracted from
`portfolio_datas[strategy_idx]`. This is distinct from the **market reference**: the first
column of `multi_portfolio_data.benchmark_prices`, used for both holdings-beta fits.
For example, compare TAA with SAA while using SPY to remove the same market exposure.
No benchmark is selected by its p-value or outcome.

| Row | Paired Sharpe difference | Marginal holdings alpha |
|---|---|---|
| Summary table | Full, recent five and recent three years: arithmetic Sharpe difference, paired HAC 95% CI and two-sided p-value | The same named windows: annualised mean alpha difference, HAC 95% CI and two-sided p-value |
| Expanding history | Expanding paired Sharpe difference and pointwise HAC band | Expanding mean alpha difference and pointwise HAC band |
| Rolling window | Five-year paired Sharpe difference and pointwise HAC band | Five-year mean alpha difference and pointwise HAC band |

Positive values favour the strategy. The nulls are equal arithmetic Sharpes and zero mean
incremental exposure-adjusted return, respectively. Let $s$ denote strategy, $c$ comparison,
and $m$ the common market reference. The raw marginal payoff is

$$
\Delta a_t = a_{s,t}-a_{c,t}
 = r_{s,t}-r_{c,t}
 - (\widehat{\beta}^{\mathrm{holdings}}_{s,t-1}
    -\widehat{\beta}^{\mathrm{holdings}}_{c,t-1})(r_{m,t}-r_{f,t}).
$$

The same cash return cancels between the two portfolio legs but remains in the market excess
return. The [holdings-alpha definition](#holdings-weighted-alpha-inference), lagged exposure
timing, EWMA demeaning and per-instrument warm-up apply separately to each portfolio.
Both portfolios must have complete holdings-beta coverage on a date for marginal alpha to
be available. NAV costs are retained without further deductions; comparable fee, cash and
currency bases remain the caller's responsibility. Intraperiod exposure changes and omitted
factors can remain in this residual: it is not pure manager skill or a NAV-regression alpha.

The left column delegates to public `qis.sharpe_difference_test` on the two aligned return
series. Its paired influence series preserves cross-portfolio covariance; subtracting two
standalone CI bounds or adding their variances would be wrong. The right column tests the
**unsmoothed difference** directly through the canonical internal mean-only
`qis.utils.regression.estimate_hac_regression(x=None, span=None)`, using Bartlett HAC and the
one-parameter finite-sample correction. Alpha mean/SE/bounds scale by $\mathrm{af}$;
Sharpe differences/SE/bounds by $\sqrt{\mathrm{af}}$. Identical non-degenerate portfolios
give difference zero, zero SE and p-value one, not independent-sample uncertainty.

Sharpe uses `perf_params.freq_vol`; alpha uses the report's `freq_beta` and
`factor_beta_span` (direct-call defaults `B` and 260; reporting presets supply their own).
Each test uses paired dates within its column; alpha warm-up does not shorten the Sharpe
sample. All [sample safeguards](#inference-sample-safeguards-and-interpretation) apply,
including no condensation of interior gaps, minimum 24 returns and two observation-equivalent
years, and the warned 5/3/2-year rolling fallback. Thus quarterly three- and five-year windows
are unavailable under the 24-observation minimum. CIs remain exploratory, pointwise and
asymptotic; repeated windows and the two diagnostics are not a multiplicity-adjusted family.
No further calibration is performed, Stage 5 remains incomplete, and legacy defaults and
acceptance gates are unchanged.

Run `Locals.PAIRED_INFERENCE` through `run_local` in the
[strategy-benchmark example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/strategy_benchmark.py).
It compares VolParity with EqualWeight on free Yahoo ETF prices, uses SPY as the market
reference, monthly reporting, zero cash and the same stated trading-cost setting. Output goes
through YAML-configured `qis.local_path`, or the explicit `output_path` override. The three-page
result is summary, paired inference, then Brinson. Footnotes link to this guide, holdings-beta
methodology and HAC limitations. The report-only adapter is
[`_paired_performance_inference.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/_paired_performance_inference.py);
the estimators remain in their canonical modules.

### Multi-strategy inference against a base

From the 5.47.2 development source, `generate_multi_portfolio_factsheet` adds a default-on
**Incremental performance vs base** overview immediately after its unchanged first page,
before group reports and standalone strategy appendices. `base_portfolio_id=0` selects the
first portfolio by position, not by its observed performance. All other portfolios are
compared with this base in their original order; the base has no self-comparison row.

| Row | Paired Sharpe difference | Marginal holdings alpha |
|---|---|---|
| Full eligible history | Each portfolio minus base: estimate, HAC 95% CI, raw and Holm p-values | Same comparisons, after subtracting the difference in lagged holdings-market exposure |
| Recent five years | Same statistics over the recent paired sample | Same statistics over the recent paired alpha sample |
| Recent three years | Same statistics over the recent paired sample | Same statistics over the recent paired alpha sample |

Each panel is a compact forest plot with numerical results, actual dates, observation count
and HAC lag. At most six non-base portfolios appear on a page; additional portfolios continue
on subsequent overview pages, without selecting winners or omitting poor results. Axis scales
are shared across windows and pages within each metric, but Sharpe and alpha have different
units. Portfolio positions identify rows even if display names coincide or are shortened.

The [paired estimators and marginal-payoff identity](#paired-strategy-benchmark-inference)
are unchanged. The first **market benchmark** in `benchmark_prices` supplies instrument betas
for both legs; the **base portfolio** is the comparison, not a replacement regression factor.
Actual signed holdings, EWMA demeaning, span-length warm-up, prior exposures and complete
coverage remain required. No NAV-regression or beta-one fallback is introduced. Sharpe uses
`perf_params.freq_vol`; alpha uses `freq_beta` and `factor_beta_span`. Supply comparable cash,
currency, costs and mandate conventions. Positive differences favour the non-base portfolio.

Each portfolio/base pair uses its own eligible sample within the requested window; a late
strategy does not truncate earlier pairs. Leading/trailing missing data are trimmed, not
interior gaps. Alpha eligibility does not shorten the paired Sharpe sample. The same minimum
24 returns, two observation-equivalent years and HAC requirement `N >= 5*(L+1)` apply. An
unavailable test remains **N/A** with a warning, not zero or an invented non-rejection. Since
pairs can have different dates, the overview is not a matched-universe ranking.

The overview predeclares **one Holm family** before computing any results: all non-base
portfolios × two metrics × three windows, including every page. With $K$ comparisons there
are $6K$ tests. The canonical `adjust_pvalues` applies Holm once to their two-sided raw
p-values, allowing dependence among comparisons and overlapping windows. If **any** declared
p-value is unavailable, all Holm values are N/A; valid raw results are still displayed. The
family is never reduced, split by page or completed with fabricated p-values. The 95% CIs are
**pointwise**, not simultaneous or Holm-adjusted. Holm's familywise error-control justification
requires valid input p-values: it does not repair finite-sample HAC miscalibration, account
for a hidden strategy search, or turn these diagnostics into a certificate of skill.

Set `add_performance_inference_sheet=False` to suppress the overview and inference in
appended standalone factsheets. Independently, `add_pairwise_inference_sheets=True`
(default False) adds the existing expanding/rolling paired page for each non-base portfolio,
after the overview and before other appendices. Detail-page p-values remain raw; repeated
curve endpoints are not extra members of the overview's declared family. A single-portfolio
input warns and skips comparison pages. Other page-one settings are not changed by the base.

Run `Locals.PERFORMANCE_INFERENCE` through `run_local` in the
[multi-strategy VolParity example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/multi_strategy.py).
It compares spans 10/20/40/60/120 with span 5 on free Yahoo ETF prices, using SPY as market
reference, monthly reporting, explicit zero cash and equal trading-cost settings. It produces
the summary plus one overview page; output follows YAML-configured `qis.local_path` or the
explicit `output_path` argument. Footnotes link to this section, holdings-alpha methodology
and the significance guide. The private report adapter
[`_multi_performance_inference.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/_multi_performance_inference.py)
only assembles canonical paired estimators and the established Holm correction. No further
calibration is performed; Stage 5 remains incomplete, and legacy estimator defaults and
acceptance gates are unchanged.

### Configurable alpha inference and same-sample sensitivity

All three inference-enabled report generators accept the keyword-only `alpha_inference`:
`generate_strategy_factsheet`, `generate_strategy_benchmark_factsheet_plt`, and
`generate_multi_portfolio_factsheet`. It is also forwarded through `factsheet` and to any
embedded standalone or pair-detail pages. The option belongs to the alpha column only;
the arithmetic Sharpe estimand, IID option, paired covariance and Bartlett HAC calculations
are unchanged, as is page one.

<!-- docs-test: skip -->
```python
from qis.perfstats.significance import InterceptInference

figures = qis.generate_strategy_factsheet(
    portfolio_data=portfolio,
    benchmark_prices=benchmarks,  # Only the first market reference supplies holdings betas.
    alpha_inference=InterceptInference.PREWHITENED_AR1,
)
# Omit alpha_inference (or pass BARTLETT_HAC) to retain the existing default.
```

| Alpha method | What changes | What does not change |
|---|---|---|
| `BARTLETT_HAC` (default; **B**) | Bartlett covariance of the centred raw alpha returns | The holdings-adjusted payoff, arithmetic sample mean, eligible dates, annualisation and Sharpe |
| `PREWHITENED_AR1` (opt-in; **PW**) | AR(1) filter on the intercept influence, Bartlett covariance of its innovations, then recolouring | The same payoff and mean; it does not re-estimate holdings betas or regress portfolio NAV on a factor |

The report always compares both methods on **one validated sample** per window/endpoint,
using the same original observation count, dates and automatic Newey–West lag count.
Prewhitening produces one fewer innovation; it does not remove another return from the
reported sample. Both methods retain the original-sample finite-sample normalisation and a
normal reference. The selected method supplies the primary CI and p-value. The other is
labelled sensitivity: a second CI/p-value in the alpha table, dashed bounds in the time-series
panels, and dashed intervals plus raw p-values in the multi-strategy overview. Annual alpha
means, SEs and bounds are in return units per year, not Sharpe units. Internal report tables
retain the SEs and fitted AR coefficient for reproducibility.

The multi-strategy Holm family remains **6K primary tests**: K portfolio/base pairs, two
metrics, three windows. Only the selected alpha method enters that family; alternate-method
p-values are **unadjusted sensitivity diagnostics outside Holm**, not a second inferential
claim or a reason to select the smaller p-value. Choose the primary method before inspecting
results. Missing primary results disable all family adjustments as before; failure of an
alternate fit does not invalidate otherwise available primary tests.
Raw Sharpe estimates, CIs and p-values are identical across alpha settings. Their **Holm-adjusted**
p-values can change when primary alpha p-values change, because Holm operates on the joint
predeclared family; this is not a change to the Sharpe estimator.

Prewhitening can help when the **alpha return process** has persistent serial dependence.
Slow-moving weights or betas alone do not establish that. An AR(1) filter adds estimation
noise, can be unstable and need not capture all dependence; it is not uniformly preferable,
particularly in short samples. The existing minimum-sample rules apply to both methods.
Degenerate influence, unstable AR coefficient or invalid variance yield N/A and a warning,
without clipping, method substitution or a manufactured p-value. An exactly identical paired
portfolio retains Bartlett's neutral zero difference/SE and p=1; its degenerate AR fit stays
unavailable. Neither method propagates uncertainty in an unknown true beta. See the
[prewhitening derivation and limitations](factor_and_backtest_significance.md#ar1-prewhitening-of-the-intercept-score).

Do not assume every call produces one page. Close figures after saving or inspection in
batch processes.

For direct generator calls, use `qis.fetch_default_report_kwargs` to build the preset.
Caller overrides take precedence, so overrides to one frequency/window should be reviewed
alongside the related fields. See the [full-control example](factsheets.md#full-control).

The [facade source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/factsheet_facade.py)
defines dispatch and output handling; the
[configuration source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/config.py)
defines presets. The [batch analytics scripts](https://github.com/ArturSepp/QuantInvestStrats/tree/main/tools/docs_analytics)
generate figures, supporting tables and provenance together.

## Interpretation and limitations

<a id="how-to-interpret-and-choose-a-report"></a>
<a id="constraints-and-common-failure-modes"></a>

- A single-strategy `PortfolioData` report without `benchmark_prices` raises
  `ValueError`. A `kind` override must match the supplied object; it does not construct
  portfolios from raw prices.
- In multi-portfolio reports, the comparison portfolio and the market reference series are
  separate inputs. Supply the reference panel on `MultiPortfolioData`; the facade's
  standalone `benchmark_prices` argument is used on the raw-pandas and single-strategy paths.
- Missing observations, stale values and ragged histories can give panels different effective
  samples. Neither the facade nor its frequency guard certifies tradability or data quality.
- `add_rates_data=True` attempts a download through the optional `data` extra.
  Missing yfinance or an empty result can leave `rates_data=None`. Verify coverage and
  conventions; do not infer successful excess-return calculation from that flag alone.
- Dense tables and legends require visual review at the intended display size. Layout tests
  check geometry, not readability. Reduce the series count or adjust documented display settings.
- Synthetic reports demonstrate behaviour under specified inputs. Their returns, Sharpe ratios
  and drawdowns are not estimates of an investable strategy's expected performance.

## See also

- [Factsheet API and configuration reference](factsheets.md)
- [Factsheet gallery and regeneration](gallery.md)
- [Frequency and annualisation](frequency_convention_note.md)
- [Portfolio backtesting](portfolio_backtesting.md)
- [Brinson attribution](brinson_attribution.md)
- [Documentation and figure standard](documentation_standard.md)

## References

1. Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd edition. Wiley. The performance-measurement conventions behind factsheet statistics.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
