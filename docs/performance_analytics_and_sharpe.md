---
myst:
  html_meta:
    description: >-
      Define performance, drawdowns, and three Sharpe conventions, with explicit return,
      sampling, funding, and annualisation contracts and reproducible qis examples.
---

# Performance analytics and Sharpe conventions

*[author / affiliation / date — placeholder]*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Performance analytics summarises the return, variability, and losses of a price or net asset
value (NAV) history. A Sharpe ratio relates a return measure to its variability; its meaning
depends on the return, funding, sampling, and annualisation conventions. This article describes
the conventions in qis's static tables and distinguishes them from rolling and regime displays.

## Overview

Use this layer to describe an observed or simulated history. It does not construct allocations,
recover missing economic observations, or turn a full-sample estimate into a forecast.

[Sharpe (1994)](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm) defines the historical ratio
using the mean and standard deviation of periodic differential returns. qis also provides
compound-return and log-return reporting ratios. Their named columns make that choice visible;
the three numbers need not agree.

<a id="data-and-calculation-contract"></a>

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units or convention |
|---|---|---|
| $P_t$ | Positive price or NAV at time $t$ | Any consistent scale per asset; columns identify assets or strategies |
| $r_t$ | Simple periodic return | Decimal fraction |
| $\ell_t$ | Log periodic return | Log price ratio |
| $s(x)$ | Sample standard deviation of a return series | `ddof=1` |
| $a$ | Periods per year used for sampled risk | 12 for `ME`; inferred from the sampled index |
| $v_t$ | Returns selected for table volatility | Log by default; simple with `ReturnTypes.RELATIVE` |
| $Y$ | Elapsed years between the ratio's sampled endpoints | Calendar days divided by 365.25 |
| `rates_data` | Cash rate used by excess-return calculations | Annualised decimal rate, applied with a one-period lag |

Inputs are a `pandas.Series` or `pandas.DataFrame` with a sorted `DatetimeIndex`.
A risk output of `0.10` means 10%; Sharpe ratios are dimensionless.

`PerfParams.return_type` chooses the return basis for volatility and higher moments.
`freq_vol`, `freq_skewness`, `freq_drawdown`, and `freq_reg` are separate sampling grids.
A 260-observation business-day rolling window is a window choice; it does not set the
volatility annualisation factor. The generic business-day risk factor is 252 unless overridden
by the relevant API. See the [frequency convention](frequency_convention_note.md).

Static tables evaluate each asset on its own observed support. Their visible return columns
retain native start/end observations. P.a., log, excess, and Sortino **ratio numerators** instead
use the complete `freq_vol` boundaries used by the risk denominator. Dividing a visible
`PA_RETURN` column by `VOL` need not reproduce `SHARPE_RF0` when native endpoints fall between
reporting boundaries.

Interior gaps retain the table's established filling policy, but an asset is not extended beyond
its last sampled observed price merely because another column continues. Benchmark regressions
use joint sampled support. Elsewhere, `qis.to_returns` forward-fills by default; use
`ffill_nans=False` when a gap must remain missing. Filling a stale mark creates no new information.

## Methodology

### Returns and annualised volatility

The two return transformations and the table's volatility are:

$$
r_t=\frac{P_t}{P_{t-1}}-1,
\qquad
\ell_t=\log\left(\frac{P_t}{P_{t-1}}\right),
\qquad
\sigma_v=\sqrt{a}\,s(v).
$$

Simple returns compound through time; log returns add through time. Neither permits unrestricted
addition of asset log returns to obtain a portfolio log return.

### The three Sharpe objects

Let $P_s,P_e$ be the first and last complete sampled boundaries for a particular asset.
The table's p.a. return helper uses the following $C$ on that support:

$$
C=
\begin{cases}
(P_e/P_s)^{1/Y}-1, & Y>1,\\
P_e/P_s-1, & 0<Y\leq 1.
\end{cases}
$$

Thus histories of at most one year retain their total return by default; they are not
automatically extrapolated to a one-year return. The standalone `compute_pa_return` helper
offers `annualize_less_1y=True` for linear scaling, but the static table does not request it.

The zero-rate static-table ratios are:

$$
S_{\mathrm{PA}}=\frac{C}{\sigma_v},
\qquad
S_{\mathrm{arith}}=\frac{\sqrt{a}\,\overline r}{s(r)},
\qquad
S_{\mathrm{log}}=\frac{\log(1+C)}{\sigma_v}.
$$

| Convention | Full-table columns | Interpretation |
|---|---|---|
| P.a. / compound | `SHARPE_RF0`, `SHARPE_EXCESS` | Compound-return reporting numerator over the selected table volatility |
| Arithmetic | `SHARPE_ARITH`, `SHARPE_ARITH_EXCESS` | Matched mean and standard deviation of simple returns or simple excess returns |
| Log | `SHARPE_LOG_AN`, `SHARPE_LOG_EXCESS` | Log of the compound-return numerator over the selected table volatility |

The p.a. and log columns share `VOL`. Consequently, switching `return_type` to
`ReturnTypes.RELATIVE` also changes the denominator of the log-labelled ratio. It does not
change the arithmetic family's matched simple-return denominator. Even with log volatility,
calendar-year endpoint scaling need not exactly equal scaling the sample mean by $a$.

The `_RF0` column assumes zero cash return. Excess p.a./log columns replace the numerator with
the corresponding compounded excess-return measure and retain the selected table volatility.
The arithmetic-excess family uses both the mean **and** standard deviation of simple excess
returns. Funding uses the lagged annual rate times the elapsed calendar-day fraction over 365;
this funding day count is separate from the 365.25-day return annualisation above. With no cash
series, each excess column coincides with its zero-rate counterpart.

`PerfParams.sharpe_convention` selects the convention for regime-conditional Sharpe displays.
`compute_ra_perf_table` keeps all named families side by side. Choose an explicit `PerfStat`
column when consuming its results.

### Rolling statistics and drawdowns

`compute_rolling_perf_stat` returns `(data, label)`. For volatility, Sharpe, skewness, and total
returns, `roll_periods` counts observations on `roll_freq`: 36 on `ME` is a 36-month window.
The `PA_RETURNS` branch instead applies its window to the supplied native price rows; resample
prices explicitly before calling that branch when a different grid is required. `EWMA_VOL`
uses an exponentially weighted estimator rather than a fixed-window sample standard deviation.

Rolling `SHARPE` does not read `PerfParams`. Its current helper uses log returns and
$\sqrt{a}\,[\exp(\overline{\ell})-1]/s(\ell)$ within each window. This is a separate reporting
calculation from the three static-table ratios above.

Running drawdown measures the loss from the running observed peak:

$$
D_t=\frac{P_t}{\max_{s\leq t}P_s}-1.
$$

`compute_rolling_drawdowns` starts at the first valid price and carries the drawdown state over
later missing prices. `compute_drawdowns_stats_table` adds episode dates, depths, and recovery
durations. A drawdown of −20% describes a level below the peak; it is not a return to subtract
from the NAV again.

<a id="minimal-offline-example"></a>

## Worked example

Three monthly simple returns of 5%, −2%, and 5% have mean $0.08/3$, annualised sample
volatility 14%, and an arithmetic Sharpe of $0.32/0.14=16/7$, approximately 2.285714.
The following fixed arithmetic illustration checks the named qis column. Three returns are
enough to check arithmetic, not to support a reliable performance assessment.

```python
from math import isclose

import pandas as pd
import qis

illustration = pd.Series(
    [100.0, 105.0, 102.9, 108.045],
    index=pd.date_range('2024-01-31', periods=4, freq='ME'),
    name='Arithmetic illustration',
)
illustration_table = qis.compute_ra_perf_table(
    prices=illustration,
    perf_params=qis.PerfParams(freq='ME', return_type=qis.ReturnTypes.RELATIVE),
)
arithmetic_sharpe = illustration_table.loc[
    illustration.name, qis.PerfStat.SHARPE_ARITH.to_str()
]
assert isclose(arithmetic_sharpe, 16.0 / 7.0, abs_tol=1e-12)
```

## Implementation in qis

For a longer offline example, use the frozen synthetic universe with an explicit seed and
sample. Disabling its reporting quirks here isolates the performance calculations.

```python
import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2014-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY']]
params = qis.PerfParams(
    freq='ME', freq_drawdown='B', return_type=qis.ReturnTypes.LOG,
    sharpe_convention=qis.SharpeConvention.PA,
)
performance = qis.compute_ra_perf_table(prices=prices, perf_params=params)
sharpe_columns = [
    qis.PerfStat.SHARPE_RF0.to_str(),
    qis.PerfStat.SHARPE_ARITH.to_str(),
    qis.PerfStat.SHARPE_LOG_AN.to_str(),
]
sharpe_table = performance[sharpe_columns]
rolling_vol, rolling_label = qis.compute_rolling_perf_stat(
    prices=prices, rolling_perf_stat=qis.RollingPerfStat.VOL,
    roll_freq='ME', roll_periods=36,
)
drawdowns = qis.compute_rolling_drawdowns(prices=prices)
```

`performance` and `sharpe_table` are indexed by asset. `rolling_vol` and `drawdowns` are indexed
by time; `rolling_label` describes the window. Fixed-window statistics remain missing until
their window has enough observations. A full-sample row and a trailing-window observation
describe different periods.

The implementation owners are
[performance tables](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/perf_stats.py),
[returns and funding](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py),
and [rolling statistics](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/stats/rolling_stats.py).
The [packaged convention note](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/sharpe_conventions.md)
is also available to installed users; its [rendered copy](_included/sharpe_conventions.md)
is included in the documentation site.

<a id="constraints-and-common-failure-modes"></a>

## Interpretation and limitations

- Report the sampling grid, return basis, cash series, and exact Sharpe column together.
- Square-root annualisation is a convention whose time-aggregation justification assumes
  additive returns and no serial correlation; it does not make finite-sample statistics
  invariant to resampling. Compounding introduces a further distinction.
- Constant prices, short histories, or too few negative returns can produce zero risk,
  missing values, or undefined ratios. Inspect the observations before interpreting a ratio.
- Native-endpoint returns and complete-boundary ratio numerators may differ intentionally.
  Histories of at most one year also require the short-history convention above.
- Whole-sample tables are descriptive. Using future observations to evaluate an earlier
  allocation decision introduces look-ahead.

## See also

- [Frequency convention](frequency_convention_note.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Portfolio backtesting](portfolio_backtesting.md)
- {doc}`PerfParams API <api/generated/qis.PerfParams>` and
  [source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/config.py)
- {doc}`Risk-adjusted table API <api/generated/qis.compute_ra_perf_table>`
- [Reporting-frequency note](_included/reporting_frequencies.md) and
  [packaged source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/reporting_frequencies.md)
- [Canonical performance examples](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/perfstats)

## References

1. Sharpe, W. F. (1994). [The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm).
   *The Journal of Portfolio Management*, 21(1), 49–58. The historical differential-return
   definition and its time-aggregation assumptions.
2. Sepp, A., and qis contributors. [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
   Software, MIT licence. Use [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
   for citation metadata and identify the version/source used for a calculation.
