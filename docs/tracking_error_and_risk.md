---
myst:
  html_meta:
    description: >-
      Define ex-ante and realised tracking error, information ratio, and additive Euler risk
      contributions, with explicit units, timing, and offline qis examples.
---

# Tracking error and benchmark-relative risk

*[author / affiliation / date — placeholder]*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Tracking error is the variability of portfolio returns relative to a benchmark. Ex-ante tracking
error evaluates current active weights under a covariance estimate; ex-post tracking error
measures realised return differences. Both may be annualised percentages, but they describe
different objects and need not agree.

<a id="choose-the-object-that-matches-the-question"></a>

## Overview

| Question | qis entry point | Required input | Output |
|---|---|---|---|
| What active risk do these weights carry? | `RiskModel` | Dated covariance matrices and portfolio/benchmark weights | Risk in the covariance's square-root units |
| How has realised active-return risk evolved? | `compute_ewma_realised_tracking_error` | Portfolio and benchmark NAVs | Annualised EWMA tracking-error Series |
| What were whole-sample tracking error and information ratio? | `compute_te_ir_errors` | Periodic return-difference DataFrame | TE and IR Series |
| How do several sets of strategies compare? | `compute_info_ratio_table` | Mapping of labels to return-difference DataFrames | TE and IR DataFrames |

Active-risk analysis follows the benchmark-relative mean/variance setting discussed by
[Roll (1992)](https://www.anderson.ucla.edu/documents/areas/fac/finance/1992-2.pdf).
A low tracking error means that relative returns vary little; it does not establish a high
absolute return or low total portfolio volatility.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and alignment |
|---|---|---|
| $w_p,w_b$ | Portfolio and benchmark weights | Decimal capital fractions; asset labels must match the covariance universe |
| $d=w_p-w_b$ | Active weights | May be positive or negative |
| $\Sigma$ | Asset-return covariance | Periodic or annualised fractional covariance; state which |
| $x_t=r_{p,t}-r_{b,t}$ | Realised active return | Difference of returns using the same convention and grid |
| $s(x)$ | Sample standard deviation | `ddof=1`; NaNs omitted per column |
| $a$ | Periods per year | Inferred from the return-difference index; 12 for regular month-end observations |
| `ewma_span` | EWMA span | Count of sampled return periods, not calendar days |

Supply positive NAVs over an explicitly aligned sample. Choose simple or log returns with
`is_log_returns`; the difference of simple returns and the log of a relative NAV are different
quantities.

A covariance used as a risk forecast must be estimated from information available on that
date. `RiskModel` validates finite, square, symmetric, identically labelled matrices; the caller
remains responsible for the estimate's economic meaning and suitability as a covariance.

## Methodology

### Ex-ante covariance risk

For the active weights and covariance at the same risk date:

$$
\mathrm{TE}_{\mathrm{ex\ ante}}
=\sqrt{d^\mathsf{T}\Sigma d}.
$$

`RiskModel` applies no annualisation. An annualised covariance produces annualised tracking
error; a monthly covariance produces monthly tracking error. The canonical estimator
`estimate_rolling_ewma_covar(..., apply_an_factor=True)` supplies annualised covariance.

Covariance labels define the asset universe. Missing in-universe weights become zero.
Material weights outside that universe are rejected in strict mode. Methods accepting dated
weight histories use the last observation at or before each covariance date, with zero weights
before the first observation. They do not interpolate or supply a backtest execution lag.
Single-date methods require an exact covariance-grid date.

### Standalone group risk and additive contributions

`compute_tre_at_date` returns a scalar, or standalone sleeve risks when given `group_data`.
A sleeve's active weights are set to zero outside the group before evaluating the quadratic form.
The resulting group risks do not add to portfolio tracking error because cross-group covariance
is omitted from each standalone calculation.

For positive total tracking error, the Euler contribution of asset $i$ is:

$$
c_i=\frac{d_i(\Sigma d)_i}{\mathrm{TE}},
\qquad
\sum_i c_i=\mathrm{TE}.
$$

`compute_marginal_tre_at_date` returns these contributions in `mcte`. They include the
active-weight multiplier; they are not merely the derivative of TE with respect to a weight.
A negative contribution can represent a reduction in total active risk. At zero TE, qis defines
all contributions as zero.

Group rows aggregate asset contributions additively. If the returned table also contains
`Total`, do not add that row to its constituent groups. Optional systematic and residual
contributions reconcile when the supplied factor covariance decomposition is consistent;
qis does not silently repair inconsistent model views.

### Ex-post realised tracking error

For a regularly sampled active-return series, the whole-sample estimators are:

$$
\widehat{\mathrm{TE}}=\sqrt{a}\,s(x),
\qquad
\widehat{\mathrm{IR}}=\frac{\sqrt{a}\,\overline{x}}{s(x)}.
$$

TE is in annualised return units; IR is dimensionless. The difference must be formed before
estimating its standard deviation. Subtracting the two standalone volatilities does not recover
tracking error.

`compute_te_ir_errors` omits NaNs column by column. A zero-variance difference has TE zero and
IR missing; an insufficient sample can make both missing. An irregular input index may trigger
the annualisation helper's fallback, so construct the intended regular return grid explicitly.

`compute_ewma_realised_tracking_error` concatenates the NAVs, removes jointly missing rows,
forward-fills, samples to `freq`, forms the chosen returns, and passes their difference to qis's
EWMA volatility estimator. The first `ewma_span` estimates are masked for warm-up. Clip both
ends to the intended common sample before calling it: forward-filling a terminated NAV would
otherwise imply a stale investment value.

The EWMA series describes evolving realised variability. It is not the same estimator as
whole-sample sample-standard-deviation TE or covariance-based ex-ante risk.

<a id="independent-identity-checks"></a>

## Worked example

Consider two assets with annualised volatilities 20% and 10%, correlation 0.10, and active
weights $(0.10,-0.10)$. Their covariance matrix is

$$
\Sigma=\begin{pmatrix}0.04&0.002\\0.002&0.01\end{pmatrix}.
$$

Active variance is $0.00046$, so ex-ante TE is approximately **2.1448%**.

Separately, four monthly simple active returns of 1%, −1%, 2%, and 0% have mean 0.5% and
sample variance $1/6000$. Their annualised TE is $\sqrt{0.002}$, approximately **4.4721%**,
and IR is $0.06/\sqrt{0.002}$, approximately **1.3416**. These are fixed arithmetic illustrations,
not forecasts or a comparison of predicted and realised risk for the same portfolio.

```python
from math import isclose, sqrt

import pandas as pd
import qis

assets = ['Asset A', 'Asset B']
date = pd.Timestamp('2024-12-31')
covariance = pd.DataFrame(
    [[0.04, 0.002], [0.002, 0.01]], index=assets, columns=assets
)
risk = qis.RiskModel(covar={date: covariance})
benchmark_weights = pd.Series([0.5, 0.5], index=assets)
portfolio_weights = pd.Series([0.6, 0.4], index=assets)
ex_ante = risk.compute_tre_at_date(
    benchmark_weights=benchmark_weights, portfolio_weights=portfolio_weights, date=date
)
contributions = risk.compute_marginal_tre_at_date(
    benchmark_weights=benchmark_weights, portfolio_weights=portfolio_weights, date=date
)
assert isclose(ex_ante, sqrt(0.00046), abs_tol=1e-12)
assert isclose(contributions['mcte'].sum(), ex_ante, abs_tol=1e-12)

return_diffs = pd.DataFrame(
    {'Active return': [0.01, -0.01, 0.02, 0.00]},
    index=pd.date_range('2024-01-31', periods=4, freq='ME'),
)
te, ir = qis.compute_te_ir_errors(return_diffs=return_diffs)
assert isclose(te.iloc[0], sqrt(0.002), abs_tol=1e-12)
assert isclose(ir.iloc[0], 0.06 / sqrt(0.002), abs_tol=1e-12)
```

The Euler sum checks attribution arithmetic. It does not test covariance forecasting accuracy.

## Implementation in qis

[RiskModel](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/risk_model.py)
owns ex-ante tracking error, benchmark beta, factor exposures, and marginal risk.
The adjacent [ex-post module](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/ex_post_tracking_error.py)
owns realised TE and IR. Use these implementations when extending analytics in qis.

Two complete examples use the frozen synthetic universe offline:

- [Ex-ante example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_anti_tracking_error_and_risk.py):
  monthly log-return EWMA covariance, quarterly risk dates, a 36-month span, benchmark beta,
  and standalone/Euler group risk.
- [Ex-post example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_post_tracking_error_and_risk.py):
  quarterly rebalanced synthetic portfolios, simple monthly active returns, 36-month EWMA TE,
  whole-sample TE/IR, and EWMA benchmark beta/alpha.

From the repository, run:

```bash
python -m examples.portfolios.ex_anti_tracking_error_and_risk
python -m examples.portfolios.ex_post_tracking_error_and_risk
```

Both examples keep 2014-01-02–2025-12-31 inputs fixed and disable the fixture's reporting quirks
to isolate the calculations. The spelling `ex_anti` in the existing filename is retained for
compatibility. On the maintainer's Windows host, use the external interpreter and C-local
execution setup in [AGENTS.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/AGENTS.md).

<a id="constraints-and-common-failure-modes"></a>

## Interpretation and limitations

- Ex-ante and ex-post TE can differ because their weights, return basis, covariance estimate,
  sampling period, and estimation method differ. Equal annualisation labels do not make them
  interchangeable.
- A low TE does not imply low absolute portfolio risk or a positive mean active return.
- Covariance and weights must be available at the stated risk date. As-of selection does not
  remove look-ahead already embedded in an input estimate.
- Standalone group TE is non-additive. Use `mcte` for additive attribution and exclude a
  precomputed `Total` row when summing groups.
- Specify the common NAV sample and investigate warm-up or missing values before interpreting
  the last EWMA point or a whole-sample IR.
- The conventional square-root annualisation does not correct for serial dependence or make
  a short sample precise.

## See also

- [Performance and Sharpe conventions](performance_analytics_and_sharpe.md)
- [Frequency convention](frequency_convention_note.md)
- [Portfolio backtesting](portfolio_backtesting.md)
- {doc}`RiskModel API <api/generated/qis.RiskModel>`
- {doc}`EWMA realised-TE API <api/generated/qis.compute_ewma_realised_tracking_error>`
- {doc}`Whole-sample TE/IR API <api/generated/qis.compute_te_ir_errors>`

Ordinary source links for these APIs are provided in [Implementation in qis](#implementation-in-qis)
for readers using a Markdown viewer without Sphinx roles.

## References

1. Roll, R. (1992). [A Mean/Variance Analysis of Tracking Error](https://www.anderson.ucla.edu/documents/areas/fac/finance/1992-2.pdf).
   *The Journal of Portfolio Management*, 18(4), 13–22.
   [DOI: 10.3905/jpm.1992.701922](https://doi.org/10.3905/jpm.1992.701922).
2. Sepp, A., and qis contributors. [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
   Software, MIT licence. Citation metadata:
   [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
