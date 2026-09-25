---
myst:
  html_meta:
    description: >-
      Point-in-time investable-universe, capital-breadth, risk-breadth, and allocation-efficiency
      diagnostics for portfolio target weights in qis.
---

# Portfolio breadth and allocation efficiency

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-06](https://github.com/ArturSepp/QuantInvestStrats/commit/1e42e13562707bc34686695c1bf2ad425a4af0e9)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Portfolio breadth describes how many opportunities, capital positions, or risk contributors a
portfolio effectively contains. An instrument count measures participation; an effective count
also accounts for concentration. These are allocation diagnostics, not measures of realised alpha.

## Overview

Portfolio breadth separates three questions that a raw instrument count cannot answer:

1. How large and independent is the opportunity set?
2. How broadly does the portfolio deploy capital across that set?
3. How broadly is portfolio risk distributed?

Use `qis.compute_portfolio_breadth` with asset returns and target weights. The calculation is made
only on the dates supplied in `weights`; weights are not forward-filled and realised, drifted
holdings are not substituted for the allocation decision.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Caller's choice of simple or log returns, stated explicitly |
| Sampling grid | Native return rows; breadth is evaluated only on target-weight dates |
| Annualisation | None: counts and ratios are scale-free |
| Mean adjustment | None: EWM second moment about zero |
| Timing | Point in time: covariance dated at or before each weight date |
| Output units | Counts and dimensionless ratios |
| qis default | `span=36` return rows; `position_threshold=1e-4` |

| Symbol or input | Meaning | Convention |
|---|---|---|
| $i,t$ | Asset and evaluation date | Evaluate only on supplied target-weight dates. |
| $w_{i,t}$ | Signed target weight | Decimal capital fraction; shorts are negative. |
| $a_{i,t}$ | Availability indicator | Defined by the selected data path below. |
| $\varepsilon$ | Material-position threshold | Strict absolute-weight cutoff; default `1e-4`. |
| $\mathcal I_t$ | Material, available positions | Both availability and the threshold must hold. |
| $\Sigma_t$ | Covariance used for risk breadth | Use one consistent return convention and frequency. |
| $\sigma_{p,t}$ | Volatility of the measurable invested subportfolio | Square root of its covariance-based variance. |
| `returns` | Date-by-asset return observations | Decimal simple or log returns, explicitly chosen by the caller. |
| `span` | EWM estimation span | Native return rows; no internal resampling. |

The calculation reports dimensionless counts and ratios. A common positive scaling of covariance
does not change correlation eigenvalue shares or normalized risk shares. That does not make
different sampling frequencies or simple/log return covariances interchangeable.

## Methodology

### Point-in-time availability

Availability is derived rather than accepted as a second caller-maintained mask:

- If `covar_dict` is supplied, the latest covariance dated at or before each weight date is used.
  An asset is investable exactly when its covariance diagonal is finite and strictly positive.
- Otherwise, availability is inferred causally from `returns`: an asset becomes available at its
  first finite return. QIS also estimates an EWM second-moment covariance using observations no
  later than the evaluation date. A return-available asset with zero estimated variance remains
  investable, but is not yet risk-measurable.

There is no look-ahead in either path. `span` is expressed in native return rows, so `span=36`
means 36 months only when monthly returns are supplied. The function does not resample. Cash is
excluded only when its columns are explicitly named in `cash_columns`; neither a ticker name nor
zero variance is interpreted as cash.

### Breadth measures

Let $a_{i,t}$ indicate availability and let a material position satisfy
$|w_{i,t}| > \varepsilon$, where $\varepsilon$ is `position_threshold`. The two direct counts are

$$
N^{\mathrm{investable}}_t = \sum_i a_{i,t}, \qquad
N^{\mathrm{invested}}_t = \sum_i a_{i,t}\mathbf{1}\{|w_{i,t}|>\varepsilon\}.
$$

Effective counts below use the reciprocal squared-share concentration construction. Its
effective-number interpretation is the order-two case in
[Hill's diversity-number framework](https://doi.org/10.2307/1934352). The choices of availability,
capital shares, and absolute Euler-risk shares are the specific qis conventions defined here.

### Effective independent assets

For the point-in-time correlation matrix of risk-measurable assets, let $\nu_{j,t}$ be its
non-negative eigenvalues and $\pi_{j,t}=\nu_{j,t}/\sum_k\nu_{k,t}$. The participation ratio

$$
N^{\mathrm{universe}}_{\mathrm{eff},t}
= \frac{1}{\sum_j \pi_{j,t}^{2}}
$$

is the effective number of independent correlation directions. It equals the asset count for an
identity correlation matrix and falls towards one as assets become redundant.

### Effective capital assets

For material, investable positions, define absolute capital shares

$$
p_{i,t}
= \frac{|w_{i,t}|}{\sum_{j\in\mathcal I_t}|w_{j,t}|}.
$$

This definition applies to $i\in\mathcal I_t$; other capital shares are zero.

The effective number of capital positions is

$$
N^{\mathrm{capital}}_{\mathrm{eff},t}
= \frac{1}{\sum_i p_{i,t}^{2}}.
$$

Absolute shares make the definition valid for both long-only and long-short portfolios.

### Effective risk contributors

QIS uses the canonical Euler contribution
$c_{i,t}=w_{i,t}(\Sigma_t w_t)_i/\sigma_{p,t}$. After normalising absolute contributions,
$q_{i,t}=|c_{i,t}|/\sum_j|c_{j,t}|$, risk breadth is

$$
N^{\mathrm{risk}}_{\mathrm{eff},t}
= \frac{1}{\sum_i q_{i,t}^{2}}.
$$

This is a concentration diagnostic. It does not introduce another risk decomposition beside
`qis.compute_portfolio_risk_contributions`.

The covariance and weights in this calculation are restricted to risk-measurable assets; weights
outside the material invested set are zero. Euler contributions may be negative for hedges.
Taking their absolute values measures concentration of contributions, while losing their signs;
it is not an additive decomposition of net portfolio volatility. For the underlying allocation
principle, see [Tasche](https://arxiv.org/abs/0708.2542).

### Allocation-efficiency decomposition

The result reports four dimensionless ratios:

$$
\begin{aligned}
\mathrm{SelectionCoverage}_t
  &= \frac{N^{\mathrm{invested}}_t}{N^{\mathrm{investable}}_t}, \\
\mathrm{SizingEvenness}_t
  &= \frac{N^{\mathrm{capital}}_{\mathrm{eff},t}}{N^{\mathrm{invested}}_t}, \\
\mathrm{CapitalUtilisation}_t
  &= \frac{N^{\mathrm{capital}}_{\mathrm{eff},t}}{N^{\mathrm{investable}}_t}
   = \mathrm{SelectionCoverage}_t\,\mathrm{SizingEvenness}_t, \\
\mathrm{RiskBreadthEfficiency}_t
  &= \frac{N^{\mathrm{risk}}_{\mathrm{eff},t}}{N^{\mathrm{invested}}_t}.
\end{aligned}
$$

The exact reconciliation distinguishes *selection* (how much of the available universe is used)
from *sizing* (how evenly capital is spread over selected assets). Audit columns also expose gross
target weight, unavailable positions and unavailable gross weight.

## Worked example

The following fixed teaching panel contains eight monthly **simple** returns and two allocation
decisions. It is synthetic, uses no random generator or network, and illustrates the capital
calculation with inputs that can be checked by hand. The short covariance history is illustrative,
not a recommendation for a production estimation window.

```python
import pandas as pd
import qis

dates = pd.date_range("2024-01-31", periods=8, freq="ME")
returns = pd.DataFrame(
    {
        "Equity": [0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, 0.03],
        "Rates": [0.01, 0.01, -0.01, 0.00, 0.02, -0.01, 0.01, 0.00],
        "Credit": [0.01, 0.00, 0.02, 0.01, -0.01, 0.01, 0.02, 0.01],
        "Gold": [-0.01, 0.02, 0.00, 0.01, 0.03, -0.01, 0.02, 0.01],
    },
    index=dates,
)
weights = pd.DataFrame(
    [[0.50, 0.20, 0.20, 0.10], [0.35, 0.25, 0.20, 0.20]],
    index=dates[[-2, -1]],
    columns=returns.columns,
)

breadth = qis.compute_portfolio_breadth(returns=returns, weights=weights, span=4)
history_figure = qis.plot_portfolio_breadth_history(breadth)
concentration_figure = qis.plot_portfolio_breadth_concentration(breadth)
```

All four assets are available and invested at both decision dates. The first allocation has
squared capital shares summing to 0.34; the second sums to 0.265:

| Decision | Invested count | Effective capital count | Selection coverage | Sizing evenness |
|---|---:|---:|---:|---:|
| July 2024 | 4 | 2.9412 | 1.0000 | 0.7353 |
| August 2024 | 4 | 3.7736 | 1.0000 | 0.9434 |

Capital utilisation equals sizing evenness here because the entire available universe is held.
The second decision spreads capital more evenly; this alone does not establish more independent
risks or better performance. Risk breadth additionally depends on the estimated covariance.

## Implementation in qis

`compute_portfolio_breadth` returns a `PortfolioBreadthResult`. Its `counts`, `efficiency`, and
`audit` properties select the corresponding metric tables. `availability`, capital/risk shares,
and `covariance_dates` provide the per-asset and as-of evidence behind each row. NaN target weights
are treated as zero; invalid labels and infinite targets are rejected.

The [implementation source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/portfolio_breadth.py)
and [independent numerical contracts](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/tests/portfolio_breadth_test.py)
are ordinary source links for readers outside the hosted documentation. The checks include
hand-computed counts, identical-asset covariance, and changing future observations without changing
earlier results. Record the installed qis version and source revision when reproducing a report.

### Comparing model layers

Compute one result for each layer with that layer's own target-weight schedule, then pass the
ordered mapping to the comparison plot. This is schematic: the four result objects must first be
computed from the corresponding layer's actual inputs, rather than copied from the example above.

<!-- docs-test: skip -->

```python
layer_results = {
    "Static benchmark": static_breadth,
    "Risk layer": risk_breadth,
    "Signal layer": signal_breadth,
    "Full model": full_model_breadth,
}
figure = qis.plot_portfolio_breadth_current_comparison(layer_results)
```

The three public views serve different purposes:

- `plot_portfolio_breadth_history` shows counts and efficiencies through time for one portfolio.
- `plot_portfolio_breadth_current_comparison` compares the latest observation across layers.
- `plot_portfolio_breadth_concentration` ranks current absolute capital and risk shares and shows
  how many positions account for 50% and 80% of each allocation.

Breadth can be shown beside model-layer alpha attribution to explain how the portfolio used its
opportunity set, but breadth does not by itself establish that diversification caused alpha.
Benchmark-relative active breadth is intentionally outside the current API.

### API reference

- {doc}`PortfolioBreadthResult <api/generated/qis.PortfolioBreadthResult>`
- {doc}`compute_portfolio_breadth <api/generated/qis.compute_portfolio_breadth>`
- {doc}`plot_portfolio_breadth_history <api/generated/qis.plot_portfolio_breadth_history>`
- {doc}`plot_portfolio_breadth_current_comparison <api/generated/qis.plot_portfolio_breadth_current_comparison>`
- {doc}`plot_portfolio_breadth_concentration <api/generated/qis.plot_portfolio_breadth_concentration>`

For a Markdown viewer without Sphinx roles, use the
[rendered breadth guide and API links](https://quantinveststrats.readthedocs.io/en/latest/portfolio_breadth.html).

## Interpretation and limitations

- Return-derived availability remains true after the first finite observation. It is not a
  delisting or execution-eligibility detector. Use an explicitly dated covariance/universe policy
  when availability must change later.
- In the return-estimation path, missing observations use the existing zero-fill covariance
  convention. Inspect information frequency and missingness before interpreting estimated breadth.
- A zero-variance available asset can count toward capital breadth while being absent from risk
  breadth. The audit table exposes the difference between investable and risk-measurable counts.
- An empty set has effective count zero. Ratios with zero denominators use the implementation's
  zero convention; they do not indicate a successful allocation. Inspect counts alongside ratios.
- Normalizing absolute shares removes common leverage scale and hedge signs. Equal breadth does
  not imply equal volatility, liquidity, cost, or directional exposure.
- These diagnostics evaluate supplied allocation decisions; they do not reconstruct realised
  holdings between decisions or establish that diversification caused alpha.

## See also

- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Model-layer attribution](model_layer_attribution.md)
- [Targets and held units in a backtest](portfolio_backtesting.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)

## References

1. Hill, M. O. (1973). Diversity and evenness: a unifying notation and its consequences. *Ecology*, 54(2), 427–432. [DOI: 10.2307/1934352](https://doi.org/10.2307/1934352). This supplies the effective-number construction, not a portfolio performance model.
2. Tasche, D. (2008). Capital allocation to business units and sub-portfolios: the Euler principle. Working paper. [arXiv:0708.2542](https://arxiv.org/abs/0708.2542).
3. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
