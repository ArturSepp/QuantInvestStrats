---
myst:
  html_meta:
    description: >-
      Point-in-time investable-universe, capital-breadth, risk-breadth, and allocation-efficiency
      diagnostics for portfolio target weights in qis.
---

# Portfolio breadth and allocation efficiency

Portfolio breadth separates three questions that a raw instrument count cannot answer:

1. How large and independent is the opportunity set?
2. How broadly does the portfolio deploy capital across that set?
3. How broadly is portfolio risk distributed?

Use `qis.compute_portfolio_breadth` with asset returns and target weights. The calculation is made
only on the dates supplied in `weights`; weights are not forward-filled and realised, drifted
holdings are not substituted for the allocation decision.

```python
import numpy as np
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

breadth = qis.compute_portfolio_breadth(
    returns=returns,
    weights=weights,
    span=4,
)

history_figure = qis.plot_portfolio_breadth_history(breadth)
concentration_figure = qis.plot_portfolio_breadth_concentration(breadth)
```

## Point-in-time availability

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

## Breadth measures

Let $a_{i,t}$ indicate availability and let a material position satisfy
$|w_{i,t}| > \varepsilon$, where $\varepsilon$ is `position_threshold`. The two direct counts are

```{math}
N^{\mathrm{investable}}_t = \sum_i a_{i,t}, \qquad
N^{\mathrm{invested}}_t = \sum_i a_{i,t}\mathbf{1}\{|w_{i,t}|>\varepsilon\}.
```

### Effective independent assets

For the point-in-time correlation matrix of risk-measurable assets, let $\lambda_{j,t}$ be its
non-negative eigenvalues and $u_{j,t}=\lambda_{j,t}/\sum_k\lambda_{k,t}$. The participation ratio

```{math}
N^{\mathrm{universe}}_{\mathrm{eff},t}
= \frac{1}{\sum_j u_{j,t}^{2}}
```

is the effective number of independent correlation directions. It equals the asset count for an
identity correlation matrix and falls towards one as assets become redundant.

### Effective capital assets

For material, investable positions, define absolute capital shares

```{math}
p_{i,t}
= \frac{|w_{i,t}|}{\sum_{j\in\mathcal I_t}|w_{j,t}|}.
```

The effective number of capital positions is

```{math}
N^{\mathrm{capital}}_{\mathrm{eff},t}
= \frac{1}{\sum_i p_{i,t}^{2}}.
```

Absolute shares make the definition valid for both long-only and long-short portfolios.

### Effective risk contributors

QIS uses the canonical Euler contribution
$c_{i,t}=w_{i,t}(\Sigma_t w_t)_i/\sigma_{p,t}$. After normalising absolute contributions,
$q_{i,t}=|c_{i,t}|/\sum_j|c_{j,t}|$, risk breadth is

```{math}
N^{\mathrm{risk}}_{\mathrm{eff},t}
= \frac{1}{\sum_i q_{i,t}^{2}}.
```

This is a concentration diagnostic. It does not introduce another risk decomposition beside
`qis.compute_portfolio_risk_contributions`.

## Allocation-efficiency decomposition

The result reports four dimensionless ratios:

```{math}
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
```

The exact reconciliation distinguishes *selection* (how much of the available universe is used)
from *sizing* (how evenly capital is spread over selected assets). Audit columns also expose gross
target weight, unavailable positions and unavailable gross weight.

## Comparing model layers

Compute one result for each layer with that layer's own target-weight schedule, then pass the
ordered mapping to the comparison plot:

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

## API reference

- {doc}`PortfolioBreadthResult <api/generated/qis.PortfolioBreadthResult>`
- {doc}`compute_portfolio_breadth <api/generated/qis.compute_portfolio_breadth>`
- {doc}`plot_portfolio_breadth_history <api/generated/qis.plot_portfolio_breadth_history>`
- {doc}`plot_portfolio_breadth_current_comparison <api/generated/qis.plot_portfolio_breadth_current_comparison>`
- {doc}`plot_portfolio_breadth_concentration <api/generated/qis.plot_portfolio_breadth_concentration>`
