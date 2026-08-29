---
myst:
  html_meta:
    description: >-
      Distinguish covariance-based ex-ante tracking error from realised ex-post tracking error
      and information ratio with qis.
---

# Tracking error and benchmark-relative risk

Tracking error has two different inputs and two different uses. Ex-ante analysis asks how much
active risk a set of current weights carries under a point-in-time covariance model. Ex-post
analysis asks how variable the realised strategy-minus-benchmark return series was. They should
not be substituted for one another merely because both are reported as annualised percentages.

## Choose the object that matches the question

| Question | qis object | Required input | Output |
|---|---|---|---|
| What active risk do these weights carry under this covariance estimate? | `RiskModel` | dated covariance matrices plus portfolio and benchmark weights | scalar/Series/DataFrame in the covariance's square-root units |
| How has realised active-return volatility evolved? | `compute_ewma_realised_tracking_error` | portfolio and benchmark NAVs | annualised EWMA tracking-error Series |
| What were whole-sample TE and IR? | `compute_te_ir_errors` | periodic strategy-minus-benchmark return DataFrame | annualised TE Series and dimensionless IR Series |
| How do several panels compare? | `compute_info_ratio_table` | mapping of labels to return-difference DataFrames | TE and IR DataFrames |

## Ex-ante covariance risk

For active weights `d = w_portfolio - w_benchmark` and covariance `Sigma`, `RiskModel` computes
`sqrt(d' Sigma d)`. Weights are decimal capital fractions and covariance labels define the
authoritative asset universe. `RiskModel` applies **no annualisation**: if `Sigma` is annualised
fractional covariance, tracking error is an annualised fraction; if it is periodic covariance,
the result is periodic. `estimate_rolling_ewma_covar(..., apply_an_factor=True)` produces the
annualised convention used by the canonical example.

Covariance matrices must be finite, square, symmetric, and identically labelled on rows and
columns. Material weights outside that universe are rejected in strict mode; missing in-universe
weights are filled with zero. Dated weight histories are selected as-of each covariance date,
without interpolation or look-ahead. Dates before the first weight observation receive zero
weights.

`compute_tre_at_date` returns a float, or a Series when `group_data` is supplied. Those group
values are standalone sleeve risks and are not additive. `compute_marginal_tre_at_date` returns
Euler contributions; its `mcte` column sums to total TE and group rows are additive.

Run the verified, core/offline
[ex-ante example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_anti_tracking_error_and_risk.py).
It builds monthly log-return EWMA covariance matrices on a quarterly risk grid, with a 36-month
span, then reports ex-ante TE, benchmark beta, and marginal TE.

## Ex-post realised tracking error

`compute_ewma_realised_tracking_error` aligns and forward-fills the two NAV series, resamples to
`freq`, forms portfolio and benchmark returns with the explicit `is_log_returns` choice, and
applies an annualised EWMA volatility to their difference. `ewma_span` is measured in periods of
`freq`; the first full span is masked as NaN while the estimator warms up. Clip a benchmark that
starts earlier than the portfolio so the intended joint sample is explicit.

For whole-sample numbers, form the periodic return difference explicitly:
`strategy_return - benchmark_return`. `compute_te_ir_errors` calculates
`sqrt(a) * std(diff, ddof=1)` and `sqrt(a) * mean(diff) / std(diff, ddof=1)`, where `a` is inferred
from the DatetimeIndex. Inputs and TE are decimal fractions; IR is dimensionless. NaNs are omitted
column by column, but a sparse or zero-volatility difference produces an undefined or NaN ratio.

Run the verified, core/offline
[ex-post example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_post_tracking_error_and_risk.py).
It backtests synthetic portfolio and benchmark NAVs, explicitly forms simple monthly return
differences, and reports 36-month EWMA TE plus whole-sample TE/IR and EWMA beta/alpha.

## Independent identity checks

Two small checks keep units and formulae honest:

1. For ex-ante risk, compare `RiskModel.compute_tre_at_date(...)` with an independent
   `sqrt(d @ Sigma @ d)` calculation using the same labelled weights and covariance date.
2. For whole-sample ex-post risk, compare the returned TE with
   `return_diffs.std(ddof=1) * sqrt(12)` for monthly differences. Forming the difference first is
   essential; subtracting separately estimated volatilities is not tracking error.

These identities test implementation equivalence, not forecast accuracy. Ex-ante and ex-post TE
can differ legitimately because the covariance forecast, realised sample, weights, frequency,
and estimator are different.

## Constraints and common failure modes

- State whether a covariance is periodic or annualised; `RiskModel` inherits its units.
- State simple versus log returns and the sampling frequency for ex-post results.
- Do not use a covariance estimated with future observations at an earlier backtest date.
- Do not sum standalone group TE values. Use Euler marginal contributions for an additive
  decomposition.
- Align portfolio and benchmark NAV windows before interpreting warm-up NaNs or whole-sample IR.
- Do not recreate tracking error from a second formula elsewhere in the package; use `RiskModel`
  for ex-ante analysis and the exported ex-post functions for realised analysis.

## See also

- {doc}`Generated RiskModel API <api/generated/qis.RiskModel>`
- {doc}`Generated EWMA realised-TE API <api/generated/qis.compute_ewma_realised_tracking_error>`
- {doc}`Generated whole-sample TE/IR API <api/generated/qis.compute_te_ir_errors>`
- [Frequency convention note](frequency_convention_note.md)
- [Canonical ex-ante example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_anti_tracking_error_and_risk.py)
- [Canonical ex-post example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_post_tracking_error_and_risk.py)
