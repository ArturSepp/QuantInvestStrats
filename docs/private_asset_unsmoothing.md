---
myst:
  html_meta:
    description: >-
      Distinguish leverage adjustment from AR unsmoothing and analyse appraisal-based private
      asset returns with point-in-time and full-sample qis methods.
---

# Private-asset unsmoothing and de-levering

Use unsmoothing when reported appraisal returns are serially filtered versions of an underlying
economic return. Use de-levering when an observed vehicle amplifies asset returns with debt. They
are different transformations: AR unsmoothing inverts lagged reporting, while de-levering inverts
the financing identity. Applying one does not correct the other.

## Data and calculation contract

- **Inputs:** return Series/DataFrames for `delever_returns` and the return-level unsmoothers, or
  a positive price/NAV DataFrame for `compute_ar_unsmoothed_prices`. Indexes are dates and columns
  are assets.
- **Units:** returns, annualised financing rates, and diagnostics are decimals. `leverage` is
  debt/equity: `1.0` means one unit of debt per unit of equity, not a 1x gross multiplier.
- **Return convention:** `delever_returns` applies the simple-return identity
  `r_asset = (r_vehicle + L*r_financing)/(1+L)`. `compute_ar_unsmoothed_prices` uses log returns
  by default for AR estimation, then returns unsmoothed simple returns and NAVs.
- **Frequency and annualisation:** use the appraisal frequency. Pass `periods_per_year=12` for
  monthly or `4` for quarterly de-levering unless the index permits reliable inference. A
  per-asset `freq` Series can keep monthly and quarterly sleeves on distinct grids.
- **NaNs:** rolling unsmoothing has a warm-up and returns NaN where parameters are unidentified.
  Static GLM returns NaN when all required lags are not observed. It does not pass an
  uncorrectable observation through as if it had been unsmoothed.

## Minimal offline example

The synthetic `SAL_HF` and `SAL_PE` series have known appraisal smoothing. The latter reports only
monthly; this example deliberately estimates it quarterly to illustrate a mixed private-assets
panel without treating repeated daily marks as information.

```python
import pandas as pd
import qis
from qis.datasets.synthetic import generate_synthetic_prices

prices = generate_synthetic_prices()[['SAL_HF', 'SAL_PE']]
frequencies = pd.Series({'SAL_HF': 'ME', 'SAL_PE': 'QE'})

unsmoothed_navs, unsmoothed_returns, betas, r_squared = (
    qis.compute_ar_unsmoothed_prices(
        prices=prices,
        ar_order=1,
        freq=frequencies,
        span=20,
        warmup_period=8,
        mean_adj_type=qis.MeanAdjType.EWMA,
        is_log_returns=True,
    )
)

monthly_vehicle_returns = qis.to_returns(
    prices=prices['SAL_HF'], freq='ME', is_log_returns=False, drop_first=True
)
delevered_returns = qis.delever_returns(
    returns=monthly_vehicle_returns,
    leverage=0.50,
    financing_rate=0.04,
    periods_per_year=12,
)
```

The four unsmoothing outputs are DataFrames: reconstructed NAV, unsmoothed simple returns,
estimated AR coefficient sums, and regression R-squared. Their valid histories differ because
monthly and quarterly estimators warm up at different calendar speeds. `delevered_returns` is a
Series on the monthly grid. It is a financing adjustment only; it contains no AR estimate.

## Rolling versus full-sample unsmoothing

`compute_ar_unsmoothed_prices` and `unsmooth_returns_ar1_ewma` estimate rolling EWMA AR states.
For a backtest, keep the default `MeanAdjType.EWMA` or another explicitly point-in-time mean.
`MeanAdjType.INSAMPLE` subtracts the full-sample mean and is forward-looking, so it is suitable
for a fixed-sample exhibit but not for a trading path.

`unsmooth_returns_glm` fits one static AR(q) model to the whole sample. That is useful for an
academic full-sample estimate or a supplied fixed `theta`, but an estimated GLM result is
descriptive rather than point-in-time. Inspect `theta_sum`, the volatility-inflation factor, and
`is_severe`: as the coefficient sum approaches one, inversion becomes unstable. A negative or
greater-than-one coefficient sum may also invalidate the intended smoothing interpretation.

The canonical OCSL/GCF walkthrough uses a bundled parquet panel and therefore needs the optional
I/O dependency: install with `pip install "qis[io]"`, then run
`python -m examples.perfstats.unsmoothing_and_delevering`. It does not fetch market data.

## Constraints and failure modes

- Do not estimate daily AR dynamics from a quarterly appraisal series that was merely
  forward-filled onto business days. The reporting frequency controls the observations and
  annualisation.
- Short samples are weakly identified. Static GLM requires at least four times the AR order, and
  roughly 30 quarterly observations can still produce unstable diagnostics.
- AR unsmoothing can amplify noise and outliers. Beta caps keep `1 - beta` away from zero but do
  not prove that the economic model is correct.
- De-levering assumes constant debt/equity and one financing rate. Filings-based, time-varying
  leverage and interest expense are preferable when available.
- Do not clip negative unsmoothed returns unless that constraint is economically justified;
  clipping changes the return distribution.
- Neither transformation creates liquidity, an executable price, or a point-in-time valuation.

## See also

- {doc}`Generated rolling unsmoother API <api/generated/qis.compute_ar_unsmoothed_prices>`
- {doc}`Generated de-levering API <api/generated/qis.delever_returns>`
- [Frequency convention note](frequency_convention_note.md)
- [Canonical unsmoothing and de-levering example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/unsmoothing_and_delevering.py)
