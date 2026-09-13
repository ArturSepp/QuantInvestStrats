---
myst:
  html_meta:
    description: >-
      Construct correlated factor stress scenarios, reprice portfolio exposures, reconcile
      attribution, and compute analytical conditional-covariance prediction bands with qis.
---

# Factor stress testing: shocks, valuation and prediction bands

*[author / affiliation / date — placeholder]*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Factor stress testing evaluates portfolio exposures under specified factor moves. A conditional
scenario additionally assigns co-moves to factors whose shocks were not fixed. Its valuation
centre and its uncertainty band answer separate questions.

## Overview

`qis.portfolio.risk.stress_testing` turns an assigned factor model and current holdings into
portfolio stress results. It separates the specified market move, the implied co-moves of other
factors, exact valuation under frozen log-return loadings, and uncertainty around that valuation.

The functions are also exported directly from `qis`. They require no particular model provider,
optimiser, asset taxonomy or data vendor. The caller supplies the fitted snapshot, chooses the
stress anchors and owns their economic interpretation. This guide describes the API added in
5.24.0. Its runnable source is the
[offline factor-stress example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/factor_stress_testing.py).

<a id="inputs-dimensions-and-units"></a>

## Inputs, notation, and assumptions

Let $n$ be the number of assets, $k$ the number of factors and $m$ the number of scenarios.

| Input | Shape and labels | Convention |
|---|---|---|
| `betas`, $B$ | Assets by factors, $n\times k$ | Frozen loadings of asset **log returns** on factor log returns |
| `covariance`, $\Sigma$ | Factors by the same ordered factors, $k\times k$ | Annualised log-return covariance for bands |
| `residual_variances`, $d$ | Series in asset order, length $n$ | Annual independent asset residual variances, not volatilities |
| `amounts`, $a$ | Series in asset order | Signed MTM in one reference currency; shorts are negative |
| `nav`, $N$ | Positive finite scalar | Explicit portfolio NAV in that currency |
| `factor_log_shocks`, $Z$ | Scenarios by factors, $m\times k$ | Decimal log-return shocks |
| `adjustments`, $\delta$ | Scenarios by assets, $m\times n$ | Optional additional asset log returns |
| `horizon_years`, $T$ | Positive scalar | One month is $1/12$ |
| `confidence`, $c$ | Scalar strictly between zero and one | 95% is `0.95` |

Let $w_i=a_i/N$ be the signed portfolio weight and $R_s$ the simple NAV return under scenario $s$.
In the method below, $A$ and $F$ index anchored and free factors; $C$ is conditional factor
covariance, $\Omega_{\mid A}$ is conditional asset covariance, $e$ is the factor exposure vector,
and $v$ is annual portfolio variance. The subscript $F$ denotes free factors, not a forward price.

Weights are $w_i=a_i/N$. They are **not renormalised** to net or gross exposure.
Every modelled asset must have a loading row, even when that row is all zero.
A zero-loading cash position still needs an explicit residual variance, normally zero.
Coverage decisions, proxy selection, unsmoothing, private-asset priors and estimation quality
belong to the model-estimation layer, before these functions are called.

Covariance may use any consistent frequency for conditional **mean shocks** and the raw Schur
complement, because a common scalar cancels from the regression coefficients. Prediction bands
require annual covariance and annual residual variances. Do not annualise one but not the other.

Factor definitions matter. A bond-factor return is not a yield change, a credit-factor return
is not a spread change, and an FX return must use the assigned quote direction. A commodity
spot target is not automatically a rolling-futures factor target when their bases differ.

## Methodology

### 1. Convert the requested market move

#### Price target

For positive current price $P_0$ and target $P_1$,

$$
z = \log\!\left(\frac{P_1}{P_0}\right).
$$

`price_target_log_shock(2500, 3000)` represents a +20% price move:
$z=\log(1.2)\approx0.182322$. Both endpoints must be positive and expressed in identical units.
The example uses synthetic price anchors, not current market quotes.

#### Simple return and fixed exposure

With simple underlying return $r$ and fixed effective exposure $h$,

$$
z=\log(1+h r), \qquad r>-1,\quad 1+h r>0.
$$

`return_log_shock(-0.20)` is $\log(0.8)\approx-0.223144$.
For a partially invested basket, `effective_weight` explicitly supplies $h$;
the uninvested part has zero instantaneous carry. This is a frozen exposure conversion:
it does not rerun volatility targeting, rebalancing or dynamic leverage.

#### Duration and yield changes

With nonnegative effective duration $D$ and decimal yield change $\Delta y$,

$$
r_{\mathrm{rates}}\approx-D\Delta y,
\qquad z_{\mathrm{rates}}=\log(1-D\Delta y).
$$

For $D=8$:

| Yield move | `delta_y` | Approximate simple rates-factor return | Log shock |
|---|---:|---:|---:|
| Down 50bp | -0.0050 | +4.00% | $\log(1.04)$ |
| Up 50bp | +0.0050 | -4.00% | $\log(0.96)$ |
| From 5.00% to 5.25% | +0.0025 | -2.00% | $\log(0.98)$ |

These signs follow the generic duration convention. An externally calibrated scenario proxy
can imply a different vector; supply that explicit vector to valuation and label its source.
There is no implicit duration-eight default in qis, no convexity term, and no yield-level lookup.

### 2. Construct joint correlated shocks

Partition the factors into anchored factors $A$ and free factors $F$. Anchors are the specified
log moves $z_A$. Under a zero-mean jointly Gaussian factor model, the conditional mean of the free factors is

$$
\begin{aligned}
\Sigma &=
\begin{pmatrix}
\Sigma_{AA} & \Sigma_{AF}\\
\Sigma_{FA} & \Sigma_{FF}
\end{pmatrix},\\
\widehat z_F &= \Sigma_{FA}\Sigma_{AA}^{-1}z_A.
\end{aligned}
$$

Implementation uses a linear solve, not an explicit inverse. The returned vector retains the
original covariance order. There is no alpha or drift estimate in this conditioning operation.
For nonzero means, a caller must centre the anchors and restore the conditional means consistently;
passing raw nonzero-mean returns is not equivalent.

With one anchor $j$, each free factor $i$ has

$$
\widehat z_i
=\frac{\Sigma_{ij}}{\Sigma_{jj}}z_j
=\rho_{ij}\frac{\sigma_i}{\sigma_j}z_j.
$$

The Gaussian conditioning result is derived in [Geyer (2019), slides 136–140](https://www.stat.umn.edu/geyer/s19/5101/slides/s5.pdf).
Without joint normality, the same covariance regression gives a linear projection; a covariance
matrix alone does not determine a general nonlinear conditional expectation.

Correlation alone is insufficient: the volatility ratio determines shock magnitude.
If annual equity volatility is 20%, rates volatility is 10%, and their correlation is $-0.4$,
an equity shock $\log(0.8)$ implies rates log shock $-0.2\log(0.8)=0.044629$,
or approximately **+4.56%** in simple returns.

An omitted factor is free; an explicitly supplied zero is fixed. An anchor mapping containing
only `"Equity"` leaves rates free. Adding `"Rates": 0.0` fixes rates as well and jointly conditions
other factors on both constraints. The complete example below executes both cases.
A zero in a full scenario DataFrame is an actual zero shock;
valuation does not reinterpret it as a missing anchor.

#### Multiple credit anchors

For two credit factors, supply both `"Credit IG"` and `"Credit HY"` in **one** anchor mapping, each with log
shock `qis.return_log_shock(-0.10)`. The canonical five-factor example below demonstrates this call.

Adding two single-anchor conditional vectors generally double-counts dependence and need not
preserve either requested anchor. The joint solve accounts for dependence within $A$.

#### Direct, correlated and historical scenarios

A **direct** scenario fills all unspecified factor returns with zero. A **correlated** scenario
fills free factors using the joint-conditioning formula above.
Price targets are not automatically classified by qis: the caller explicitly requests either
construction. The worked example reports both so the role of co-moves is visible.

A historical scenario already supplies a complete realised factor vector. Pass those vectors
directly to `project_factor_scenarios`; do not condition them again. Historical-month selection,
ranking and point-in-time covariance selection remain caller decisions.

### 3. Reprice assets and reconcile attribution

For scenario $s$ and asset $i$, the modelled asset log return, simple return and currency P&L are

$$
g_{si}=\sum_{j=1}^{k} B_{ij}z_{sj}+\delta_{si},\qquad
r_{si}=\exp(g_{si})-1,\qquad
\operatorname{PnL}_{si}=a_i r_{si}.
$$

The implementation uses `expm1` for accuracy near zero. “Exact” means exact under these frozen
log-return loadings. It does not mean instrument-level option, bond or cash-flow repricing.
No coupons, carry, rebalancing or parameter changes are added automatically.

Portfolio return and asset attribution as fractions of NAV are

$$
R_s=\frac{\sum_i\operatorname{PnL}_{si}}{N},
\qquad A^{\mathrm{asset}}_{si}=\frac{\operatorname{PnL}_{si}}{N}.
$$

For additive factor attribution define the continuous scaling function

$$
\begin{aligned}
q(g)&=
\begin{cases}
(\exp(g)-1)/g,&g\ne0,\\
1,&g=0,
\end{cases}\\
C_{sj}&=\sum_i a_i B_{ij}z_{sj}q(g_{si}),\\
C_{s,\mathrm{adj}}&=\sum_i a_i\delta_{si}q(g_{si}),\\
\sum_j C_{sj}+C_{s,\mathrm{adj}}&=\sum_i\operatorname{PnL}_{si}.
\end{aligned}
$$

This allocates the nonlinear asset P&L proportionally to its additive log-return components.
It is a reconciliation convention, not a causal attribution or a marginal risk contribution.
Opposing components can be large even when total P&L is near zero.
The adjustment component appears in `factor_attribution` as
`Anchor / residual adjustment`, including when its value is zero.
Implementation uses the continuous limit for $|g|\le10^{-14}$ and checks reconciliation.

For a +20% factor return, an asset with beta two has return
$(1.2)^2-1=44\%$, not 40%. A USD 10m long position gains USD 4.4m.
A USD 2m short position in the same asset loses USD 0.88m.

Divide attribution by **NAV** to report percentage-point contributions:
a cell displayed as “Company A +1.20%” means +1.20 percentage points of NAV.
Do not divide by scenario net P&L. The latter is a different attribution convention and can
be unstable around zero. Top-ten tables should retain an “other” remainder if totals must add up.

`FactorScenarioProjection` contains `asset_pnl`, `factor_attribution` and
`asset_log_returns`. It does not choose company labels or truncate contributors.

### 4. Compute conditional covariance

Holding the anchored factors fixed removes their uncertainty. The free-factor covariance is
the Schur complement

$$
C_{FF}=\Sigma_{FF}-\Sigma_{FA}\Sigma_{AA}^{-1}\Sigma_{AF}.
$$

`conditional_factor_covariance` returns this as a **full factor-order matrix** $C$,
with anchor rows and columns exactly zero. If all factors are anchored, the entire matrix is zero.
For a Gaussian model $C$ does not depend on the realised anchor values.

A useful independent interpretation is the covariance of regression residuals:
$f_F-\Sigma_{FA}\Sigma_{AA}^{-1}f_A$. Conditioning removes explained variance rather than
adding uncertainty to an already fully specified anchor.

In the two-factor example above, equity held fixed leaves rates annual variance
$0.10^2(1-(-0.4)^2)=0.0084$, hence conditional annual rates volatility about **9.17%**.
Zeroing all free-factor covariance would retain only idiosyncratic risk and yield a narrower band.

### 5. Add analytical prediction bands

The conditional asset covariance, baseline portfolio exposures and annual variance are

$$
\begin{aligned}
\Omega_{\mid A}&=B C B^\top+\operatorname{diag}(d),\\
e&=B^\top w,\\
v_{\mathrm{factor}}&=e^\top C e,\\
v_{\mathrm{idio}}&=\sum_i w_i^2 d_i,\\
v&=v_{\mathrm{factor}}+v_{\mathrm{idio}}.
\end{aligned}
$$

qis builds the conditioned asset/factor snapshot and delegates portfolio aggregation to
`RiskModel`. Residuals are independent across assets and independent of factors; a full residual
covariance is not an input to this API. Annual variances add; their square-root volatilities do not.

For central probability $c$, horizon $T$ in years and standard-normal quantile $\Phi^{-1}$,

$$
[L_s,U_s]
=R_s\ \pm\ \Phi^{-1}\!\left(\frac{1+c}{2}\right)\sqrt{T v}.
$$

For $c=0.95$, the multiplier is approximately 1.96. One month uses $\sqrt{1/12}$.
Changing the horizon scales the **band width**; it does not scale the stress target or centre.

For example, take 50% exposure to the free rates factor from the preceding two-factor example,
and annual portfolio idiosyncratic volatility 2%. Annual conditional variance is
$0.5^2(0.0084)+0.02^2=0.0025$. Total annual conditional volatility is 5% and the
one-month 95% half-width is approximately **2.83% of NAV**.

## Worked example

The two-factor illustration fixes annual equity/rates volatilities at 20%/10% and correlation
at -0.4. An equity fall of 20% conditions the free rates log return to approximately 0.044629,
or a 4.564% simple gain. Holding rates explicitly at zero instead removes that response.

Take a single rates fund worth USD 50 in a portfolio with NAV USD 100; the remaining USD 50 is
zero-risk cash. The fund's rates loading is one, its equity loading zero, and its annual
independent residual volatility 4%. The portfolio's residual volatility is therefore 2%.
The conditioned log-shock vector gives a **2.281978%** scenario NAV return. Conditional annual
variance is 0.0025 and the one-month 95% half-width is approximately **2.828964% of NAV**.

These are fixed teaching inputs. The assertions compare the implementation with the
single-anchor covariance ratio, cash P&L, and separately calculated horizon variance.

```python
from math import isclose, sqrt
from statistics import NormalDist
import pandas as pd
import qis

factors = ['Equity', 'Rates']
covariance = pd.DataFrame(
    [[0.04, -0.008], [-0.008, 0.01]], index=factors, columns=factors,
)
equity_anchor = qis.return_log_shock(-0.20)
equity_only = qis.conditional_factor_shock(
    covariance, {'Equity': equity_anchor},
)
rates_fixed = qis.conditional_factor_shock(
    covariance, {'Equity': equity_anchor, 'Rates': 0.0},
)
conditional = qis.conditional_factor_covariance(covariance, ['Equity'])
assert isclose(equity_only['Rates'], -0.2 * equity_anchor, abs_tol=1e-12)
assert isclose(rates_fixed['Rates'], 0.0, abs_tol=1e-12)
assert isclose(conditional.loc['Rates', 'Rates'], 0.0084, abs_tol=1e-12)

assets = ['Rates fund']
betas = pd.DataFrame([[0.0, 1.0]], index=assets, columns=factors)
amounts = pd.Series([50.0], index=assets)
residual_variances = pd.Series([0.04 ** 2], index=assets)
nav = 100.0
shocks = equity_only.to_frame('Equity -20%').T
projection = qis.project_factor_scenarios(betas, amounts, shocks)
centres = projection.asset_pnl.sum(axis=1) / nav
expected_return = 0.5 * (0.8 ** (-0.2) - 1.0)
assert isclose(centres.iloc[0], expected_return, abs_tol=1e-12)
assert isclose(
    projection.factor_attribution.sum(axis=1).iloc[0],
    projection.asset_pnl.sum(axis=1).iloc[0], abs_tol=1e-12,
)

band = qis.compute_conditional_scenario_band(
    covariance=covariance, betas=betas, residual_variances=residual_variances,
    weights=amounts / nav, anchors=['Equity'], centres=centres,
    horizon_years=1.0 / 12.0, confidence=0.95,
)
expected_half_width = NormalDist().inv_cdf(0.975) * sqrt(0.0025 / 12.0)
assert isclose(band.annual_total_vol, 0.05, abs_tol=1e-12)
assert isclose(band.summary.band_half_width.iloc[0], expected_half_width, abs_tol=1e-12)
```

## Implementation in qis

### Public entry points

| Function | Purpose | Result |
|---|---|---|
| `price_target_log_shock` | Positive price endpoints to a log shock | Scalar |
| `return_log_shock` | Simple underlying return and fixed exposure to a log shock | Scalar |
| `duration_log_shock` | Decimal yield move and duration to a log shock | Scalar |
| `conditional_factor_shock` | Jointly condition free factors on explicit anchors | Full ordered Series |
| `conditional_factor_covariance` | Remove uncertainty in fixed factors | Full ordered DataFrame |
| `project_factor_scenarios` | Value holdings and allocate P&L | `FactorScenarioProjection` |
| `compute_conditional_scenario_band` | Add conditional-factor and residual risk around supplied centres | `ConditionalScenarioBand` |
| `compute_factor_sensitivity` | Run a supplied joint-anchor grid through all three steps | `FactorSensitivityResult` |

See the [generated API reference](api/index.rst) for signatures and defaults.
The shipped concise note is {doc}`_included/stress_testing`; its
[ordinary source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/stress_testing.md)
is available outside Sphinx. The
[canonical calculation source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/stress_testing.py)
owns conditioning, valuation and the band contract.

### Result fields

`ConditionalScenarioBand.summary` contains decimal NAV returns:

| Column | Meaning |
|---|---|
| `portfolio_return` | Supplied or computed scenario centre |
| `lower_bound`, `upper_bound` | Pointwise prediction bounds |
| `residual_vol_horizon` | $\sqrt{T v_{\mathrm{idio}}}$ |
| `conditional_factor_vol_horizon` | $\sqrt{T v_{\mathrm{factor}}}$ |
| `conditional_vol_horizon` | $\sqrt{T v}$ |
| `band_half_width` | Normal quantile times horizon total volatility |

The result also retains the full annual `conditional_factor_covariance`, annual idiosyncratic,
factor and total volatilities, horizon and confidence. `FactorSensitivityResult` adds the ordered
`anchors`, complete `factor_log_shocks` and the structured `projection`.

### 6. End-to-end offline use case

The runnable example creates a USD 100m synthetic portfolio with four positions, including
a short gold hedge. Five synthetic return histories represent equity, rates, investment-grade
credit, high-yield credit and gold. They come from the frozen `qis.datasets.synthetic` generator
with quirks disabled, default seed 20260725 and fixed dates 2014-01-02–2025-12-31. Loadings and independent residual volatilities are explicit teaching inputs;
the script does not pretend to estimate asset betas.

It estimates annual covariance from Wednesday weekly log returns with EWMA span 52, sampled
at quarter ends, and uses the last returned covariance date. The risk date is printed.
It is an EWMA span, **not a hard 52-observation rolling window**. The zero-mean scenario
conditioning is separate from the estimator's demeaning convention.

From the repository root, with qis 5.24.0 or later installed:

~~~console
python -m examples.portfolios.factor_stress_testing
python -m examples.portfolios.factor_stress_testing --output-dir /path/to/output
~~~

On Windows, use the external repository interpreter
`C:\Python\QuantInvestStrats312\Scripts\python.exe` and an explicit C-local output directory.
The example creates figures in memory and prints the scenario comparison by default.
With `--output-dir` it saves two figures in PNG/PDF and 22 CSV tables; reuse of an output directory
overwrites matching filenames. The example is repository-only, not installed in the wheel.

The workflow demonstrates:

1. Isolated and correlated versions of equity, yield and gold targets.
2. One joint negative shock to both credit factors.
3. The effect of explicitly fixing rates at zero during an equity sell-off.
4. Full factor and asset attribution, expressed as fractions of NAV.
5. Equity sensitivity from -30% to +30%, and rates, joint credit and gold from -20% to +20%,
   each in 1% simple-return increments.
6. One-month 95% analytical bands, including free-factor covariance and idiosyncratic risk.

Every anchor in a sensitivity panel receives the **same** simple return $x$,
converted to $\log(1+x)$. To impose different relative magnitudes on multiple anchors,
construct the vectors with `conditional_factor_shock`, call `project_factor_scenarios`,
then pass portfolio return centres to `compute_conditional_scenario_band`.

#### Canonical runnable source

The site includes the canonical example directly. In ordinary Markdown, open the
[complete runnable source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/factor_stress_testing.py)
from the same source revision.

~~~{literalinclude} ../examples/portfolios/factor_stress_testing.py
:language: python
:linenos:
~~~

#### Adapting the example to an assigned model

Replace the synthetic covariance, loadings, residual variances and MTM with aligned point-in-time
model inputs. Set NAV explicitly and choose factors from that model's definitions. Keep
`horizon_years` and `confidence` explicit. Select the correct source covariance date before calling
the API: these functions do not look up dates, interpolate snapshots or enforce a calendar.

The example uses `qis.plot_bars` and `qis.plot_scatter`. Matplotlib supplies layout and shades the
already-computed analytical bounds. Scatter regression fitting and regression confidence bands
are disabled. Call `qis.plot_df_table` on formatted result tables when composing a report page.

### Verification

The offline numerical suite covers single and joint conditioning, explicit zero anchors,
Schur-complement identities, long/short nonlinear valuation, adjustment reconciliation and
independently computed horizon variance. Run it with:

~~~console
python -m pytest --pyargs qis.portfolio.risk.tests.stress_testing_test
~~~

The repository's existing examples harness automatically executes the new offline example:

~~~console
python -m pytest src/qis/tests/test_examples.py -k factor_stress_testing
~~~

### Instrument portfolios and derivative reports

For positions with observed marks, intrinsic calls/puts, futures or composite payoffs,
use the [instrument portfolio stress interface](portfolio_stress.md), available from 5.30.1.
It uses these conditioning and risk primitives and adds original-holding valuation,
nonlinear historical ranking, factor/family Euler exhibits and the unified report.
The low-level APIs on this page keep their existing contracts.

## Interpretation and limitations

### Interpretation and limits

The centre uses the nonlinear valuation in the asset-valuation formula above.
The band is a **linearised, additive return-risk approximation around that centre**, with fixed
baseline exposures and covariance. Its width is therefore constant within a fixed-anchor grid,
although different anchor sets generally have different widths.

It is a pointwise prediction band conditional on the prescribed factor move. It is not a
confidence interval for estimated betas, a regression-fit interval, a simultaneous band covering
the entire curve, or a probability assigned to the stress target. It excludes parameter
uncertainty, covariance regime changes, nonlinear dispersion effects and non-normal tails.
There is no conditional Monte Carlo in this implementation. It is possible for the additive
lower bound to cross -100%; bounds are not clipped or interpreted as exact return quantiles.

When all factors are anchored, only idiosyncratic risk remains. With zero residual variances as
well, the band collapses to the scenario centre.

### Validation and failure modes

The API rejects nonfinite inputs, duplicate or mismatched ordered labels, nonpositive price/NAV
endpoints, simple returns at or below -100%, negative residual variance, invalid horizon/confidence,
unknown or duplicate anchors, and ill-conditioned anchor blocks (condition number above $10^8$).
Covariance must be symmetric positive semidefinite within numerical tolerances. No pseudoinverse,
automatic missing-exposure filling or covariance repair is applied.

Negative duration is rejected; the first-order duration endpoint must remain positive.
Short **positions** are supported independently of these factor-endpoint restrictions.
Scenario attribution is checked against total asset P&L.

<a id="related-guides"></a>

## See also

- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Factsheets and reporting](factsheets_and_reporting.md)
- [Private-asset unsmoothing](private_asset_unsmoothing.md)
- [Concise shipped stress-testing note](_included/stress_testing.md) and
  [packaged source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/stress_testing.md)


~~~{toctree}
:hidden:

_included/stress_testing
~~~

## References

1. Geyer, C. J. (2019). [Stat 5101 Lecture Slides: Deck 5](https://www.stat.umn.edu/geyer/s19/5101/slides/s5.pdf),
   University of Minnesota, slides 136–140. Conditional means and covariance for a partitioned
   multivariate normal distribution.
2. Sepp, A., and qis contributors. [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
   Software, MIT licence. The proportional log-component P&L allocation and baseline-exposure
   band are implementation conventions described above. Use
   [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
   and identify the version/source used for a calculation.
