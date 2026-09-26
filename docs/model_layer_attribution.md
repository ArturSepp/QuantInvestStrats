---
myst:
  html_meta:
    description: >-
      Attribute layered portfolio log returns to systematic exposure, risk, signals and
      integration with qis, distinguish descriptive and lagged estimates, and reproduce
      HAC intervals and factorial or Shapley feature effects.
---

# Model-layer attribution: risk-layer, signal-layer, and integration alpha

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-29](https://github.com/ArturSepp/QuantInvestStrats/commit/e61077f07098e118d1304e0da99aced93bcc9c60)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Model-layer attribution decomposes a portfolio model's log returns into benchmark exposure,
risk-layer alpha, signal-layer alpha and an integration residual. In qis these components
reconstruct every observed full-model return. Full-sample OLS and endpoint EWMA regressions
also attach confidence intervals to alpha; a separate lagged-beta estimator measures realised
alpha using beta information available before each return.

## Overview

A layered allocation model combines a risk estimate, signals and an optimiser subject to
constraints. Running the risk and signal layers separately produces useful counterfactuals,
but their effects need not sum to the integrated result. The integration term records that
difference on a stated benchmark basis. Annualised mean log-return contributions add exactly;
they are not additive compounded returns.

The estimators are derived in [Regression and HAC inference](regression_and_hac.md) and
[Exponentially weighted estimators](ewm_estimators.md). This chapter adds the layer definitions,
the exact return bridge, the contrasts between layers and feature runs, and the result objects.

Start with the [exact return bridge](#the-exact-return-bridge), then choose
[full-sample inference](#full-sample-olshac-inference),
[lagged realised alpha](#lagged-no-look-ahead-ewma-beta-realised-and-cumulative-alpha), or a
[current endpoint estimate](#current-endpoint-geometric-ewma-wls-regression).
The [worked example](#worked-example) provides both arithmetic and seeded examples.

### Choose the object that matches the question

| Question | Method and output |
|---|---|
| What did each layer add over the full sample? | [OLS/HAC](#full-sample-olshac-inference): alpha, beta, intervals and exact return components |
| How did alpha accrue under previously estimated betas? | [Lagged EWMA](#lagged-no-look-ahead-ewma-beta-realised-and-cumulative-alpha): realised components and cumulative paths |
| What is the current estimate with recency weights? | [Endpoint EWMA-WLS](#current-endpoint-geometric-ewma-wls-regression): weighted estimates and HAC intervals |
| How did the current estimate evolve? | [Expanding-prefix fits](#current-endpoint-geometric-ewma-wls-regression): descriptive alpha history |
| How much return per unit of risk did each component add? | [Common-denominator contributions](#current-endpoint-geometric-ewma-wls-regression): additive full-sample or EWMA ratios |
| Which model changes explain the difference? | [Feature attribution](#alphabeta-attribution-by-multiple-model-features): factorial and Shapley effects |

For a single strategy's ex-post TE or IR, use [tracking-error analytics](tracking_error_and_risk.md).
For ex-ante active risk from a covariance model, use `qis.RiskModel`. Neither replaces the
counterfactual layer NAVs required here.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns of the supplied layer NAVs; no cash or risk-free return is subtracted |
| Sampling grid | `freq`: `QE` for full-sample attribution, `ME` for the EWMA estimators and feature attribution |
| Annualisation | Linear, $\mathrm{AN}\,\hat\alpha$ and $\mathrm{AN}$ times mean components; Sharpe-contribution volatilities scale by $\sqrt{\mathrm{AN}}$; betas and $R^2$ are not annualised |
| Mean adjustment | OLS and WLS fit an intercept; lagged betas use point-in-time EWMA means (`MeanAdjType.EWMA`) |
| Timing | Full-sample and endpoint fits are descriptive; lagged EWMA betas are applied `beta_lag` periods after estimation |
| Output units | Annualised log-return contributions; dimensionless betas; periodic HAC standard errors |
| qis default | `compute_model_layer_alpha_beta_attribution(freq='QE', hac_lags=3, confidence_level=0.95)`; EWMA estimators use `freq='ME'`, span 36, `beta_lag=1`, `beta_init_value=1.0` |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $V_{L,t}$ | Positive NAV of layer $L$ at $t$ | Common currency, dates and valuation basis |
| $B,R,S,F,F^{\mathrm{net}}$ | Benchmark, risk layer, signal layer, gross and net full model | Supplied NAVs; $F^{\mathrm{net}}$ is optional |
| $I$, $C$ | Integration and trading-cost drag | Return differences, not supplied NAVs |
| $\ell_{L,t}$ | Log return of layer $L$ over $(t-1,t]$ | Decimal per period of `freq`; every return in this chapter is a log return |
| $\bar\ell_L$, $\bar\ell^{\,\omega}_L$ | Sample mean and $\omega$-weighted mean of $\ell_{L,t}$ | Per period |
| $\alpha_L,\beta_L,\varepsilon_{L,t}$ | Intercept, benchmark slope and residual of layer $L$ | $\alpha$ per period; $\beta$ dimensionless |
| $a_{L,t}$ | Alpha component of layer $L$ in a return bridge | Periodic log return |
| $\mathcal{A}_{L,t}$ | Cumulative alpha component $\sum_{t'\le t}a_{L,t'}$ | Log-return points, not a wealth index |
| $\hat\beta_{L,t}$, $b_{L,t}$, $\beta_0$, $h$ | EWM beta after return $t$; beta applied to return $t$; prior; lag | $b_{L,t}=\hat\beta_{L,t-h}$ or $\beta_0$; `beta_init_value` 1.0, `beta_lag` 1 |
| $N,\lambda$ | EWM span and decay, $\lambda=1-2/(N+1)$ | `span` or `beta_span`, default 36 |
| $\mathcal{E}_t[z]$ | EWM of a series $z$ through $t$, seeded at its first row | Same span as the betas |
| $\omega_t$, $T_{\mathrm{eff}}$ | Geometric WLS weight $\lambda^{T-1-t}$; Kish effective sample size | Latest row has weight one |
| $T,\mathrm{AN}$ | Retained observations and periods per year | `ME` uses $\mathrm{AN}=12$; `QE` uses $\mathrm{AN}=4$ |
| $q$, $\gamma$, $z_{\gamma}$ | Bartlett lag count, confidence level, normal quantile $\Phi^{-1}((1+\gamma)/2)$ | `hac_lags` (3), `confidence_level` (0.95) |
| $\hat\Sigma_{\alpha}$, $c$, $x_t$ | Joint HAC covariance of layer alphas; contrast vector; regressor row $(1,\ell_{B,t})^{\top}$ | Ordered $(F,R,S)$ |
| $\hat\mu_K$, $\pi_K$, $\sigma_B$, $\sigma_{F^{*}}$ | Annualised mean of bridge component $K$, its Sharpe contribution; benchmark and endpoint-model volatility | $K\in\{\mathrm{sys},R,S,I,C\}$; $F^{*}$ is $F^{\mathrm{net}}$ when supplied, else $F$ |
| $\ell^{(0)}_{L,t}$, $\ell^{(1)}_{L,t}$, $\Delta_{L,t}$ | Layer return without and with one feature; their difference | Log return of a NAV ratio |
| $\mathcal{M}$, $n$; $\mathcal{C}$, $\mathcal{D}$; $\ell^{\mathcal{C}}_{L,t}$, $d^{\mathcal{C}}_{L,t}$, $\phi_{f,L,t}$ | Feature set and its size; coalitions; coalition return, Harsanyi dividend, Shapley effect of feature $f$ | $\varnothing$ is production; periodic log returns |

Frequency and annualisation must describe the NAVs supplied. Full-sample attribution defaults
to quarter-end; the endpoint EWMA regression defaults to month-end. The examples explicitly
choose month-end. A quarterly lag is not a monthly lag.

### Inputs and the common sample

The inputs are four NAV series and an optional fifth:

1. The benchmark $B$.
1. The risk-layer model $R$ (the full model run with every alpha signal set to zero).
1. The signal layer $S$ (a separately specified portfolio built from the signals alone).
1. The full model $F$.
1. Optionally, the full model net of trading costs $F^{\mathrm{net}}$.

Each layer $L$ is converted at frequency `freq` to log returns,

$$
\ell_{L,t}=\log V_{L,t}-\log V_{L,t-1},
\qquad t=1,\dots,T.
$$

The residual definition below makes the within-period bridge exact. Log returns additionally
[sum across time](notation_and_conventions.md#simple-and-log-returns), so cumulative
contributions and their annualised means reconcile on the same scale. This does not make the log
return of an arbitrary portfolio a weighted sum of its sleeves' log returns. Reported annualised
means are not compounded annual growth rates; exponentiating each component separately destroys
additivity.

The common sample is set before any resampling. The NAVs are trimmed to the range between the
latest first valid observation and the earliest last valid observation, forward-filled inside
that range, and then converted to returns at `freq`. A layer that starts later or ends earlier
than the others therefore shortens the sample for every layer, and no layer contributes flat
forward-filled returns outside its own history. Any date on which a periodic return is still not
finite is dropped for all layers.

**Definition (integration return and cost drag).** The integration return is the log-return
residual of the full model against the sum of the risk and signal layers run on their own, and
the trading-cost drag is the difference between the net and gross full models,

$$
\ell_{I,t}=\ell_{F,t}-\ell_{R,t}-\ell_{S,t},
\qquad
\ell_{C,t}=\ell_{F^{\mathrm{net}},t}-\ell_{F,t}.
$$

## Methodology

### Full-sample OLS layer regressions

**Definition (layer regression).** For each layer $L\in\{R,S,I,F,F^{\mathrm{net}}\}$,
`qis.compute_model_layer_alpha_beta_attribution` fits

$$
\ell_{L,t}=\alpha_L+\beta_L\,\ell_{B,t}+\varepsilon_{L,t}
$$

by [ordinary least squares with an intercept](regression_and_hac.md#ordinary-least-squares) on
the common sample, so $\hat\alpha_L=\bar\ell_L-\hat\beta_L\bar\ell_B$. The integration return is
regressed as a layer of its own. The benchmark row of the table is fixed at $\hat\alpha_B=0$ and
$\hat\beta_B=1$ rather than estimated. Alpha is
[annualised](regression_and_hac.md#alpha-annualisation) linearly, $\mathrm{AN}\,\hat\alpha_L$,
because a mean log return scales with the number of periods; beta, $R^2$ and the periodic
standard error are not annualised.

### The exact return bridge

**Definition (return bridge).** The gross full-model return is separated into the systematic
return $\hat\beta_F\,\ell_{B,t}$ and three alpha components, risk $a_{R,t}$, signal $a_{S,t}$
and integration $a_{I,t}$:

$$
\begin{aligned}
a_{R,t}&=\ell_{R,t}-\hat\beta_R\,\ell_{B,t},\\
a_{S,t}&=\ell_{S,t}-\hat\beta_S\,\ell_{B,t},\\
a_{I,t}&=\ell_{F,t}-\hat\beta_F\,\ell_{B,t}-a_{R,t}-a_{S,t}.
\end{aligned}
$$

**Identity (exact bridge).** In every period,

$$
\ell_{F,t}=\hat\beta_F\,\ell_{B,t}+a_{R,t}+a_{S,t}+a_{I,t},
\qquad
\ell_{F^{\mathrm{net}},t}=\ell_{F,t}+\ell_{C,t}.
$$

**Proof.** The first equation rearranges the definition of $a_{I,t}$; the second rearranges the
definition of $\ell_{C,t}$. $\square$

Three properties turn this bookkeeping into an estimator.

#### Linearity: the integration term is an estimated alpha

**Proposition (integration coefficients).** On the common sample,

$$
\hat\beta_I=\hat\beta_F-\hat\beta_R-\hat\beta_S,
\qquad
\hat\alpha_I=\hat\alpha_F-\hat\alpha_R-\hat\alpha_S,
\qquad
a_{I,t}=\hat\alpha_I+\hat\varepsilon_{I,t}.
$$

**Proof.** All layer regressions share one design $[\mathbf{1},\ell_B]$, because they share the
common sample, and OLS coefficients and residuals are
[linear in the response](regression_and_hac.md#linearity-in-the-response). Applied to
$\ell_I=\ell_F-\ell_R-\ell_S$ this gives the first two equations and
$\hat\varepsilon_I=\hat\varepsilon_F-\hat\varepsilon_R-\hat\varepsilon_S$. Substituting the
bridge definitions gives
$a_{I,t}=\ell_{I,t}-(\hat\beta_F-\hat\beta_R-\hat\beta_S)\,\ell_{B,t}$, which is
$\ell_{I,t}-\hat\beta_I\,\ell_{B,t}=\hat\alpha_I+\hat\varepsilon_{I,t}$. $\square$

This is why the function regresses the integration return as an additional layer and reports its
beta, alpha and interval on the same footing as the observed layers. The integration alpha is not
an unexplained plug. It is the OLS alpha of the log-return series $\ell_F-\ell_R-\ell_S$.

#### Bar heights are OLS alphas

**Identity (bar heights).** The annualised sample mean of each alpha component equals the
annualised OLS alpha of its layer, and the four annualised components add to the annualised mean
full-model return:

$$
\frac{\mathrm{AN}}{T}\sum_{t=1}^{T}a_{L,t}=\mathrm{AN}\,\hat\alpha_L,\quad L\in\{R,S,I\},
\qquad
\mathrm{AN}\big(\hat\beta_F\bar\ell_B+\hat\alpha_R+\hat\alpha_S+\hat\alpha_I\big)
=\mathrm{AN}\,\bar\ell_F .
$$

**Proof.** $a_{L,t}=\hat\alpha_L+\hat\varepsilon_{L,t}$ for $L\in\{R,S\}$ by the definition of
the residual, and for $L=I$ by the previous proposition. OLS residuals have zero sample mean when
the regression includes an intercept, which gives the first equation. By linearity
$\hat\alpha_R+\hat\alpha_S+\hat\alpha_I=\hat\alpha_F=\bar\ell_F-\hat\beta_F\bar\ell_B$, which gives
the second. $\square$

The return bridge and the regression table are therefore one object: the bars of a bridge chart
are the annualised alphas, and the whiskers on them are the intervals of those same alphas.

#### Invariance to the excess-return basis

qis does not subtract a risk-free rate in this attribution. State whether the supplied NAVs
represent funded total returns or an already specified excess-return strategy. The algebra below
concerns subtracting the **benchmark log return**, not an arbitrary risk-free series. Those are
different transformations; the latter need not leave regression alphas unchanged.

**Proposition (benchmark-excess invariance).** Replace the signal-layer return by its excess
over the benchmark, $\ell'_{S,t}=\ell_{S,t}-\ell_{B,t}$. Every alpha, residual, HAC standard
error, confidence bound and p-value is unchanged, and two betas move, with their $R^2$:
$\hat\beta'_S=\hat\beta_S-1$ and $\hat\beta'_I=\hat\beta_I+1$. The same holds for the risk
layer.

**Proof.** The benchmark regressed on itself has intercept 0, slope 1 and zero residuals. By
linearity, subtracting $\ell_B$ from a response subtracts 0 from its alpha, 1 from its beta and
nothing from its residuals, and the integration response $\ell_F-\ell_R-\ell'_S$ gains $\ell_B$.
The HAC covariance depends on the regressor and the residuals only. $\square$

> **Insight.** The invariance settles how to read the integration beta. A fully invested signal
> layer carries a benchmark beta near one, as does the risk layer. Adding the two as total
> returns doubles the benchmark exposure, so the integration beta is near $-1$ by construction,
> for example $0.85-1.05-1.00=-1.20$. On the excess basis for the signal layer the same
> integration term has beta near $-0.20$, with identical alphas and intervals. The integration
> term is a log-return residual, not a tradeable portfolio.

### Additive cumulative alpha with fixed full-sample betas

The annualised alpha table answers how much each component contributed on average. The same
`component_returns` output also shows when the contribution accumulated.

**Identity (additive cumulative alpha).** Let
$\mathcal{A}_{L,t}=\sum_{t'=1}^{t}a_{L,t'}$ for $L\in\{R,S,I\}$, and let the cumulative total model
alpha be $\mathcal{A}_{F,t}=\sum_{t'=1}^{t}\big(\ell_{F,t'}-\hat\beta_F\,\ell_{B,t'}\big)$. Then at
every date

$$
\mathcal{A}_{F,t}=\mathcal{A}_{R,t}+\mathcal{A}_{S,t}+\mathcal{A}_{I,t}.
$$

**Proof.** Sum the exact bridge over $t'\le t$. $\square$

No new regression is run for this chart: the full-sample OLS betas behind `Risk Layer Alpha`,
`Signal Layer Alpha` and `Integration Alpha` stay fixed, and the chart cumulatively sums those
exact periodic components. The runnable workflow below produces percentage-point paths with an
explicit 0% origin one attribution period before the first return. The vertical axis is
cumulative log-return contribution, not a wealth index, and the paths are descriptive because
their betas are full-sample estimates; they are not point-in-time alpha forecasts.

> **Pitfall.** Do not plot `100 * exp(cumsum(alpha))` for an additive alpha-attribution chart.
> That construction is a compounded pseudo-NAV: its components reconcile multiplicatively,
> whereas the alpha bridge is defined and interpreted additively.

### Lagged no-look-ahead EWMA-beta realised and cumulative alpha

Use the rolling estimator when the question is how alpha accumulated under betas that were
available before each realised return.

**Definition (lagged realised components).** For $L\in\{R,S,F\}$, let $\hat\beta_{L,t}$ be the
EWM beta of $\ell_L$ on $\ell_B$ after observing return $t$, with span $N$ (`beta_span`, default
36), and let $b_{L,t}=\hat\beta_{L,t-h}$ be the beta applied to return $t$, with lag $h$
(`beta_lag`, default 1) and the prior $\beta_0$ (`beta_init_value`, default 1) substituted while
that estimate is missing. The realised components are

$$
a_{L,t}=\ell_{L,t}-b_{L,t}\,\ell_{B,t},\quad L\in\{R,S,F\},
\qquad
a_{I,t}=a_{F,t}-a_{R,t}-a_{S,t},
$$

and with a net NAV the net total alpha is $a_{F,t}+\ell_{C,t}$, on the unchanged gross-model beta.

**Identity (lagged bridge).** In every period
$\ell_{F,t}=b_{F,t}\,\ell_{B,t}+a_{R,t}+a_{S,t}+a_{I,t}$, and $b_{L,t}$ uses returns up to $t-h$
only.

**Proof.** The first statement rearranges the definitions of $a_{F,t}$ and $a_{I,t}$. The second
holds because $\hat\beta_{L,t-h}$ comes from EWM recursions over rows up to $t-h$ with
point-in-time seeds and centring, and $\beta_0$ is a constant. $\square$

Thus return $t$ may update $\hat\beta_{L,t}$, but it cannot change the beta applied to itself.
The betas are those of `qis.compute_ewm_beta_alpha_forecast` with
`mean_adj_type=MeanAdjType.EWMA` and `init_type=InitType.X0`: ratios of EWM moments about
point-in-time EWMA means of the same span, derived in
[point-in-time EWM regressions](regression_and_hac.md#point-in-time-ewm-regressions) and
[mean adjustment](ewm_estimators.md#mean-adjustment). `MeanAdjType.INSAMPLE` is rejected because
a full-sample mean looks ahead. Under the `X0` seed the first centred return is zero, so the first
estimated beta is deliberately missing; a missing estimate after the first finite one raises.

> **Pitfall.** The prior is more than a fallback. It replaces the first informative observation
> and [seeds](ewm_estimators.md#initial-conditions) both moment recursions, so the first finite
> estimate equals $\beta_0$ exactly and the prior is the whole state of both moments on that
> date. It keeps weight $\lambda^{k}$ in the estimate $k$ returns later, as much as
> $(N+1)/2=18.5$ ordinary returns of its date and above 5% for about $1.5N=54$ returns at $N=36$,
> so early realised alpha depends on `beta_init_value`.

The expanding annualised estimate through $t$ is $\mathrm{AN}\,t^{-1}\sum_{t'\le t}a_{L,t'}$. The
post-warm-up exhibit is the unannualised cumulative sum of $a_{L,t'}$ after a chosen base date,
with an explicit zero row on that date; with a one-period lag qis checks that the first accrued
return uses the beta available on the base date. The current estimate is
$\mathrm{AN}\,\mathcal{E}_t[a_L]$, the same-span
[EWM](ewm_estimators.md#the-recursion-and-its-weights) of the step-ahead residuals seeded with
`InitType.X0`. It is a point-in-time EWMA of realised
out-of-sample alpha, not the contemporaneous EWM alpha of the lower-level beta routine and
not a refitted 36-observation regression, and linearity keeps it additive across layers. It is a
point estimate, not a confidence interval, and its
[effective sample size](ewm_estimators.md#span-mean-lag-effective-sample-size-and-half-life) is
the span.

`estimated_betas` records estimates after each return; `applied_betas` records the betas actually
used for each return, which is the audit for the lag. The runnable workflow below uses a 36-month
EWMA beta, lags it by one month, allows 12 monthly returns for estimator warm-up, and accumulates
realised alpha from the next month.

### Current endpoint geometric EWMA-WLS regression

Use `compute_model_layer_ewma_regression_attribution` when the question is the current
recency-weighted decomposition rather than a historical sequence of investable beta estimates.
This is an endpoint descriptive regression: every return in the sample is used to estimate the
single alpha and beta shown at the final date. It must therefore not be substituted for the
lagged-beta realised attribution above.

**Definition (endpoint EWMA-WLS attribution).** On common-sample log returns (monthly at the
default `freq='ME'`), qis constructs $\ell_I=\ell_F-\ell_R-\ell_S$ and fits
$\ell_{L,t}=\alpha_L+\beta_L\,\ell_{B,t}+\varepsilon_{L,t}$ for $L\in\{R,S,I,F\}$, and
$F^{\mathrm{net}}$ when supplied, jointly with `qis.estimate_ewma_alpha_beta_hac`: weighted least
squares with the common weights $\omega_t=\lambda^{T-1-t}$ on returns $t=0,\ldots,T-1$ from oldest
to newest, so that the latest has weight one, and a stacked-score Bartlett HAC covariance.

The normal equations, the weighted $R^2$, the statsmodels $T/(T-2)$ correction and the Kish (1965)
effective sample size $T_{\mathrm{eff}}=N(1-\lambda^{T})/(1+\lambda^{T})$ are derived in
[weighted least squares with geometric weights](regression_and_hac.md#weighted-least-squares-with-geometric-weights).
At $N=36$, $\lambda=35/37\simeq0.9459$, and the 240 monthly returns of the simulated example give
`effective_nobs` 35.9999. A value displayed as 36.0 does not mean a hard 36-month window, since all
$T$ returns enter, and it is not the degrees of freedom of the correction. Three layer-specific
properties follow.

- **Integration is a contrast.** With fixed weights the linearity argument above carries over,
  and the [integration-variance proposition](#full-sample-olshac-inference) holds with weighted
  scores. Fitting the precomputed integration response is algebraically equivalent to the
  contrast but avoids cancellation when the component equations are nearly collinear or returns
  use very small units.
- **Bars are the WLS alphas.** The WLS normal equations with an intercept give
  $\sum_t\omega_t\hat\varepsilon_{L,t}=0$, so the $\omega$-weighted mean of each alpha component is
  its WLS alpha, exactly the midpoint of its own confidence interval up to rounding.
- **The covariances are stored.** `annualised_alpha_covariance` holds the $\mathrm{AN}^2$-scaled
  joint covariance of the risk, signal and integration alphas, and `parameter_covariance` the
  periodic alpha and beta covariance of the directly observed layers.

The endpoint return bridge remains exact in every period: systematic return is
$\hat\beta_F\,\ell_{B,t}$, risk and signal contributions are their beta-adjusted returns, and
integration is the residual required to reconstruct $\ell_{F,t}$. Its displayed annualised bars are
$\omega$-weighted means of those periodic log-return components. Supplying `full_model_net_nav`
adds the realised trading-cost drag and a net endpoint; it does not estimate a fee inside qis.

The final endpoint bar makes that regression identity visible. It starts with gross-model
systematic return $\mathrm{AN}\,\hat\beta_F\bar\ell^{\,\omega}_B$, applies realised trading-cost
drag when a net NAV is present, and then adds gross total alpha $\mathrm{AN}\,\hat\alpha_F$. The
label above the bar is gross return without a net NAV and net return with one. The black
gross-alpha interval is translated by the systematic and cost segments, so its midpoint is the
displayed endpoint. The beta and $R^2$ rows under this bar therefore come from the gross
`Full Model` regression, as do its alpha and interval. The risk, signal and integration alpha
bars also show their own regression $R^2$ below beta. The standalone Systematic bar uses that same
gross `Full Model` regression $R^2$: it reports the fit of the observed full model to the
benchmark, not the tautological $R^2=1$ obtained by regressing the constructed series
$\hat\beta_F\,\ell_B$ back on $\ell_B$.

An EWMA interval is not guaranteed to be narrower than its full-sample OLS/HAC counterpart.
Recency weighting usually reduces effective information, and the weighted residual variance,
serial dependence and cross-equation covariance can all change. EWMA answers a different question
more responsively; it is not a mechanical confidence-band shrinkage method.

#### Rolling descriptive EWMA-WLS alpha

`compute_model_layer_rolling_ewma_regression_alpha` repeats the same geometric EWMA-WLS fit on
every expanding prefix for which the joint regression is nonsingular. At date $t$, it uses only
returns through $t$, with weights $\omega_{t',t}=\lambda^{t-t'}$ for $t'\le t$. It returns
annualised `Total Model Alpha`, `Risk Layer Alpha`, `Signal Layer Alpha`, and `Integration Alpha`, plus
`Total Model Net Alpha` when the supplied attribution has a net NAV. Linearity gives

$$
\hat\alpha_{F,t}=\hat\alpha_{R,t}+\hat\alpha_{S,t}+\hat\alpha_{I,t}
$$

at every plotted date. Its final gross row is checked directly against the `Full Model`,
`Risk Layer`, `Signal Layer`, and `Integration` rows of the current regression table. This path is
descriptive and contemporaneous: unlike lagged-beta realised attribution, return $t$ participates
in the regression displayed at $t$. Accordingly, the generic qis plot title calls this an
annualised model-layer alpha path, not an out-of-sample alpha estimate.
`plot_model_layer_rolling_ewma_regression_alpha(..., start_date=...)` still estimates every
expanding prefix from the complete history and only then clips the displayed path. Its `avg` and
`last` legend statistics therefore describe the displayed date range without resetting EWMA state.

#### Common-denominator EWMA Sharpe contributions

**Definition (common-denominator contributions).**
`compute_model_layer_ewma_sharpe_contributions` divides the annualised means $\hat\mu_K$ of the
current EWMA return bridge by one common endpoint-model volatility,
$\pi_K=\hat\mu_K/\sigma_{F^{*}}$, and the benchmark reference by benchmark volatility,
$\mathrm{SR}_B=\hat\mu_B/\sigma_B$. Here $K$ runs over the systematic return, the risk, signal
and integration alphas and the optional cost drag $C$, and $F^{*}$ is the net model when supplied
and the gross model otherwise.

**Identity (additive Sharpe bridge).**
$\sum_K\pi_K=\hat\mu_{F^{*}}/\sigma_{F^{*}}=\mathrm{SR}_{F^{*}}$, and each alpha contribution
has the sign of its alpha.

**Proof.** By the exact bridge the numerators add to $\hat\mu_{F^{*}}$, and every model term has
the same positive denominator. $\square$

These are contribution ratios, not standalone sleeve Sharpes. Both volatilities are the final row
of `qis.compute_ewm_vol` with the attribution span, `MeanAdjType.EWMA` and `InitType.ZERO`,
annualised by $\sqrt{\mathrm{AN}}$ ([EWM volatility](ewm_estimators.md#volatility-and-annualisation)).
For IID returns EWMA centring makes them about 4% low at $N=36$
([mean adjustment](ewm_estimators.md#mean-adjustment)), which raises every ratio alike and leaves
additivity intact.

The older `compute_model_layer_ewma_stage_sharpes` remains available as a separate diagnostic. It
computes the EWMA Sharpe path after adding risk, signal and integration returns sequentially. Its
deltas are order-dependent: a positive signal alpha can produce a negative stage-Sharpe increment
when it adds enough volatility or covariance. The public bridge plot uses the additive
common-denominator measure instead.

`compute_model_layer_in_sample_sharpe_contributions` uses the identical two-denominator
construction with the exact full-sample alpha-attribution numerators and full-sample annualised
log-return volatilities, $\sigma=\sqrt{\mathrm{AN}}\,s(\ell)$. It therefore reports full-history
realised return per unit of full-history realised risk, with no EWMA window in either numerator or
denominator. The endpoint bar in both public Sharpe plots is split into systematic, optional
realised cost, and combined alpha contributions.

The return and Sharpe bridges use the final endpoint, while the rolling-alpha figure shows every
displayed estimable prefix. The return-bridge, rolling-alpha, EWMA-Sharpe and in-sample-Sharpe
plot functions accept `detailed_mode=False` for a clean export without title, subtitle or
methodology footnote.

### Full-sample OLS/HAC inference

Alpha inference in `compute_model_layer_alpha_beta_attribution` uses the Bartlett HAC covariance
of [HAC covariance of the OLS coefficients](regression_and_hac.md#hac-covariance-of-the-ols-coefficients):
$q$ lags (`hac_lags`, default 3, capped at $T-1$), the statsmodels `use_correction=True` factor
$T/(T-2)$, a normal reference distribution, and a two-sided interval at level $\gamma$
(`confidence_level`, default 0.95). Each layer, integration included, is its own regression; the
table reports the periodic $\mathrm{se}(\hat\alpha_L)$ as `Alpha HAC SE`, the two-sided normal
p-value, and the annualised interval

$$
\mathrm{AN}\big(\hat\alpha_L\pm z_{\gamma}\,\mathrm{se}(\hat\alpha_L)\big),
$$

with $z_{\gamma}=1.960$ at the default level. The estimator is the internal
`qis.utils.regression.estimate_ols_alpha_beta_hac`, which `src/qis/utils/tests/regression_test.py`
checks against a hand-rolled Newey and West matrix calculation. The default of three lags is
fixed for any frequency; the internal `qis.utils.regression.newey_west_lag_rule(nobs)` returns the
rule of thumb associated with Newey and West (1994), four for the 240 monthly returns of the
simulated example. The result records `freq`, `hac_lags` and `confidence_level`, so a table or a
footnote can quote the settings that produced it.

**Proposition (integration variance).** Let $\hat\Sigma_{\alpha}$ be the joint HAC covariance of
$(\hat\alpha_F,\hat\alpha_R,\hat\alpha_S)$ built from the stacked scores with the same kernel and
correction. The HAC variance of $\hat\alpha_I$ from its own regression is

$$
\widehat{\operatorname{Var}}(\hat\alpha_I)=c^{\top}\hat\Sigma_{\alpha}\,c,
\qquad
c=(1,-1,-1)^{\top},
$$

and the same holds for the endpoint EWMA-WLS fit with weighted scores.

**Proof.** By the integration-coefficients proposition the integration scores
$x_t\hat\varepsilon_{I,t}$, with $x_t=(1,\ell_{B,t})^{\top}$, are the combination $c$ of the three
layer scores. The HAC estimator is a quadratic form in the scores with a shared bread and
correction, so this is the
[variance of a linear contrast](regression_and_hac.md#weighted-least-squares-with-geometric-weights)
with unit weights for OLS. $\square$

A wide integration interval is a statement about the joint estimation error of three layers, not
a computational artefact; adding three marginal variances instead would ignore the cross-layer
covariances. The three intervals on a bridge chart are marginal intervals. They are not
independent, and their widths do not add. The interval of the total alpha
$\mathrm{AN}\,\hat\alpha_F$ is the `Full Model` row of the table.

The kernel follows [Newey and West (1987)](https://www.nber.org/papers/t0055);
the correction and lag-rule convention follow the [statsmodels HAC implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).

### Measuring the impact of a model feature

The same bridge isolates the effect of one model feature on the risk side and on the signal side.
With superscripts $(0)$ and $(1)$ for the model without and with the feature, the feature return
of layer $L\in\{R,S,F\}$ is $\Delta_{L,t}=\ell^{(1)}_{L,t}-\ell^{(0)}_{L,t}$, the log return of
the NAV ratio $V^{(1)}_{L,t}/V^{(0)}_{L,t}$.

**Proposition (feature decomposition).** On a common sample, regressing $\Delta_L$ on the
benchmark gives $\hat\alpha_{\Delta L}=\hat\alpha^{(1)}_L-\hat\alpha^{(0)}_L$ and
$\hat\beta_{\Delta L}=\hat\beta^{(1)}_L-\hat\beta^{(0)}_L$, with the HAC interval of the
difference from a single regression, and the feature's total effect decomposes exactly into a
risk-side, a signal-side and an integration effect,

$$
\hat\alpha_{\Delta F}=\hat\alpha_{\Delta R}+\hat\alpha_{\Delta S}+\hat\alpha_{\Delta I},
\qquad
\Delta_{I,t}=\Delta_{F,t}-\Delta_{R,t}-\Delta_{S,t}=\ell^{(1)}_{I,t}-\ell^{(0)}_{I,t}.
$$

**Proof.** Linearity in the response on one design gives the differences of coefficients. The
decomposition is the integration-coefficients proposition applied to the responses $\Delta_R$,
$\Delta_S$ and $\Delta_F$. $\square$

In code this is one call with the benchmark NAV and the three ratio NAVs in place of the layer
NAVs; the `Integration` row then gives $\hat\alpha_{\Delta I}$. Two conditions apply. The two
models of each layer must share the same date index, because a date missing from one NAV makes the
ratio missing there and the forward fill then replaces a return difference by a level jump. And
the identity between the ratio regression and the difference of two separate attributions holds
only when both land on the same common sample. A feature that changes only the covariance
estimator has $\Delta_{S,t}\equiv0$, and its effect is read from the risk and integration rows. A
feature that changes only a signal has $\Delta_{R,t}\equiv0$.

> **Pitfall.** Subtracting two regression tables is not a substitute for the ratio regression:
> the standard-error and interval columns of a table difference have no meaning, because the
> standard error of a difference is not a difference of standard errors.

### Alpha/beta attribution by multiple model features

`qis.compute_model_feature_alpha_beta_attribution` extends the single-feature ratio analysis to a
complete factorial experiment. A scenario is keyed by the `frozenset` of features enabled in that
run; the empty coalition is the production baseline. For $n$ features, all $2^n$ coalitions must be
supplied, and every coalition must use the same benchmark path. The scenario NAVs are aligned on
the intersection of their dates, without forward filling, and rebased to one; a benchmark that
differs by more than $10^{-10}$ between coalitions raises.

**Definition (factorial and Shapley effects).** Let $\mathcal{M}$ be the set of the $n$ features
and $\ell^{\mathcal{C}}_{L,t}$ the log return of layer $L$ in the run with coalition
$\mathcal{C}\subseteq\mathcal{M}$ enabled, $\ell^{\varnothing}_{L,t}$ being production. For a
non-empty $\mathcal{C}$ the Harsanyi dividend is the factorial effect, and for a feature $f$ the
[Shapley value](https://www.rand.org/pubs/papers/P295.html) is its order-free feature effect:

$$
\begin{aligned}
d^{\mathcal{C}}_{L,t}&=\sum_{\mathcal{D}\subseteq\mathcal{C}}
  (-1)^{\lvert\mathcal{C}\rvert-\lvert\mathcal{D}\rvert}\,\ell^{\mathcal{D}}_{L,t},\\
\phi_{f,L,t}&=\sum_{\mathcal{D}\subseteq\mathcal{M}\setminus\{f\}}
  \frac{\lvert\mathcal{D}\rvert!\,(n-\lvert\mathcal{D}\rvert-1)!}{n!}
  \big(\ell^{\mathcal{D}\cup\{f\}}_{L,t}-\ell^{\mathcal{D}}_{L,t}\big).
\end{aligned}
$$

Singleton dividends are direct effects and larger coalitions interactions. qis forms each effect
as a product of scenario NAVs raised to these coefficients, so its log return is the stated
combination at every observation.

**Proposition (reconstruction of the joint effect).** At every date,

$$
\sum_{\varnothing\ne\mathcal{C}\subseteq\mathcal{M}}d^{\mathcal{C}}_{L,t}
=\sum_{f\in\mathcal{M}}\phi_{f,L,t}
=\ell^{\mathcal{M}}_{L,t}-\ell^{\varnothing}_{L,t}.
$$

**Proof.** The dividends are the Möbius transform of $\mathcal{C}\mapsto\ell^{\mathcal{C}}_{L,t}$,
whose inverse gives
$\ell^{\mathcal{M}}_{L,t}=\sum_{\mathcal{C}\subseteq\mathcal{M}}d^{\mathcal{C}}_{L,t}$ with
$d^{\varnothing}_{L,t}=\ell^{\varnothing}_{L,t}$. The Shapley value splits each dividend
equally among the members of its coalition,
$\phi_{f,L,t}=\sum_{\mathcal{C}\ni f}d^{\mathcal{C}}_{L,t}/\lvert\mathcal{C}\rvert$, so the
Shapley effects sum to the same total, the efficiency property of Shapley (1952). $\square$

For two features $f$ and $g$ the dividend representation gives
$\phi_{f,L,t}=d^{\{f\}}_{L,t}+\tfrac12\,d^{\{f,g\}}_{L,t}$, the average of the feature's effect
without and with the other feature: each feature receives half of the interaction.

Each Shapley path is passed to `compute_model_layer_alpha_beta_attribution`; its alpha, beta and
HAC interval are therefore estimated from one effect-return series rather than by subtracting two
regression tables, and by the feature-decomposition proposition its total alpha splits exactly
into risk, signal and integration effects. The return rows of the summary have no regressor; their
intervals are Bartlett HAC intervals of a mean with the $T/(T-1)$ correction (the internal
`qis.utils.regression.estimate_hac_mean`, a special case in
[HAC covariance](regression_and_hac.md#hac-covariance-of-the-ols-coefficients)).

When net full-model NAVs are supplied, they must be present in every coalition. The summary then
includes both gross and net total-return intervals and the net-model regression. Scenario
construction remains outside qis: the caller decides what enabling a feature means and supplies
the resulting NAVs. Interactions are calculated for the supplied experiment.

## Worked example

### A four-month arithmetic example

Let benchmark log returns be −2%, 1%, 3% and 0%, with risk-layer alpha 0.1% and beta 1.1,
signal-layer alpha 0.2% and beta 0.9, and full-model alpha 0.25% and beta 0.8 each month.
There is no noise in this arithmetic example. Integration alpha is −0.05% per month and
integration beta is −1.2. Multiplying means by 12 gives these log-return contributions.

| Component | Annualised mean log return |
|---|---:|
| Systematic | 4.8% |
| Risk-layer alpha | 1.2% |
| Signal-layer alpha | 2.4% |
| Integration alpha | −0.6% |
| Full model | 7.8% |

These are decomposition checks, not estimated investment opportunities. The exact linear data
give essentially zero residual uncertainty and cannot illustrate realistic confidence intervals.

~~~python
import numpy as np
import pandas as pd
import qis

dates = pd.date_range('2024-01-31', periods=5, freq='ME')
benchmark = np.array([-0.02, 0.01, 0.03, 0.00])

def nav(log_returns):
    return pd.Series(np.exp(np.r_[0.0, np.cumsum(log_returns)]), index=dates)

attribution = qis.compute_model_layer_alpha_beta_attribution(
    benchmark_nav=nav(benchmark),
    risk_layer_nav=nav(0.001 + 1.1 * benchmark),
    signal_layer_nav=nav(0.002 + 0.9 * benchmark),
    full_model_nav=nav(0.0025 + 0.8 * benchmark),
    freq='ME',
    hac_lags=0,
)
components = ['Systematic Return', 'Risk Layer Alpha',
              'Signal Layer Alpha', 'Integration Alpha']
np.testing.assert_allclose(
    attribution.annualised_components.loc[components],
    [0.048, 0.012, 0.024, -0.006], atol=1e-12,
)
np.testing.assert_allclose(
    attribution.component_returns[components].sum(axis=1),
    attribution.periodic_returns['Full Model'], atol=1e-12,
)
assert np.isclose(attribution.annualised_components['Full Model Return'], 0.078)
~~~

### Runnable layer and feature workflow

From a checkout, this block uses the canonical teaching simulation below. It creates every input
it needs and uses monthly log returns throughout. The fixed-beta, lagged-beta and endpoint
estimates answer different questions; their alpha values need not match.

~~~python
import numpy as np
import pandas as pd
import qis
from examples.portfolios.model_layer_attribution_simulated import (
    simulate_layer_navs, simulate_feature_scenarios,
)

navs = simulate_layer_navs(seed=169)
full_sample_attribution = qis.compute_model_layer_alpha_beta_attribution(
    **navs, freq='ME', hac_lags=3, confidence_level=0.95,
)
alpha_columns = ['Risk Layer Alpha', 'Signal Layer Alpha', 'Integration Alpha']
alpha_paths = full_sample_attribution.component_returns[alpha_columns].cumsum().mul(100)
alpha_paths.insert(0, 'Total model alpha', alpha_paths.sum(axis=1))
origin = alpha_paths.index[0] - pd.tseries.frequencies.to_offset('ME')
alpha_paths = pd.concat([
    pd.DataFrame(0.0, index=[origin], columns=alpha_paths.columns), alpha_paths,
])

rolling = qis.compute_model_layer_ewma_alpha_attribution(
    **navs, freq='ME', beta_span=36, beta_lag=1, beta_init_value=1.0,
    mean_adj_type=qis.MeanAdjType.EWMA,
)
post_warmup = qis.compute_model_layer_cumulative_alpha_after_warmup(
    attribution=rolling, base_date=rolling.periodic_returns.index[11],
    warmup_periods=12,
)
current_ewma_alpha = rolling.current_ewma_annualised_alpha
current = qis.compute_model_layer_ewma_regression_attribution(
    **navs, freq='ME', span=36, hac_lags=3, confidence_level=0.95,
)
rolling_alpha = qis.compute_model_layer_rolling_ewma_regression_alpha(current)
sharpe_contributions = qis.compute_model_layer_ewma_sharpe_contributions(current)
in_sample_sharpe_contributions = qis.compute_model_layer_in_sample_sharpe_contributions(
    attribution=full_sample_attribution,
)
np.testing.assert_allclose(
    rolling.component_returns[alpha_columns].sum(axis=1),
    rolling.component_returns['Total Model Alpha'], atol=1e-12,
)
np.testing.assert_allclose(
    rolling.applied_betas.to_numpy(),
    rolling.estimated_betas.shift(1).fillna(1.0).to_numpy(), atol=1e-12,
)

scenarios = simulate_feature_scenarios(navs, seed=170)
decomposition = qis.compute_model_feature_alpha_beta_attribution(
    scenario_layer_navs=scenarios, freq='ME', hac_lags=3,
)
feature_table = decomposition.summary.loc['Shapley']
risk_span = decomposition.feature_attributions['beta_span']
pair_interaction = decomposition.factorial_effect_attributions[
    frozenset({'beta_span', 'signal_horizon'})
]
risk_span_alpha_paths = risk_span.component_returns[alpha_columns].cumsum().mul(100)
risk_span_alpha_paths.insert(0, 'Total feature alpha', risk_span_alpha_paths.sum(axis=1))
assert np.max(np.abs(decomposition.identity_errors.to_numpy())) < 1e-10
~~~

### Simulated example

The example below simulates 240 monthly log returns for a benchmark (6% mean, 10% volatility), a
risk layer (beta 1.05, alpha 1% per year), a signal layer (beta 1.00, alpha 3% per year) and a
full model that runs at beta 0.85, keeps all of the risk-layer alpha and 60% of the signal-layer
alpha, and carries its own residual. Residuals are AR(1) with autocorrelation 0.3, so the HAC
intervals differ from OLS intervals. By construction the population integration alpha is
$-0.4 \times 3\% = -1.2\%$ per year: the constrained model gives up 40% of the signal-layer
alpha. The fixed seed is 169, with 170 for the feature experiment. NAVs start on 2005-12-31;
the 240 month-end returns cover January 2006 through December 2025. The seed is a teaching
choice whose realised values are close to the design values; it is not a random model-selection
study. Residual volatility parameters scale AR(1) innovations, not unconditional residual volatility.
The script uses core qis dependencies and no data service.

Ordinary source: [model_layer_attribution_simulated.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/model_layer_attribution_simulated.py).
Run this plotting script from a C-local source export on the maintainer's host:

```console
python -m examples.portfolios.model_layer_attribution_simulated
```

```{literalinclude} ../examples/portfolios/model_layer_attribution_simulated.py
:language: python
:linenos:
```

The printed regression table (alphas and bounds in decimal log-return units, beta dimensionless,
and the HAC standard error measured per observation) is:

```text
                  Alpha  An Alpha     Beta       R2  p-Alpha  Alpha HAC SE  An Alpha CI Low  An Alpha CI High
Benchmark       0.00000   0.00000  1.00000  1.00000  1.00000       0.00000          0.00000           0.00000
Risk Layer      0.00103   0.01232  1.04715  0.96061  0.04831       0.00052          0.00009           0.02455
Signal Layer    0.00229   0.02743  0.98753  0.85319  0.01073       0.00090          0.00636           0.04851
Integration    -0.00071  -0.00849 -1.17503  0.94954  0.21851       0.00058         -0.02203           0.00504
Full Model      0.00260   0.03126  0.85966  0.83225  0.00354       0.00089          0.01025           0.05227
Full Model Net  0.00248   0.02976  0.85966  0.83225  0.00550       0.00089          0.00875           0.05077
```

The annualised mean log-return components in percent are benchmark return 5.05, systematic
return 4.34, risk-layer alpha 1.23, signal-layer alpha 2.74, integration alpha −0.85, full-model
return 7.47, trading-cost drag −0.15 and net return 7.32. The identity checks print residuals of
order $10^{-15}$ or smaller for linearity and bar heights, and the excess-basis run changes no
alpha, standard error, bound or p-value while shifting the signal-layer beta by exactly −1 and the
integration beta by exactly +1. The lag-rule check moves from three to four Bartlett lags and
changes the interval half-widths by at most 5 basis points per year, from 122 to 126 for the risk
layer and from 210 to 215 for the full model. The headline directions in this illustration remain
the same, but marginal significance and reported bounds still depend on the lag choice.

![Annualised model-layer return bridge showing systematic return, layer alphas, trading costs and net return with HAC intervals](images/model_layer_attribution_simulated.png)

[Open full-resolution preview](images/model_layer_attribution_simulated.png).

Read the exhibit left to right. The benchmark's annualised mean log return was 5.05%. The full
model runs at $\hat\beta_F = 0.86$, so its systematic return is 4.34%, and the 0.71% gap between the two
blue bars is the systematic return given up by running below beta one. The risk layer added
1.23% of alpha with an interval that just excludes zero (p = 0.048). The signal layer added
2.74% with an interval of 0.64% to 4.85%. The integration alpha is −0.85% with an interval of
−2.20% to +0.50%: the constrained model kept less than the full signal-layer alpha, and the
loss is not distinguishable from zero at this sample length. The design value of −1.2% lies
inside the interval. Total alpha is 3.13% with an interval of 1.03% to 5.23%, which is the sum of
the three alpha bars and the `Full Model` row of the table. Trading costs at 15 basis points per
year take the net return to 7.32%.

The integration beta is −1.18 on the total-return basis because both the risk and signal layers
carry betas near one. On the excess basis for the signal layer it is −0.18, with the same alpha
and the same whisker. The example checks this invariance at machine precision.

### Additive cumulative exhibit

![Additive cumulative model-layer alpha on simulated layers](images/model_layer_attribution_cumulative_alpha_simulated.png)

[Open full-resolution preview](images/model_layer_attribution_cumulative_alpha_simulated.png).

The seeded example accumulates 62.5 log-return percentage points of total model alpha over the
20-year sample: approximately 24.6 points from the risk layer and 54.9 from the signal layer,
offset by 17.0 points of negative integration. The terminal identity is only one reading of the
chart. The paths also
show when each source added or detracted and whether the total was diversified across sources.
At every intermediate date, not just at the end, the dark-green total is the exact sum of the
teal, amber and brown paths.

### Two-feature sensitivity exhibit

The simulated example represents a controlled $2 \times 2$ experiment: production, a doubled
beta-estimation span, a doubled signal horizon, and both changes together. In a production study,
the caller reruns the complete model under the same data, constraints and cost assumptions for
all four coalitions. The example supplies seeded illustrative NAVs for those four completed runs;
QIS performs the factorial, Shapley and layer attribution, not the model reruns themselves.

![Two-feature Shapley model sensitivity with HAC intervals](images/model_feature_attribution_simulated.png)

[Open full-resolution preview](images/model_feature_attribution_simulated.png).

Read each colour across the five groups. The first group is the feature's annualised net-return
change, estimated as a HAC mean. The remaining groups are benchmark-OLS alphas. The doubled
beta-estimation span has a +0.31% total-alpha effect: +0.17% through the risk layer, -0.02%
through the signal layer and +0.16% through integration. The doubled signal horizon has a +0.26%
total-alpha effect: +0.02% risk, +0.48% signal and -0.24% integration. The component bars sum to
the total-alpha bar for each feature; the net-return bar additionally reflects the simulated
implementation-cost change.

The black whiskers are 95% Bartlett HAC(3) intervals from one regression on each Shapley effect
path, and the black point marks the estimate at the interval midpoint. All intervals cross zero
in this illustration: the chart communicates both the estimated direction and the uncertainty,
not a tuning recommendation. One colour is retained for every bar belonging to the same feature,
so the viewer follows a feature across layers rather than mistaking the layers for independent
experiments.

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| Common-sample layer log returns | $\ell_{L,t}=\log V_{L,t}-\log V_{L,t-1}$ on the intersection of valid ranges | `periodic_returns` of every result |
| Full-sample layer regressions and intervals | OLS of $\ell_L$ on $\ell_B$; $\mathrm{AN}(\hat\alpha_L\pm z_{\gamma}\,\mathrm{se})$ | `qis.compute_model_layer_alpha_beta_attribution`, `regression_table` |
| Exact return bridge | $\ell_{F,t}=\hat\beta_F\ell_{B,t}+a_{R,t}+a_{S,t}+a_{I,t}$ | `component_returns`, `annualised_components` |
| Lagged realised alpha | $a_{L,t}=\ell_{L,t}-b_{L,t}\ell_{B,t}$ with $b_{L,t}=\hat\beta_{L,t-h}$ | `qis.compute_model_layer_ewma_alpha_attribution` |
| Current EWMA of realised alpha | $\mathrm{AN}\,\mathcal{E}_t[a_L]$ | `current_ewma_annualised_alpha` property |
| Post-warm-up cumulative alpha | Sum of $a_{L,t'}$ after a base date, zero on it | `qis.compute_model_layer_cumulative_alpha_after_warmup` |
| Endpoint EWMA-WLS attribution | WLS with $\omega_t=\lambda^{T-1-t}$; stacked Bartlett HAC | `qis.compute_model_layer_ewma_regression_attribution` |
| Expanding-prefix EWMA-WLS alpha | $\mathrm{AN}\,\hat\alpha_{L,t}$ with $\omega_{t',t}=\lambda^{t-t'}$ | `qis.compute_model_layer_rolling_ewma_regression_alpha` |
| Common-denominator Sharpe contributions | $\pi_K=\hat\mu_K/\sigma_{F^{*}}$, $\mathrm{SR}_B=\hat\mu_B/\sigma_B$ | `qis.compute_model_layer_ewma_sharpe_contributions`, `qis.compute_model_layer_in_sample_sharpe_contributions` |
| Sequential stage Sharpes | EWM Sharpe of cumulative bridge stages | `qis.compute_model_layer_ewma_stage_sharpes` |
| Factorial and Shapley feature effects | $d^{\mathcal{C}}_{L,t}$, $\phi_{f,L,t}$, then the layer attribution of each | `qis.compute_model_feature_alpha_beta_attribution` |

The runnable workflow covers the public attribution entry points. Plot a result with
`qis.plot_model_layer_ewma_return_bridge`, `qis.plot_model_layer_rolling_ewma_regression_alpha`,
`qis.plot_model_layer_ewma_sharpe_bridge`, or `qis.plot_model_layer_in_sample_sharpe_bridge`;
each accepts the matching attribution object. Canonical sources are
[model_layer.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/model_layer.py),
[model_feature.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/model_feature.py),
and the [plotting module](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/plots/derived/model_layer_attribution.py).
The [regression module](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/regression.py)
owns the OLS and weighted HAC estimators. Its qualified helper names are contributor references,
not a promise that every helper is exported as `qis.<name>`.

The three previews use the dedicated simulation and are registered to the
[model-layer producer](https://github.com/ArturSepp/QuantInvestStrats/blob/main/tools/docs_analytics/model_layer.py).
The [batch runner instructions](https://github.com/ArturSepp/QuantInvestStrats/blob/main/tools/docs_analytics/README.md)
describe how to regenerate all 18 documentation analytics images with their tables and
source fingerprints in one run. The [published provenance record](images/analytics_manifest.json)
identifies the actual source version, parameters, checks and image hashes. These previews share
the gallery's typography while preserving the simulation and computed values. Their fixed sample
endpoint is not a market-data freshness claim.

### Reading the result

`ModelLayerAlphaBetaAttribution` has seven fields.

`periodic_returns`
: The common-sample log returns of the supplied NAVs, one column per layer.

`regression_table`
: One row per layer (`Benchmark`, `Risk Layer`, `Signal Layer`, `Integration`, `Full Model`, and
  `Full Model Net` when a net NAV is supplied). Columns are the `PerfStat` labels for alpha,
  annualised alpha, beta, $R^2$ and the alpha p-value, then `Alpha HAC SE` (periodic),
  `An Alpha CI Low` and `An Alpha CI High` (annualised). The column-label constants are exported
  from `qis.portfolio.attribution.model_layer`.

`component_returns`
: The exact periodic components: `Benchmark Return`, `Risk Layer Return`, `Signal Layer Return`,
  `Systematic Return`, `Risk Layer Alpha`, `Signal Layer Alpha`, `Integration Alpha`,
  `Full Model Return`, then `Trading Cost Drag` and `Full Model Net Return` when a net NAV is
  supplied.

`annualised_components`
: $\mathrm{AN}$ times the column means of `component_returns`. These are the bar heights of a return
  bridge, and by the bar-height identity the three alpha entries equal the annualised alphas in
  `regression_table`.

`freq`, `hac_lags`, `confidence_level`
: The return frequency, Bartlett lag count and interval level used in the estimation.

`ModelLayerEwmaAlphaAttribution` contains `periodic_returns`, `estimated_betas`, `applied_betas`,
`component_returns`, `cumulative_alpha`, `expanding_annualised_alpha`, and the settings `freq`,
`beta_span`, `beta_lag`, `beta_init_value`, and `mean_adj_type`. Its computed properties
`ewma_annualised_components` and `ewma_annualised_alpha` apply `beta_span` to the exact realised
component returns. `current_ewma_annualised_components` and `current_ewma_annualised_alpha` return
their final rows. The final row of `estimated_betas` is the beta estimate after the latest return,
available for the next period; the final row of `applied_betas` is the lagged beta that explained
the latest return.

`ModelLayerEwmaRegressionAttribution` contains `periodic_returns` (monthly at the default
`freq='ME'`), geometric `weights`, `regression_table`, exact `component_returns`, weighted
`annualised_components`, joint `annualised_alpha_covariance`, direct-equation
`parameter_covariance`, and the settings `freq`, `span`, `ewm_lambda`, `nobs`, `effective_nobs`,
`hac_lags`, and `confidence_level`. This result is the input to both EWMA bridge plots, the rolling
EWMA-WLS alpha path, the additive Sharpe contributions, and the legacy sequential stage Sharpes.

`ModelLayerCumulativeAlphaAttribution` contains the post-warm-up `alpha_returns` and
`cumulative_alpha`, the `base_date`, `first_alpha_date`, `warmup_periods`, and the inherited beta
estimator settings.

### Reading the feature-attribution result

`ModelFeatureAlphaBetaAttribution` keeps the scenario paths, effect paths and statistical
attributions separate:

- `scenario_layer_navs` contains the aligned, rebased `ModelLayerNavs` input for every coalition.
- `factorial_effect_navs` and `factorial_effect_attributions` contain the Harsanyi direct effects
  and interactions. Singleton coalitions are direct effects; larger coalitions are interactions.
- `shapley_feature_navs` and `feature_attributions` contain the order-independent allocation of
  all interactions to individual features.
- `joint_effect_navs` and `joint_attribution` measure all features enabled versus production.
- `summary` provides factorial, Shapley and joint return estimates, alpha estimates, betas,
  p-values and confidence intervals in one table.
- `identity_errors` audits that both the factorial effects and Shapley effects reconstruct the
  joint path, and that every model-layer alpha bridge reconciles.

Every value in `feature_attributions` is a normal `ModelLayerAlphaBetaAttribution`. Consequently,
the additive cumulative-alpha construction above applies without modification to one feature's
Shapley effect. The runnable workflow constructs the corresponding `risk_span_alpha_paths`.

The resulting total path answers when the Shapley-allocated alpha effect of `beta_span`
accumulated. Its three component paths show whether that effect came through the risk layer, the
signal layer, or integration in the constrained full model. As for the base model, the
paths are additive log-return percentage points, not compounded feature NAVs.

## Interpretation and limitations

<a id="conventions-and-limitations"></a>

- Returns are log returns at `freq`, and alphas are annualised linearly by the periods per year
  of `freq`. State the frequency when quoting the numbers.
- The OLS/HAC regressions are full-sample and descriptive. Their betas are not point-in-time
  estimates and their components must not be used as backtest inputs or read as forecasts.
- The rolling estimator is point-in-time only when `mean_adj_type` is `EWMA`, `EXPANDING`, or
  `NONE` and the estimated beta is applied after a positive lag. The API rejects `INSAMPLE`.
- The endpoint EWMA-WLS estimator is also descriptive. Its geometric weights emphasise recent
  monthly returns, but its final beta is estimated with the same final return it describes.
- The default lag count of three applies at any frequency. At the default quarterly frequency
  three lags span nine months, and `newey_west_lag_rule` gives the sample-size rule instead.
  Normal-reference HAC inference is asymptotic. Small samples, structural breaks and strong
  dependence can make nominal 95% intervals unreliable; see
  [serial dependence](serial_dependence.md) for diagnosing persistent residuals.
- The common sample is the intersection of the layers' valid ranges. A short layer shortens the
  sample for every layer, so check `periodic_returns.index` when a layer has a late start or an
  early end.
- The integration term is a log-return residual, not a portfolio. Its beta depends on the
  stated total-return or benchmark-excess basis for the signal layer.
- Cumulative alpha paths are arithmetic sums of the periodic log-return components. Compounding
  them into NAV indices changes the question from additive alpha attribution to wealth impact.
- The EWMA Sharpe bridge divides all model components by the same full-model EWMA volatility, so
  it is additive and order-free. The benchmark reference uses benchmark volatility. The legacy
  `compute_model_layer_ewma_stage_sharpes` diagnostic remains order-dependent.
- Layer and feature effects depend on the supplied counterfactual runs and the chosen benchmark.
  They do not establish causal effects of changing an optimiser or prove investable future alpha.
- Forward filling within a common sample can hide stale marks. Validate positive NAVs, duplicate
  dates, calendar alignment and valuation timing before fitting; dropping nonfinite returns is
  not a data-quality repair.
- The Sharpe bridges use log-return numerators on the supplied NAV basis and do not automatically
  subtract risk-free returns. They are not the separately specified Sharpe conventions in
  [performance analytics](performance_analytics_and_sharpe.md).
- HAC intervals quantify sampling uncertainty conditional on the experiment. They do not correct
  for choosing a benchmark, seed, features or settings after seeing the results, and separate
  feature intervals are not a simultaneous multiple-testing guarantee.

## See also

- {doc}`Generated attribution API <api/generated/qis.compute_model_layer_alpha_beta_attribution>`
- {doc}`Generated result API <api/generated/qis.ModelLayerAlphaBetaAttribution>`
- {doc}`Generated rolling attribution API <api/generated/qis.compute_model_layer_ewma_alpha_attribution>`
- {doc}`Generated rolling result API <api/generated/qis.ModelLayerEwmaAlphaAttribution>`
- {doc}`Generated endpoint EWMA API <api/generated/qis.compute_model_layer_ewma_regression_attribution>`
- {doc}`Generated endpoint EWMA result API <api/generated/qis.ModelLayerEwmaRegressionAttribution>`
- {doc}`Generated rolling EWMA-WLS alpha API <api/generated/qis.compute_model_layer_rolling_ewma_regression_alpha>`
- {doc}`Generated rolling EWMA-WLS alpha plot API <api/generated/qis.plot_model_layer_rolling_ewma_regression_alpha>`
- {doc}`Generated EWMA Sharpe-contribution API <api/generated/qis.compute_model_layer_ewma_sharpe_contributions>`
- {doc}`Generated in-sample Sharpe-contribution API <api/generated/qis.compute_model_layer_in_sample_sharpe_contributions>`
- {doc}`Generated EWMA Sharpe-stage API <api/generated/qis.compute_model_layer_ewma_stage_sharpes>`
- {doc}`Generated EWMA return bridge API <api/generated/qis.plot_model_layer_ewma_return_bridge>`
- {doc}`Generated EWMA Sharpe bridge API <api/generated/qis.plot_model_layer_ewma_sharpe_bridge>`
- {doc}`Generated in-sample Sharpe bridge API <api/generated/qis.plot_model_layer_in_sample_sharpe_bridge>`
- {doc}`Generated warm-up cumulative API <api/generated/qis.compute_model_layer_cumulative_alpha_after_warmup>`
- {doc}`Generated cumulative result API <api/generated/qis.ModelLayerCumulativeAlphaAttribution>`
- {doc}`Generated feature API <api/generated/qis.compute_model_feature_alpha_beta_attribution>`
- {doc}`Generated feature result API <api/generated/qis.ModelFeatureAlphaBetaAttribution>`
- [Regression and HAC inference](regression_and_hac.md): OLS, linearity, Bartlett HAC, lag rule and EWMA-WLS
- [Exponentially weighted estimators](ewm_estimators.md): spans, seeds, mean adjustment and EWM volatility
- [Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md)
- [Serial dependence and autocorrelation](serial_dependence.md)
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Simulated example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/model_layer_attribution_simulated.py)
- [Performance and Sharpe conventions](performance_analytics_and_sharpe.md)
- [Reproducibility and bootstrap conventions](reproducibility.md)

## References

1. Kish, L. (1965). *Survey Sampling*. Wiley. Defines the effective sample size reported for the geometric weights.
2. Newey, W. K., and West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. [Working paper and published-version record](https://www.nber.org/papers/t0055).
3. Newey, W. K., and West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *The Review of Economic Studies*, 61(4), 631–653. [DOI: 10.2307/2297912](https://doi.org/10.2307/2297912). The lag rule of thumb returned by `newey_west_lag_rule` is associated with this paper.
4. Shapley, L. S. (1952). *A Value for N-Person Games*. RAND, P-295. [Original report](https://www.rand.org/pubs/papers/P295.html). Published in *Contributions to the Theory of Games II* (1953); [publisher's reprint record](https://doi.org/10.1515/9781400829156-012).
5. statsmodels developers. statsmodels. Software. [HAC covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).
6. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
