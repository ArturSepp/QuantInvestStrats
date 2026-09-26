---
myst:
  html_meta:
    description: >-
      Linear factor risk models in qis: the covariance identity, factor exposures, systematic
      and residual risk, Euler contributions, benchmark beta, factor groups and EWM loading
      estimation, implemented by RiskModel, EwmLinearModel and LinearModel.
---

# Factor risk models

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A linear factor risk model writes each asset return as a loading-weighted sum of a few factor
returns plus a residual that is uncorrelated with the factors and with other residuals. The
asset covariance is then a low-rank factor term plus a diagonal matrix, and portfolio risk splits
exactly into a systematic and a residual part. In qis, `EwmLinearModel` estimates time-varying
loadings by exponentially weighted regressions, and `RiskModel` evaluates exposures, systematic
and residual risk, Euler contributions and benchmark beta on dated snapshots that the caller
assembles.

## Overview

The chapter is the reference for the factor block that the
[tracking-error chapter](tracking_error_and_risk.md) and the
[factor stress-testing chapter](stress_testing.md) use. It answers five questions:

1. **What does the model assume, and what covariance does it imply?** The linear model and the
   identity $\Sigma=B\Sigma_fB^{\top}+\Psi$.
2. **How much of a portfolio's risk is systematic?** Exposures $x=B^{\top}w$, the exact variance
   split, and additive Euler contributions by factor and by residual.
3. **How does the portfolio co-move with a benchmark?** The ex-ante benchmark beta and its
   regression interpretation.
4. **How are loadings estimated without look-ahead?** EWM regressions, their warm-up, and the
   one-period lag between an estimate and the return it explains.
5. **How good is the fit?** The diagnostics of `LinearModel`, stated as implemented.

qis separates estimation from evaluation. `EwmLinearModel` and `estimate_ewm_factor_model`
produce loadings from return panels. `RiskModel` never estimates anything: it validates and
evaluates the covariance snapshots, loadings, factor covariances and residual variances it is
given. Tracking-error formulas and stress scenarios are covered in their own chapters and are
only referenced here.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Total log returns of the supplied prices in `estimate_ewm_factor_model`; `RiskModel` inherits the returns its covariance describes |
| Sampling grid | Weekly `W-WED` returns in `estimate_ewm_factor_model`; `RiskModel` evaluates on its covariance date grid |
| Annualisation | None inside `RiskModel` or the EWM fit; the caller scales moments by $\mathrm{AN}$ (52 weekly) and volatilities by $\sqrt{\mathrm{AN}}$ |
| Mean adjustment | None by default: EWM moments about zero (regression through the origin) and an uncentred $R^2$ |
| Timing | $\hat B_t$ uses data up to and including $t$; apply $\hat B_{t-1}$ to $f_t$; weights as of each covariance date |
| Output units | Exposures in weight units; variances in covariance units; contributions in volatility units or shares |
| qis default | `estimate_ewm_factor_model(freq='W-WED', span=26)`; `EwmLinearModel.fit(span=31, init_type=InitType.X0, warmup_period=20)` |

Let $n$ be the number of assets and $K$ the number of factors.

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $n$, $K$ | Number of assets and of factors | Counts |
| $q$ | Factor index | $q=1,\ldots,K$; $i,j$ index assets |
| $f_t$ | Factor returns over $(t-1,t]$ | $K$-vector; the code's `x` panel |
| $r_t$ | Asset returns over $(t-1,t]$ | $n$-vector; the code's `y` panel |
| $B$, $B_{iq}$ | Factor loadings | $n\times K$, assets by factors, as in `RiskModel.factor_loadings` |
| $\varepsilon_t$ | Residual returns | $n$-vector, uncorrelated with $f_t$ |
| $\Sigma_f$ | Factor covariance | $K\times K$, `factor_covar` or `x_covars` |
| $\Psi=\operatorname{diag}(\psi_1,\ldots,\psi_n)$ | Residual covariance | Diagonal; `residual_vars` holds variances, not volatilities |
| $\Sigma$ | Asset covariance | $n\times n$, `covar` |
| $w$, $w_p$, $w_b$ | Portfolio, benchmark weights | Signed capital fractions |
| $d=w_p-w_b$ | Active weights | As in [tracking error](tracking_error_and_risk.md) |
| $x=B^{\top}w$, $x_a=B^{\top}d$ | Factor exposures; active factor exposures | $K$-vectors in weight units; not the code attribute `LinearModel.x` |
| $\sigma$, $\sigma_{\mathrm{sys}}$, $\sigma_{\mathrm{res}}$ | Total, systematic, residual volatility | Square-root units of the covariance |
| $c_q$, $c^{\varepsilon}_i$ | Euler contribution of factor $q$ and of residual $i$ | Volatility units; they sum to $\sigma$ |
| $\eta_{iq}$ | Holding-by-factor systematic contribution | Volatility units |
| $\beta_b$, $\gamma$ | Benchmark beta; benchmark-beta loading vector | Dimensionless; $\gamma$ is an $n$-vector |
| $G$, $\omega_q$ | Factor group; its bump weights | $\omega_q\ge0$, $\sum_{q\in G}\omega_q=1$ |
| $x_G$, $x^{\omega}_G$ | Summed and bump-weighted group exposures | Weight units |
| $\kappa$, $v$ | Size and direction of a factor bump | Decimal return; $K$-vector |
| $m$ | Summation index over past periods | $m\le t$ |
| $M_{ff,t}$, $M_{fr,t}$ | EWM factor second moment; factor-asset cross moment | $K\times K$ and $K\times n$, about zero |
| $W$ | Warm-up | `warmup_period`, count of sampled periods |
| $k$ | Loading lag in the diagnostics | $k\in\{0,1\}$ |
| $\hat\varepsilon^{(k)}_{i,t}$ | Model residual with lag $k$ | The code's "factor alpha" |
| $\mathcal{T}_{i,t}$, $n_{i,t}(m)$ | Dates up to $t$ with finite residual and return of asset $i$; number of those dates after $m$ | Common sample of the two sums in $R^2_{i,t}$ |
| $Q^{\varepsilon}_{i,t}$, $Q^{r}_{i,t}$ | EWM-weighted sums of squared residual and squared return over $\mathcal{T}_{i,t}$ | About zero |
| $R^2_{i,t}$ | EWM coefficient of determination | Dimensionless, clipped to $[0,1]$ |
| $C$, $\bar\rho_i$ | Residual correlation matrix; mean off-diagonal correlation of asset $i$ | Dimensionless |
| $X_{q,t}$ | Aggregated factor exposure on a loading date | Weight units |
| $w_{i,(t)}$ | Weight of asset $i$ in force at $t$: the last weight row dated at or before $t$ | A missing entry means not held |

The model rests on three assumptions:

- **Linearity.** $r_t=Bf_t+\varepsilon_t$ with loadings that are constant over the horizon of the
  forecast. Estimated loadings vary through time; a snapshot freezes them.
- **Orthogonality.** $\operatorname{Cov}(f_t,\varepsilon_t)=0$.
- **Diagonal residuals.** $\operatorname{Cov}(\varepsilon_t)=\Psi$ is diagonal: all co-movement
  between assets passes through the factors.

Supply loadings and moments on the same return basis and grid. A weekly log-return beta paired
with a monthly simple-return covariance is not a model.

## Methodology

### The linear factor model and its covariance

**Definition.** A linear factor model for $n$ asset returns and $K$ factors is

$$
r_t=Bf_t+\varepsilon_t,
\qquad
\operatorname{Cov}(f_t)=\Sigma_f,
\qquad
\operatorname{Cov}(\varepsilon_t)=\Psi,
\qquad
\operatorname{Cov}(f_t,\varepsilon_t)=0,
$$

with $B$ of shape $n\times K$, $\Sigma_f$ of shape $K\times K$ and $\Psi$ diagonal of shape
$n\times n$.

**Proposition (factor covariance identity).** The asset covariance implied by the model is

$$
\Sigma=B\Sigma_fB^{\top}+\Psi .
$$

**Proof.** Covariance is bilinear, so $\operatorname{Cov}(Bf+\varepsilon)$ equals
$B\Sigma_fB^{\top}+\Psi$ plus the cross terms $B\operatorname{Cov}(f,\varepsilon)$ and
$\operatorname{Cov}(\varepsilon,f)B^{\top}$, which vanish by orthogonality. $\square$

The identity is the reason factor models are used for risk. The first term has rank at most $K$.
The model has $nK+K(K+1)/2+n$ parameters instead of the $n(n+1)/2$ of an unrestricted covariance.
For 500 stocks and 10 factors that is 5,555 parameters instead of 125,250, and the implied
$\Sigma$ is positive definite whenever $\Sigma_f$ is positive semi-definite and every $\psi_i>0$.
The treatment follows Grinold and Kahn (2000), whose exposure matrix is our $B$.

> **Pitfall.** qis uses two orientations for the same loadings, by design, because changing either
> would break the public API. `RiskModel.factor_loadings[date]` and the stress-testing `betas`
> are assets by factors ($B$). `LinearModel.get_loadings_at_date` returns factors by assets
> ($B^{\top}$), and `EwmLinearModel.loadings` stores one dates-by-assets frame per factor. Every
> docstring states its orientation. Transpose before passing an estimated snapshot to `RiskModel`.

### Exposures and the systematic-residual variance split

**Definition.** The factor exposures of weights $w$ are $x=B^{\top}w$, so $x_q=\sum_iw_iB_{iq}$.
The portfolio return is $r_{p,t}=w^{\top}r_t=x^{\top}f_t+w^{\top}\varepsilon_t$.

**Proposition (variance split).** For any weights $w$ with $x=B^{\top}w$,

$$
w^{\top}\Sigma w
=\underbrace{x^{\top}\Sigma_f x}_{\sigma_{\mathrm{sys}}^2}
+\underbrace{w^{\top}\Psi w}_{\sigma_{\mathrm{res}}^2},
\qquad
\sigma_{\mathrm{res}}^2=\sum_{i=1}^{n}\psi_iw_i^2 .
$$

**Proof.** Substitute the covariance identity:
$w^{\top}B\Sigma_fB^{\top}w=(B^{\top}w)^{\top}\Sigma_f(B^{\top}w)=x^{\top}\Sigma_fx$, and
$w^{\top}\Psi w=\sum_i\psi_iw_i^2$ because $\Psi$ is diagonal. $\square$

Variances add; volatilities do not. $\sigma_{\mathrm{sys}}+\sigma_{\mathrm{res}}\ge\sigma$, with
equality only when one of the two parts is zero. The same split holds for active weights $d$:
with active exposures $x_a=B^{\top}d$,
$\mathrm{TE}^2=x_a^{\top}\Sigma_fx_a+d^{\top}\Psi d$. The
[tracking-error chapter](tracking_error_and_risk.md) states how `RiskModel` reports it, following
the benchmark-relative framework of [Roll (1992)](https://doi.org/10.3905/jpm.1992.701922). A
portfolio is the special case $w_b=0$, which is how the worked example obtains total volatility.

### Euler contributions of factors and residuals

**Definition.** With $\sigma>0$, the Euler contribution of factor $q$ and of the residual of asset
$i$ are

$$
c_q=\frac{x_q\,(\Sigma_fx)_q}{\sigma},
\qquad
c^{\varepsilon}_i=\frac{\psi_iw_i^2}{\sigma}.
$$

**Proposition (full allocation).** The contributions add up to total volatility, and the factor
contributions add up to $\sigma$ times the systematic variance share:

$$
\sum_{q=1}^{K}c_q+\sum_{i=1}^{n}c^{\varepsilon}_i=\sigma,
\qquad
\sum_{q=1}^{K}c_q=\frac{\sigma_{\mathrm{sys}}^2}{\sigma}.
$$

**Proof.** $\sum_qx_q(\Sigma_fx)_q=x^{\top}\Sigma_fx=\sigma_{\mathrm{sys}}^2$ and
$\sum_i\psi_iw_i^2=\sigma_{\mathrm{res}}^2$. By the variance split the sum is
$\sigma^2/\sigma=\sigma$. For the Euler reading, write
$\sigma(x,w)=(x^{\top}\Sigma_fx+w^{\top}\Psi w)^{1/2}$ as a function of exposures and residual
weights. It is positively homogeneous of degree one, $x_q\,\partial\sigma/\partial x_q=c_q$ and
$w_i\,\partial\sigma/\partial w_i=c^{\varepsilon}_i$, so Euler's theorem gives the same
sum. $\square$

The factor contributions treat exposures as the decision variables, which is the "hot spots"
view of Litterman (1996). [Tasche (2008)](https://arxiv.org/abs/0708.2542) reviews why the Euler
principle is the natural allocation for homogeneous risk measures, and Qian (2006) reads a
component's share of risk as its expected share of a large portfolio loss. The general
asset-level identity is in [Portfolio risk and Euler contributions](risk_contributions.md).

**Identity (two slicings of systematic risk).** Define the holding-by-factor cells
$\eta_{iq}=w_iB_{iq}(\Sigma_fx)_q/\sigma$. Their column sums are the factor contributions $c_q$ and
their row sums are the per-asset systematic contributions $w_i(B\Sigma_fx)_i/\sigma$.

**Proof.** $\sum_iw_iB_{iq}=x_q$ gives the column sums, and
$\sum_qB_{iq}(\Sigma_fx)_q=(B\Sigma_fx)_i$ gives the row sums. $\square$

`RiskModel.compute_marginal_tre_at_date` reports the row sums as `mcte_systematic` and the
residual contributions as `mcte_residual`; the factor column sums are not a `RiskModel` output.
The `mcte_systematic` total equals $\sigma_{\mathrm{sys}}^2/\sigma$ only when the supplied
covariance satisfies the identity, because `RiskModel` divides by the volatility of the supplied
`covar`, not by the factor-model volatility.

> **Insight.** Standalone volatilities and Euler contributions answer different questions. In
> the worked example the standalone systematic and residual volatilities are 13.39% and 4.10%,
> which sum to 17.49%, while their Euler contributions are 12.80% and 1.20% and sum to the 14%
> total. Use Euler contributions for risk budgets and standalone volatilities for "what if only
> this part existed".

### Benchmark beta

**Proposition (benchmark beta as a regression slope).** Let $r_p=w_p^{\top}r$ and
$r_b=w_b^{\top}r$ with $\operatorname{Cov}(r)=\Sigma$ and $w_b^{\top}\Sigma w_b>0$. The
population least-squares slope of $r_p$ on $r_b$, with an intercept, is

$$
\beta_b=\frac{w_p^{\top}\Sigma w_b}{w_b^{\top}\Sigma w_b},
$$

and the residual $r_p-\beta_br_b$ is uncorrelated with $r_b$.

**Proof.** The slope of a population regression with an intercept is
$\operatorname{Cov}(r_p,r_b)/\operatorname{Var}(r_b)$. Bilinearity gives
$\operatorname{Cov}(r_p,r_b)=w_p^{\top}\Sigma w_b$ and
$\operatorname{Var}(r_b)=w_b^{\top}\Sigma w_b$. Then
$\operatorname{Cov}(r_p-\beta_br_b,r_b)=w_p^{\top}\Sigma w_b-\beta_bw_b^{\top}\Sigma w_b=0$.
$\square$

The beta is linear in the portfolio weights: $\beta_b=\gamma^{\top}w_p$ with
$\gamma=\Sigma w_b/(w_b^{\top}\Sigma w_b)$, and $\gamma^{\top}w_b=1$. Under the factor structure
$\beta_b=(x_p^{\top}\Sigma_fx_b+w_p^{\top}\Psi w_b)/(x_b^{\top}\Sigma_fx_b+w_b^{\top}\Psi w_b)$,
so a portfolio sharing no names with the benchmark has a purely systematic beta.

`RiskModel.compute_benchmark_beta_at_date` evaluates exactly this ratio with the supplied
covariance `covar`, not with the factor block. A non-positive benchmark variance raises
`ValueError` at one date and gives NaN with a `UserWarning` in the history. Benchmark weights
outside the covariance universe are rejected in strict mode: estimate the covariance on the joint
universe of holdings and benchmark constituents. The ex-post regression beta from return series is
covered in [Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md).

### Factor groups

A factor group, or family, names several fitted factors that a scenario or a report treats
together, for example two credit factors. `FactorGroupSpec(group_id, members, weights, label)`
stores the members in order and bump weights $\omega_q\ge0$ that sum to one within $10^{-12}$;
`weights=None` means an equal split. The weights are never renormalised silently. `RiskModel`
accepts a mapping `factor_groups` whose keys equal each `group_id`, requires `factor_loadings`,
rejects a group name that collides with a fitted factor, and requires every member to be a
loading column on every date.

**Definition.** For a group $G$, the implemented group exposures are

$$
x_G=\sum_{q\in G}x_q,
\qquad
x^{\omega}_G=\sum_{q\in G}\omega_qx_q ,
$$

returned as `exposure_sum` and `split_bump_exposure` by
`RiskModel.compute_factor_group_exposures_at_date`.

**Proposition (group exposures as sensitivities).** Let the systematic return be $x^{\top}f$. If
every member factor moves by $\kappa$, its derivative in $\kappa$ is $x_G$. If the factors move
by $\kappa\,\omega_q$ for $q\in G$ and not at all outside $G$, its derivative is $x^{\omega}_G$.

**Proof.** With $f=f_0+\kappa\,v$ the derivative in $\kappa$ is $x^{\top}v$. Take $v_q=1$ for
$q\in G$, or $v_q=\omega_q$ for $q\in G$, and $v_q=0$ otherwise. $\square$

Both are local, first-order sensitivities in the units of the weights, not finite-scenario P&L.
The [instrument stress interface](portfolio_stress.md) uses the same weights to split a family
bump before converting it to log returns.

For risk, a group's Euler contribution is $c_G=\sum_{q\in G}c_q$. It is additive only across a
partition of the factors; overlapping groups double count. The standalone systematic volatility
of a group, $(x_G^{\top}\Sigma_{f,GG}x_G)^{1/2}$ on the member block, is not additive and is not
a `RiskModel` output. The stress report's family exhibit, built by an internal helper in
`qis.portfolio.stress._diagnostics`, sums factor Euler contributions over non-overlapping groups
and falls back to single factors when groups overlap.

### Estimating loadings with EWM regressions

`EwmLinearModel.fit` estimates one set of loadings per date from the factor panel $f_t$ and the
asset panel $r_t$. Rows are indexed $0,1,\ldots$ in sample order, and $m$ runs over past rows.

**Definition (EWM factor regression).** With span $N$, decay $\lambda=1-2/(N+1)$ and zero initial
moments, the EWM moments about zero and the loadings are

$$
\begin{aligned}
M_{ff,t}&=(1-\lambda)\sum_{m=0}^{t}\lambda^{t-m}f_mf_m^{\top},\\
M_{fr,t}&=(1-\lambda)\sum_{m=0}^{t}\lambda^{t-m}f_mr_m^{\top},\\
\hat B_t^{\top}&=M_{ff,t}^{-1}M_{fr,t},\qquad t>W .
\end{aligned}
$$

The code runs the recursion $M_t=\lambda M_{t-1}+(1-\lambda)f_tf_t^{\top}$ in
`compute_ewm_xy_beta_tensor` and returns a time-by-factors-by-assets tensor, whose slice at $t$ is
$\hat B_t^{\top}$.

**Proposition (EWM weighted least squares).** If $M_{ff,t}$ is invertible, $\hat B_t$ minimises
$\sum_{m\le t}\lambda^{t-m}\lVert r_m-Bf_m\rVert^2$ over $B$. In particular, if $r_m=Bf_m$ for
all $m\le t$, then $\hat B_t=B$.

**Proof.** The objective separates into one weighted least-squares problem without an intercept
per asset. The normal equations for the transposed row $b_i$ of $B$ are
$\big(\sum_m\lambda^{t-m}f_mf_m^{\top}\big)b_i=\sum_m\lambda^{t-m}f_mr_{i,m}$, which is
$M_{ff,t}b_i=(M_{fr,t})_{\cdot i}$ after multiplying both sides by $1-\lambda$. If $r_m=Bf_m$,
then $M_{fr,t}=M_{ff,t}B^{\top}$. $\square$

Estimation theory for this estimator, including inference, is in
[Regression and HAC inference](regression_and_hac.md); span, half-life and effective sample size
are in [Exponentially weighted estimators](ewm_estimators.md). Three consequences matter here.

1. **No start-up bias, only start-up noise.** Both moments start at zero and carry the same
   factor $1-\lambda^{t+1}$, which cancels in the ratio. The warm-up exists because an estimate
   from a handful of observations is noisy, not because it is biased by the seed. Positions
   $0,\ldots,W$ are masked, which is $W+1$ rows: 21 rows for the default $W=20$.
2. **Full or diagonal inversion.** `is_x_correlated=True`, the default, inverts the full
   $M_{ff,t}$. With `False`, qis divides by its diagonal only, giving the univariate slope
   $(M_{fr,t})_{qi}/(M_{ff,t})_{qq}$ on each factor. In the noise-free case this is
   $\operatorname{diag}(M_{ff,t})^{-1}M_{ff,t}B^{\top}$, which equals $B^{\top}$ only when the
   weighted factor sample is orthogonal. Otherwise it carries omitted-variable bias.
3. **Numerical fall-backs.** A singular $M_{ff,t}$ is inverted on its diagonal. If the smallest
   diagonal element of $M_{ff,t}$ is at most $10^{-8}$, qis replaces the inverse by the identity,
   so the reported value is the cross moment $M_{fr,t}$ itself, not a slope. On decimal returns
   this triggers only for a factor with a per-period volatility near $10^{-4}$ or below. Missing
   values hold the previous moment element by element.

`mean_adj_type` chooses the regression. `MeanAdjType.NONE`, the default, regresses through the
origin on moments about zero. `MeanAdjType.EWMA` and `MeanAdjType.EXPANDING` first subtract a
point-in-time mean from both panels, which turns the moments into covariances. `init_type` seeds
that EWM mean and has no effect under `NONE`. Its default, `InitType.X0`, seeds with the first
observation, so every loading dated $t$ uses returns up to $t$ only. The demeaned panels serve the
moments only: the model's `x` and `y` keep the returns as supplied. `InitType.MEAN`, the default
until the handbook follow-up and still available, seeds with the full-sample mean and so leaks
later data into early estimates with weight $\lambda^t$, about 0.26 at the first reported loading
for span 31 ($\lambda^{21}$). On monthly synthetic returns with span 36 the two seeds differed by
up to 0.28 in beta at the first reported loading and by less than 0.01 three spans later.

`estimate_ewm_factor_model(asset_prices, factor_prices, freq='W-WED', span=26,
mean_adj_type=MeanAdjType.NONE)` is the price-level entry point. It forms log returns on the
`W-WED` grid, drops the first period, aligns the factor returns to the asset-return index, and
calls `fit` with the full inversion and the default warm-up. Span 26 gives
$\lambda=25/27\approx0.926$ and a half-life of about 9.0 weeks; the first loadings appear at the
22nd weekly return, about five months into the sample. `EwmLinearModel.fit` on its own defaults to
span 31, $\lambda=0.9375$ and a half-life of about 10.7 periods of whatever grid the panels use.
When `span` is given it overrides `ewm_lambda`, whose value 0.94 is the daily decay of
RiskMetrics (J.P. Morgan and Reuters, 1996).

`EwmLinearModel` estimates loadings only. A risk snapshot also needs $\Sigma_f$ and $\Psi$. The
caller estimates them, for example as $\mathrm{AN}$ times EWM second moments of factor returns and
of model residuals from `qis.compute_ewm_covar`, and assembles $\Sigma=B\Sigma_fB^{\top}+\Psi$;
[Stress testing with options](stress_testing_with_options.md) shows this assembly on weekly data.

### Point-in-time use of loadings

$\hat B_t$ uses returns up to and including $t$. It therefore explains $r_t$ in sample and is a
forecast only for returns after $t$. The implementation follows this rule by default:

- `LinearModel.get_factor_alpha(lag=1)` applies $\hat B_{t-1}$ to $f_t$;
  `get_asset_factor_attribution` always shifts the loadings by one period, and its `Total` is
  missing on the dates where a lagged loading is missing.
- `lag=0` applies $\hat B_t$ to $f_t$ and is an in-sample fit. `get_model_ewm_r2` defaults to it.
- A `RiskModel` snapshot dated $t$ should hold $\hat B_t$, $\hat\Sigma_{f,t}$ and
  $\hat\psi_t$ estimated from data up to $t$, and describes risk over the following
  periods. `RiskModel` checks alignment, not provenance: a snapshot built with later data carries
  its look-ahead into every result.

`RiskModel` evaluates single-date methods only on an exact covariance date, raising `KeyError`
with the nearest earlier date otherwise. History methods select, for every covariance date, the
last weight row dated at or before it (forward fill), and use zero weights before the first
weight row. A weight row dated after a covariance date never reaches it.

### LinearModel diagnostics as implemented

`LinearModel(x, y, loadings, x_covars, residual_vars)` is the container that `EwmLinearModel`
extends. Its `loadings` map each factor to a dates-by-assets frame, `x_covars` maps dates to
factor covariances, and `residual_vars` is a dates-by-assets frame. The diagnostics below are
stated as the code computes them.

**Definition (model residual).** For lag $k\in\{0,1\}$, `get_factor_alpha(lag=k)` returns

$$
\hat\varepsilon^{(k)}_{i,t}=r_{i,t}-\sum_{q=1}^{K}\hat B_{iq,t-k}\,f_{q,t},
$$

with the loadings forward-filled onto the return index before the shift; the optional `span`
smooths the result with `qis.compute_ewm`. The code calls this "factor alpha". It is the residual
of a regression without an intercept, so it contains any intercept plus noise; it is not an
intercept estimate.

**Definition (EWM $R^2$).** With $\mathcal{T}_{i,t}$ the dates up to $t$ on which both
$\hat\varepsilon^{(k)}_{i}$ and $r_i$ are finite, and $n_{i,t}(m)$ the number of those dates after
$m$, `get_model_ewm_r2(span=52, lag=0)` returns

$$
\begin{aligned}
Q^{\varepsilon}_{i,t}&=\sum_{m\in\mathcal{T}_{i,t}}\lambda^{n_{i,t}(m)}\big(\hat\varepsilon^{(k)}_{i,m}\big)^2,
\qquad
Q^{r}_{i,t}=\sum_{m\in\mathcal{T}_{i,t}}\lambda^{n_{i,t}(m)}r_{i,m}^2,\\
R^2_{i,t}&=\min\big\{1,\ \max\{0,\ 1-Q^{\varepsilon}_{i,t}/Q^{r}_{i,t}\}\big\},
\end{aligned}
$$

with span $N=52$ and $k=0$ by default. The two sums run over the same dates with the same weights,
so no seed and no normalisation enters the ratio: it is the uncentred $R^2$ of the lag-$k$ fit on
the weighted window, and on the first date after the warm-up it is the single-observation ratio
$1-(\hat\varepsilon^{(k)}_{i,m})^2/r_{i,m}^2$. It differs from a textbook $R^2$ in two deliberate
ways. It is uncentred: the denominator is a second moment about zero, not a variance, matching the
regression through the origin that `fit` runs by default. At the default lag it is in sample; pass
`lag=1` for the point-in-time version. Until the handbook follow-up the residual moment started
from zero after the warm-up while the return moment had run since the first return, which pushed
$R^2$ towards one for roughly a span after the warm-up.

**Definition (average residual correlation).** `get_model_residuals_corrs(span=52)` returns the
EWM correlation matrix $C$ of the lag-0 residuals at the last date only, from moments about zero
with a zero seed, and for each asset the mean off-diagonal correlation

$$
\bar\rho_i=\frac{1}{n-1}\sum_{j\ne i}C_{ij},
$$

which is NaN for a single asset. Until the handbook follow-up the function returned
$\tfrac{n-1}{2n}\bar\rho_i$ instead, one third of $\bar\rho_i$ for three assets. Material residual
correlation means the diagonal $\Psi$ misses common risk.

**Definition (aggregated exposures).** `compute_agg_factor_exposures(weights)` returns
$X_{q,t}=\sum_i\hat B_{iq,t}w_{i,(t)}$ on the loading dates, with same-date loadings and the
weights in force at $t$, selected as of $t$; weight rows dated off the loading grid therefore
count from their own date on. The sum runs over the assets with a non-zero weight. $X_{q,t}$ is
missing before the first weight row and wherever a held asset has no loading, which includes the
warm-up rows; a portfolio of zero weights has exposure zero. Until the handbook follow-up weights
were matched to loading dates exactly and missing loadings counted as zero.

**Definition (factor risk contribution shares).** For each date $t$ of `x_covars`,
`compute_factor_risk_contribution(weights)` takes the last weights, loadings and residual
variances at or before $t$, forms $x=\hat B_t^{\top}w$ over the held assets and returns

$$
\frac{x_q(\Sigma_fx)_q}{\sigma^2},
\qquad
\frac{\sigma_{\mathrm{res}}^2}{\sigma^2},
\qquad
\frac{x_q(\Sigma_fx)_q}{\sigma_{\mathrm{sys}}^2},
$$

with $\sigma^2=\sigma_{\mathrm{sys}}^2+\sigma_{\mathrm{res}}^2$. The factor shares and the
residual share sum to one; they equal $c_q/\sigma$ and $\sum_ic^{\varepsilon}_i/\sigma$ and form
the second output, with the residual share in an `Idiosyncratic` column. The third expression is
the third output. The first output divides the second by its row sum, which is one wherever the
shares are defined, and the fourth holds $\sigma_{\mathrm{sys}}^2$ and $\sigma_{\mathrm{res}}^2$.
A date is missing in every output when a held asset has no loading or residual variance; a factor
of $\Sigma_f$ on which no asset loads has exposure zero. Until the handbook follow-up missing
exposures were set to zero, residual variances were read on exactly $t$, and undefined ratios
were reported as zero.

## Worked example

Take three assets and two factors, Equity and Rates, with loadings, factor covariance and residual
variances

$$
B=\begin{pmatrix}1&0\\0.5&0.5\\0&1\end{pmatrix},
\qquad
\Sigma_f=\begin{pmatrix}0.04&0.004\\0.004&0.01\end{pmatrix},
\qquad
\psi=(0.01,\ 0.0004,\ 0.0004).
$$

The factor volatilities are 20% and 10% with correlation 0.2, and the residual volatilities are
10%, 2% and 2%. All inputs are annual and hand-chosen. The identity gives

$$
\Sigma=\begin{pmatrix}0.05&0.022&0.004\\0.022&0.0149&0.007\\0.004&0.007&0.0104\end{pmatrix}.
$$

For weights $w=(0.4,0.4,0.2)$ the exposures are $x=(0.6,0.4)$ and $\Sigma_fx=(0.0256,0.0064)$.
Systematic variance is $0.01792$, residual variance is $0.00168$, and they add to $0.0196$: total
volatility is exactly **14%**, with standalone systematic and residual volatilities of 13.39% and
4.10%.

```python
from math import isclose

import numpy as np
import pandas as pd
import qis

assets = ['A1', 'A2', 'A3']
factors = ['Equity', 'Rates']
date = pd.Timestamp('2025-12-31')
loadings = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], index=assets, columns=factors)
factor_covar = pd.DataFrame([[0.04, 0.004], [0.004, 0.01]], index=factors, columns=factors)
residual_vars = pd.Series([0.01, 0.0004, 0.0004], index=assets)
covar = loadings @ factor_covar @ loadings.T + pd.DataFrame(
    np.diag(residual_vars), index=assets, columns=assets)
np.testing.assert_allclose(covar.to_numpy(), [[0.05, 0.022, 0.004],
                                              [0.022, 0.0149, 0.007],
                                              [0.004, 0.007, 0.0104]], atol=1e-15)

model = qis.RiskModel(covar={date: covar}, factor_loadings={date: loadings},
                      factor_covar={date: factor_covar}, residual_vars={date: residual_vars})
weights = pd.Series([0.4, 0.4, 0.2], index=assets)
no_benchmark = pd.Series(0.0, index=assets)  # w_b = 0 turns tracking error into volatility

exposures = model.compute_exposures_at_date(portfolio_weights=weights, date=date)
np.testing.assert_allclose(exposures.to_numpy(), [0.6, 0.4], atol=1e-15)

vol = model.compute_tre_at_date(benchmark_weights=no_benchmark, portfolio_weights=weights,
                                date=date)
split = model.compute_tre_decomposition_at_date(benchmark_weights=no_benchmark,
                                                portfolio_weights=weights, date=date)
assert isclose(vol, 0.14, abs_tol=1e-12)
assert isclose(split['factor_te'] ** 2, 0.6 * 0.0256 + 0.4 * 0.0064, abs_tol=1e-12)
assert isclose(split['residual_te'] ** 2, 0.16 * 0.01 + 0.16 * 0.0004 + 0.04 * 0.0004,
               abs_tol=1e-12)
assert isclose(split['factor_te'] ** 2 + split['residual_te'] ** 2, vol ** 2, abs_tol=1e-12)
assert isclose(split['tracking_error'], vol, abs_tol=1e-12)  # the two views reconcile
```

The last check is the covariance identity seen through `RiskModel`: the covariance view
(`compute_tre_at_date`) and the factor-block view (`tracking_error` of the decomposition) agree
because the supplied `covar` equals $B\Sigma_fB^{\top}+\Psi$.

The factor Euler contributions are $0.6\times0.0256/0.14=10.97\%$ for Equity and
$0.4\times0.0064/0.14=1.83\%$ for Rates. They sum to **12.80%**, which is $32/35\approx91.43\%$
of total volatility, the systematic variance share. The residual contributions sum to 1.20%. As
shares of variance the three parts are 78.37%, 13.06% and 8.57%; within systematic risk the split
is $6/7$ and $1/7$.

```python
factor_euler = exposures * (factor_covar @ exposures) / vol
residual_euler = residual_vars * weights ** 2 / vol
np.testing.assert_allclose(factor_euler.to_numpy(), [0.01536 / 0.14, 0.00256 / 0.14],
                           atol=1e-12)
assert isclose(factor_euler.sum() + residual_euler.sum(), vol, abs_tol=1e-12)
assert isclose(factor_euler.sum() / vol, 32 / 35, abs_tol=1e-12)
assert isclose(residual_euler.sum(), 0.012, abs_tol=1e-12)

# Euler on the factor block alone sums to the systematic volatility; rescale to total risk.
on_factor_block = qis.compute_portfolio_risk_contributions(exposures, factor_covar)
np.testing.assert_allclose(on_factor_block * split['factor_te'] / vol, factor_euler,
                           atol=1e-12)

by_asset = model.compute_marginal_tre_at_date(benchmark_weights=no_benchmark,
                                              portfolio_weights=weights, date=date)
np.testing.assert_allclose(by_asset['mcte_systematic'] + by_asset['mcte_residual'],
                           by_asset['mcte'], atol=1e-12)
assert isclose(by_asset['mcte_systematic'].sum(), factor_euler.sum(), abs_tol=1e-12)
np.testing.assert_allclose(by_asset['mcte_residual'], residual_euler, atol=1e-12)

linear_model = qis.LinearModel(
    x=pd.DataFrame(0.0, index=[date], columns=factors),
    y=pd.DataFrame(0.0, index=[date], columns=assets),
    loadings={q: loadings[[q]].T.set_axis([date]) for q in factors},  # dates by assets
    x_covars={date: factor_covar},
    residual_vars=residual_vars.to_frame(date).T)
ratios, total_shares, systematic_shares, variances = (
    linear_model.compute_factor_risk_contribution(weights=weights.to_frame(date).T))
np.testing.assert_allclose(total_shares.loc[date], [192 / 245, 32 / 245, 3 / 35], atol=1e-12)
np.testing.assert_allclose(systematic_shares.loc[date], [6 / 7, 1 / 7], atol=1e-12)
```

Take the benchmark to be asset A1 alone. Then $\beta_b=(\Sigma w)_1/\Sigma_{11}=0.0296/0.05=$
**0.592**, and the position $w-\beta_bw_b$ has zero covariance with the benchmark. A family
`Macro` with both factors and an equal split has `exposure_sum` $0.6+0.4=1$ and
`split_bump_exposure` $0.5\times0.6+0.5\times0.4=0.5$. Finally, with covariance dates 30 June and
31 December 2025 and weight rows dated 31 March, 30 September 2025 and 15 January 2026, each
covariance date sees the last earlier weight row; the 2026 row is never used.

```python
benchmark = pd.Series([1.0, 0.0, 0.0], index=assets)
beta = model.compute_benchmark_beta_at_date(benchmark_weights=benchmark,
                                            portfolio_weights=weights, date=date)
assert isclose(beta, 0.0296 / 0.05, abs_tol=1e-12)
gamma = model.compute_benchmark_beta_loadings_at_date(benchmark_weights=benchmark, date=date)
assert isclose(float(gamma @ weights), beta, abs_tol=1e-12)
assert isclose(float(gamma @ benchmark), 1.0, abs_tol=1e-12)
assert isclose(float((weights - beta * benchmark) @ covar @ benchmark), 0.0, abs_tol=1e-15)

macro = qis.FactorGroupSpec(group_id='Macro', members=('Equity', 'Rates'),
                            label='Equity and rates')
grouped = qis.RiskModel(covar={date: covar}, factor_loadings={date: loadings},
                        factor_covar={date: factor_covar}, residual_vars={date: residual_vars},
                        factor_groups={'Macro': macro})
group_table = grouped.compute_factor_group_exposures_at_date(portfolio_weights=weights,
                                                             date=date)
assert isclose(group_table.loc['Macro', 'exposure_sum'], 1.0, abs_tol=1e-12)
assert isclose(group_table.loc['Macro', 'split_bump_exposure'], 0.5, abs_tol=1e-12)

risk_dates = [pd.Timestamp('2025-06-30'), date]
dated = qis.RiskModel(covar={risk_date: covar for risk_date in risk_dates},
                      factor_loadings={risk_date: loadings for risk_date in risk_dates})
weight_history = pd.DataFrame([[1.0, 0.0, 0.0], [0.4, 0.4, 0.2], [0.0, 0.0, 1.0]],
                              index=pd.to_datetime(['2025-03-31', '2025-09-30', '2026-01-15']),
                              columns=assets)
exposure_history = dated.compute_exposures_history(portfolio_weights=weight_history)
np.testing.assert_allclose(exposure_history.to_numpy(), [[1.0, 0.0], [0.6, 0.4]], atol=1e-15)
```

Now estimate the loadings. Daily factor log returns are drawn from a fixed seed over 2023–2024,
and asset log prices are built as exactly $B$ times factor log prices, so weekly log returns
satisfy $r_t=Bf_t$ with no noise. `estimate_ewm_factor_model` with its defaults gives 103 weekly
`W-WED` returns. The first 21 rows are warm-up; from the 22nd row on, every estimate equals $B$
to within $10^{-10}$, as the weighted-least-squares proposition predicts. With `lag=1` the
residual is missing for the first 22 rows and zero afterwards.

```python
rng = np.random.default_rng(20260725)
days = pd.bdate_range('2023-01-02', '2024-12-31')
daily_factor_logs = pd.DataFrame(rng.normal(0.0, [0.012, 0.006], size=(len(days), 2)),
                                 index=days, columns=factors)
factor_prices = 100.0 * np.exp(daily_factor_logs.cumsum())
asset_prices = 100.0 * np.exp((daily_factor_logs @ loadings.T).cumsum())  # log r = B f exactly

fit = qis.estimate_ewm_factor_model(asset_prices=asset_prices, factor_prices=factor_prices)
weekly_index = fit.x.index
assert weekly_index.freqstr == 'W-WED' and len(weekly_index) == 103
first_estimate = fit.loadings['Equity'].first_valid_index()
assert first_estimate == weekly_index[21]  # rows 0, ..., 20 are the warm-up
for q in factors:
    estimated = fit.loadings[q].loc[first_estimate:]
    np.testing.assert_allclose(estimated.to_numpy(),
                               np.tile(loadings[q].to_numpy(), (len(estimated), 1)), atol=1e-10)

snapshot = fit.get_loadings_at_date(weekly_index[-1]).T  # factors by assets -> assets by factors
np.testing.assert_allclose(snapshot.loc[assets, factors], loadings, atol=1e-10)

point_in_time, explained = fit.get_factor_alpha()  # lag=1: last week's loadings
assert point_in_time.iloc[:22].isna().all().all()
assert point_in_time.iloc[22:].abs().max().max() < 1e-12
```

With noise the proposition still identifies the estimator. Add independent residuals with weekly
volatilities 2%, 1% and 1% to 156 weeks of factor returns and fit with span 26. The final
loadings equal an explicit weighted least-squares solve with weights $\lambda^{155-s}$: about
$1.009$, $0.468$ and $-0.042$ on Equity and $-0.110$, $0.484$ and $0.908$ on Rates. The
diagonal fit instead returns the univariate slopes, and its Rates loading for A1 is about $-0.73$
against a true value of zero: the weighted sample correlation of the two factors leaks into it.

```python
weeks = pd.date_range('2022-01-05', periods=156, freq='W-WED')
factor_returns = pd.DataFrame(rng.normal(0.0, [0.03, 0.015], size=(156, 2)),
                              index=weeks, columns=factors)
noise = pd.DataFrame(rng.normal(0.0, [0.02, 0.01, 0.01], size=(156, 3)),
                     index=weeks, columns=assets)
asset_returns = factor_returns @ loadings.T + noise
joint = qis.EwmLinearModel(x=factor_returns, y=asset_returns)
joint.fit(span=26)
diagonal = qis.EwmLinearModel(x=factor_returns, y=asset_returns)
diagonal.fit(span=26, is_x_correlated=False)

lam = 1.0 - 2.0 / 27.0
decay = lam ** np.arange(155, -1, -1)
f_np, r_np = factor_returns.to_numpy(), asset_returns.to_numpy()
m_ff = (f_np * decay[:, None]).T @ f_np
m_fr = (f_np * decay[:, None]).T @ r_np
joint_last = joint.get_loadings_at_date(weeks[-1])
np.testing.assert_allclose(joint_last, np.linalg.solve(m_ff, m_fr), atol=1e-12)
np.testing.assert_allclose(joint_last.to_numpy(), [[1.009, 0.468, -0.042],
                                                   [-0.110, 0.484, 0.908]], atol=5e-4)
diagonal_last = diagonal.get_loadings_at_date(weeks[-1])
np.testing.assert_allclose(diagonal_last, m_fr / np.diag(m_ff)[:, None], atol=1e-12)
assert isclose(diagonal_last.loc['Rates', 'A1'], -0.731, abs_tol=5e-4)
```

The residual-correlation diagnostic returns each asset's mean correlation with the other two
assets: about 0.13, 0.18 and $-0.02$. The check recomputes $C$ from the lag-0 residuals with
span-52 weights.

```python
corr, avg_corr = joint.get_model_residuals_corrs()  # span 52, lag-0 residuals, last date
in_sample, _ = joint.get_factor_alpha(lag=0)
e_np = in_sample.to_numpy()
valid = np.isfinite(e_np).all(axis=1)
lam52 = 1.0 - 2.0 / 53.0
w52 = lam52 ** (len(weeks) - 1 - np.flatnonzero(valid))
moments = (e_np[valid] * w52[:, None]).T @ e_np[valid]
c = moments / np.sqrt(np.outer(np.diag(moments), np.diag(moments)))
np.testing.assert_allclose(corr.to_numpy(), c, atol=1e-12)
mean_off_diagonal = (c.sum(axis=1) - 1.0) / 2.0
np.testing.assert_allclose(avg_corr.to_numpy(), mean_off_diagonal, atol=1e-12)
np.testing.assert_allclose(avg_corr.to_numpy(), [0.126, 0.176, -0.024], atol=5e-4)
```

The in-sample EWM $R^2$ at the last date is about 0.80, 0.73 and 0.72. It is reproduced below from
its definition: two sums with the same span-52 weights over the dates with a finite residual.
On the first date after the warm-up it is the single-week ratio, about 0.15, 0.00 (clipped) and
0.79; the misaligned seeds used until the handbook follow-up reported 0.999, 0.997 and 0.970
there. With `lag=1` the last-date values fall to about 0.75, 0.66 and 0.63.

```python
r2 = joint.get_model_ewm_r2()  # span 52, lag 0: in sample
numerator = (w52[:, None] * e_np[valid] ** 2).sum(axis=0)
denominator = (w52[:, None] * r_np[valid] ** 2).sum(axis=0)
np.testing.assert_allclose(r2.iloc[-1].to_numpy(), 1.0 - numerator / denominator, atol=1e-12)
np.testing.assert_allclose(r2.iloc[-1].to_numpy(), [0.80, 0.73, 0.72], atol=5e-3)
first = np.flatnonzero(valid)[0]
single_week = np.clip(1.0 - e_np[first] ** 2 / r_np[first] ** 2, 0.0, 1.0)
np.testing.assert_allclose(r2.iloc[first].to_numpy(), single_week, atol=1e-12)
np.testing.assert_allclose(single_week, [0.15, 0.00, 0.79], atol=5e-3)
r2_lag1 = joint.get_model_ewm_r2(lag=1)
np.testing.assert_allclose(r2_lag1.iloc[-1].to_numpy(), [0.75, 0.66, 0.63], atol=5e-3)
```

Finally, month-end weights meet the weekly loadings as of each week: a week that starts a new
month uses the previous month-end's weights, and the warm-up weeks have no exposure.

```python
month_ends = pd.date_range('2022-01-31', periods=36, freq='ME')
month_weights = pd.DataFrame(rng.dirichlet(np.ones(3), size=36), index=month_ends,
                             columns=assets)
agg = joint.compute_agg_factor_exposures(weights=month_weights)
last_month_end = np.searchsorted(month_ends.to_numpy(), weeks.to_numpy(), side='right') - 1
in_force = month_weights.to_numpy()[np.maximum(last_month_end, 0)]
in_force[last_month_end < 0] = np.nan  # weeks before the first weight row
for q in factors:
    expected = (joint.loadings[q].to_numpy() * in_force).sum(axis=1)  # NaN if any term is NaN
    np.testing.assert_allclose(agg[q].to_numpy(), expected, atol=1e-15)
assert agg.iloc[:21].isna().all().all() and agg.iloc[21:].notna().all().all()
```

These are fixed teaching inputs. The asserts compare qis with hand arithmetic, with the explicit
matrix products of the propositions, and with direct NumPy solves; they test the algebra, not the
forecasting accuracy of any model.

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| Dated model snapshot | $\Sigma$, $B$, $\Sigma_f$, $\psi$ on one date grid | `qis.RiskModel(covar, factor_loadings, factor_covar, residual_vars, factor_groups)` |
| Factor exposures | $x=B^{\top}w$ | `RiskModel.compute_exposures_at_date`, `RiskModel.compute_exposures_history` |
| Systematic and residual risk | $\sigma_{\mathrm{sys}}$, $\sigma_{\mathrm{res}}$, their root-sum-square | `RiskModel.compute_tre_decomposition_at_date` and `_history`: `factor_te`, `residual_te`, `tracking_error` |
| Covariance-view risk | $(d^{\top}\Sigma d)^{1/2}$ | `RiskModel.compute_tre_at_date`, `compute_tre_history`, `compute_tre_by_group_loadings_at_date` |
| Asset Euler contributions | $d_i(\Sigma d)_i/\mathrm{TE}$, split by $B\Sigma_fB^{\top}d$ and $\Psi d$ | `RiskModel.compute_marginal_tre_at_date`: `mcte`, `mcte_systematic`, `mcte_residual` |
| Factor Euler contributions | $c_q=x_q(\Sigma_fx)_q/\sigma$ | `qis.compute_portfolio_risk_contributions(x, factor_covar)` times $\sigma_{\mathrm{sys}}/\sigma$ |
| Contribution shares | $x_q(\Sigma_fx)_q/\sigma^2$, $\sigma_{\mathrm{res}}^2/\sigma^2$ | `LinearModel.compute_factor_risk_contribution` |
| Benchmark beta | $w_p^{\top}\Sigma w_b/(w_b^{\top}\Sigma w_b)$ | `RiskModel.compute_benchmark_beta_at_date`, `compute_benchmark_beta_history` |
| Beta loadings | $\gamma=\Sigma w_b/(w_b^{\top}\Sigma w_b)$ | `RiskModel.compute_benchmark_beta_loadings_at_date` |
| Group exposures | $x_G$, $x^{\omega}_G$ | `qis.FactorGroupSpec`, `RiskModel.compute_factor_group_exposures_at_date` |
| EWM loadings | $\hat B_t^{\top}=M_{ff,t}^{-1}M_{fr,t}$ | `qis.EwmLinearModel.fit`, `qis.compute_ewm_xy_beta_tensor` |
| Loadings from prices | weekly log returns, span 26 | `qis.estimate_ewm_factor_model` |
| Loadings snapshot | $\hat B_t^{\top}$, factors by assets | `LinearModel.get_loadings_at_date` |
| Model residual | $\hat\varepsilon^{(k)}_{i,t}$ | `LinearModel.get_factor_alpha(lag=1, span=None)` |
| EWM $R^2$ | $1-Q^{\varepsilon}_{i,t}/Q^{r}_{i,t}$ on common dates, uncentred, lag 0 | `LinearModel.get_model_ewm_r2(span=52, lag=0)` |
| Residual correlation | $C$ and $\bar\rho_i=\tfrac{1}{n-1}\sum_{j\ne i}C_{ij}$ | `LinearModel.get_model_residuals_corrs(span=52)` |
| Aggregated exposures | $X_{q,t}=\sum_i\hat B_{iq,t}w_{i,(t)}$, weights as of $t$ | `LinearModel.compute_agg_factor_exposures` |
| Asset attribution | $\hat B_{iq,t-1}f_{q,t}$ and their total, missing while a lagged loading is | `LinearModel.get_asset_factor_attribution` |
| Portfolio benchmark betas | EWMA-demeaned betas aggregated by weights | `qis.compute_portfolio_ewm_benchmark_betas`, `qis.compute_portfolio_benchmark_ewm_beta_alpha_attribution` |
| Factor price panel | validated factor prices | `qis.FactorsData(factors_prices, factors=None)` |

The implementations are in
[risk_model.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/risk_model.py),
[factor_groups.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/factor_groups.py),
[ewm_factor_model.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/ewm_factor_model.py),
[factor_model.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/factor_model.py),
[contributions.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/contributions.py)
and [factors_data.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/factors_data.py).
The EWM beta recursion is in
[ewm.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py).

### The RiskModel contract

- **Construction.** `covar` is required and non-empty; every field is a dict keyed by date, and
  keys are converted to `pd.Timestamp` and sorted. `factor_covar` and `residual_vars` come
  together and require `factor_loadings`; `factor_groups` requires `factor_loadings`.
- **Validation.** Each covariance and factor covariance must be a labelled, finite, square
  DataFrame with index equal to columns and symmetric within $10^{-12}$. Loadings and residual
  variances must cover exactly the covariance assets and every optional field exactly the
  covariance dates. Loadings and residual variances are reordered to the covariance assets, and
  the factor covariance to the loading columns. Both covariances must be positive
  semi-definite and residual variances non-negative, each up to a rounding tolerance of
  $10^{-10}$ times the largest diagonal element or variance (at least $10^{-10}$); the check
  raises `ValueError` with the smallest eigenvalue or the offending variances and never alters
  the inputs. These checks were added in the handbook follow-up; the stress module applied them
  only at evaluation time before.
- **Consistency.** `RiskModel` never builds $\Sigma$ from the factor block. Covariance-view
  methods use `covar`; decomposition methods use the factor block; they agree only when the caller
  supplies $\Sigma=B\Sigma_fB^{\top}+\Psi$, and qis does not reconcile them silently.
- **Weights.** Missing in-universe weights become zero; weights above $10^{-10}$ in absolute
  value outside the covariance universe raise `ValueError` when `strict=True`, the default.
- **Units.** No annualisation is applied and results inherit the covariance units. The stack
  convention is annual covariance, which `qis.estimate_rolling_ewma_covar` returns by default
  (`apply_an_factor=True`).

### Loading layouts across modules

| Object | Layout | Residual variances |
|---|---|---|
| `RiskModel.factor_loadings[date]` | Assets by factors, $B$ | `residual_vars[date]`: Series by asset |
| Stress-testing `betas` | Assets by factors, $B$ | Series by asset |
| `EwmLinearModel.loadings[factor]` | Dates by assets, one frame per factor | Not estimated |
| `LinearModel.get_loadings_at_date` | Factors by assets, $B^{\top}$ | `LinearModel.residual_vars`: dates by assets |
| `qis.compute_ewm_xy_beta_tensor` | Time by factors by assets | Not estimated |

`LinearModel.compute_active_factor_risk` is deprecated and warns on every call; use the
`RiskModel` decomposition and contributions instead. API pages:
{doc}`RiskModel <api/generated/qis.RiskModel>`,
{doc}`EwmLinearModel <api/generated/qis.EwmLinearModel>`,
{doc}`LinearModel <api/generated/qis.LinearModel>`,
{doc}`FactorGroupSpec <api/generated/qis.FactorGroupSpec>` and
{doc}`estimate_ewm_factor_model <api/generated/qis.estimate_ewm_factor_model>`.

## Interpretation and limitations

- **The model is only as good as its diagonal.** Residual correlation that the factors miss, such
  as a sector the model has no factor for, is set to zero in $\Psi$ and understates the risk of
  concentrated portfolios. Inspect `get_model_residuals_corrs`, whose average is each asset's mean
  off-diagonal residual correlation.
- **Time-series loadings are statistical.** `EwmLinearModel` regresses on observed factor
  returns. Fundamental models, which take loadings as observed characteristics and estimate
  factor returns cross-sectionally, are not implemented in qis; `RiskModel` accepts their
  snapshots all the same.
- **In-sample diagnostics flatter the model.** The default EWM $R^2$ uses lag 0 and is
  uncentred; use `lag=1` for the point-in-time fit. Its first values after the warm-up rest on a
  few dates and are noisy.
- **Estimation noise is not propagated.** Loadings, $\Sigma_f$ and $\Psi$ enter as known
  quantities. Contributions, betas and bands carry no parameter uncertainty, and a snapshot of
  noisy loadings reports noisy exposures with full confidence.
- **Look-ahead enters through choices, not through `RiskModel`.** An explicit full-sample mean
  seed (`init_type=InitType.MEAN` with a mean adjustment), lag-0 residuals and snapshots
  assembled with later data all leak information that no validation step detects. The defaults
  of `EwmLinearModel.fit` are point in time.

> **Pitfall.** Missing is not zero. Exposures, contribution shares and attribution totals of
> `LinearModel` are missing while a held asset has no loading, including the warm-up, and a
> weight row dated off the loading grid applies as of its own date, as in
> `RiskModel.compute_exposures_history`. Until the handbook follow-up, weights were matched to the
> loading dates exactly, so month-end weights against weekly `W-WED` loadings survived only on
> Wednesdays, and missing loadings counted as zero exposure.

> **Insight.** A factor model is a structured covariance estimator. With $K$ factors and diagonal
> residuals it accepts some bias in exchange for far fewer parameters, which tends to make its
> contributions more stable through time than those of a sample covariance of the same assets.

## See also

- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Portfolio risk and Euler contributions](risk_contributions.md)
- [Factor stress testing](stress_testing.md)
- [Instrument portfolio stress](portfolio_stress.md)
- [Stress testing with options](stress_testing_with_options.md)
- [Regression and HAC inference](regression_and_hac.md)
- [Exponentially weighted estimators](ewm_estimators.md)
- [Covariance, correlation and principal components](covariance_correlation_pca.md)
- [Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md)
- [Notation and conventions](notation_and_conventions.md)

## References

1. Grinold, R. C., and Kahn, R. N. (2000). *Active Portfolio Management*, 2nd edition. McGraw-Hill. The structure of factor risk models, exposures and the systematic-specific split.
2. Litterman, R. (1996). Hot Spots and Hedges. *Goldman Sachs Risk Management Series*. Marginal contributions to portfolio risk as a decomposition tool.
3. Tasche, D. (2008). Capital allocation to business units and sub-portfolios: the Euler principle. Working paper. [arXiv:0708.2542](https://arxiv.org/abs/0708.2542). The Euler allocation of a homogeneous risk measure.
4. Qian, E. (2006). On the Financial Interpretation of Risk Contribution: Risk Budgets Do Add Up. *Journal of Investment Management*, 4(4), 41–51. The loss-contribution reading of Euler risk shares.
5. Roll, R. (1992). A Mean/Variance Analysis of Tracking Error. *The Journal of Portfolio Management*, 18(4), 13–22. [DOI: 10.3905/jpm.1992.701922](https://doi.org/10.3905/jpm.1992.701922). Benchmark-relative risk for the active-weight form of the variance split.
6. J.P. Morgan and Reuters (1996). *RiskMetrics — Technical Document*, 4th edition. J.P. Morgan. Exponentially weighted moments and the 0.94 daily decay.
7. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
