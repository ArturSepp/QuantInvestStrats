---
myst:
  html_meta:
    description: >-
      Ordinary, geometrically weighted and point-in-time exponentially weighted return
      regressions in qis, with Bartlett-kernel Newey-West HAC inference for alpha, Kish effective
      sample sizes and exact contracts for every estimator.
---

# Regression and HAC inference

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A return regression explains a periodic return by an intercept, alpha, and by slopes, betas, on
benchmark or factor returns, leaving a residual. qis estimates these coefficients in three ways:
full-sample ordinary least squares, endpoint least squares with geometric recency weights, and
point-in-time ratios of exponentially weighted moments. Inference on alpha uses the Bartlett-kernel
heteroskedasticity and autocorrelation consistent (HAC) covariance of
[Newey and West (1987)](https://www.nber.org/papers/t0055), with the statsmodels small-sample
correction and a normal reference distribution.

## Overview

The chapter is the reference for the regressions behind benchmark tables, model-layer
attribution and linear factor models. It states what each estimator computes, proves the
properties the rest of the library relies on, and records the implementation contracts,
including the defaults that look ahead.

| Question | Estimator | qis entry point |
|---|---|---|
| Full-sample alpha, beta, $R^2$ and a classical p-value | OLS | `qis.fit_multivariate_ols`; internal `estimate_ols_alpha_beta` |
| Full-sample alpha with a dependence-robust interval | OLS with Bartlett HAC | internal `estimate_ols_alpha_beta_hac`, `estimate_hac_mean` |
| Current recency-weighted alphas of related series, jointly | Geometric WLS with stacked HAC | `qis.estimate_ewma_alpha_beta_hac` |
| Exposures at every date | Ratios of EWM moments | `qis.compute_ewm_xy_beta_tensor`, `qis.EwmLinearModel.fit`, `qis.compute_one_factor_ewm_betas` |
| One-factor EWM alpha and $R^2$ paths | EWM beta, EWM residual mean | `qis.compute_ewm_beta_alpha_forecast`, `qis.compute_ewm_alpha_r2_given_prediction`, `qis.LinearModel` |

Three results carry the chapter:

1. **Linearity.** For a fixed design, least-squares coefficients, residuals and scores are linear
   in the response. Additive attribution of alphas across layers is exact because of it.
2. **Quadratic-form inference.** The Bartlett HAC covariance is a quadratic form in the scores
   and is positive semidefinite for every lag count. The variance of a linear contrast of alphas
   is therefore $c^{\top}\hat\Sigma_{\alpha}c$, and it equals the variance obtained by fitting the
   contrast series directly.
3. **Point-in-time EWM betas are prefix regressions.** The ratio of zero-seeded EWM moments at
   date $t$ is the weighted least-squares slope through the origin on the rows up to $t$, with
   the same geometric weights that the endpoint EWMA-WLS fit uses.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Any periodic return series supplied by the caller; qis callers pass log returns (model-layer attribution, `qis.estimate_ewm_factor_model`) or simple returns (`qis.compute_ra_perf_table_with_benchmark` by default) |
| Sampling grid | Rows as supplied; the estimators have no calendar logic and do not sort rows. The examples use month-end (`ME`) log returns |
| Annualisation | None inside the estimators. Tables annualise alpha linearly, $\mathrm{AN}\,\hat\alpha$; chart legends compound, $e^{\mathrm{AN}\hat\alpha}-1$, only when `alpha_an_factor` is passed |
| Mean adjustment | OLS and WLS fit an intercept, which removes the sample (or weighted) means. EWM betas use moments about zero by default (`MeanAdjType.NONE`) |
| Timing | OLS/HAC and EWMA-WLS are descriptive endpoint fits over every retained row. An EWM beta dated $t$ uses rows up to and including $t$; lag it one period before applying it |
| Output units | Coefficients in the units of the inputs: $\hat\alpha$ per period, $\hat\beta$ dimensionless, $R^2$ a fraction, standard errors per period |
| qis default | `qis.estimate_ewma_alpha_beta_hac(span=36.0, hac_lags=3, confidence_level=0.95)`; `EwmLinearModel.fit(span=31, is_x_correlated=True, mean_adj_type=MeanAdjType.NONE, init_type=InitType.MEAN, warmup_period=20)` |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $y_t$, $x_t$ | Response and regressor row at row $t$ | Periodic returns; $x_t$ starts with a one when an intercept is fitted |
| $X$, $y$ | Design matrix ($T\times p$) and response vector | Rows retained after dropping non-finite values |
| $p$ | Number of estimated coefficients, the columns of $X$ | 2 for $(\alpha,\beta)$ with one regressor |
| $\theta$, $\hat\theta$ | Coefficient vector and its estimate | $\theta=(\alpha,\beta)^{\top}$ with one regressor |
| $\hat\varepsilon_t$ | Residual $y_t-x_t^{\top}\hat\theta$ | Periodic return units |
| $\mathrm{SSR}$, $\mathrm{SST}$ | Residual and total sums of squares | Squared return units |
| $\hat\sigma^2_{\varepsilon}$ | Classical residual variance $\mathrm{SSR}/(T-p)$ | Squared return units |
| $g_t$ | Score: $x_t\hat\varepsilon_t$ for OLS, $\omega_tx_t\hat\varepsilon_t$ for WLS | Stacked across equations in a joint fit |
| $\hat\Gamma_k$ | Lag-$k$ cross-product of scores | Sum over $t$, not divided by $T$ |
| $q$, $\kappa_k$ | Bartlett lag count and weight $1-k/(q+1)$ | Rows; `hac_lags` capped at $T-1$ |
| $\hat S$ | Bartlett-weighted long-run cross-product of the scores | Positive semidefinite |
| $\hat\Sigma_{\theta}$ | Estimated covariance matrix of $\hat\theta$ | Per-period units |
| $\hat\Theta$, $\hat\Sigma_{\Theta}$, $\hat\Sigma_{\alpha}$ | Stacked coefficients of $J$ equations, their joint covariance, and its $J\times J$ intercept block | Ordered by equation, then intercept and slope |
| $\gamma$, $z_{\gamma}$ | Confidence level and normal quantile $\Phi^{-1}((1+\gamma)/2)$ | `confidence_level`, default 0.95 |
| $\Phi$ | Standard normal distribution function | |
| $\omega_t$, $\Omega$ | Objective weight $\lambda^{T-1-t}$ of row $t=0,\ldots,T-1$; $\Omega=\operatorname{diag}(\omega_t)$ | Latest row has weight one |
| $T_{\mathrm{eff}}$ | Kish effective sample size of the weights | Rows |
| $J$, $c$ | Number of equations fitted jointly; contrast vector in $\mathbb{R}^{J}$ | |
| $K$, $f$ | Number of factors; factor index | |
| $A$ | Weighted cross-product $X^{\top}\Omega X$ | Bread of the WLS sandwich is $A^{-1}$ |
| $G_t$ | Moving sum of $q+1$ consecutive scores | Used in the positive-semidefiniteness proof |
| $M_t$, $C_t$, $B_t$ | EWM factor second moment ($K\times K$), cross moment ($K\times J$), loadings $M_t^{-1}C_t$ | Point in time at $t$ |
| $\mathcal{E}_t[z]$ | EWM recursion $\lambda\mathcal{E}_{t-1}[z]+(1-\lambda)z_t$, seeded at the first row | Seed set by `InitType` |
| $m^{y}_t$ | EWM mean $\mathcal{E}_t[y]$ seeded at $y_0$ | Point in time |
| $\eta_t$ | First-stage residual $y_t-\hat\beta_tx_t$ of the one-factor EWM fit | Periodic return units |
| $a_t$, $h$ | Linear-model alpha $y_t-\sum_fB_{f,t-h}x_{f,t}$; loading lag | $h=1$ point in time, $h=0$ in sample |
| $\phi$ | AR(1) coefficient of the residuals in the examples | Dimensionless |

The estimators assume that rows are consecutive observations in time order, that $X$ has full
column rank, and that the regressors are exogenous, $\mathbb{E}[x_t\varepsilon_t]=0$. The HAC
covariance also assumes weakly dependent scores whose autocovariances beyond lag $q$ are small.
OLS and EWMA-WLS drop every row in which any variable is not finite, so two responses with
different missing patterns are fitted on different samples. The EWM recursions instead hold their
state across a missing value (`NanBackfill.FFILL`).

## Methodology

### Ordinary least squares

**Definition (OLS).** For a $T\times p$ design $X$ of full column rank and a response $y$, the
ordinary least-squares estimator minimises the sum of squared residuals,

$$
\hat\theta=\arg\min_{\theta}\,(y-X\theta)^{\top}(y-X\theta).
$$

**Proposition (normal equations).** The minimiser is unique and solves
$X^{\top}X\hat\theta=X^{\top}y$:

$$
\hat\theta=(X^{\top}X)^{-1}X^{\top}y,
\qquad
X^{\top}\hat\varepsilon=0 .
$$

**Proof.** The objective has gradient $-2X^{\top}(y-X\theta)$ and Hessian $2X^{\top}X$, which is
positive definite when $X$ has full column rank. The objective is therefore strictly convex, its
unique stationary point is the global minimum, and setting the gradient to zero gives the normal
equations. The second statement is the same equation written for
$\hat\varepsilon=y-X\hat\theta$. $\square$

**Identity (one regressor with an intercept).** With $X=[\mathbf{1},x]$,

$$
\hat\beta=\frac{\sum_{t=1}^{T}(x_t-\bar x)(y_t-\bar y)}{\sum_{t=1}^{T}(x_t-\bar x)^2},
\qquad
\hat\alpha=\bar y-\hat\beta\,\bar x .
$$

**Proof.** The intercept row of $X^{\top}\hat\varepsilon=0$ states $\sum_t\hat\varepsilon_t=0$,
that is $\bar y=\hat\alpha+\hat\beta\bar x$. Substituting this $\hat\alpha$ into the slope row
$\sum_t x_t\hat\varepsilon_t=0$ gives $\sum_t x_t\big((y_t-\bar y)-\hat\beta(x_t-\bar x)\big)=0$.
Replacing $x_t$ by $x_t-\bar x$ changes nothing, because the bracket sums to zero; solving for
$\hat\beta$ gives the ratio. $\square$

The slope is a sample covariance divided by a sample variance, so the divisor ($T$ or $T-1$)
cancels. With an intercept the residuals have zero mean, which is why an annualised average of
beta-adjusted returns equals an annualised OLS alpha.

**Definition ($R^2$).** With an intercept,
$R^2=1-\mathrm{SSR}/\mathrm{SST}$ with $\mathrm{SSR}=\sum_t\hat\varepsilon_t^2$ and
$\mathrm{SST}=\sum_t(y_t-\bar y)^2$; for one regressor it is the squared sample correlation of $x$
and $y$. Without an intercept, statsmodels, and therefore qis, reports the uncentred
$R^2=1-\mathrm{SSR}/\sum_t y_t^2$. The two are not comparable.

**Definition (classical covariance).** For homoskedastic, serially uncorrelated errors,

$$
\hat\Sigma_{\theta}^{\mathrm{OLS}}=\hat\sigma^2_{\varepsilon}\,(X^{\top}X)^{-1},
\qquad
\hat\sigma^2_{\varepsilon}=\frac{\mathrm{SSR}}{T-p}.
$$

For one regressor this gives
$\mathrm{se}(\hat\beta)=\hat\sigma_{\varepsilon}/\sqrt{\sum_t(x_t-\bar x)^2}$ and
$\mathrm{se}(\hat\alpha)=\hat\sigma_{\varepsilon}\sqrt{1/T+\bar x^2/\sum_t(x_t-\bar x)^2}$.
The classical alpha p-value is two-sided against a Student $t$ distribution with $T-p$ degrees of
freedom.

#### The contract of `estimate_ols_alpha_beta`

The internal helper `qis.utils.regression.estimate_ols_alpha_beta(x, y, order=1,
fit_intercept=True)` feeds the `Alpha`, `An Alpha`, `Beta`, `R2` and `p-Alpha` columns of
`qis.compute_ra_perf_table_with_benchmark`. It returns the tuple (alpha, beta, $R^2$, classical
alpha p-value) and behaves as follows.

- Rows where $x$ or $y$ is not finite are dropped jointly before the fit.
- `order` 2, 3 or 4 adds powers of $x$ as extra regressors; beta is then the linear coefficient.
- `fit_intercept=False` fits through the origin, returns alpha 0.0 and alpha p-value 0.0, and the
  uncentred $R^2$. The zero p-value is a placeholder, not a significance statement.
- When the fit raises inside the helper (non-numeric input, pandas indexes that differ, or no
  finite row), it emits a `UserWarning` and returns `(0.0, 0.0, 0.0, 0.0)`: zeros, not NaN.
- A constant regressor, including a single observation, raises `IndexError` instead.
  statsmodels' `add_constant` does not add an intercept to a column that is already constant, the
  fit has one coefficient, and the slope is read outside the protected block.
- Two observations fit exactly: $R^2=1$ and the p-value is NaN.

`qis.compute_ra_perf_table_with_benchmark` writes NaN itself when fewer than two joint rows exist,
so the zero fallback reaches a table only through the other failure modes.

### Linearity in the response

**Proposition (linearity).** Fix a design $X$ of full column rank. The maps
$y\mapsto\hat\theta(y)=(X^{\top}X)^{-1}X^{\top}y$ and
$y\mapsto\hat\varepsilon(y)=\big(I_T-X(X^{\top}X)^{-1}X^{\top}\big)y$ are linear: for responses
$y_1,\ldots,y_J$ and constants $c_1,\ldots,c_J$,

$$
\hat\theta\Big(\sum_{j}c_jy_j\Big)=\sum_{j}c_j\,\hat\theta(y_j),
\qquad
\hat\varepsilon\Big(\sum_{j}c_jy_j\Big)=\sum_{j}c_j\,\hat\varepsilon(y_j).
$$

The same holds for weighted least squares with fixed weights, and for the EWM loadings
$B_t=M_t^{-1}C_t$, whose cross moment $C_t$ is linear in $y$ while $M_t$ depends on the regressors
only.

**Proof.** Each map multiplies $y$ by a matrix that depends only on $X$ (and on the weights).
Matrix multiplication distributes over linear combinations. $\square$

Model-layer attribution rests on this proposition: the integration return
$r_F-r_R-r_S$ has OLS alpha $\hat\alpha_F-\hat\alpha_R-\hat\alpha_S$, and its residual is the same
combination of residuals. Three conditions matter in practice.

1. **One design.** The regressor values and the retained rows must be identical. Because each call
   drops its own non-finite rows, responses with different missing patterns are fitted on
   different designs and the identity fails. Trim to a common sample first.
2. **Only first-order objects are linear.** Coefficients, fitted values, residuals and scores are
   linear; $R^2$, standard errors, t-statistics and p-values are not. The standard error of a sum
   of alphas is not the sum of their standard errors.
3. **EWM paths are linear for one missing pattern.** The seeds `InitType.X0`, `ZERO` and `MEAN`
   are all linear in the data, so EWM betas and alphas of a sum are sums, provided the series
   share their missing values.

> **Insight.** The HAC covariance is a quadratic form in the scores, and the scores are linear in
> the response. The standard error of a fixed linear combination of alphas is therefore obtained
> by fitting the combined return series as its own equation. No covariance bookkeeping is needed,
> provided every series shares one design.

### HAC covariance of the OLS coefficients

**Identity (estimation error).** If $y=X\theta+\varepsilon$, then

$$
\hat\theta-\theta=(X^{\top}X)^{-1}\sum_{t=1}^{T}x_t\varepsilon_t .
$$

**Proof.** Substitute $y$ into $\hat\theta=(X^{\top}X)^{-1}X^{\top}y$ and write
$X^{\top}\varepsilon=\sum_t x_t\varepsilon_t$. $\square$

Conditional on $X$, the covariance of $\hat\theta$ is the sandwich
$(X^{\top}X)^{-1}\operatorname{Var}\big(\sum_t x_t\varepsilon_t\big)(X^{\top}X)^{-1}$. The middle
term contains every autocovariance of the score $x_t\varepsilon_t$. The classical formula keeps
only $\sigma^2X^{\top}X$, which is correct for homoskedastic, uncorrelated errors and wrong for
overlapping, smoothed or volatility-clustered returns.

**Definition (Bartlett HAC covariance, as implemented).** With scores $g_t=x_t\hat\varepsilon_t$
and $q$ equal to `hac_lags` capped at $T-1$,

$$
\begin{aligned}
\hat\Gamma_k&=\sum_{t=k+1}^{T}g_t\,g_{t-k}^{\top},
\qquad \kappa_k=1-\frac{k}{q+1},\\
\hat S&=\hat\Gamma_0+\sum_{k=1}^{q}\kappa_k\big(\hat\Gamma_k+\hat\Gamma_k^{\top}\big),\\
\hat\Sigma_{\theta}&=\frac{T}{T-p}\,(X^{\top}X)^{-1}\hat S\,(X^{\top}X)^{-1}.
\end{aligned}
$$

The factor $T/(T-p)$ is statsmodels' `use_correction=True`, with $p$ the number of columns of the
design: $p=2$ in `estimate_ols_alpha_beta_hac` with one regressor, $p=1$ in `estimate_hac_mean`,
and $p=2$ in `estimate_ewma_alpha_beta_hac`, whatever the number of equations. The helpers call
`get_robustcov_results(cov_type='HAC', maxlags=q, use_correction=True, use_t=False)` or reproduce
it in numpy. Inference uses the standard normal as reference distribution:

$$
\mathrm{se}(\hat\alpha)=\sqrt{(\hat\Sigma_{\theta})_{11}},
\qquad
\text{p-value}=2\big(1-\Phi(\lvert\hat\alpha\rvert/\mathrm{se}(\hat\alpha))\big),
\qquad
\hat\alpha\pm z_{\gamma}\,\mathrm{se}(\hat\alpha).
$$

**Proposition (positive semidefiniteness).** For every score sequence and every $q\ge 0$, the
Bartlett estimator $\hat S$ is positive semidefinite, and so is $\hat\Sigma_{\theta}$.

**Proof.** Set $g_t=0$ outside $1\le t\le T$ and form the moving sums
$G_t=\sum_{j=0}^{q}g_{t-j}$ for $t=1,\ldots,T+q$. Two scores $k\le q$ rows apart appear together
in exactly $q+1-k$ of these sums, so

$$
\sum_{t=1}^{T+q}G_tG_t^{\top}=(q+1)\,\hat S .
$$

A sum of outer products is positive semidefinite, and so is its congruence transform by
$(X^{\top}X)^{-1}$. $\square$

[Newey and West (1987)](https://www.nber.org/papers/t0055) chose the Bartlett weights to guarantee
this property. Equal weights do not: a zigzag score $1,-1,1,-1,1,-1$ has $\hat\Gamma_0=6$ and
$\hat\Gamma_1=-5$, so one lag with unit weight gives $6-10=-4$, a negative variance, whereas the
Bartlett weight $\kappa_1=1/2$ gives $6-5=1$.

**Special cases.** With $q=0$ the estimator is the heteroskedasticity-consistent covariance of
[White (1980)](https://doi.org/10.2307/1912934), $\hat S=\sum_t\hat\varepsilon_t^2x_tx_t^{\top}$;
with the $T/(T-p)$ factor it is the version often labelled HC1. The internal
`qis.utils.regression.estimate_hac_mean(y, hac_lags=3, confidence_level=0.95)` regresses $y$ on a
constant, so $p=1$, $g_t=y_t-\bar y$ and $\operatorname{se}(\bar y)^2=\hat S/(T(T-1))$.

**Identity (HAC mean with no lags).** With $q=0$, `estimate_hac_mean` returns the classical
standard error of the mean, $s(y)/\sqrt{T}$.

**Proof.** With $q=0$, $\hat S=\sum_t(y_t-\bar y)^2=(T-1)\,s(y)^2$, and the sandwich with
$X^{\top}X=T$ and the factor $T/(T-1)$ gives $s(y)^2/T$. $\square$

#### Choosing the lag count

qis defaults to `hac_lags=3` for every sampling grid. Three monthly lags cover a quarter; three
daily lags cover three days. A fixed lag count is not consistent under general dependence:
consistency requires $q\to\infty$ with $q/T\to 0$ (Newey and West, 1987), and
[Andrews (1991)](https://doi.org/10.2307/2938229) shows that the mean-squared-error optimal
Bartlett bandwidth grows like $T^{1/3}$ and gives data-dependent rules for it. For callers who
prefer a rule, the internal `qis.utils.regression.newey_west_lag_rule(nobs)` returns the
rule of thumb associated with [Newey and West (1994)](https://doi.org/10.2307/2297912),

$$
q_{\mathrm{NW}}=\Big\lfloor 4\Big(\frac{T}{100}\Big)^{2/9}\Big\rfloor ,
$$

and raises `ValueError` for $T<1$. It gives $q_{\mathrm{NW}}=3$ at $T=60$, 4 at $T=120$ and
$T=240$, 5 at $T=520$ and 8 at $T=2520$. No qis estimator calls it by default; pass its value as
`hac_lags`.

> **Pitfall.** Few lags understate the standard error of persistent residuals even with unlimited
> data. For AR(1) errors with coefficient $\phi$ and a regressor independent of them, the long-run
> variance of the intercept score is $(1+\phi)/(1-\phi)$ times its variance, while the Bartlett
> estimator with $q$ lags targets $1+2\sum_{k=1}^{q}\kappa_k\phi^k$. At $\phi=0.6$ the first is 4
> and the second, for $q=3$, is 2.37: when the regressor's mean is small relative to its
> volatility, the alpha standard error is too small by a factor of about 1.30.

### Weighted least squares with geometric weights

**Definition (EWMA-WLS, as implemented).** For retained rows $t=0,\ldots,T-1$, oldest first,
`qis.estimate_ewma_alpha_beta_hac` minimises
$\sum_t\omega_t\big(y_t-\alpha-\beta x_t\big)^2$ with

$$
\omega_t=\lambda^{T-1-t},\qquad \lambda=1-\frac{2}{N+1},
$$

so the latest row has weight one. All equations share the regressor, the weights and the common
finite sample.

**Proposition (WLS estimator).** With $\Omega=\operatorname{diag}(\omega_0,\ldots,\omega_{T-1})$,

$$
\hat\theta=(X^{\top}\Omega X)^{-1}X^{\top}\Omega\,y .
$$

**Proof.** With $\tilde y=\Omega^{1/2}y$ and $\tilde X=\Omega^{1/2}X$ the objective is
$(\tilde y-\tilde X\theta)^{\top}(\tilde y-\tilde X\theta)$, and the normal equations of OLS give
$\hat\theta=(\tilde X^{\top}\tilde X)^{-1}\tilde X^{\top}\tilde y$. $\square$

The reported $R^2$ is the weighted, centred one,
$1-\sum_t\omega_t\hat\varepsilon_t^2/\sum_t\omega_t(y_t-\bar y_{\omega})^2$ with
$\bar y_{\omega}=\sum_t\omega_ty_t/\sum_t\omega_t$. Multiplying every weight by a constant changes
neither $\hat\theta$ nor the HAC covariance below, because the bread scales by the inverse of the
constant and the meat by its square. The latest-weight-one normalisation is a labelling choice.

**Definition (Kish effective sample size).** Following Kish (1965), the
effective sample size of weights $\omega_t$ is

$$
T_{\mathrm{eff}}=\frac{\big(\sum_t\omega_t\big)^2}{\sum_t\omega_t^2}.
$$

**Proposition (closed form).** For $\omega_t=\lambda^{T-1-t}$ with $\lambda=1-2/(N+1)$,

$$
T_{\mathrm{eff}}=\frac{1+\lambda}{1-\lambda}\cdot\frac{1-\lambda^{T}}{1+\lambda^{T}}
=N\,\frac{1-\lambda^{T}}{1+\lambda^{T}} .
$$

**Proof.** The geometric sums are $\sum_t\omega_t=(1-\lambda^T)/(1-\lambda)$ and
$\sum_t\omega_t^2=(1-\lambda^{2T})/(1-\lambda^2)$. Divide the square of the first by the second
and use $1-\lambda^{2T}=(1-\lambda^T)(1+\lambda^T)$ and $1-\lambda^2=(1-\lambda)(1+\lambda)$.
Finally $\lambda=(N-1)/(N+1)$ gives $(1+\lambda)/(1-\lambda)=N$. $\square$

$T_{\mathrm{eff}}$ is below $N$ for every finite $T$ and approaches it geometrically: $T=239$ and
$N=36$ give 35.9999. The estimator reports it as `effective_nobs` and requires it to exceed two;
it also raises `ValueError` with fewer than three common finite rows, a regressor that does not
vary, or a response that is constant.
It is not the degrees of freedom of the small-sample factor, which is $T/(T-2)$ with the raw row
count, as in a statsmodels WLS fit.

**Definition (stacked-score joint HAC).** For $J$ equations sharing $X$ and $\Omega$, stack
$\hat\Theta=(\hat\theta_1^{\top},\ldots,\hat\theta_J^{\top})^{\top}$ and the scores
$g_t=(\hat\varepsilon_{1,t},\ldots,\hat\varepsilon_{J,t})^{\top}\otimes\omega_tx_t$, a vector of
length $2J$ ordered by equation. With $A=X^{\top}\Omega X$ and $\hat S$ the Bartlett estimator of
the previous section applied to the stacked $g_t$,

$$
\hat\Sigma_{\Theta}=\frac{T}{T-2}\,(I_J\otimes A^{-1})\,\hat S\,(I_J\otimes A^{-1}),
$$

symmetrised by averaging with its transpose. The result's `parameter_covariance` holds
$\hat\Sigma_{\Theta}$ with `(equation, parameter)` labels and `Intercept`, `Beta` as parameters.

**Proposition (variance of a linear contrast).** For $c\in\mathbb{R}^{J}$, the combined response
$\sum_jc_jy_j$ has intercept $\sum_jc_j\hat\alpha_j$ and

$$
\widehat{\operatorname{Var}}\Big(\sum_{j}c_j\hat\alpha_j\Big)=c^{\top}\hat\Sigma_{\alpha}\,c
=\sum_{j}c_j^2\,\widehat{\operatorname{Var}}(\hat\alpha_j)
+2\sum_{j<l}c_jc_l\,\widehat{\operatorname{Cov}}(\hat\alpha_j,\hat\alpha_l),
$$

where $\hat\Sigma_{\alpha}$ is the $J\times J$ intercept block of $\hat\Sigma_{\Theta}$. It equals
the variance obtained by fitting $\sum_jc_jy_j$ as its own equation.

**Proof.** By linearity the combined equation has scores $(c^{\top}\otimes I_2)\,g_t$. Every
$\hat\Gamma_k$ is bilinear in the scores, so the combined $\hat S$ is
$(c^{\top}\otimes I_2)\hat S(c\otimes I_2)$, and the bread $A^{-1}$ and the factor $T/(T-2)$ are
unchanged. The intercept element of the result is the stated quadratic form. $\square$

The sum of marginal variances is the right answer only when the cross covariances vanish. Layers
and variants of one strategy share most of their residual variation, so their alpha estimates are
strongly positively correlated and a difference of alphas is far more precise than the sum
suggests. The integration alpha of model-layer attribution is the contrast $c=(1,-1,-1)$ of the
full, risk and signal equations. With $\Omega=I_T$ the same argument covers full-sample OLS/HAC.

### Point-in-time EWM regressions

**Definition (EWM moment regression, as implemented).** For factor rows $x_t\in\mathbb{R}^{K}$ and
response rows $y_t\in\mathbb{R}^{J}$, `qis.compute_ewm_xy_beta_tensor` runs from zero seeds

$$
\begin{aligned}
M_t&=\lambda M_{t-1}+(1-\lambda)\,x_tx_t^{\top},\\
C_t&=\lambda C_{t-1}+(1-\lambda)\,x_ty_t^{\top},\qquad M_{-1}=C_{-1}=0,\\
B_t&=M_t^{-1}C_t,
\end{aligned}
$$

and returns $B_t$ as a $(T,K,J)$ array. The moments are about zero: there is no intercept and no
demeaning unless the caller demeans first. `span` overrides `ewm_lambda` through
$\lambda=1-2/(N+1)$.

**Proposition (prefix least squares through the origin).** Without missing values,

$$
B_t=\arg\min_{B}\sum_{s=0}^{t}\lambda^{t-s}\,
\big(y_s-B^{\top}x_s\big)^{\top}\big(y_s-B^{\top}x_s\big).
$$

**Proof.** Unrolling the recursions from zero seeds gives
$M_t=(1-\lambda)\sum_{s\le t}\lambda^{t-s}x_sx_s^{\top}$ and
$C_t=(1-\lambda)\sum_{s\le t}\lambda^{t-s}x_sy_s^{\top}$. The factor $1-\lambda$ cancels in
$M_t^{-1}C_t$, and $M_tB=C_t$ are the normal equations of the weighted problem, one response
column at a time. $\square$

An EWM beta is therefore point in time, since no later row enters $B_t$, but contemporaneous,
since row $t$ does. To decompose return $t$ ex ante, apply $B_{t-1}$ to $x_t$;
`LinearModel.get_factor_alpha(lag=1)` does exactly that.

**Identity (diagonal versus full inversion).** With $B^{\mathrm{full}}_t=M_t^{-1}C_t$ and
$B^{\mathrm{diag}}_t=\operatorname{diag}(M_t)^{-1}C_t$, the row of factor $f$ satisfies

$$
B^{\mathrm{diag}}_{t,f}=B^{\mathrm{full}}_{t,f}
+\sum_{f'\ne f}\frac{(M_t)_{ff'}}{(M_t)_{ff}}\,B^{\mathrm{full}}_{t,f'}.
$$

**Proof.** $C_t=M_tB^{\mathrm{full}}_t$; divide row $f$ by $(M_t)_{ff}$. $\square$

The diagonal version runs $K$ separate one-factor regressions and absorbs the exposure to
correlated factors, the omitted-variable term above. It equals the full version only when the
factors are orthogonal in the EWM second moment. `is_x_correlated=True` (the default of both
`compute_ewm_xy_beta_tensor` and `EwmLinearModel.fit`) inverts $M_t$ in full; `False` uses its
diagonal. A one-dimensional $x$ always takes the scalar path.

The implementation adds four rules.

- **Warm-up.** $B_t$ is NaN for $t\le$ `warmup_period` (default 20): the first 21 rows are
  missing.
- **Singular and small moments.** If inverting $M_t$ fails, the diagonal inverse is used. If the
  smallest diagonal element of $M_t$ is at most $10^{-8}$, an absolute threshold in squared return
  units, the inverse is replaced by the identity and $B_t=C_t$, a cross moment rather than a beta.
- **Missing values.** With `NanBackfill.FFILL` a missing factor row holds both $M_t$ and $C_t$. A
  missing response holds its column of $C_t$ while $M_t$ keeps updating.
- **One factor.** `qis.compute_one_factor_ewm_betas(x, y, span=None, ewm_lambda=0.94)` returns the
  $K=1$ slice as a frame, requires identical indexes and uses the fixed warm-up of 20 rows. Without
  a span the decay 0.94 applies, the daily RiskMetrics value (J.P. Morgan and Reuters, 1996),
  which corresponds to a span of about 32.

`qis.EwmLinearModel.fit(span=31, ewm_lambda=0.94, is_x_correlated=True,
mean_adj_type=MeanAdjType.NONE, init_type=InitType.MEAN, warmup_period=20)` requires identical
factor and asset indexes and stores one $(T\times J)$ loadings frame per factor. With
`mean_adj_type` other than `NONE` it first subtracts a mean from both panels and overwrites the
model's `x` and `y` with the demeaned panels, so later calls to `get_factor_alpha` and
`get_model_ewm_r2` see demeaned returns. `MeanAdjType.INSAMPLE` subtracts the full-sample mean,
`EXPANDING` the expanding mean, and `EWMA` the running EWM mean $m_s$ at each row $s$, seeded by
`init_type`. The `EWMA` result is point in time only with a point-in-time seed, and it is not a
prefix regression with an intercept, because each row is centred on its own running mean. `qis.estimate_ewm_factor_model` fits the same model
on `W-WED` log returns with span 26.

#### One-factor alpha, prediction and $R^2$

`qis.compute_ewm_beta_alpha_forecast(x_data, y_data, span=None, ewm_lambda=0.94,
mean_adj_type=MeanAdjType.NONE, init_type=InitType.MEAN, beta_init_value=None, annualize=False)`
regresses each asset on one factor (a Series broadcast to every asset, or paired columns); when
the indexes differ, the assets are reindexed to the factor index with a forward fill. Let
$\mathcal{E}_t$ be the qis recursion: its value at the first row is the seed, which replaces the
first observation, and later rows follow
$\mathcal{E}_t[z]=\lambda\mathcal{E}_{t-1}[z]+(1-\lambda)z_t$. With
$\operatorname{clip}_{[0,1]}$ truncating to the unit interval, per asset

$$
\begin{aligned}
\hat\beta_t&=\frac{\mathcal{E}_t[xy]}{\mathcal{E}_t[x^2]},
\qquad \eta_t=y_t-\hat\beta_tx_t,
\qquad \hat\alpha_t=\mathcal{E}_t[\eta],\\
\hat y_t&=\hat\beta_tx_t+\hat\alpha_t,
\qquad \hat\sigma^2_{\varepsilon,t}=\mathcal{E}_t\big[(y-\hat y)^2\big],\\
R^2_t&=\operatorname{clip}_{[0,1]}\Big(1-
\frac{\hat\sigma^2_{\varepsilon,t}}{\mathcal{E}_t\big[(y-m^{y})^2\big]}\Big).
\end{aligned}
$$

It returns $(\hat\beta,\hat\alpha,\hat y,\mathcal{E}[x^2],\hat\sigma^2_{\varepsilon},R^2)$. The
$R^2$ is centred: the residual is taken about the EWM alpha and $y$ about its EWM mean $m^y$,
which is seeded at $y_0$, with the denominator seeded at zero. The first four recursions, and
the optional demeaning under `mean_adj_type`, are seeded by `init_type`. `annualize=True`
multiplies $\mathcal{E}[x^2]$, $\hat\sigma^2_{\varepsilon}$ and the $R^2$ denominator by
$\mathrm{AN}$ inferred from the index, leaving $R^2$ unchanged. Beta is NaN where
$\mathcal{E}_t[x^2]$ is within $10^{-8}$ of zero. `beta_init_value` replaces the first finite,
non-zero factor observation by a one-observation prior, so the first finite beta equals the
prior. The two moment recursions always carry a missing value forward (`NanBackfill.FFILL`); the
`nan_backfill` argument reaches only the demeaning, alpha, residual and $R^2$ recursions.

Two properties of the defaults matter:

- **The default seed looks ahead.** With `InitType.MEAN`, $\mathcal{E}_0[xy]$ and
  $\mathcal{E}_0[x^2]$ are full-sample means, so $\hat\beta_0=\sum_tx_ty_t/\sum_tx_t^2$ is the
  full-sample slope through the origin. Its weight decays like $\lambda^t$ but is present at
  every date, and the residual-mean and variance seeds are full-sample means too. `InitType.X0` is point in
  time and gives $\hat\beta_0=y_0/x_0$.
- **The "forecast" is a fitted value.** $\hat\beta_t$ and $\hat\alpha_t$ both use row $t$. An
  ex-ante prediction of $y_t$ is $\hat\beta_{t-1}x_t+\hat\alpha_{t-1}$, which the caller forms by
  lagging the outputs.

#### Alpha and $R^2$ given a prediction

`qis.compute_ewm_alpha_r2_given_prediction(y_data, y_prediction, span=None, ewm_lambda=0.94)`
returns the EWM alpha and centred $R^2$ of an arbitrary prediction $\hat y$:

$$
\hat\alpha_t=\mathcal{E}_t[y-\hat y],
\qquad
R^2_t=\operatorname{clip}_{[0,1]}\Big(1-
\frac{\mathcal{E}_t\big[(y-\hat y-\hat\alpha)^2\big]}{\mathcal{E}_t\big[(y-m^{y})^2\big]}\Big),
$$

with the alpha seeded at its first-row value (`InitType.X0`, zero when that row is missing) and
both variances seeded at zero.
It is point in time whenever $\hat y$ is, for example the lag-one explained return of a
`LinearModel`.

#### Linear-model alpha and uncentred $R^2$

`qis.LinearModel.get_factor_alpha(x=None, y=None, lag=1, span=None)` returns the pair (alpha,
explained return) with

$$
a_t=y_t-\sum_{f=1}^{K}B_{f,t-h}\,x_{f,t},\qquad h\in\{0,1\},
$$

where the loadings are forward-filled onto the factor index and shifted by `lag` $=h$. Lag one
is point in time; lag zero is in sample. There is no intercept: $a_t$ contains whatever mean of
$y$ the factors do not explain. With `span`, the alpha is smoothed to $\mathcal{E}_t[a]$.

`qis.LinearModel.get_model_ewm_r2(span=52, lag=0)` computes

$$
R^2_t=\operatorname{clip}_{[0,1]}\Big(1-\frac{\mathcal{E}_t[a^2]}{\mathcal{E}_t[y^2]}\Big),
$$

an uncentred ratio (neither $a$ nor $y$ is demeaned) evaluated by default with lag-zero, in-sample
loadings. After the warm-up the numerator starts from a zero seed while the denominator has run
since the first row, so the first values after the warm-up are close to one by construction.

> **Pitfall.** Three defaults leak or overstate. `InitType.MEAN`, the default seed of
> `compute_ewm_beta_alpha_forecast` and of the demeaning step in `EwmLinearModel.fit`, uses
> full-sample means. The prediction of `compute_ewm_beta_alpha_forecast` uses the same row it
> predicts. `get_model_ewm_r2` defaults to in-sample loadings. For backtests, pass
> `init_type=InitType.X0`, lag betas and alphas by one period, and use `lag=1`.

> **Insight.** A point-in-time EWM beta is an endpoint EWMA-WLS slope without the intercept. The
> last row of `compute_one_factor_ewm_betas` equals the slope of
> $\sum_t\omega_t(y_t-\beta x_t)^2$ with the weights reported by `estimate_ewma_alpha_beta_hac`.
> The two estimators differ only in whether the mean is modelled by an intercept.

### Alpha annualisation

Alpha is estimated per period. qis annualises it in two ways, depending on the output.

| Output | Annualised alpha | Where it is used |
|---|---|---|
| `PerfStat.ALPHA_AN` (`An Alpha`) | $\mathrm{AN}\,\hat\alpha$, $\mathrm{AN}$ of `PerfParams.freq_reg` (default `QE`, 4) | `qis.compute_ra_perf_table_with_benchmark` |
| Model-layer tables, bars and intervals | $\mathrm{AN}\,\hat\alpha$ and $\mathrm{AN}(\hat\alpha\pm z_{\gamma}\,\mathrm{se})$ | `qis.compute_model_layer_alpha_beta_attribution` and its EWMA variants |
| Scatter-plot legends with `alpha_an_factor` | $e^{\mathrm{AN}\hat\alpha}-1$, formatted `'{:+0.0%}'` | internal `reg_model_params_to_str`, through `qis.plot_scatter` and `qis.plot_returns_scatter` keyword arguments |
| Scatter-plot legends by default | raw $\hat\alpha$, formatted `'{0:+0.2f}'` | the same, and the label of `qis.fit_multivariate_ols` |

The linear form keeps alphas additive, which the attribution identities need. The compound form is
the annual growth of a constant periodic log alpha: exact when the regression is on log returns,
an approximation on simple returns. Since $e^{a}-1\ge a$ with a gap of about $a^2/2$, the legend
exceeds the table: a monthly alpha of 1.3% is 15.6% in a table and 16.9% (displayed `+17%`) in a
legend. No qis report passes `alpha_an_factor`; by default a legend shows the periodic alpha to two
decimals, so a monthly alpha of 0.013 reads `+0.01`.

## Worked example

The examples run in order and share one namespace. The first block uses five hand-checkable
observations; the second a fixed-seed numpy sample with serially correlated residuals; the rest
the frozen synthetic universe at month-end, 239 log returns from February 2006 to December 2025.

### Five observations by hand

Benchmark returns 1%, 2%, 3%, 4%, 5% and fund returns 2%, 3%, 5%, 4%, 6% have means 3% and 4%.
The centred cross-product is $9\times10^{-4}$ and the centred benchmark sum of squares
$10\times10^{-4}$, so $\hat\beta=0.9$ and $\hat\alpha=0.04-0.9\times0.03=0.013$. The residuals are
$-0.2\%$, $-0.1\%$, $1.0\%$, $-0.9\%$ and $0.2\%$, so $\mathrm{SSR}=1.9\times10^{-4}$,
$\mathrm{SST}=10\times10^{-4}$ and $R^2=0.81$. With $\hat\sigma^2_{\varepsilon}=1.9\times10^{-4}/3$,
$\mathrm{se}(\hat\alpha)=0.00835$ and the classical two-sided p-value on three degrees of freedom
is 0.217. A second fund shows linearity, and the last lines compare the two alpha annualisations.

```python
import numpy as np
import pandas as pd
from scipy import stats
import qis
from qis.utils.regression import estimate_ols_alpha_beta, fit_ols, reg_model_params_to_str

dates = pd.date_range('2024-01-31', periods=5, freq='ME')
x = pd.DataFrame({'bench': [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)
y = pd.Series([0.02, 0.03, 0.05, 0.04, 0.06], index=dates, name='fund')

# hand arithmetic from centred values
x_c = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * 0.01
y_c = np.array([-2.0, -1.0, 1.0, 0.0, 2.0]) * 0.01
beta_hand = (x_c @ y_c) / (x_c @ x_c)
alpha_hand = 0.04 - beta_hand * 0.03
resid_hand = y.to_numpy() - alpha_hand - beta_hand * x['bench'].to_numpy()
ssr, sst = resid_hand @ resid_hand, y_c @ y_c
se_alpha = np.sqrt(ssr / 3 * (1 / 5 + 0.03 ** 2 / (x_c @ x_c)))
np.testing.assert_allclose([beta_hand, alpha_hand, 1 - ssr / sst], [0.9, 0.013, 0.81])
np.testing.assert_allclose(resid_hand, [-0.002, -0.001, 0.010, -0.009, 0.002], atol=1e-15)

prediction, params, label = qis.fit_multivariate_ols(x=x, y=y, verbose=False)
np.testing.assert_allclose(params[['intercept', 'bench']], [alpha_hand, beta_hand])
np.testing.assert_allclose(y - prediction, resid_hand, atol=1e-15)
assert label == 'y=+0.01+0.90*bench, R²=81%'

alpha, beta, r2, alpha_pvalue = estimate_ols_alpha_beta(x=x['bench'], y=y)
np.testing.assert_allclose([alpha, beta, r2], [0.013, 0.9, 0.81])
np.testing.assert_allclose(alpha_pvalue, 2 * stats.t.sf(alpha_hand / se_alpha, df=3))
assert round(se_alpha, 5) == 0.00835 and round(alpha_pvalue, 3) == 0.217

# linearity: the fit of a difference is the difference of the fits
y_other = pd.Series([0.01, 0.00, 0.02, 0.03, 0.01], index=dates, name='other')
pred_other, params_other, _ = qis.fit_multivariate_ols(x=x, y=y_other, verbose=False)
pred_diff, params_diff, _ = qis.fit_multivariate_ols(x=x, y=(y - y_other).rename('diff'),
                                                     verbose=False)
np.testing.assert_allclose(params_diff, params - params_other, atol=1e-14)
np.testing.assert_allclose((y - y_other) - pred_diff,
                           (y - prediction) - (y_other - pred_other), atol=1e-15)

# annualisation: linear in tables, compounded in legends
model = fit_ols(x=x['bench'], y=y)
table_alpha, legend_alpha = 12 * model.params[0], np.expm1(12 * model.params[0])
np.testing.assert_allclose([table_alpha, legend_alpha], [0.156, np.exp(0.156) - 1.0])
assert round(legend_alpha, 3) == 0.169
assert reg_model_params_to_str(reg_model=model, order=1) == 'y=+0.90X+0.01, R²=81%'
assert reg_model_params_to_str(reg_model=model, order=1,
                               alpha_an_factor=12) == 'y=+0.90X+17%, R²=81%'
```

### HAC standard errors with serially correlated residuals

A fixed-seed sample of $T=120$ monthly observations has benchmark returns with mean 0.6% and
volatility 4.5%, and a fund with true alpha 0.2% and beta 0.8 whose errors follow an AR(1) with
$\phi=0.6$. The fit gives $\hat\alpha=0.0046$ and $\hat\beta=0.829$, with lag-one residual
autocorrelation 0.47. The classical standard error of alpha is 0.00113; White's ($q=0$) is 0.00114;
the default $q=3$ gives 0.00163, 1.44 times the classical value; the Newey–West rule for $T=120$,
$q=4$, gives 0.00169. An independent numpy sandwich reproduces every value, the Bartlett
cross-product equals the moving-sum form of the proof, and the zigzag series checks the
positive-semidefinite example.

```python
from scipy.stats import norm
from qis.utils.regression import (estimate_hac_mean, estimate_ols_alpha_beta_hac,
                                  newey_west_lag_rule)

rng = np.random.default_rng(20260725)
T, phi = 120, 0.6
bench = 0.006 + 0.045 * rng.standard_normal(T)
innovations = 0.01 * rng.standard_normal(T)
errors = np.zeros(T)
for t in range(T):
    errors[t] = (phi * errors[t - 1] if t > 0 else 0.0) + innovations[t]
fund = 0.002 + 0.8 * bench + errors


def bartlett_sandwich(design, response, lags):
    """Independent OLS with a Bartlett HAC covariance and the T/(T-p) correction."""
    bread = np.linalg.inv(design.T @ design)
    coef = bread @ design.T @ response
    scores = design * (response - design @ coef)[:, None]
    meat = scores.T @ scores
    for k in range(1, lags + 1):
        gamma_k = scores[k:].T @ scores[:-k]
        meat = meat + (1.0 - k / (lags + 1.0)) * (gamma_k + gamma_k.T)
    n, p = design.shape
    return coef, n / (n - p) * bread @ meat @ bread, meat, scores


design = np.column_stack([np.ones(T), bench])
coef = np.linalg.solve(design.T @ design, design.T @ fund)
resid = fund - design @ coef
classical_se = np.sqrt(resid @ resid / (T - 2) * np.linalg.inv(design.T @ design)[0, 0])
hac_se, hac_pvalue = {}, {}
for q in (0, 3, 4):
    coef_q, cov_q, meat_q, scores = bartlett_sandwich(design, fund, q)
    result = estimate_ols_alpha_beta_hac(x=bench, y=fund, hac_lags=q)
    np.testing.assert_allclose([result.alpha, result.beta], coef_q, rtol=1e-10)
    np.testing.assert_allclose(result.alpha_hac_se, np.sqrt(cov_q[0, 0]), rtol=1e-10)
    z = result.alpha / result.alpha_hac_se
    np.testing.assert_allclose(result.alpha_pvalue, 2 * norm.sf(abs(z)), rtol=1e-10)
    half_width = norm.ppf(0.975) * result.alpha_hac_se
    np.testing.assert_allclose(result.alpha_confidence_interval,
                               (result.alpha - half_width, result.alpha + half_width), rtol=1e-10)
    # moving sums of q + 1 scores: sum of G G' equals (q + 1) S, hence S is PSD
    padded = np.vstack([np.zeros((q, 2)), scores, np.zeros((q, 2))])
    moving = np.array([padded[t:t + q + 1].sum(axis=0) for t in range(T + q)])
    np.testing.assert_allclose(moving.T @ moving / (q + 1), meat_q, rtol=1e-10)
    assert np.linalg.eigvalsh(meat_q).min() > 0.0
    hac_se[q], hac_pvalue[q] = result.alpha_hac_se, result.alpha_pvalue

assert round(coef[0], 4) == 0.0046 and round(coef[1], 3) == 0.829
assert round(np.corrcoef(resid[1:], resid[:-1])[0, 1], 2) == 0.47
np.testing.assert_allclose([classical_se, hac_se[0], hac_se[3], hac_se[4]],
                           [0.00113, 0.00114, 0.00163, 0.00169], atol=5e-6)
assert round(hac_se[3] / classical_se, 2) == 1.44
classical_pvalue = 2 * stats.t.sf(coef[0] / classical_se, df=T - 2)
assert round(hac_pvalue[3], 3) == 0.004 and round(classical_pvalue, 4) == 0.0001
assert round(hac_pvalue[3] / classical_pvalue, -1) == 60

# lag rule and the population target of a three-lag Bartlett estimator
assert newey_west_lag_rule(T) == int(np.floor(4 * (T / 100) ** (2 / 9))) == 4
assert [newey_west_lag_rule(n) for n in (60, 240, 520, 2520)] == [3, 4, 5, 8]
bartlett_target = 1 + 2 * sum((1 - k / 4) * phi ** k for k in (1, 2, 3))
np.testing.assert_allclose([bartlett_target, (1 + phi) / (1 - phi)], [2.368, 4.0])
assert round(np.sqrt(4.0 / bartlett_target), 2) == 1.30

# zigzag: Bartlett S = 6 - 5 = 1, whereas equal weights would give 6 - 10 = -4
zigzag = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
np.testing.assert_allclose(estimate_hac_mean(zigzag, hac_lags=1).hac_se,
                           np.sqrt(6 / 5 * 1.0 / 36))
# with no lags the HAC mean has the classical standard error of the mean
np.testing.assert_allclose(estimate_hac_mean(fund, hac_lags=0).hac_se,
                           np.std(fund, ddof=1) / np.sqrt(T), rtol=1e-12)
```

The alpha HAC p-value at $q=3$ is 0.004 against a classical 0.0001: the conclusion survives, but
the p-value is about 60 times larger.

### EWMA-WLS with a joint contrast

A core sleeve `Core` (the synthetic European equity `SEQ_EU`) and a variant `Tilted` that moves
20% into gold (`SCM_GLD`) are regressed on the synthetic 60/40 benchmark `SBM_6040` with the
default span $N=36$, so $\lambda=35/37$. The difference `Tilted - Core` is fitted as a third
equation. The EWMA-WLS alphas are 0.00250 (Core), 0.00186 (Tilted) and $-0.00064$ (difference),
with HAC standard errors 0.00577, 0.00469 and 0.00131. The two sleeve alphas have estimated
correlation 0.99. Adding their variances as if independent would give a standard error of
0.00743, 5.7 times the correct 0.00131. $T_{\mathrm{eff}}=35.9999$ matches the closed form.

```python
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(start='2006-01-02', end='2025-12-31', seed=20260725,
                                       apply_quirks=False)
prices = pd.concat([universe.prices, universe.benchmark_prices], axis=1)
returns = qis.to_returns(prices=prices, is_log_returns=True, freq='ME', drop_first=True)
bench_m = returns['SBM_6040']
layers = pd.DataFrame({'Core': returns['SEQ_EU'],
                       'Tilted': 0.8 * returns['SEQ_EU'] + 0.2 * returns['SCM_GLD']})
layers['Tilted - Core'] = layers['Tilted'] - layers['Core']
wls = qis.estimate_ewma_alpha_beta_hac(x=bench_m, y=layers, span=36.0, hac_lags=3)

# independent weighted least squares with the square roots of the weights
n_obs, lam = len(bench_m), 1.0 - 2.0 / 37.0
omega = lam ** np.arange(n_obs - 1, -1, -1)
root = np.sqrt(omega)[:, None]
design_m = np.column_stack([np.ones(n_obs), bench_m.to_numpy()])
wls_coef = np.linalg.lstsq(root * design_m, root * layers.to_numpy(), rcond=None)[0]
np.testing.assert_allclose(wls.alpha, wls_coef[0], rtol=1e-8)
np.testing.assert_allclose(wls.beta, wls_coef[1], rtol=1e-8)
np.testing.assert_allclose(wls.weights, omega, rtol=1e-12)
resid_m = layers.to_numpy() - design_m @ wls_coef
y_bar = omega @ layers.to_numpy() / omega.sum()
np.testing.assert_allclose(wls.r_squared, 1 - omega @ resid_m ** 2
                           / (omega @ (layers.to_numpy() - y_bar) ** 2), rtol=1e-10)

# Kish effective size and its closed form
kish = omega.sum() ** 2 / (omega ** 2).sum()
closed_form = 36.0 * (1 - lam ** n_obs) / (1 + lam ** n_obs)
np.testing.assert_allclose([wls.effective_nobs, kish], closed_form, rtol=1e-12)
assert n_obs == 239 and round(wls.effective_nobs, 4) == 35.9999

# variance of the contrast from the joint covariance equals its own equation's variance
cov = wls.parameter_covariance
core, tilted = ('Core', 'Intercept'), ('Tilted', 'Intercept')
contrast_var = cov.loc[tilted, tilted] + cov.loc[core, core] - 2 * cov.loc[tilted, core]
np.testing.assert_allclose(wls.alpha_hac_se['Tilted - Core'], np.sqrt(contrast_var), rtol=1e-8)
naive_se = np.sqrt(cov.loc[tilted, tilted] + cov.loc[core, core])
correlation = cov.loc[tilted, core] / np.sqrt(cov.loc[tilted, tilted] * cov.loc[core, core])
np.testing.assert_allclose(wls.alpha, [0.00250, 0.00186, -0.00064], atol=5e-6)
np.testing.assert_allclose(wls.alpha_hac_se, [0.00577, 0.00469, 0.00131], atol=5e-6)
assert round(naive_se, 5) == 0.00743 and round(correlation, 2) == 0.99
assert round(naive_se / wls.alpha_hac_se['Tilted - Core'], 1) == 5.7
```

### Point-in-time EWM betas

Two factors, US equities `SEQ_US` and Treasuries `SBD_TSY` (full-sample correlation $-0.19$),
explain three assets with `EwmLinearModel.fit(span=36)`. At December 2025 the full-inversion
loadings equal the prefix weighted regression through the origin; the diagonal loadings equal the
one-factor regressions and satisfy the omitted-factor identity. The Treasury loading of `SEQ_EU` is
0.256 with full inversion and 0.317 with the diagonal. The first 21 rows are missing, so the first
loading is dated November 2007. The one-factor EWM beta of `SEQ_EU` on the benchmark is 1.037,
against the EWMA-WLS slope of 1.026 with an intercept and identical weights.

```python
factors = returns[['SEQ_US', 'SBD_TSY']]
assets = returns[['SEQ_EU', 'SBD_HY', 'SAL_HF']]
full = qis.EwmLinearModel(x=factors, y=assets)
full.fit(span=36)
diag = qis.EwmLinearModel(x=factors, y=assets)
diag.fit(span=36, is_x_correlated=False)

x_f, y_a = factors.to_numpy(), assets.to_numpy()
moment = x_f.T @ (omega[:, None] * x_f)        # proportional to M_T
cross = x_f.T @ (omega[:, None] * y_a)          # proportional to C_T
b_full = np.linalg.solve(moment, cross)
b_diag = cross / np.diag(moment)[:, None]
last_date = factors.index[-1]
np.testing.assert_allclose(full.get_loadings_at_date(last_date), b_full, rtol=1e-8)
np.testing.assert_allclose(diag.get_loadings_at_date(last_date), b_diag, rtol=1e-8)
off_diagonal = moment - np.diag(np.diag(moment))
np.testing.assert_allclose(b_diag, b_full + off_diagonal @ b_full / np.diag(moment)[:, None],
                           rtol=1e-10)
assert round(b_full[1, 0], 3) == 0.256 and round(b_diag[1, 0], 3) == 0.317
assert all(full.loadings[f].isna().sum().eq(21).all() for f in factors.columns)
assert full.loadings['SEQ_US'].first_valid_index() == pd.Timestamp('2007-11-30')

one_factor = qis.compute_one_factor_ewm_betas(x=bench_m, y=assets, span=36)
x_b = bench_m.to_numpy()
np.testing.assert_allclose(one_factor.iloc[-1], (omega * x_b) @ y_a / (omega @ x_b ** 2),
                           rtol=1e-8)
assert round(one_factor['SEQ_EU'].iloc[-1], 3) == 1.037 and round(wls.beta['Core'], 3) == 1.026
```

### Look-ahead in the default seed

With the default `InitType.MEAN`, the first beta of `compute_ewm_beta_alpha_forecast` is the
full-sample slope through the origin, 1.289 for `SEQ_EU`, although only one month has been
observed. Adding 10% to the final month's returns moves that first beta by about 0.0002. With
`InitType.X0` every beta before the final month is unchanged, and the first beta is $y_0/x_0$.

```python
from qis import InitType

beta_mean = qis.compute_ewm_beta_alpha_forecast(x_data=bench_m, y_data=assets, span=36)[0]
bumped = assets.copy()
bumped.iloc[-1] = bumped.iloc[-1] + 0.10
beta_mean_bumped = qis.compute_ewm_beta_alpha_forecast(x_data=bench_m, y_data=bumped,
                                                       span=36)[0]
np.testing.assert_allclose(beta_mean.iloc[0], x_b @ y_a / (x_b @ x_b), rtol=1e-8)
assert round(beta_mean['SEQ_EU'].iloc[0], 3) == 1.289
np.testing.assert_allclose((beta_mean.iloc[0] - beta_mean_bumped.iloc[0]).abs(), 0.0002,
                           atol=1e-5)

beta_x0 = qis.compute_ewm_beta_alpha_forecast(x_data=bench_m, y_data=assets, span=36,
                                              init_type=InitType.X0)[0]
beta_x0_bumped = qis.compute_ewm_beta_alpha_forecast(x_data=bench_m, y_data=bumped, span=36,
                                                     init_type=InitType.X0)[0]
np.testing.assert_array_equal(beta_x0.iloc[:-1], beta_x0_bumped.iloc[:-1])
np.testing.assert_allclose(beta_x0.iloc[0], y_a[0] / x_b[0], rtol=1e-12)
```

### EWM $R^2$ definitions

The last block reproduces the two $R^2$ definitions with an independent loop that mirrors the qis
recursion: the seed replaces the first row, and a series that starts missing starts from a zero
seed at its first finite value. The in-sample, uncentred `get_model_ewm_r2` is 0.999 for `SEQ_EU`
in November 2007, the first month after the warm-up, and 0.512 in December 2025. The centred
$R^2$ of the lag-one, point-in-time prediction from `compute_ewm_alpha_r2_given_prediction` is
0.410 in December 2025. The two numbers answer different questions and are not comparable.

```python
def ewm_path(values, seeds, decay):
    """The qis EWM recursion with NanBackfill.FFILL, written as an explicit loop."""
    out = np.full(values.shape, np.nan)
    for j in range(values.shape[1]):
        state = np.nan
        for t in range(values.shape[0]):
            value = values[t, j]
            if np.isnan(state):
                if np.isfinite(value):
                    state = seeds[j] if t == 0 else decay * seeds[j] + (1 - decay) * value
            elif np.isfinite(value):
                state = decay * state + (1 - decay) * value
            out[t, j] = state
    return out


def first_row_seed(values):
    return np.where(np.isfinite(values[0]), values[0], 0.0)


# uncentred, in-sample R2 of LinearModel
r2_uncentred = full.get_model_ewm_r2(span=36)
alpha_lag0, _ = full.get_factor_alpha(lag=0)
a2, y2 = alpha_lag0.to_numpy() ** 2, y_a ** 2
expected = np.clip(1 - ewm_path(a2, first_row_seed(a2), lam) / ewm_path(y2, y2[0], lam), 0, 1)
np.testing.assert_allclose(r2_uncentred, expected, rtol=1e-9)
assert round(r2_uncentred.loc['2007-11-30', 'SEQ_EU'], 3) == 0.999
assert round(r2_uncentred['SEQ_EU'].iloc[-1], 3) == 0.512

# centred R2 of a point-in-time prediction: lag-one loadings times the current factor return
_, explained_lag1 = full.get_factor_alpha(lag=1)
alpha_oos, r2_oos = qis.compute_ewm_alpha_r2_given_prediction(y_data=assets,
                                                              y_prediction=explained_lag1,
                                                              span=36)
resid_1 = y_a - explained_lag1.to_numpy()
alpha_path = ewm_path(resid_1, first_row_seed(resid_1), lam)
resid_var = ewm_path((resid_1 - alpha_path) ** 2, np.zeros(3), lam)
y_var = ewm_path((y_a - ewm_path(y_a, y_a[0], lam)) ** 2, np.zeros(3), lam)
np.testing.assert_allclose(alpha_oos, alpha_path, rtol=1e-9)
np.testing.assert_allclose(r2_oos, np.clip(1 - resid_var / y_var, 0, 1), rtol=1e-9)
assert round(r2_oos['SEQ_EU'].iloc[-1], 3) == 0.410
```

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| OLS fit, prediction and legend label | $(X^{\top}X)^{-1}X^{\top}y$ | `qis.fit_multivariate_ols(x, y, fit_intercept=True, verbose=True)`, returns (prediction, params, label) |
| Scalar OLS statistics for tables | $\hat\alpha$, $\hat\beta$, $R^2$, classical p-value | internal `qis.utils.regression.estimate_ols_alpha_beta`; `PerfStat.ALPHA`, `ALPHA_AN`, `BETA`, `R2`, `ALPHA_PVALUE` of `qis.compute_ra_perf_table_with_benchmark` |
| OLS with Bartlett HAC for alpha | $\hat\Sigma_{\theta}$ with $p=2$, normal reference | internal `qis.utils.regression.estimate_ols_alpha_beta_hac(x, y, hac_lags=3, confidence_level=0.95)`, returns `OlsAlphaBetaHacResult` |
| Mean with Bartlett HAC | $\hat\Sigma_{\theta}$ with $p=1$ | internal `qis.utils.regression.estimate_hac_mean(y, hac_lags=3, confidence_level=0.95)`, returns `HacMeanResult` |
| Lag rule | $\lfloor 4(T/100)^{2/9}\rfloor$ | internal `qis.utils.regression.newey_west_lag_rule(nobs)` |
| Geometric WLS with stacked HAC | $(X^{\top}\Omega X)^{-1}X^{\top}\Omega y$, $\hat\Sigma_{\Theta}$, $T_{\mathrm{eff}}$ | `qis.estimate_ewma_alpha_beta_hac(x, y, span=36.0, hac_lags=3, confidence_level=0.95)`, returns `qis.EwmaAlphaBetaHacResult` |
| Legend annualisation | $e^{\mathrm{AN}\hat\alpha}-1$ | internal `qis.utils.regression.reg_model_params_to_str(..., alpha_an_factor=None)`, via `qis.plot_scatter` |
| EWM loadings tensor | $B_t=M_t^{-1}C_t$ | `qis.compute_ewm_xy_beta_tensor(x, y, span=None, ewm_lambda=0.94, warmup_period=20, is_x_correlated=True)` |
| One-factor EWM betas | $\mathcal{E}_t[xy]/\mathcal{E}_t[x^2]$, zero seeds | `qis.compute_one_factor_ewm_betas(x, y, span=None, ewm_lambda=0.94)` |
| Linear model loadings | $B_t$ per factor | `qis.EwmLinearModel.fit(span=31, is_x_correlated=True, init_type=InitType.MEAN, warmup_period=20)`; `qis.estimate_ewm_factor_model` |
| One-factor EWM alpha, prediction, $R^2$ | centred, seeds from `init_type` | `qis.compute_ewm_beta_alpha_forecast(..., init_type=InitType.MEAN)` |
| EWM alpha and $R^2$ of a prediction | centred, X0 and zero seeds | `qis.compute_ewm_alpha_r2_given_prediction` |
| Linear-model alpha | $y_t-\sum_fB_{f,t-h}x_{f,t}$ | `qis.LinearModel.get_factor_alpha(lag=1, span=None)` |
| Linear-model $R^2$ | $1-\mathcal{E}_t[a^2]/\mathcal{E}_t[y^2]$, uncentred | `qis.LinearModel.get_model_ewm_r2(span=52, lag=0)` |

The OLS, HAC and EWMA-WLS code is in
[regression.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/regression.py);
the EWM recursions in
[ewm.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py);
the linear models in
[ewm_factor_model.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/ewm_factor_model.py)
and
[factor_model.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/factor_model.py).
Of the regression helpers, only `fit_multivariate_ols`, `estimate_ewma_alpha_beta_hac` and
`EwmaAlphaBetaHacResult` are exported in `qis.__all__`; the others are internal to
`qis.utils.regression` and may change without a deprecation cycle.
`src/qis/utils/tests/regression_test.py` checks the HAC and EWMA-WLS estimators against
independent matrix calculations.

API pages: {doc}`fit_multivariate_ols <api/generated/qis.fit_multivariate_ols>`,
{doc}`estimate_ewma_alpha_beta_hac <api/generated/qis.estimate_ewma_alpha_beta_hac>`,
{doc}`EwmaAlphaBetaHacResult <api/generated/qis.EwmaAlphaBetaHacResult>`,
{doc}`compute_ewm_xy_beta_tensor <api/generated/qis.compute_ewm_xy_beta_tensor>`,
{doc}`compute_one_factor_ewm_betas <api/generated/qis.compute_one_factor_ewm_betas>`,
{doc}`compute_ewm_beta_alpha_forecast <api/generated/qis.compute_ewm_beta_alpha_forecast>`,
{doc}`compute_ewm_alpha_r2_given_prediction <api/generated/qis.compute_ewm_alpha_r2_given_prediction>`,
{doc}`EwmLinearModel <api/generated/qis.EwmLinearModel>` and
{doc}`LinearModel <api/generated/qis.LinearModel>`.

## Interpretation and limitations

- **HAC corrects inference, not bias.** Smoothed or stale returns bias the beta itself towards
  zero; a HAC interval around a biased beta is still centred on the wrong value. See
  [private-asset unsmoothing](private_asset_unsmoothing.md).
- **The default lag count is a choice.** `hac_lags=3` on every grid understates uncertainty when
  residuals are persistent; compare it with `newey_west_lag_rule` and with larger values.
- **Normal critical values are optimistic in small samples.** With a few dozen observations, or an
  EWMA-WLS fit whose effective size is about $N$, treat p-values as approximate. The EWMA-WLS
  correction $T/(T-2)$ uses the raw row count, not $T_{\mathrm{eff}}$.
- **Row order is the caller's responsibility.** `estimate_ewma_alpha_beta_hac` weights rows by
  position and does not sort them; an unsorted index silently reweights the sample.
- **Failure modes differ by helper.** `estimate_ols_alpha_beta` returns zeros with a warning on
  most failures and raises `IndexError` on a constant regressor; the HAC helpers raise
  `ValueError`. A zero alpha in a table can therefore be a failed fit.
- **Units matter for EWM betas.** `compute_ewm_xy_beta_tensor` replaces the inverse by the
  identity when a factor's EWM second moment is at most $10^{-8}$, which returns a cross moment
  for very low-volatility factors; `compute_ewm_beta_alpha_forecast` returns NaN in the same
  regime. Scale such series before regressing.
- **Uncentred and centred $R^2$ differ.** `get_model_ewm_r2` is uncentred, in sample by default
  and biased towards one just after the warm-up; the $R^2$ of `compute_ewm_beta_alpha_forecast`
  and `compute_ewm_alpha_r2_given_prediction` is centred. Do not rank models across the two.
- **Linearity needs one sample.** Exact additive attribution of alphas requires the same rows and
  regressor values in every equation.

## See also

- [Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md)
- [Exponentially weighted estimators](ewm_estimators.md)
- [Serial dependence and autocorrelation](serial_dependence.md)
- [Factor risk models](factor_risk_models.md)
- [Model-layer attribution](model_layer_attribution.md)
- [The performance-statistic catalogue](performance_statistics.md)
- [Private-asset unsmoothing](private_asset_unsmoothing.md)
- [Notation and conventions](notation_and_conventions.md)
- [Bibliography](bibliography.md)

## References

1. Newey, W. K., and West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. [Working paper and published-version record](https://www.nber.org/papers/t0055). The Bartlett-weighted HAC estimator and its positive semidefiniteness.
2. Newey, W. K., and West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *The Review of Economic Studies*, 61(4), 631–653. [DOI: 10.2307/2297912](https://doi.org/10.2307/2297912). Lag selection and the rule of thumb behind `newey_west_lag_rule`.
3. White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. *Econometrica*, 48(4), 817–838. [DOI: 10.2307/1912934](https://doi.org/10.2307/1912934). The zero-lag, heteroskedasticity-consistent case.
4. Andrews, D. W. K. (1991). Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation. *Econometrica*, 59(3), 817–858. [DOI: 10.2307/2938229](https://doi.org/10.2307/2938229). Kernel comparison and optimal bandwidth rates.
5. Kish, L. (1965). *Survey Sampling*. Wiley. The effective sample size of a weighted sample.
6. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Least-squares algebra and long-run variance of dependent series.
7. J.P. Morgan and Reuters (1996). *RiskMetrics — Technical Document*, 4th edition. J.P. Morgan. The 0.94 daily decay used as the EWM default.
8. statsmodels developers. statsmodels. Software. [HAC covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html). The HAC implementation and small-sample correction that qis follows.
9. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
