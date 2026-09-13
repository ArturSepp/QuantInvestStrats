---
myst:
  html_meta:
    description: >-
      Attribute layered portfolio log returns to systematic exposure, risk, signals and
      integration with qis, distinguish descriptive and lagged estimates, and reproduce
      HAC intervals and factorial or Shapley feature effects.
---

# Model-layer attribution: risk-layer, signal-layer, and integration alpha

*[author / affiliation / date — placeholder]*

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

| Symbol or input | Meaning | Units and timing |
|---|---|---|
| $N_L(t)$ | Positive NAV of layer $L$ | Common currency, dates and valuation basis |
| $B,R,S,F,I$ | Benchmark, risk, signal, full model and integration | $I$ is a return residual, not a supplied NAV |
| $r_L(t)$ | Log change in the layer NAV | Decimal per retained observation at `freq` |
| $\alpha_L,\beta_L$ | Regression intercept and benchmark slope | Periodic log return and dimensionless exposure |
| $a_L(t),c(t)$ | Realised alpha component and net-minus-gross cost drag | Periodic log-return contributions |
| $T,A$ | Retained observations and periods per year | `ME` uses $A=12$; `QE` uses $A=4$ |
| $h,\lambda$ | EWMA span and decay | Observations and dimensionless decay |
| $q$ | Bartlett HAC lag count | Retained return observations |

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
r_L(t) = \log N_L(t) - \log N_L(t-1),
\qquad t = 1, \dots, T.
$$

The residual definition makes the within-period bridge exact. Log returns additionally sum
across time, so cumulative contributions and their annualised means reconcile on the same scale.
This does not make the log return of an arbitrary portfolio a weighted sum of its sleeves' log
returns. Reported annualised means are not compounded annual growth rates; exponentiating each
component separately destroys additivity.

The common sample is set before any resampling. The NAVs are trimmed to the range between the
latest first valid observation and the earliest last valid observation, forward-filled inside
that range, and then converted to returns at `freq`. A layer that starts later or ends earlier
than the others therefore shortens the sample for every layer, and no layer contributes flat
forward-filled returns outside its own history. Any date on which a periodic return is still not
finite is dropped for all layers.

The integration return is the log-return residual of the full model against the sum of the risk and signal
layers run on their own, and the trading-cost drag is the difference between the net and gross full
models,

$$
r_I(t) = r_F(t) - r_R(t) - r_S(t), \qquad c(t) = r_F^{\mathrm{net}}(t) - r_F(t).
$$

## Methodology

### Full-sample OLS layer regressions

For each layer $L \in \{R, S, I, F, F^{\mathrm{net}}\}$ the function estimates the
full-sample regression on the benchmark,

$$
r_L(t) = \alpha_L + \beta_L \, r_B(t) + \epsilon_L(t),
$$

by ordinary least squares. With $\bar r_L$ the sample mean over the common sample,

$$
\hat\beta_L = \frac{\sum_{t=1}^T (r_L(t) - \bar r_L)(r_B(t) - \bar r_B)}{\sum_{t=1}^T (r_B(t) - \bar r_B)^2},
\qquad
\hat\alpha_L = \bar r_L - \hat\beta_L \, \bar r_B.
$$

The benchmark row of the regression table is fixed at $\hat\alpha_B = 0$ and
$\hat\beta_B = 1$ rather than estimated. The annualised alpha is $A \hat\alpha_L$
with $A$ the number of periods per year implied by `freq`. Beta, $R^2$ and the
periodic standard error are not annualised. Annualisation is linear because $\hat\alpha_L$
is a mean log return, and a mean log return scales with the number of periods.

### The exact return bridge

The gross full-model return is separated into systematic return $s_F$ and three alpha
components: risk $a_R$, signal $a_S$, and integration $a_I$.

$$
\begin{aligned}
s_F(t) &= \hat\beta_F \, r_B(t), \\
a_R(t) &= r_R(t) - \hat\beta_R \, r_B(t), \\
a_S(t) &= r_S(t) - \hat\beta_S \, r_B(t), \\
a_I(t) &= r_F(t) - \hat\beta_F \, r_B(t) - a_R(t) - a_S(t).
\end{aligned}
$$

The fourth line defines the integration alpha as the residual, so the identity

$$
r_F(t) = \hat\beta_F \, r_B(t) + a_R(t) + a_S(t) + a_I(t)
$$

holds in every period by construction. When a net NAV is supplied,
$r_F^{\mathrm{net}}(t) = r_F(t) + c(t)$ extends the identity to the net return. Three
properties turn this bookkeeping into an estimator.

#### Linearity: the integration term is an estimated alpha

On the common sample, the integration coefficients are exact linear combinations of the layer
coefficients,

$$
\hat\beta_I = \hat\beta_F - \hat\beta_R - \hat\beta_S, \qquad
\hat\alpha_I = \hat\alpha_F - \hat\alpha_R - \hat\alpha_S,
$$

and the residual bridge term is the beta-adjusted integration return,
$a_I(t) = r_I(t) - \hat\beta_I \, r_B(t) = \hat\alpha_I + \hat\epsilon_I(t)$. The reason is
that the OLS estimator $(X^{\intercal}X)^{-1} X^{\intercal} y$ with
$X = [\mathbf{1}, r_B]$ is linear in $y$ for a fixed regressor matrix, and all layer
regressions share the regressor matrix because they share the common sample. The residual vector
is linear in $y$ for the same reason, so
$\hat\epsilon_I = \hat\epsilon_F - \hat\epsilon_R - \hat\epsilon_S$. This is why the
function regresses the integration return as an additional layer and reports its beta, alpha and
interval on the same footing as the observed layers. The integration alpha is not an unexplained
plug. It is the OLS alpha of the log-return series $r_F - r_R - r_S$.

#### Bar heights are OLS alphas

The annualised sample mean of each alpha component equals the annualised OLS alpha of its layer,

$$
\frac{A}{T} \sum_{t=1}^T a_L(t) = A \, \hat\alpha_L, \qquad L \in \{R, S, I\},
$$

because OLS residuals have zero sample mean when the regression includes an intercept. The
annualised systematic component equals $A \hat\beta_F \bar r_B$, and the four annualised
components sum to $A(\hat\beta_F \bar r_B + \hat\alpha_R + \hat\alpha_S + \hat\alpha_I) =
A(\hat\beta_F \bar r_B + \hat\alpha_F) = A \bar r_F$. The return bridge and the regression table
are therefore one object: the bars of a bridge chart are the annualised alphas, and the whiskers
on them are the intervals of those same alphas.

#### Invariance to the excess-return basis

qis does not subtract a risk-free rate in this attribution. State whether the supplied NAVs
represent funded total returns or an already specified excess-return strategy. The algebra below
concerns subtracting the **benchmark log return**, not an arbitrary risk-free series. Those are
different transformations; the latter need not leave regression alphas unchanged.

Replace the signal-layer return by its excess over the benchmark,
$r_S'(t) = r_S(t) - r_B(t)$. Every alpha, every residual, every HAC standard error, every
confidence bound and every p-value is unchanged. Two betas move, and their $R^2$ with them,

$$
\hat\beta_S' = \hat\beta_S - 1, \qquad \hat\beta_I' = \hat\beta_I + 1.
$$

The benchmark regressed on itself has intercept 0, slope 1 and zero residuals, so subtracting
$r_B$ from a layer subtracts 0 from its alpha, 1 from its beta and nothing from its
residuals, and the HAC covariance depends on the regressor and the residuals only. The same holds
for the risk layer.

The invariance settles how to read the integration beta. A fully invested signal layer carries a
benchmark beta near one, as does the risk layer. Adding the two as total returns doubles the
benchmark exposure, so the integration beta is near $-1$ by construction, for example
$0.85 - 1.05 - 1.00 = -1.20$. On the excess basis for the signal layer the same integration term
has beta near $-0.20$, with identical alphas and intervals. The integration term is a
log-return residual, not a tradeable portfolio, and its beta must be interpreted on the stated total-return or benchmark-excess basis.

### Additive cumulative alpha with fixed full-sample betas

The annualised alpha table answers how much each component contributed on average. The same
`component_returns` output also shows when the contribution accumulated. For return date $t_k$,
define the cumulative alpha contribution of layer $L$ by

$$
C_L(t_k)=\sum_{t=1}^{k} a_L(t), \qquad L \in \{R,S,I\}.
$$

Because the bridge identity holds in every period, the cumulative total model alpha
$C_F(t_k) = \sum_{t=1}^{k} \big(r_F(t) - \hat\beta_F \, r_B(t)\big)$ is additive at every date,

$$
C_F(t_k)=C_R(t_k)+C_S(t_k)+C_I(t_k).
$$

No new regression is run for this chart. The full-sample OLS betas used to construct
`Risk Layer Alpha`, `Signal Layer Alpha` and `Integration Alpha` remain fixed, and the chart simply
cumulatively sums those exact periodic components. The runnable workflow below produces
percentage-point paths with an explicit 0% origin one attribution period before the first return.

The vertical axis is cumulative log-return contribution in percentage points. It is not a wealth
index. In particular, do not use `100 * exp(cumsum(alpha))` for an additive alpha-attribution
chart. That construction is a compounded pseudo-NAV: its components reconcile multiplicatively,
whereas the alpha bridge is defined and interpreted additively. The cumulative paths are
descriptive because their betas are full-sample estimates; they are not point-in-time alpha
forecasts.

### Lagged no-look-ahead EWMA-beta realised and cumulative alpha

Use the rolling estimator when the question is how alpha accumulated under betas that were
available before each realised return. For the default span $h=36$, QIS uses
$\lambda=1-2/(h+1)$ and removes a same-span point-in-time EWMA mean from the benchmark and each
layer. For a generic return $x_t$, the mean recursion is

$$
m_t^x=\lambda m_{t-1}^x+(1-\lambda)x_t, \qquad \tilde x_t=x_t-m_t^x.
$$

The beta estimate after observing date $t$ is the ratio of EWMA cross moment to EWMA benchmark
variance,

$$
\begin{aligned}
q_t^{BL}&=\lambda q_{t-1}^{BL}+(1-\lambda)\tilde r_B(t)\tilde r_L(t),\\
q_t^{BB}&=\lambda q_{t-1}^{BB}+(1-\lambda)\tilde r_B(t)^2,\\
\hat\beta_L(t)&=\frac{q_t^{BL}}{q_t^{BB}}.
\end{aligned}
$$

The estimator uses `MeanAdjType.EWMA`, the point-in-time `InitType.X0` initial condition, and an
explicit beta prior (one by default). `MeanAdjType.INSAMPLE` is rejected because subtracting a
full-sample mean would introduce look-ahead. Under EWMA mean adjustment and `InitType.X0`, the
first centered observation is zero, so the first estimated beta is deliberately retained as
missing. The applied beta remains finite: QIS uses the prior until a lagged estimate is available.

With the default one-period lag, the beta estimated at $t-1$ is applied to return $t$,

$$
\tilde\beta_L(t)=\hat\beta_L(t-1), \qquad
a_L(t)=r_L(t)-\tilde\beta_L(t)r_B(t),
$$

with the prior substituted before the first lagged finite estimate. Thus return $t$ may update
$\hat\beta_L(t)$, but it cannot change the beta applied to itself. Total and integration alpha are

$$
a_F(t)=r_F(t)-\tilde\beta_F(t)r_B(t), \qquad
a_I(t)=a_F(t)-a_R(t)-a_S(t),
$$

so $a_F(t)=a_R(t)+a_S(t)+a_I(t)$ at every date. The expanding annualised estimate through $t_k$
is $A k^{-1}\sum_{t=1}^{k}a_L(t)$. The post-warm-up exhibit is different: it is the unannualised
cumulative sum $\sum a_L(t)$ after a chosen base date, with an explicit zero row on that date.

For a current estimate that gives more weight to recent realised alpha, QIS applies the same EWMA
span to these step-ahead residuals. With $h=36$ and
$\lambda=1-2/(h+1)$,

$$
\bar a_L(t)=\lambda\bar a_L(t-1)+(1-\lambda)a_L(t),
\qquad
\bar\alpha_L^{\mathrm{ann}}(t)=A\bar a_L(t).
$$

This is a point-in-time EWMA of realised OOS alpha. It is not the contemporaneous alpha forecast
returned by the lower-level beta routine, and it does not refit a 36-observation in-sample
regression. Because EWMA is linear, total alpha continues to equal risk-layer alpha plus
signal-layer alpha plus integration alpha at every date. The current estimate is the final row of
the EWMA path. It is a responsive point estimate, not a confidence interval; a 36-period EWMA has
less effective information than the full-sample HAC regression.

The runnable workflow below uses a 36-month EWMA beta, lags it by one month, allows 12 monthly returns
for estimator warm-up, and then accumulates realised alpha from the next month:

`estimated_betas` records estimates after each return; `applied_betas` records the betas actually
used for each return. That distinction is the audit for the one-period lag. The result also exposes
the exact periodic `component_returns`, cumulative alpha from inception, the estimator settings,
and `mean_adj_type`. The post-warm-up result records the base date, first accrued alpha date and the
same settings.

### Current endpoint geometric EWMA-WLS regression

Use `compute_model_layer_ewma_regression_attribution` when the question is the current
recency-weighted decomposition rather than a historical sequence of investable beta estimates.
This is an endpoint descriptive regression: every return in the sample is used to estimate the
single alpha and beta shown at the final date. It must therefore not be substituted for the
lagged-beta realised attribution above.

The default endpoint exhibit converts all NAVs to common-sample monthly log returns with
`freq='ME'`. For $t=0,\ldots,T-1$, counting retained monthly observations from oldest to newest,
the span-$h$ objective weight is

$$
w_t=\lambda^{T-1-t}, \qquad \lambda=1-\frac{2}{h+1}.
$$

Thus the latest month has weight one. With the default $h=36$,
$\lambda=35/37\simeq0.9459$. The reported Kish effective sample size is

$$
T_{\mathrm{eff}}=\frac{\left(\sum_t w_t\right)^2}{\sum_t w_t^2}.
$$

Equivalently,

$$
T_{\mathrm{eff}}=h\frac{1-\lambda^T}{1+\lambda^T}.
$$

For a long history it approaches 36; for a finite history it is smaller. Thus “effective sample
size 36.0” does not mean that QIS discards older observations or fits a 36-month hard window. All
$T$ returns enter with geometric weights; for example, $T=260$ and $h=36$ give
$T_{\mathrm{eff}}=35.999962$, which displays as 36.0. The value is an information-concentration
diagnostic, not the degrees of freedom used in the statsmodels-compatible HAC correction.

Let $X=[\mathbf 1,r_B]$, $W=\operatorname{diag}(w_0,\ldots,w_{T-1})$, and first construct
$r_I=r_F-r_R-r_S$. The columns of $Y=[r_R,r_S,r_I,r_F]$ contain the observed layer returns plus
this exact integration response. The common EWMA-WLS coefficient matrix is

$$
\hat\Theta=(X^{\intercal}WX)^{-1}X^{\intercal}WY.
$$

Inference uses weighted regression scores $w_t x_t\hat\epsilon_L(t)$. QIS stacks those scores
across the risk, signal, integration and full-model equations before applying a Bartlett HAC kernel
with three lags by default. It uses the same $T/(T-2)$ correction as a two-parameter statsmodels WLS
fit and a normal reference distribution. The full joint covariance retains cross-equation terms.
In exact arithmetic,

$$
\begin{aligned}
\hat\alpha_I&=\hat\alpha_F-\hat\alpha_R-\hat\alpha_S,\\
\widehat{\operatorname{Var}}(\hat\alpha_I)
  &=\hat V_{II}=c^{\intercal}\hat V_{RSF}c,\\
c&=(-1,-1,1)^{\intercal}.
\end{aligned}
$$

Fitting the precomputed integration response in the joint system is algebraically equivalent to
the contrast but avoids cancellation when the component equations are nearly collinear or returns
use very small units. Using the joint covariance, rather than a sum of three marginal variances,
gives integration its valid confidence interval. Each return bridge contribution uses the same
EWMA-WLS alpha as its interval, so the alpha bar is exactly the midpoint of its lower and upper
confidence bounds, up to floating-point rounding.

The endpoint return bridge remains exact in every period: systematic return is
$\hat\beta_F r_B$, risk and signal contributions are their beta-adjusted returns, and integration
is the residual required to reconstruct $r_F$. Its displayed annualised bars are geometrically
weighted means of those periodic log-return components. Supplying `full_model_net_nav` adds the
realised trading-cost drag and a net endpoint; it does not estimate a fee inside QIS.

The final endpoint bar makes that regression identity visible. It starts with gross-model
systematic return $\hat\beta_F\bar r_B$, applies realised trading-cost drag when a net NAV is
present, and then adds gross total alpha $\hat\alpha_F$. The label above the bar is gross return
without a net NAV and net return with one. The black gross-alpha interval is translated by the
systematic and cost segments, so its midpoint is the displayed endpoint. The beta and $R^2$ rows
under this bar therefore come from the gross `Full Model` regression, as do its alpha and interval.
The risk, signal and integration alpha bars also show their own regression $R^2$ below beta.
The standalone Systematic bar uses that same gross `Full Model` regression $R^2$: it reports the
fit of the observed full model to the benchmark, not the tautological $R^2=1$ obtained by
regressing the constructed series $\hat{\beta}_F r_B$ back on $r_B$.

An EWMA interval is not guaranteed to be narrower than its full-sample OLS/HAC counterpart.
Recency weighting usually reduces effective information, and the weighted residual variance,
serial dependence and cross-equation covariance can all change. EWMA answers a different question
more responsively; it is not a mechanical confidence-band shrinkage method.

#### Rolling descriptive EWMA-WLS alpha

`compute_model_layer_rolling_ewma_regression_alpha` repeats the same geometric EWMA-WLS fit on
every expanding prefix for which the joint regression is nonsingular. At date $t$, it uses only
returns through $t$, with weights $w_{j,t}=\lambda^{t-j}$. It returns annualised `Total Model
Alpha`, `Risk Layer Alpha`, `Signal Layer Alpha`, and `Integration Alpha`, plus `Total Model Net
Alpha` when the supplied attribution has a net NAV. Linearity gives

$$
\hat\alpha_F(t)=\hat\alpha_R(t)+\hat\alpha_S(t)+\hat\alpha_I(t)
$$

at every plotted date. Its final gross row is checked directly against the `Full Model`, `Risk
Layer`, `Signal Layer`, and `Integration` rows of the current regression table. This path is
descriptive and contemporaneous: unlike lagged-beta realised attribution, return $t$ participates
in the regression displayed at $t$. Accordingly, the generic QIS plot title calls this an
annualised model-layer alpha path, not an out-of-sample alpha estimate.
`plot_model_layer_rolling_ewma_regression_alpha(..., start_date=...)` still estimates every
expanding prefix from the complete history and only then clips the displayed path. Its `avg` and
`last` legend statistics therefore describe the displayed date range without resetting EWMA state.

#### Common-denominator EWMA Sharpe contributions

`compute_model_layer_ewma_sharpe_contributions` uses the annualised return and alpha estimates from
the current EWMA return bridge. The benchmark reference uses benchmark EWMA volatility, while all
model contributions use one common endpoint-model EWMA volatility:

$$
S_B=\frac{\bar r_B^{\mathrm{ann}}}{\sigma_B^{\mathrm{EWMA}}}, \qquad
C_j=\frac{\bar r_j^{\mathrm{ann}}}{\sigma_{F^*}^{\mathrm{EWMA}}}, \qquad
S_{F^*}=\sum_j C_j,
$$

where $j$ runs over systematic return, risk alpha, signal alpha, integration alpha and optional
realised cost drag, and $F^*$ is the net model when supplied and gross otherwise. Both volatility
series use the attribution span and the QIS centred EWMA-volatility recursion. Because every model
term has the same positive denominator, the bridge is additive and each alpha contribution retains
the sign of its alpha. These are contribution ratios, not standalone sleeve Sharpes.

The older `compute_model_layer_ewma_stage_sharpes` remains available as a separate diagnostic. It
computes the EWMA Sharpe path after adding risk, signal and integration returns sequentially. Its
deltas are order-dependent: a positive signal alpha can produce a negative stage-Sharpe increment
when it adds enough volatility or covariance. The public bridge plot uses the additive
common-denominator measure instead.

`compute_model_layer_in_sample_sharpe_contributions` uses the identical two-denominator
construction with the exact full-sample alpha-attribution numerators and full-sample annualised
log-return volatilities. For a return frequency with annualisation factor $A$,
$\sigma=\sqrt{A}\,\operatorname{std}(r,\mathrm{ddof}=1)$. It therefore reports full-history
realised return per unit of full-history realised risk, with no EWMA window in either numerator or
denominator. The endpoint bar in both public Sharpe plots is split into systematic, optional
realised cost, and combined alpha contributions.

The return and Sharpe bridges use the final endpoint, while the rolling-alpha figure shows every
displayed estimable prefix. The return-bridge, rolling-alpha, EWMA-Sharpe and in-sample-Sharpe plot functions accept `detailed_mode=False` for a clean export
without title, subtitle or methodology footnote.

### Full-sample OLS/HAC inference

Alpha inference uses a Bartlett-kernel heteroskedasticity and autocorrelation consistent (HAC)
covariance with $q$ lags (`hac_lags`, default 3), the statsmodels small-sample correction,
a normal reference distribution and a two-sided interval at `confidence_level` (default 0.95).
With $x_t = (1, r_B(t))^{\intercal}$ and OLS residuals $\hat\epsilon_L(t)$,

$$
\begin{aligned}
\hat\Gamma_\ell
  &=\sum_{t=\ell+1}^{T}x_t\,\hat\epsilon_L(t)\,\hat\epsilon_L(t-\ell)\,x_{t-\ell}^{\intercal},\\
\hat S
  &=\hat\Gamma_0+\sum_{\ell=1}^{q}
    \Big(1-\frac{\ell}{q+1}\Big)
    \big(\hat\Gamma_\ell+\hat\Gamma_\ell^{\intercal}\big).
\end{aligned}
$$

$$
\begin{aligned}
\widehat{\mathrm{Var}}(\hat\alpha_L,\hat\beta_L)
  &=\frac{T}{T-2}(X^{\intercal}X)^{-1}\hat S(X^{\intercal}X)^{-1},\\
\mathrm{se}(\hat\alpha_L)&=\sqrt{\widehat{\mathrm{Var}}_{11}}.
\end{aligned}
$$

The factor $T/(T-2)$ is the `use_correction=True` degrees-of-freedom adjustment for two
regressors. The annualised interval and the p-value are

$$
\begin{aligned}
\mathrm{CI}_{95\%}(A\hat\alpha_L)
  &=A\big(\hat\alpha_L\pm z_{0.975}\,\mathrm{se}(\hat\alpha_L)\big),\\
p_L&=2\big(1-\Phi(\lvert\hat\alpha_L\rvert/\mathrm{se}(\hat\alpha_L))\big).
\end{aligned}
$$

with $z_{0.975} = 1.960$ at the default level. The generic estimator lives in
`qis.utils.regression` as `estimate_ols_alpha_beta_hac`, and
`src/qis/utils/tests/regression_test.py` checks it against a hand-rolled Newey and West matrix
calculation. The lag count is a choice. The default of three is fixed for any frequency, and
`qis.utils.regression.newey_west_lag_rule(nobs)` returns the Newey and West rule
$\lfloor 4 (T/100)^{2/9} \rfloor$ for callers who prefer it, which gives four at
$T = 240$ and three at $T = 86$. The result records `freq`, `hac_lags` and
`confidence_level`, so a table or a footnote can quote the settings that produced it.

The interval of the integration alpha carries the covariance between the layers. Let
$\hat V$ be the joint HAC covariance of $(\hat\alpha_F, \hat\alpha_R, \hat\alpha_S)$
built from the stacked score process with the same kernel and correction. Then the HAC variance
of $\hat\alpha_I$ from its own regression equals

$$
\widehat{\mathrm{Var}}(\hat\alpha_I) = c^{\intercal} \hat V c, \qquad c = (1, -1, -1)^{\intercal},
$$

because each HAC estimator is a quadratic form in the scores $x_t \hat\epsilon_L(t)$, and
the integration scores are the linear combination $c$ of the three layer scores. A wide
integration interval is a statement about the joint estimation error of three layers, not a
computational artefact. The three intervals on a bridge chart are marginal intervals. They are
not independent, and their widths do not add. The interval of the total alpha
$A\hat\alpha_F$ is the `Full Model` row of the table.

The kernel follows [Newey and West (1987)](https://www.nber.org/papers/t0055);
the correction and lag-rule convention follow the [statsmodels HAC implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).

### Measuring the impact of a model feature

The same bridge isolates the effect of one feature $\phi$ on the risk side and on the
signal side. Let subscript 0 denote the model without the feature and subscript 1 the model with
it. For each layer $L \in \{R, S, F\}$ the feature return is
$\Delta_L(t) = r_{L,1}(t) - r_{L,0}(t)$, which is the log return of the NAV ratio
$N_{L,1}(t) / N_{L,0}(t)$. Regressing it on the benchmark gives
$\hat\alpha_{\Delta L} = \hat\alpha_{L,1} - \hat\alpha_{L,0}$ and
$\hat\beta_{\Delta L} = \hat\beta_{L,1} - \hat\beta_{L,0}$ by linearity, with the HAC
interval of the difference from a single regression. The feature's total effect decomposes
exactly into a risk-side, a signal-side and an integration effect,

$$
\hat\alpha_{\Delta F} = \hat\alpha_{\Delta R} + \hat\alpha_{\Delta S} + \hat\alpha_{\Delta I}.
$$

In code this is one call with the benchmark NAV and the three ratio NAVs in place of the layer
NAVs. The `Integration` row then gives $\hat\alpha_{\Delta I}$ without further work, since
$\Delta_F - \Delta_R - \Delta_S = r_{I,1} - r_{I,0}$.

Two conditions apply. The two models of each layer must share the same date index, because a
date missing from one NAV makes the ratio missing there and the forward fill then replaces a
return difference by a level jump. And the identity between the ratio regression and the
difference of two separate attributions holds only when both land on the same common sample, so
subtracting two regression tables is not a substitute: the standard-error and interval columns
of a table difference have no meaning. A feature that changes only the covariance estimator has
$\Delta_S \equiv 0$, and its effect is read from the risk and integration rows. A feature
that changes only a signal has $\Delta_R \equiv 0$.

### Alpha/beta attribution by multiple model features

`qis.compute_model_feature_alpha_beta_attribution` extends the single-feature ratio analysis to a
complete factorial experiment. A scenario is keyed by the `frozenset` of features enabled in that
run; the empty coalition is the production baseline. For $n$ features, all $2^n$ coalitions must be
supplied, and every coalition must use the same benchmark path.

For this feature experiment, let $N$ be the set of all features and $v_L(V)$ the log return
in layer $L$ for coalition $V$. QIS first computes the Harsanyi dividend for every non-empty
coalition $U$,

$$
d_L(U)=\sum_{V\subseteq U}(-1)^{|U|-|V|}v_L(V).
$$

Singleton dividends are direct feature effects; larger coalitions are interactions. Their sum is
the joint all-features-versus-production effect. QIS then assigns the interactions without an
arbitrary feature order using the Shapley value,

$$
\phi_{i,L}=\sum_{V\subseteq N\setminus\{i\}}
\frac{|V|!(n-|V|-1)!}{n!}
\left[v_L(V\cup\{i\})-v_L(V)\right].
$$

For two features the Shapley effect of feature $i$ is the average of its effect with and without
the other feature, $\phi_{i,L} = d_L(\{i\}) + \tfrac{1}{2} d_L(\{1, 2\})$, so each feature
receives half of the interaction. Both decompositions are calculated pathwise from NAV products,
so the factorial effects and the Shapley feature paths independently reconstruct the joint log
return at every observation. Each
Shapley path is passed to `compute_model_layer_alpha_beta_attribution`; its alpha, beta and HAC
interval are therefore estimated from one effect-return series rather than by subtracting two
regression tables.

When net full-model NAVs are supplied, they must be present in every coalition. The summary then
includes both gross and net total-return intervals and the net-model regression. Scenario
construction remains outside QIS: the caller decides what enabling a feature means and supplies
the resulting NAVs.


The allocation uses the [Shapley value](https://www.rand.org/pubs/papers/P295.html).
Interactions are calculated for the supplied experiment.

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

The annualised mean log-return components in percent are benchmark return 5.05, systematic return 4.34,
risk-layer alpha 1.23, signal-layer alpha 2.74, integration alpha −0.85, full-model return 7.47,
trading-cost drag −0.15 and net return 7.32. The identity checks print residuals of order
$10^{-15}$ or smaller for linearity and bar heights, and the excess-basis run changes no
alpha, standard error, bound or p-value while shifting the signal-layer beta by exactly −1 and the
integration beta by exactly +1. The lag-rule check moves from three to four Bartlett lags and
changes the interval half-widths by at most 5 basis points per year, from 122 to 126 for the risk
layer and from 210 to 215 for the full model. The headline directions in this illustration remain the same, but
marginal significance and reported bounds still depend on the lag choice.

![Annualised model-layer return bridge showing systematic return, layer alphas, trading costs and net return with HAC intervals](images/model_layer_attribution_simulated.png)

[Open full-resolution preview](images/model_layer_attribution_simulated.png).

Read the exhibit left to right. The benchmark's annualised mean log return was 5.05%. The full model runs at
$\hat\beta_F = 0.86$, so its systematic return is 4.34%, and the 0.71% gap between the two
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
describe how to regenerate all seven documentation analytics images with their tables and
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
: $A$ times the column means of `component_returns`. These are the bar heights of a return
  bridge, and by the bar-height property the three alpha entries equal the annualised alphas in
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
Shapley effect. The runnable workflow below constructs the corresponding `risk_span_alpha_paths`.

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
  dependence can make nominal 95% intervals unreliable.
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
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Simulated example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/model_layer_attribution_simulated.py)
- [Performance and Sharpe conventions](performance_analytics_and_sharpe.md)
- [Reproducibility and bootstrap conventions](reproducibility.md)

## References

- Newey, W. K., and West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.
  [Author working paper and published-version record](https://www.nber.org/papers/t0055).
- Shapley, L. S. (1952). *A Value for N-Person Games*. RAND, P-295.
  [Original report](https://www.rand.org/pubs/papers/P295.html). Published in *Contributions
  to the Theory of Games II* (1953); [publisher's reprint record](https://doi.org/10.1515/9781400829156-012).
- statsmodels. [HAC covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
