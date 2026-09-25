---
myst:
  html_meta:
    description: >-
      Volatility-normalised returns, volatility targeting, normalised return sums, EWM momentum
      filters and signal-to-weight maps, as implemented in qis/models/linear/ra_returns.py.
---

# Risk-adjusted returns and volatility targeting

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A risk-adjusted return is a periodic return divided by a volatility estimate that was known
before the return was realised. Holding a position of target volatility over estimated volatility
is volatility targeting, and the risk-adjusted return is exactly the return of that position.
qis builds this primitive on an exponentially weighted (EWM) volatility and, on top of it,
normalised return sums, unit-variance momentum filters and bounded signal-to-weight maps.

## Overview

The chapter follows one pipeline, from returns to weights:

1. **Normalise.** Divide each return by a lagged EWM volatility, `qis.compute_ra_returns`. The
   output is in units of risk: comparable across assets and summable across a panel.
2. **Target.** Scale by a volatility target to obtain a position and its returns. Evidence on
   whether this improves performance is mixed and asset-class dependent.
3. **Aggregate.** Sum risk-adjusted returns over rolling windows or calendar periods, scaled by
   the square root of the horizon.
4. **Filter.** Smooth risk-adjusted returns with EWM filters normalised to unit variance, the
   building block of time-series momentum.
5. **Map.** Turn a signal into a bounded weight through a CDF-shaped map.

Each step separates the published idea from the qis implementation choice. Four choices matter
most: the volatility is lagged one row by default; the target is per period, not per annum; the
default target is a unit, not "no scaling"; and the EWM recursion is seeded from one
observation.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Periodic simple or log returns as supplied, total or excess; `is_log_returns_to_arithmetic=True` maps a log return to $e^{\ell}-1$ before scaling |
| Sampling grid | The index of the supplied returns, usually business days (`B`); `compute_sum_freq_ra_returns` resamples to `freq` |
| Annualisation | None inside the estimators: $\hat\sigma_t$ and `vol_target` are per period; an annual target $\sigma_{\mathrm{ann}}$ enters as $\sigma_{\mathrm{ann}}/\sqrt{\mathrm{AN}}$ |
| Mean adjustment | None by default (`MeanAdjType.NONE`): an EWM second moment about zero; `EWMA` and `EXPANDING` are point in time, `INSAMPLE` is forward-looking |
| Timing | $\hat\sigma_t$ uses returns through $t$; the return $r_t$ is divided by $\hat\sigma_{t-1}$ (`weight_lag=1`); a signal dated $t$ must be applied over $(t,t+1]$ by the caller |
| Output units | Risk units (unit variance per period) when `vol_target=None`; per-period returns of the scaled position otherwise; weights as multiples of NAV |
| qis default | `qis.compute_ra_returns(span=None, ewm_lambda=0.94, vol_target=None, mean_adj_type=MeanAdjType.NONE, weight_lag=1)`, no volatility floor, no warm-up mask |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $r_t$ / `returns` | Return over $(t-1,t]$ | Decimal per period |
| $\lambda$, $N$ / `ewm_lambda`, `span` | Decay and span of the volatility EWM | $\lambda=1-2/(N+1)$; default $\lambda=0.94$, so $N\approx 32.3$ |
| $\hat\sigma_t$ | EWM volatility from returns through $t$ | Per period |
| $\sigma_{t\mid t-1}$ | True conditional volatility of $r_t$ given information through $t-1$ | Per period |
| $\mathcal{F}_{t}$ | Information available at $t$ | |
| $\sigma_{\mathrm{tgt}}$ / `vol_target` | Volatility target | Per period; `None` means 1 |
| $\delta$ / `weight_lag` | Rows between the volatility estimate and the return it scales | Default 1; `None` or 0 means no lag |
| $x_t$ | Risk-adjusted return | Risk units when $\sigma_{\mathrm{tgt}}=1$ |
| $w^{*}_t$ | Volatility-targeting weight decided at $t$ | Multiple of NAV, held over $(t,t+1]$ |
| $\kappa$ | Kurtosis of standardised returns | 3 for a normal law |
| $\theta$, $W$ | `vol_floor_quantile`, `vol_floor_quantile_roll_period` | Probability; rows |
| $h$ | Summation horizon of the rolling functions (their `span`) | Rows |
| $J$, $n_J$ | A calendar period of `freq` and its number of observations | |
| $\mathrm{AN}_f$ | Annualisation factor of `freq` | `qis.get_annualization_factor(freq)` |
| $X^{(h)}_t$, $X^{f}_{J}$ | Normalised rolling and calendar sums of $x_t$ | Risk units |
| $\lambda_m$, $N_m$ | Decay and span of the momentum EWM | `momentum_span` |
| $\lambda_L$, $\lambda_S$, $N_L$, $N_S$ | Decays and spans of the long and short filter legs | `long_span`, `short_span` |
| $c_k$ | Filter weight on $x_{t-k}$ | Dimensionless |
| $Q$ | Unit-variance normaliser of the long–short filter | Dimensionless |
| $M_t$, $F_t$ | Momentum signal and long–short filter output | Risk units |
| $y$, $y_0$, $b$ / `loc`, `scale` | Signal, centre and scale of a signal map | $b$ is in signal units for the normal and Laplace maps and in squared signal units for `ExpCDF` |
| $q$, $p_{+}$, $p_{-}$ | `tail_level`, `slope_right`, `slope_left` | Weight levels |
| $d_{+}$, $d_{-}$ | `tail_decay_right`, `tail_decay_left` | Signal units |
| $g(y)$, $\Phi$ | Signal-to-weight map; standard normal CDF | |

The propositions assume a regular sampling grid, returns whose conditional mean is negligible
relative to their volatility, and, where stated, serially uncorrelated standardised returns. The
code assumes nothing: it applies the recursions to whatever rows it receives.

## Methodology

### EWM volatility inside the normalisation

**Definition.** Let $t_0$ be the first row of the input. `compute_ra_returns` calls
`qis.compute_ewm_vol` with `annualize=False`, which runs

$$
\hat\sigma^2_{t}=\lambda\,\hat\sigma^2_{t-1}+(1-\lambda)\,r_t^2,
\qquad
\hat\sigma^2_{t_0}=r_{t_0}^2 .
$$

The seed is the first squared return (`InitType.X0`). If the first row is missing, the recursion
starts at the first finite return $t_1$ from a zero state, so $\hat\sigma^2_{t_1}=(1-\lambda)r_{t_1}^2$.
A missing return inside the sample holds the state (`NanBackfill.FFILL`). An explicit `init_value`
replaces the seed, and $r_{t_0}$ then does not enter the recursion. With `mean_adj_type` other than
`NONE`, $r_t$ is replaced by $r_t$ minus an expanding or EWM mean (same $\lambda$) through $t$, or by
$r_t$ minus the full-sample mean for `INSAMPLE`.

Two optional masks act on the estimate. With `vol_floor_quantile` $\theta$, the variance is floored
at its own rolling $\theta$-quantile over the last $W$ rows (pandas `interpolation="lower"`, at
least $0.2W$ observations), which, the square root being monotone, floors $\hat\sigma_t$ at the
same quantile of itself. With `warmup_period` $n$, the first $n$ finite estimates are set to
missing. Neither is on by default. The EWM estimators, their mean age and effective sample size
are derived in [Exponentially weighted estimators](ewm_estimators.md).

> **Pitfall.** The seed is one observation. The first risk-adjusted return is $r_{t_0+1}/\lvert r_{t_0}\rvert$,
> which can be arbitrarily large, and a leading missing row (the `qis.to_returns` default) makes
> the first estimate smaller still by the factor $\sqrt{1-\lambda}$, about 0.24 at $\lambda=0.94$.
> Use `warmup_period` of at least the span, or discard the first span of output.

### Risk-adjusted returns

**Definition.** The risk-adjusted return implemented by `qis.compute_ra_returns` is

$$
x_t=\sigma_{\mathrm{tgt}}\,\frac{r_t}{\hat\sigma_{t-\delta}},
\qquad \delta=\texttt{weight\_lag}=1 \text{ by default},
$$

with $\sigma_{\mathrm{tgt}}$ equal to `vol_target`, or 1 when it is `None`. The division is taken only
where $\hat\sigma$ is finite and positive; elsewhere $x_t$ is missing. The function returns the
triple $(x_t,\ \sigma_{\mathrm{tgt}}/\hat\sigma_{t-\delta},\ \hat\sigma_t)$: the second element is the
weight dated at the return it scales, already lagged. With `is_log_returns_to_arithmetic=True` the
numerator becomes $e^{r_t}-1$, while $\hat\sigma$ is still estimated on the supplied (log) returns.

**Proposition (unit conditional variance).** Suppose $\mathbb{E}[r_t\mid\mathcal{F}_{t-1}]=0$ and
$\operatorname{Var}(r_t\mid\mathcal{F}_{t-1})=\sigma^2_{t\mid t-1}$. If the scale equals the true
conditional volatility, $\hat\sigma_{t-1}=\sigma_{t\mid t-1}$, then $x_t=r_t/\hat\sigma_{t-1}$ satisfies

$$
\mathbb{E}[x_t\mid\mathcal{F}_{t-1}]=0,
\qquad
\operatorname{Var}(x_t\mid\mathcal{F}_{t-1})=1,
\qquad
\operatorname{Cov}(x_t,x_s)=0\ \ (s<t).
$$

**Proof.** $\hat\sigma_{t-1}$ is $\mathcal{F}_{t-1}$-measurable, so it leaves the conditional
expectation: $\operatorname{Var}(r_t/\hat\sigma_{t-1}\mid\mathcal{F}_{t-1})=\sigma^2_{t\mid t-1}/\hat\sigma^2_{t-1}=1$,
and likewise the conditional mean is zero. For $s<t$, $x_s$ is $\mathcal{F}_{t-1}$-measurable and
$\mathbb{E}[x_tx_s]=\mathbb{E}\big[x_s\,\mathbb{E}[x_t\mid\mathcal{F}_{t-1}]\big]=0$. $\square$

The lag is what makes the proposition possible: $\hat\sigma_{t-1}$ is known before $r_t$. With
`MeanAdjType.NONE` the EWM estimates the conditional second moment, not the variance. The gap is
$\mu^2/\sigma^2=\mathrm{SR}^2/\mathrm{AN}$ per period, about 0.001 for an annual Sharpe ratio of 0.5
on daily data.

**Proposition (estimation noise inflates the realised variance).** Let $r_t=\sigma z_t$ with
constant $\sigma$ and i.i.d. $z_t$, $\mathbb{E}z_t^2=1$, $\mathbb{E}z_t^4=\kappa$, and let
$\hat\sigma^2_{t-1}=\sigma^2Y_{t-1}$ with the stationary EWM $Y_{t-1}=(1-\lambda)\sum_{k\ge0}\lambda^k z^2_{t-1-k}$.
Then

$$
\mathbb{E}[x_t^2]=\mathbb{E}\big[Y_{t-1}^{-1}\big]\ \ge\ 1,
\qquad
\mathbb{E}[x_t^2]\approx 1+(\kappa-1)\frac{1-\lambda}{1+\lambda}=1+\frac{\kappa-1}{N}.
$$

**Proof.** $z_t$ is independent of $Y_{t-1}$, so $\mathbb{E}[x_t^2]=\mathbb{E}[z_t^2]\,\mathbb{E}[1/Y_{t-1}]$.
Since $\mathbb{E}Y=1$ and $1/y$ is convex, Jensen's inequality gives $\mathbb{E}[1/Y]\ge1$. A
second-order expansion of $1/y$ at 1 gives $\mathbb{E}[1/Y]\approx1+\operatorname{Var}(Y)$, and
$\operatorname{Var}(Y)=(1-\lambda)^2(\kappa-1)\sum_k\lambda^{2k}=(\kappa-1)(1-\lambda)/(1+\lambda)$.
Finally $(1-\lambda)/(1+\lambda)=1/N$ for $\lambda=1-2/(N+1)$. $\square$

For normal returns and the default $\lambda=0.94$ the second moment is about 1.062: realised
volatility runs about 3% above target. Fat tails raise the inflation, although the
second-order approximation then overstates it; a volatility floor or a longer span lowers it.

**Proposition (look-ahead bound).** With $\delta=0$ (`weight_lag=0` or `None`) and no floor, every
risk-adjusted return after the seed obeys

$$
\lvert x_t\rvert=\sigma_{\mathrm{tgt}}\frac{\lvert r_t\rvert}{\hat\sigma_t}\le\frac{\sigma_{\mathrm{tgt}}}{\sqrt{1-\lambda}}=\sigma_{\mathrm{tgt}}\sqrt{\frac{N+1}{2}} .
$$

**Proof.** $\hat\sigma^2_t=\lambda\hat\sigma^2_{t-1}+(1-\lambda)r_t^2\ge(1-\lambda)r_t^2$. At the seed,
$\lvert x_{t_0}\rvert=\sigma_{\mathrm{tgt}}$. $\square$

> **Pitfall.** `weight_lag=0` divides a return by a volatility that already contains it. The
> result is not tradeable and has artificially thin tails: at $\lambda=0.94$ no normalised return
> can exceed 4.08, whatever the size of the move. Keep the default lag of 1.

### Volatility targeting

**Definition.** The volatility-targeting weight decided at $t$ and held over $(t,t+1]$ is

$$
w^{*}_t=\frac{\sigma_{\mathrm{tgt}}}{\hat\sigma_t},
\qquad
r^{\mathrm{vt}}_{t+1}=w^{*}_t\,r_{t+1}=x_{t+1}\quad(\delta=1).
$$

With rebalancing every period, no costs and a zero cash return, the output of `compute_ra_returns`
with `vol_target` set is therefore the return series of the volatility-targeted position. The
target is per period: an annual target $\sigma_{\mathrm{ann}}$ is passed as
$\sigma_{\mathrm{ann}}/\sqrt{\mathrm{AN}}$, for example $0.15/\sqrt{252}$ on business days.

The conditional volatility of the targeted return is
$\sigma_{\mathrm{tgt}}\,\sigma_{t+1\mid t}/\hat\sigma_t$. The unmanaged asset carries
$\sigma_{t+1\mid t}$ itself, which moves with the volatility regime; the targeted position moves
only with the ratio of true to estimated volatility. That ratio departs from one for two reasons:
estimation noise, which inflates realised variance by about $(\kappa-1)/N$ as shown above, and
the lag of the EWM after a volatility jump. The mean age of the EWM is $\lambda/(1-\lambda)=(N-1)/2$
periods, 15.7 days at $\lambda=0.94$, so after a jump the position stays over-levered for several
weeks. The worked example shows both effects.

`compute_ra_returns` has no leverage cap: $w^{*}_t$ grows without bound as $\hat\sigma_t$ falls. The
only bound in the code is indirect: with a floor, $w^{*}_t\le\sigma_{\mathrm{tgt}}/\hat\sigma^{\mathrm{floor}}_t$.
A cap must be applied to the weights by the caller before execution. Executing the weights, with
held units, implementation lags and costs, is the job of `qis.backtest_model_portfolio`, described
in [Portfolio backtesting](portfolio_backtesting.md); turnover of volatility-scaled weights is
treated in [Two-sided turnover conventions](turnover_conventions.md).

> **Insight.** Volatility targeting changes the mean return only through timing:
> $\mathbb{E}[w^{*}_t r_{t+1}]=\mathbb{E}[w^{*}_t]\,\mathbb{E}[r_{t+1}]+\operatorname{Cov}(w^{*}_t,r_{t+1})$.
> With constant volatility the covariance vanishes and the strategy is a constant-leverage copy
> of the asset, with the same Sharpe ratio. A Sharpe-ratio gain requires that returns be
> relatively poor when volatility is high.

The empirical record is specific on this point. Moreira and Muir (2017) scale factor returns by
the inverse of the previous month's realised *variance*, not volatility, and report positive
alphas against the unmanaged factors for many equity factors. Harvey et al. (2018) find that
volatility targeting raises Sharpe ratios for risk assets such as equities and credit, which they
link to the leverage effect, but has a negligible effect on the Sharpe ratio of bonds,
currencies and commodities. They also find that it reduces the likelihood of extreme returns
across asset classes. Neither result is a property of the construction: both are sample
evidence, before the costs of the extra turnover and subject to leverage limits.

### Normalised sums of risk-adjusted returns

Three functions aggregate over a horizon. They differ in whether they normalise before or after
summing, and in what they divide by.

**Definition (normalise, then sum).** `qis.compute_sum_rolling_ra_returns` computes $x_t$ with
`ewm_lambda` and sums it over a rolling window of $h$ rows (its `span`):

$$
X^{(h)}_t=\frac{1}{\sqrt{h}}\sum_{j=0}^{h-1}x_{t-j}\quad(\texttt{is\_norm=True}),
\qquad X^{(1)}_t=x_t .
$$

**Definition (normalise, then sum by calendar period).** `qis.compute_sum_freq_ra_returns`
computes $x_t$ (its `span` is the volatility span) and, for `freq` other than `'B'` or `'D'`, sums
it within each calendar period $J$ of `freq`:

$$
X^{f}_{J}=\frac{1}{\sqrt{\mathrm{AN}_f}}\sum_{t\in J}x_t\quad(\texttt{is\_norm=True}).
$$

For `'B'` and `'D'` it returns $x_t$ unchanged.

**Definition (sum, then normalise).** `qis.compute_rolling_ra_returns` with $h>1$ sums the returns
first, $R^{(h)}_t=\sum_{j=0}^{h-1}r_{t-j}$, estimates the EWM volatility $\hat\sigma^{(h)}_t$ of the
overlapping sums with decay $\lambda_h=1-2/(h+1)$, and returns

$$
\sigma_{\mathrm{tgt}}\,\frac{e^{R^{(h)}_t}-1}{\hat\sigma^{(h)}_{t-1}} .
$$

The exponential map applies with the default `is_log_returns_to_arithmetic=True`, which assumes
log returns. With $h=1$ it returns $x_t$ with decay `ewm_lambda_eod`. The weight lag is one row,
not $h$ rows, and the first finite sum starts the variance from a zero state.

**Proposition (square root of the horizon).** If $x_{t-h+1},\ldots,x_t$ are uncorrelated with unit
variance, then $\operatorname{Var}\big(\sum_{j=0}^{h-1}x_{t-j}\big)=h$, so $X^{(h)}_t$ has unit variance.
For overlapping windows, $\operatorname{Corr}\big(X^{(h)}_t,X^{(h)}_{t+j}\big)=(h-j)/h$ for $0\le j<h$.

**Proof.** The variance of a sum is the sum of all covariances; only the $h$ unit variances
survive. Two windows $j$ rows apart share $h-j$ terms, so their covariance is $(h-j)/h$ after
normalisation by $\sqrt{h}\sqrt{h}$. $\square$

The first proposition of this chapter supplies the premise: correctly scaled risk-adjusted
returns are serially uncorrelated with unit variance. The second statement is why overlapping
sums need autocorrelation-robust inference; see [Serial dependence and autocorrelation](serial_dependence.md)
and [Regression and HAC inference](regression_and_hac.md).

> **Pitfall.** `compute_sum_freq_ra_returns` divides by $\sqrt{\mathrm{AN}_f}$, the number of periods of
> `freq` *per year*, not by $\sqrt{n_J}$, the number of observations *per period*. For
> unit-variance daily terms, $\operatorname{Var}(X^{f}_{J})=n_J/\mathrm{AN}_f\approx 252/\mathrm{AN}_f^2$, a
> standard deviation of about 0.31 weekly, 1.32 monthly and 3.97 quarterly. For unit variance, call
> it with `is_norm=False` and divide by the square root of the per-period observation count.

The word `span` also changes meaning: it is the volatility span in `compute_ra_returns` and
`compute_sum_freq_ra_returns`, but the summation horizon $h$ in `compute_sum_rolling_ra_returns` and
`compute_rolling_ra_returns`, where the volatility decay is `ewm_lambda`, `ewm_lambda_eod`, or
derived from $h$.

### Momentum signals from EWM filters on risk-adjusted returns

Time-series momentum takes a long position after positive past returns and a short one after
negative returns. Moskowitz, Ooi and Pedersen (2012) document it across futures markets, using
the sign of the past twelve-month excess return and sizing each position by the inverse of an
ex-ante EWM volatility. qis builds smooth versions of the signal on risk-adjusted returns, so
that one signal scale applies to every asset.

**Identity (EWM unit-variance load).** For $\lambda=1-2/(N+1)$,

$$
\sqrt{\frac{1+\lambda}{1-\lambda}}=\sqrt{N}.
$$

**Proof.** $1-\lambda=2/(N+1)$ and $1+\lambda=2N/(N+1)$; their ratio is $N$. $\square$

**Definition (EWM momentum).** `qis.compute_ewm_ra_returns_momentum` computes $x_t$ with
$\sigma_{\mathrm{tgt}}=1$, decay $1-2/(\texttt{vol\_span}+1)$ and lag `weight_shift`, then

$$
m_t=\lambda_m m_{t-1}+(1-\lambda_m)\,x_t,
\qquad
M_t=\sqrt{N_m}\;m_t=\sum_{k\ge0}c_k\,x_{t-k},
\qquad
c_k=\sqrt{1-\lambda_m^2}\;\lambda_m^{k},
$$

with $m=0$ before the first finite $x_t$. Defaults: `momentum_span=63`, `vol_span=31`,
`weight_shift=1`; `momentum_lambda` and `vol_lambda` override the spans.

**Definition (long–short filter).** `qis.compute_ewm_long_short` forms the difference of two EWMs,
each rescaled so that the result has unit variance. On the input $x_t$ it computes

$$
F_t=\sum_{k\ge0}c_k\,x_{t-k},
\qquad
c_k=\frac{\lambda_L^{k}-\lambda_S^{k}}{Q},
\qquad
Q^2=\frac{1}{1-\lambda_L^2}+\frac{1}{1-\lambda_S^2}-\frac{2}{1-\lambda_L\lambda_S}.
$$

In code the long leg is $\sqrt{N_L}\,\mathrm{EWM}_{\lambda_L}(x)/(\sqrt{1-\lambda_L^2}\,Q)$, and
$\sqrt{N_L}(1-\lambda_L)/\sqrt{1-\lambda_L^2}=1$, so each leg contributes $\sum_k\lambda^k x_{t-k}/Q$.
With `short_span=None` the output is the single-leg filter $M_t$ above, with $\lambda_L$ in place of
$\lambda_m$. `qis.compute_ewm_long_short_filter` validates the spans, applies the kernel, and blanks
the first `warmup_period` finite outputs. `qis.compute_ewm_long_short_filtered_ra_returns` first
normalises the returns with `compute_ra_returns(span=vol_span, vol_target=None, weight_lag=weight_lag)`
and then applies the filter. Defaults are $N_L=63$, $N_S=5$, `vol_span=31`, `warmup_period=21`.

**Proposition (unit-variance filters).** If $x_t$ is serially uncorrelated with unit variance,
then $\operatorname{Var}(M_t)=\operatorname{Var}(F_t)=1$ in the stationary limit, because
$\sum_{k\ge0}c_k^2=1$ for both kernels.

**Proof.** For the single leg, $\sum_k(1-\lambda_m^2)\lambda_m^{2k}=1$. For the long–short kernel,
expanding the square and summing three geometric series gives
$\sum_k(\lambda_L^k-\lambda_S^k)^2=\frac{1}{1-\lambda_L^2}+\frac{1}{1-\lambda_S^2}-\frac{2}{1-\lambda_L\lambda_S}=Q^2$.
The variance of a weighted sum of uncorrelated unit-variance terms is the sum of squared
weights. $\square$

Two properties follow from the kernel. First, $c_0=(1-1)/Q=0$: the two-leg output at $t$ does not
load on $x_t$, so it is known one row early. Second, $c_k$ is hump-shaped with its peak at

$$
k^{*}=\frac{\ln(\ln\lambda_S/\ln\lambda_L)}{\ln(\lambda_L/\lambda_S)},
$$

about 6.8 rows for spans 63 and 5; the filter is a band-pass that ignores the latest return and
the distant past. The spans are validated: each must be at least 1, and `short_span` must be
strictly less than `long_span`, since equal spans make $Q=0$.

**Proposition (drift gain).** If $x_t$ has constant mean $\mu_x$, then

$$
\mathbb{E}[M_t]=\sqrt{N_m}\,\mu_x,
\qquad
\mathbb{E}[F_t]=\frac{N_L-N_S}{2Q}\,\mu_x .
$$

**Proof.** $\sum_k\lambda^k=1/(1-\lambda)=(N+1)/2$. For the single leg,
$\sqrt{1-\lambda_m^2}\,(N_m+1)/2=\sqrt{N_m}$ by the identity above; for the long–short kernel,
$\big((N_L+1)/2-(N_S+1)/2\big)/Q$. $\square$

> **Insight.** $\mu_x$ is the per-period Sharpe ratio of the asset. An annual Sharpe ratio of 0.5
> on daily data gives $\mu_x\approx0.031$, and the default filters turn it into a mean signal of
> about 0.25 (gain $\sqrt{63}\approx7.9$ for $M_t$, 8.2 for $F_t$ with $Q\approx3.52$), against a
> unit standard deviation. Even a strong trend moves the signal by a quarter of its noise, which
> is why momentum signals flip sign often and why the map from signal to weight matters.

`compute_ewm_long_short_filtered_ra_returns` lags only the volatility normaliser through
`weight_lag`; it does not shift its output. The two-leg output at $t$ uses $x$ through $t-1$, and
the single-leg output uses $x_t$; either is applied over $(t,t+1]$.

### Signal-to-weight maps

`qis.map_signal_to_weight` maps a signal $y$ to a weight through one of the three members of
`qis.SignalMapType`. With $u=(y-y_0)/b$:

**Definition (`NormalCDF` and `LaplaceCDF`).**

$$
g_{\mathrm{N}}(y)=2\,\Phi(u)-1,
\qquad
g_{\mathrm{L}}(y)=\operatorname{sign}(u)\big(1-e^{-\lvert u\rvert}\big).
$$

Both are odd about $y_0$, bounded in $(-1,1)$ and linear near the centre, with slopes
$\sqrt{2/\pi}/b\approx0.80/b$ and $1/b$. The Laplace map approaches its bound exponentially; the
normal map approaches it faster, like a Gaussian tail. They ignore `tail_level`, the slopes and
the tail decays.

**Definition (`ExpCDF`).** With tail level $q$, anchor levels $p_{+}$ for $y\ge y_0$ and $p_{-}$ for
$y<y_0$, the code sets $s_{\pm}=1.5625\,b/\ln\big(q/(q-p_{\pm})\big)$ and returns
$g(y)=\pm q\big(1-e^{-(y-y_0)^2/s_{\pm}}\big)$, which is

$$
g_{\mathrm{E}}(y)=\pm\,q\Big[1-\Big(1-\frac{p_{\pm}}{q}\Big)^{v^2}\Big],
\qquad
v=\frac{y-y_0}{1.25\sqrt{b}} .
$$

It requires $q>p_{+}$ and $q>p_{-}$ and raises `ValueError` otherwise.

**Identity (anchor).** $g_{\mathrm{E}}(y_0\pm1.25\sqrt{b})=\pm p_{\pm}$, and $g_{\mathrm{E}}\to\pm q$ as $y\to\pm\infty$.

**Proof.** Substituting $s_{\pm}$, $e^{-(y-y_0)^2/s_{\pm}}=\big((q-p_{\pm})/q\big)^{(y-y_0)^2/(1.5625\,b)}$;
at $\lvert y-y_0\rvert=1.25\sqrt{b}$ the exponent is 1 and $g=\pm p_{\pm}$. $\square$

The constant $1.5625=1.25^2$ is not documented in the source; its only effect is to place the
anchor at $1.25\sqrt{b}$. Despite their names, `slope_right` and `slope_left` are not derivatives:
they are the weights reached at the anchor. The map is quadratic near the centre,
$g\approx q\ln\big(q/(q-p)\big)v^2$, so it has zero slope at $y_0$ and damps small signals, unlike
the normal and Laplace maps. `scale` enters under a square root, so it acts as a variance.

**Definition (`ExpCDF` tail treatment).** If both `tail_decay_right` $d_{+}$ and `tail_decay_left`
$d_{-}$ are given, the weight is multiplied by

$$
\begin{aligned}
&\exp\!\big(-(y-q-\max(y_0,0))/d_{+}\big) &&\text{if } y>q+\max(y_0,0),\\
&\exp\!\big((y+q-\min(y_0,0))/d_{-}\big) &&\text{if } y<-q+\min(y_0,0),
\end{aligned}
$$

and by 1 in between. Beyond the threshold the weight decays to zero: extreme signals are faded.
The tail level $q$ plays two roles here, the weight cap and the signal threshold, so the two
are not independent. If only one decay is given, the tail treatment is skipped without warning.

### Returns transforms and paired samples

`qis.compute_returns_transform` dispatches over `qis.ReturnsTransform`:

- `ROLLING_RA_RETURNS` returns `compute_rolling_ra_returns(returns, span=rolling_ra_returns_span, weight_shift=1)`,
  the sum-then-normalise transform with $h=31$ by default and the log-to-simple map on.
- `EWMA_RETURNS_MOMENTUM` returns `compute_ewm_ra_returns_momentum(returns, momentum_span, vol_span, weight_shift=1)`
  with defaults 31 and 33. These differ from the defaults of the underlying function (63 and 31).

Any other value raises `TypeError`.

`qis.get_paired_rareturns_signals` aligns risk-adjusted returns with a signal for predictive
diagnostics. With `is_nonoverlapping=True` it pairs $X^{f}_{J}$ from `compute_sum_freq_ra_returns`
(with the normalisation of the pitfall above) with the last signal value of the previous period,
`signal.resample(freq).last().shift(1)`. With `is_nonoverlapping=False` it pairs $X^{(h)}_t$ with
`signal.shift(1)`, the signal at $t-1$. `is_mean_adjust_returns=True` subtracts an expanding mean,
which is point in time.

> **Pitfall.** In the overlapping mode the window of $X^{(h)}_t$ covers $(t-h,t]$, so a signal
> dated $t-1$ has already seen $h-1$ of its $h$ returns. A momentum signal paired this way
> "predicts" returns it contains. For a forward-looking pair, lag the signal by $h$ rows. The
> non-overlapping mode is correctly aligned.

Diagnostics for such pairs, including the information coefficient, are the subject of
[Signal diagnostics](signal_diagnostics.md).

## Worked example

### A hand-checkable normalisation

Five daily returns of 1%, 7%, −5%, 5% and −1% with span 3, so $\lambda=0.5$ and each variance is
the average of the previous variance and the new squared return. The seed is $0.01^2$, and the
volatility path is 1%, 5%, 5%, 5% and $\sqrt{0.0013}\approx3.61\%$. Dividing each return by the
previous volatility gives 7, −1, 1 and −0.2 risk units. The first value, 7, is the one-observation
seed at work. With a per-period target of 2% the weights are 2 and then 0.4. Without the lag the
normalised returns are 1, 1.4, −1, 1 and −0.28, all within the look-ahead bound $\sqrt{2}$.

```python
import numpy as np
import pandas as pd
import qis

dates = pd.bdate_range('2024-01-01', periods=5)
returns = pd.Series([0.01, 0.07, -0.05, 0.05, -0.01], index=dates, name='asset')

# span 3 gives lambda = 1 - 2/4 = 0.5
ra, weights, vol = qis.compute_ra_returns(returns=returns, span=3)

# hand recursion: seed with the first squared return, then average old variance and new square
hand_var = [0.01 ** 2]
for r in returns.iloc[1:]:
    hand_var.append(0.5 * hand_var[-1] + 0.5 * r ** 2)
np.testing.assert_allclose(vol, np.sqrt(hand_var), atol=1e-15)
np.testing.assert_allclose(vol, [0.01, 0.05, 0.05, 0.05, np.sqrt(0.0013)], atol=1e-15)
assert np.isnan(ra.iloc[0])
np.testing.assert_allclose(ra.iloc[1:], [7.0, -1.0, 1.0, -0.2], atol=1e-12)

# a per-period target of 2%: weights 0.02 / vol(t-1), output 0.02 * x
ra_2pct, weights_2pct, _ = qis.compute_ra_returns(returns=returns, span=3, vol_target=0.02)
np.testing.assert_allclose(weights_2pct.iloc[1:], [2.0, 0.4, 0.4, 0.4], atol=1e-12)
np.testing.assert_allclose(ra_2pct.iloc[1:], [0.14, -0.02, 0.02, -0.004], atol=1e-12)

# weight_lag=0 divides by a volatility that already contains r_t: bounded by 1/sqrt(1 - 0.5)
ra_lookahead, _, _ = qis.compute_ra_returns(returns=returns, span=3, weight_lag=0)
np.testing.assert_allclose(ra_lookahead, [1.0, 1.4, -1.0, 1.0, -0.01 / np.sqrt(0.0013)],
                           atol=1e-12)
assert np.abs(ra_lookahead).max() <= np.sqrt(2.0)

# normalised two-row rolling sums of the risk-adjusted returns: (7 - 1, -1 + 1, 1 - 0.2) / sqrt(2)
summed = qis.compute_sum_rolling_ra_returns(returns=returns, span=2, ewm_lambda=0.5,
                                            is_log_returns_to_arithmetic=False)
np.testing.assert_allclose(summed.iloc[2:], np.array([6.0, 0.0, 0.8]) / np.sqrt(2.0), atol=1e-12)
```

### Volatility targeting on the frozen synthetic universe

The synthetic instruments have constant volatility, so a volatility regime is imposed by hand.
The daily returns of `SEQ_US` from 2005 to 2014 are multiplied by 0.6 in 2005–2007, by 2.5 in
2008, by 1.5 in 2009 and by 1 afterwards. The resulting asset has yearly realised volatilities
from 9.6% (2006) to 41.8% (2008), and 20.1% over the full sample, all annualised with
$\mathrm{AN}=252$. Targeting 15% per annum with the default $\lambda=0.94$ and a 21-day warm-up
mask, the yearly realised volatilities of the targeted position lie between 14.9% (2010) and
17.2% (2008), and the full-sample figure is 15.6%. The largest yearly deviation from target falls
from 26.8 to 2.2 percentage points. Leverage ranges from 0.29 (December 2008) to 2.07. Without
the warm-up mask, the one-observation seed alone sets a leverage of 3.80 on the first day.

The two residual errors are those of the Methodology section. The 2008 overshoot is the lag: the
volatility jumps fourfold on 1 January, the position enters 2008 at the calm-regime leverage of
about 1.5, and the EWM takes weeks to catch up. In the constant-volatility
years 2010–2014 the targeted volatility is 15.4%, against $15\%\times(1+2/N)^{1/2}\approx15.5\%$
predicted by the estimation-noise proposition.

```python
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(start='2005-01-03', end='2014-12-31', seed=20260725,
                                       apply_quirks=False)
base = qis.to_returns(prices=universe.prices['SEQ_US'], is_log_returns=False, drop_first=True)
year = base.index.year
regime_scale = np.select([year <= 2007, year == 2008, year == 2009], [0.6, 2.5, 1.5], default=1.0)
asset = (base * regime_scale).rename('REGIME')

AN, target, lam = 252, 0.15, 0.94
vt_returns, vt_weights, ewm_vol = qis.compute_ra_returns(
    returns=asset, ewm_lambda=lam, vol_target=target / np.sqrt(AN), warmup_period=21)


def yearly_vol(x: pd.Series) -> pd.Series:
    x = x.dropna()
    return x.groupby(x.index.year).std() * np.sqrt(AN)


raw_by_year, vt_by_year = yearly_vol(asset), yearly_vol(vt_returns)
np.testing.assert_allclose([raw_by_year.min(), raw_by_year.max()], [0.096, 0.418], atol=0.001)
np.testing.assert_allclose([vt_by_year.min(), vt_by_year.max()], [0.149, 0.172], atol=0.001)
np.testing.assert_allclose([asset.std() * np.sqrt(AN), vt_returns.std() * np.sqrt(AN)],
                           [0.201, 0.156], atol=0.001)
raw_dev, vt_dev = (raw_by_year - target).abs().max(), (vt_by_year - target).abs().max()
np.testing.assert_allclose([raw_dev, vt_dev], [0.268, 0.022], atol=0.001)
assert vt_dev < 0.1 * raw_dev
np.testing.assert_allclose([vt_weights.min(), vt_weights.max()], [0.29, 2.07], atol=0.005)
np.testing.assert_allclose(vt_weights.loc['2008-01-01'], 1.54, atol=0.005)  # calm-regime leverage
_, seed_weights, _ = qis.compute_ra_returns(returns=asset, ewm_lambda=lam,
                                            vol_target=target / np.sqrt(AN))
assert seed_weights.idxmax() == seed_weights.first_valid_index()
np.testing.assert_allclose(seed_weights.max(), 3.80, atol=0.005)

# independent numpy recursion: seed r_0^2, lag one row, 21 masked estimates
x = asset.to_numpy()
var = np.empty_like(x)
var[0] = x[0] ** 2
for t in range(1, len(x)):
    var[t] = lam * var[t - 1] + (1.0 - lam) * x[t] ** 2
direct = x[1:] * (target / np.sqrt(AN)) / np.sqrt(var[:-1])
assert vt_returns.iloc[:22].isna().all()
np.testing.assert_allclose(vt_returns.iloc[22:], direct[21:], rtol=1e-12)

# constant-volatility years: realised vol exceeds target by about 1/N, N = 2/(1 - lam) - 1
calm = vt_returns[vt_returns.index.year >= 2010].std() * np.sqrt(AN)
predicted = target * np.sqrt(1.0 + 2.0 * (1.0 - lam) / (1.0 + lam))
np.testing.assert_allclose([calm, predicted], [0.154, 0.155], atol=0.001)
```

The same numbers come out of the backtester. Target weights dated $t$ are $\sigma_{\mathrm{tgt}}/\hat\sigma_t$,
the third output inverted, not the already lagged second output. Executed at the close of $t$
with no implementation lag, no costs and daily rebalancing, the portfolio's daily returns equal
the risk-adjusted returns to machine precision.

```python
nav = pd.concat([pd.Series([100.0], index=universe.prices.index[:1]), 100.0 * (1.0 + asset).cumprod()])
target_weights = (target / np.sqrt(AN) / ewm_vol).to_frame('REGIME')
portfolio = qis.backtest_model_portfolio(prices=nav.to_frame('REGIME'), weights=target_weights,
                                         weight_implementation_lag=0)
portfolio_returns = portfolio.get_portfolio_nav().pct_change()
both = pd.concat([portfolio_returns, vt_returns], axis=1).dropna()
assert len(both) == len(vt_returns.dropna())
np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-12)
```

Across all ten clean instruments, which have constant volatility and near-normal returns, the
pooled mean of $x_t^2$ with a unit target is 1.066, against $1+2/N\approx1.062$: the normalised
returns have unit variance up to the predicted estimation-noise inflation.

```python
panel = qis.to_returns(prices=universe.prices, is_log_returns=False, drop_first=True)
x_panel, _, _ = qis.compute_ra_returns(returns=panel, ewm_lambda=lam, warmup_period=21)
pooled = float(np.nanmean(x_panel.to_numpy() ** 2))
np.testing.assert_allclose([pooled, 1.0 + 2.0 / (2.0 / (1.0 - lam) - 1.0)], [1.066, 1.062],
                           atol=0.001)
```

### Unit-variance filters and signal maps

The filter weights are read off an impulse response. A unit impulse at the second row (the
first row seeds the recursion and does not enter it) returns $c_k$ at row $k+1$. For spans 63 and
5, $Q\approx3.522$, $c_0=0$, the peak is at lag 7, and the squared weights sum to 1. The
single-leg filter and the momentum signal share the kernel $\sqrt{1-\lambda^2}\,\lambda^k$ with
$c_0=\sqrt{1-\lambda^2}\approx0.248$ for span 63.

```python
n = 4000
impulse = pd.Series(np.zeros(n), index=pd.bdate_range('2000-01-03', periods=n))
impulse.iloc[1] = 1.0
kernel = qis.compute_ewm_long_short_filter(data=impulse, long_span=63, short_span=5,
                                           warmup_period=None).to_numpy()[1:]
lam_l, lam_s = 1.0 - 2.0 / 64.0, 1.0 - 2.0 / 6.0
q_norm = np.sqrt(1 / (1 - lam_l ** 2) + 1 / (1 - lam_s ** 2) - 2 / (1 - lam_l * lam_s))
lags = np.arange(n - 1)
np.testing.assert_allclose(kernel, (lam_l ** lags - lam_s ** lags) / q_norm, atol=1e-14)
np.testing.assert_allclose([q_norm, np.sum(kernel ** 2)], [3.522, 1.0], atol=1e-3)
assert abs(np.sum(kernel ** 2) - 1.0) < 1e-12 and abs(kernel[0]) < 1e-15
assert np.argmax(kernel) == 7
np.testing.assert_allclose(np.sum(kernel) * 0.5 / np.sqrt(252), 0.259, atol=1e-3)  # drift gain

single = qis.compute_ewm_long_short_filter(data=impulse, long_span=63, short_span=None,
                                           warmup_period=None).to_numpy()[1:]
np.testing.assert_allclose(single, np.sqrt(1 - lam_l ** 2) * lam_l ** lags, atol=1e-14)
assert abs(np.sum(single ** 2) - 1.0) < 1e-12

# the momentum signal of risk-adjusted returns uses the same single-leg kernel
x_impulse = pd.Series(np.r_[np.nan, 1.0, np.zeros(n - 2)], index=impulse.index)
momentum = qis.ewm_recursion(a=x_impulse.to_numpy(), init_value=0.0, ewm_lambda=lam_l,
                             is_unit_vol_scaling=True)
np.testing.assert_allclose(momentum[1:], single, atol=1e-14)
```

The maps are checked against their closed forms: the normal map against the error function, the
Laplace map against $1-e^{-\lvert u\rvert}$, and `ExpCDF` against its anchor, where a signal of
$1.25\sqrt{b}$ returns the anchor level. With the default $q=1$, $p_{\pm}=0.5$ and $b=4$, a signal
of 2.5 maps to 0.5 and a signal of 5 to $1-0.5^4=0.9375$. With both tail decays set to 1, a signal
of 3 is faded from 0.98 to 0.13.

```python
from math import erf

signals = pd.DataFrame({'s': [-3.0, -1.25, 0.0, 1.25, 2.5, 3.0, 5.0]})
y = signals['s'].to_numpy()
normal = qis.map_signal_to_weight(signals, signal_map_type=qis.SignalMapType.NormalCDF, scale=2.0)
np.testing.assert_allclose(normal['s'], [erf(v / (2.0 * np.sqrt(2.0))) for v in y], atol=1e-12)
laplace = qis.map_signal_to_weight(signals, signal_map_type=qis.SignalMapType.LaplaceCDF)
np.testing.assert_allclose(laplace['s'], np.sign(y) * (1.0 - np.exp(-np.abs(y))), atol=1e-12)

exp_map = qis.map_signal_to_weight(signals, signal_map_type=qis.SignalMapType.ExpCDF, scale=4.0)
np.testing.assert_allclose(exp_map['s'], np.sign(y) * (1.0 - 0.5 ** ((y / 2.5) ** 2)), atol=1e-12)
np.testing.assert_allclose(exp_map['s'].iloc[[4, 6]], [0.5, 0.9375], atol=1e-12)

asymmetric = qis.map_signal_to_weight(signals, signal_map_type=qis.SignalMapType.ExpCDF,
                                      slope_right=0.8, slope_left=0.2)
np.testing.assert_allclose(asymmetric['s'].iloc[[1, 3]], [-0.2, 0.8], atol=1e-12)

faded = qis.map_signal_to_weight(signals, signal_map_type=qis.SignalMapType.ExpCDF,
                                 tail_decay_right=1.0, tail_decay_left=1.0)
plain = qis.map_signal_to_weight(signals, signal_map_type=qis.SignalMapType.ExpCDF)
np.testing.assert_allclose(faded['s'].iloc[5], plain['s'].iloc[5] * np.exp(-2.0), atol=1e-12)
np.testing.assert_allclose([plain['s'].iloc[5], faded['s'].iloc[5]], [0.98, 0.13], atol=0.005)
```

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| EWM volatility | $\hat\sigma^2_t=\lambda\hat\sigma^2_{t-1}+(1-\lambda)r_t^2$, seed $r_{t_0}^2$ | `qis.compute_ewm_vol(data, ewm_lambda=0.94, annualize=False)` |
| Risk-adjusted return | $x_t=\sigma_{\mathrm{tgt}}r_t/\hat\sigma_{t-\delta}$ | `qis.compute_ra_returns(returns, span=None, ewm_lambda=0.94, vol_target=None, weight_lag=1)` returns `(ra_returns, weights, ewm_vol)` |
| Volatility floor | $\hat\sigma_t\ge$ rolling $\theta$-quantile | `vol_floor_quantile=None`, `vol_floor_quantile_roll_period=1300` |
| Volatility-targeting weight | $w^{*}_t=\sigma_{\mathrm{tgt}}/\hat\sigma_t$ | `vol_target / ewm_vol` from `compute_ra_returns`; executed by `qis.backtest_model_portfolio` |
| Normalise-then-sum, rolling | $X^{(h)}_t$ | `qis.compute_sum_rolling_ra_returns(returns, span=1, ewm_lambda=0.94, is_norm=True, is_log_returns_to_arithmetic=True)` |
| Normalise-then-sum, calendar | $X^{f}_{J}$, divided by $\sqrt{\mathrm{AN}_f}$ | `qis.compute_sum_freq_ra_returns(returns, freq='B', span=None, ewm_lambda=0.94, is_norm=True, is_log_returns_to_arithmetic=True)` |
| Sum-then-normalise | $(e^{R^{(h)}_t}-1)/\hat\sigma^{(h)}_{t-1}$ | `qis.compute_rolling_ra_returns(returns, span=1, ewm_lambda_eod=0.94, is_log_returns_to_arithmetic=True)` |
| EWM momentum | $M_t=\sqrt{N_m}\,m_t$ | `qis.compute_ewm_ra_returns_momentum(returns, momentum_span=63, vol_span=31, weight_shift=1)` |
| Long–short kernel | $F_t$ on an array | `qis.compute_ewm_long_short(a, init_value, long_span=63, short_span=5)` |
| Long–short filter | $F_t$ with validation and warm-up | `qis.compute_ewm_long_short_filter(data, long_span=63, short_span=5, warmup_period=21)` |
| Filtered risk-adjusted returns | $F_t$ applied to $x_t$ | `qis.compute_ewm_long_short_filtered_ra_returns(returns, vol_span=31, long_span=63, short_span=5, warmup_period=21, weight_lag=1)` |
| Signal maps | $g_{\mathrm{N}}$, $g_{\mathrm{L}}$, $g_{\mathrm{E}}$ | `qis.map_signal_to_weight(signals, signal_map_type=SignalMapType.NormalCDF, loc=0.0, scale=1.0, tail_level=1.0, slope_right=0.5, slope_left=0.5)` |
| Map choice | `NormalCDF`, `LaplaceCDF`, `ExpCDF` | `qis.SignalMapType` |
| Returns transform | dispatch | `qis.compute_returns_transform(returns, returns_transform=ReturnsTransform.ROLLING_RA_RETURNS, momentum_span=31, vol_span=33, rolling_ra_returns_span=31)` |
| Transform choice | `ROLLING_RA_RETURNS`, `EWMA_RETURNS_MOMENTUM` | `qis.ReturnsTransform` |
| Returns paired with signals | $X^{f}_{J}$ or $X^{(h)}_t$ against a lagged signal | `qis.get_paired_rareturns_signals(returns, signal, freq='BQ', span=63, is_nonoverlapping=True, ra_returns_ewm_vol_lambda=0.94, is_mean_adjust_returns=False)` |

The functions live in
[ra_returns.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ra_returns.py)
and the EWM recursions in
[ewm.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py).
API pages: {doc}`compute_ra_returns <api/generated/qis.compute_ra_returns>`,
{doc}`compute_ewm_long_short_filtered_ra_returns <api/generated/qis.compute_ewm_long_short_filtered_ra_returns>`
and {doc}`map_signal_to_weight <api/generated/qis.map_signal_to_weight>`.

Contract details:

- `compute_ra_returns` never annualises. Both branches of its `vol_target` test set
  `annualize=False`, so `vol_target=0.15` on daily returns targets 15% *per day*.
- `vol_target=None` is a unit target, not "no scaling": the output is in risk units.
- The returned `weights` are already lagged by `weight_lag`; the target weight to execute at $t$ is
  `vol_target / ewm_vol`. Passing the lagged weights to a backtester that lags again delays the
  position twice.
- `span` overrides `ewm_lambda` and may be an array with one entry per column. A DataFrame is
  returned with its original column order.
- `compute_ewm_long_short` is a numba kernel on arrays that assumes validated spans and needs an
  explicit `init_value`; `compute_ewm_long_short_filter` is the validated wrapper for pandas input.
- The recursion is seeded on the first row, so a finite first observation is stored as the seed
  and does not enter the filter. The risk-adjusted inputs have a missing first row, so the issue
  does not arise inside `compute_ewm_long_short_filtered_ra_returns`.
- `qis.SignalAggType` belongs to `qis.ewm_xy_convolution` and is not used by these functions.

The example
[vol_target_and_trend.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/vol_target_and_trend.py)
sweeps volatility-targeting and trend strategies over spans through helpers in
[qis_delta1.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/strats/qis_delta1.py).
Its volatility-target strategy divides an annualised EWM volatility of log returns into the annual
target and applies the result to simple returns with a one-row lag; this reproduces
`compute_ra_returns(log_returns, span=vol_span, vol_target=vol_target/sqrt(vol_af), is_log_returns_to_arithmetic=True)`.
It annualises with `vol_af=260`, whereas qis annualises business-day statistics with 252. Its trend
strategy multiplies a unit-variance EWM signal of risk-adjusted returns by the inverse volatility,
the continuous analogue of the volatility-scaled sign of Moskowitz, Ooi and Pedersen (2012).
[optimal_leverage.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/optimal_leverage.py)
is a closed-form mean-variance illustration of leverage and beta targets and does not call the
functions of this chapter.

## Interpretation and limitations

- **Unit variance is conditional on a correct scale.** The EWM tracks volatility with a lag and
  with noise. Expect realised variance of about $1+(\kappa-1)/N$ in calm periods and overshoots
  after volatility jumps; no lag choice removes both.
- **Second moment, not variance.** With `MeanAdjType.NONE` a persistent drift remains in $x_t$. That
  is intended for momentum, where the drift is the signal, but it means $x_t$ is not demeaned.
- **Per-period target.** `vol_target` has the units of the return grid. An annual target must be
  divided by $\sqrt{\mathrm{AN}}$.
- **No leverage cap.** Weights are unbounded as the estimated volatility falls; a floor bounds them
  only relative to past volatility. Cap before execution, and model the costs of the extra
  turnover.
- **Warm-up.** Without `warmup_period`, the first span of output rests on a seed of one squared
  return and is unreliable.
- **Calendar sums are not unit-variance.** See the pitfall on `compute_sum_freq_ra_returns`; the
  same scale enters `get_paired_rareturns_signals` in its non-overlapping mode. Correlations are
  unaffected by a constant scale; regression slopes are not.
- **Overlap.** Rolling sums are autocorrelated by construction, and the overlapping mode of
  `get_paired_rareturns_signals` pairs a signal with returns it has already seen.
- **pandas 3.** Under pandas 3, the default `freq='BQ'` of `get_paired_rareturns_signals` raises
  `ValueError` (pass `'BQE'` or `'QE'`), and `is_mean_adjust_returns=True` raises `TypeError`
  because `expanding` no longer accepts `axis`.
- **Volatility floor on a Series.** `vol_floor_quantile` works on a DataFrame; on a pandas Series
  the floor is broadcast to a square array and `compute_ra_returns` raises `ValueError`. Pass a
  one-column DataFrame.
- **Evidence is not a theorem.** Moreira and Muir (2017) and Harvey et al. (2018) report
  sample-specific gains, concentrated in equity and credit. Volatility targeting reliably
  stabilises volatility; its effect on the Sharpe ratio depends on the asset.

## See also

- [Exponentially weighted estimators](ewm_estimators.md)
- [Portfolio backtesting](portfolio_backtesting.md)
- [Two-sided turnover conventions](turnover_conventions.md)
- [Signal diagnostics: information coefficient and information ratio](signal_diagnostics.md)
- [Serial dependence and autocorrelation](serial_dependence.md)
- [Regression and HAC inference](regression_and_hac.md)
- [Returns, NAVs, excess returns, fees and leverage](returns_and_navs.md)
- [Notation and conventions](notation_and_conventions.md)
- [Bibliography](bibliography.md)

## References

1. Moreira, A., and Muir, T. (2017). Volatility-Managed Portfolios. *The Journal of Finance*, 72(4), 1611–1644. [DOI: 10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513). Inverse-variance scaling of factor returns and its alphas.
2. Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., and Van Hemert, O. (2018). The Impact of Volatility Targeting. *The Journal of Portfolio Management*, 45(1), 14–33. [DOI: 10.3905/jpm.2018.45.1.014](https://doi.org/10.3905/jpm.2018.45.1.014). Asset-class evidence on Sharpe ratios and tail events under volatility targeting.
3. Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2), 228–250. [DOI: 10.1016/j.jfineco.2011.11.003](https://doi.org/10.1016/j.jfineco.2011.11.003). Time-series momentum with volatility-scaled positions.
4. J.P. Morgan and Reuters (1996). *RiskMetrics — Technical Document*, 4th edition. J.P. Morgan. Source of the EWM decay 0.94 for daily volatility, the qis default.
5. Sepp, A., and Lucic, V. (2026). The Science and Practice of Trend-Following Systems. Working paper. [arXiv:2607.19497](https://arxiv.org/abs/2607.19497). Trend-following systems built on volatility-normalised positions.
6. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
