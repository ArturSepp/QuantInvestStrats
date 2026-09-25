---
myst:
  html_meta:
    description: >-
      Valuation of funded assets, intrinsic calls and puts, futures and custom payoffs under
      factor log shocks, with local Euler risk, conditional scenarios and scenario-local stress
      bands in qis.
---

# Instrument portfolios and standard stress reports

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-12](https://github.com/ArturSepp/QuantInvestStrats/commit/5adefaa96c58e6fc225335eb10e43412f9c51c51)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

An instrument portfolio stress test revalues a frozen snapshot of original holdings under a
complete vector of factor log shocks. Every holding is anchored at its observed mark and moves by
the change in its model payoff: an exponential response for a funded asset, the change in
intrinsic value for a call or put, and the quote variation for a future. Local risk is the
payoff's Jacobian with respect to shared fitted responses, so options, futures and zero-mark
positions carry the exposures that their marks conceal.

## Overview

The chapter answers four questions about one dated position snapshot and one assigned factor
model.

1. **What is each holding worth after a factor move?** Factor shocks map to shared fitted
   responses, responses map to actual quotes and FX rates, and every payoff is evaluated in full.
2. **Which factor move is applied?** A request fixes some factors or factor families. The others
   are set to zero, or completed by the conditional mean under the factor covariance.
3. **How much risk does the book carry locally?** Dollar sensitivities to each response aggregate
   into factor exposures, a factor-model variance and signed Euler volatility contributions.
4. **How wide is the uncertainty around a scenario?** Each point of a sensitivity grid carries a
   band of one and two conditional standard deviations, computed from the exposures *at that
   point*.

Conditioning on anchors, the exponential valuation of funded exposures and the analytical band
around a scenario centre are developed in [Factor stress testing](stress_testing.md), which is
the reference for those formulas. This chapter reuses them and adds what an instrument book
needs: actual quotes kept separate from fitted proxies, currency translation, mark anchoring,
intrinsic payoffs, kink policies at strikes, a holding-by-response Jacobian, holding, family and
cluster Euler allocations, and bands whose width follows the local delta of the payoffs.

The application owns quote acquisition, factor estimation, unsmoothing, contract interpretation
and settlement decisions. qis owns valuation, risk arithmetic and the report. No estimator is
called while scenarios are evaluated or rendered.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Factor, response, quote and FX shocks are log returns; scenario P&L is in reference currency and portfolio return is P&L divided by $V$ |
| Sampling grid | One model-date snapshot, no resampling; historical replay uses at most one complete factor log-return vector per calendar month |
| Annualisation | $\Sigma$, residual variances and $\Omega$ are supplied annual; qis applies no $\mathrm{AN}$; horizon volatility is $\sqrt{\tau v}$ with $\tau$ in years |
| Mean adjustment | Zero-mean conditioning and zero-mean bands; no drift, carry, theta or roll is added |
| Timing | Holdings, marks, quotes, FX, strikes, loadings and covariance frozen at the model date; shocks are instantaneous; months after the valuation date are excluded from replay |
| Output units | Reference-currency values, P&L and dollar sensitivities per unit log return; returns, betas and volatilities as decimal fractions of $V$ |
| qis default | `StressTestConfig()`: `horizon_years=1/12`, `confidence=0.95`, `historical_count=10`, `include_conditional_comparison=True`, `ordinary_asset_bands=True`; `StressScenarios` uses `mode=INDEPENDENT`, `convention=LOG`; `PortfolioHolding.kink_policy=MIDPOINT` |

The chapter uses the following local symbols in addition to the reserved notation of
[Notation and conventions](notation_and_conventions.md).

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $f$, $z$, $z_f$ | Factor index; complete factor log-shock vector; its entry for factor $f$ | Decimal log return, model factor order |
| $A$, $F$, $a$ | Anchored (fixed) and free factor sets of one scenario; a single anchored factor | $F$ never denotes a futures price here |
| $\beta_{fa}$ | Conditional regression coefficient $\Sigma_{fa}/\Sigma_{aa}$ of factor $f$ on anchor $a$ | Dimensionless |
| $x$ | Requested simple bump for a factor or a family total | Decimal; converted with $\log(1+\cdot)$ |
| $G$, $n$, $\xi_f$ | Factor family, its number of members, member allocation weight | $\xi_f\ge0$, $\sum_{f\in G}\xi_f=1$; default $\xi_f=1/n$ |
| $j$, $B$, $y$ | Shared fitted response index; response-by-factor loadings; response log shocks $y=Bz$ | `RiskModel.factor_loadings` at the model date |
| $\Sigma$, $C$ | Annual factor covariance; conditional factor covariance in full factor order | $C$ has zero rows and columns for anchored factors |
| $\sigma^2_{\varepsilon,j}$, $\Omega$, $\sigma_{\Omega}$ | Residual variance of response $j$; supplied response covariance; total volatility under $\Omega$ | Annual; residuals independent across responses |
| $i$, $S_i$, $S_{i,0}$ | Actual quote; its scenario and baseline level | Positive, in the quote currency |
| $\rho(i)$, $c(i)$ | Fitted response and currency of quote $i$ | Several quotes may share one response |
| $X_c$, $X_{c,0}$, $\phi(c)$ | FX rate of currency $c$, baseline, and the FX quote's response | Reference currency per local unit; $X=1$ for the reference currency |
| $M^{q}$, $M^{x}$ | Loadings of log quotes and log FX rates on response shocks | Matrices of 0 and $\pm1$ |
| $h$, $p$ | Original holding; vanilla leg (primitive) of a holding | A holding keeps its ID across legs |
| $Q_p$, $K_p$, $\pi_p$ | Signed quantity times multiplier; strike; local intrinsic payoff | Units; local currency |
| $\mathrm{MTM}_h$, $\Pi_h(z)$, $b_h$ | Observed mark; model payoff; basis offset $\mathrm{MTM}_h-\Pi_h(0)$ | Reference currency |
| $g_h$, $\beta_h$, $q(g)$ | Translated log return of a funded holding; its factor loadings; proportional allocation factor | Decimal log return |
| $\mathrm{PnL}_h(z)$, $R(z)$ | Holding P&L; portfolio return $\sum_h\mathrm{PnL}_h/V$ | Currency; decimal fraction of $V$ |
| $V$ | Reporting denominator | Positive currency amount; not necessarily NAV |
| $J_{hj}$, $d_j$, $\omega_j$ | Holding dollar sensitivity to response $j$; its sum over holdings; $d_j/V$ | Currency per unit log return |
| $E_{hf}$, $E_f$, $e_f$ | Holding factor dollar exposure; portfolio dollar exposure $Ve_f$; portfolio factor beta $(B^{\top}\omega)_f$ | Currency; fraction of $V$ |
| $\delta_p$, $\eta$, $L$ | Quote slope of a leg; slope value chosen at a kink; leverage of an accumulator proxy | Dimensionless |
| $v$, $v_{\mathrm{sys}}$, $v_{\varepsilon}$, $\sigma$ | Model variance, its systematic and residual parts, $\sigma=\sqrt{v}$ | Annual; variances in squared fractions of $V$ |
| $\mathrm{RC}_f$, $\mathrm{RC}_{hf}$, $\mathrm{RC}_{\varepsilon}$ | Euler volatility contributions of a factor, a holding-factor cell and residual risk | Annual volatility units |
| $\mathrm{RC}_G$, $\mathrm{RC}_{\varepsilon,h}$ | Family Euler contribution; residual contribution allocated to holding $h$ | Annual volatility units |
| $\tau$, $\kappa$, $\gamma$ | Band horizon; band multiplier; central probability of exported bounds | Years (default $1/12$); $\kappa\in\{1,2\}$; `confidence` |
| $\Phi^{-1}$, $\hat\theta_1$, $\hat\theta_2$ | Standard-normal quantile; linear and quadratic coefficients of a grid curve fit | Decimal returns per unit bump |

The model is an assigned `qis.RiskModel` with a complete factor block (loadings, factor
covariance, residual variances) at an exact date key no later than the valuation date. The
response covariance $\Omega$ is authoritative for one reported total; the factor-model views are
used for every additive decomposition. They agree when
$\Omega=B\Sigma B^{\top}+\operatorname{diag}(\sigma^2_{\varepsilon})$, and qis reports both
rather than reconciling an inconsistent pair. Quotes and FX rates must be strictly positive
because every shock is multiplicative. The reporting denominator changes ratios only, never
holdings or currency P&L.

## Methodology

### Quotes, fitted responses and currency translation

**Definition (quote and FX shocks).** Given a complete factor log-shock vector $z$, the response
log shocks are $y=Bz$. An actual quote $i$ with response $\rho(i)$ in currency $c=c(i)$, and the
FX rate of that currency, move as

$$
\begin{aligned}
\log\frac{S_i(z)}{S_{i,0}} &= \sum_j M^{q}_{ij}\,y_j
  = y_{\rho(i)}-\mathbf{1}_{\mathrm{REF}}(i)\,y_{\phi(c)},\\
\log\frac{X_c(z)}{X_{c,0}} &= \sum_j M^{x}_{cj}\,y_j = y_{\phi(c)},
\end{aligned}
$$

where $\mathbf{1}_{\mathrm{REF}}(i)$ is one when the quote's response has
`ResponseBasis.REFERENCE` and zero for `ResponseBasis.LOCAL`. A response of `None` sets
$y_{\rho(i)}=0$, which is how explicit cash is modelled: with LOCAL basis the quote is constant
in its own currency (foreign cash still carries FX), and with REFERENCE basis its
reference-currency value is constant. The reference currency has $X\equiv1$, and an FX quote
without a response is deterministic.

**Identity (currency translation).** The reference-currency value of one unit of quote $i$ moves by

$$
\log\frac{X_c(z)\,S_i(z)}{X_{c,0}\,S_{i,0}}=
\begin{cases}
y_{\rho(i)}, & \text{REFERENCE basis},\\
y_{\rho(i)}+y_{\phi(c)}, & \text{LOCAL basis}.
\end{cases}
$$

**Proof.** Add the two log changes of the definition. For a REFERENCE quote the FX response
enters the local quote with a minus sign and the FX rate with a plus sign, and cancels. $\square$

A REFERENCE response was estimated on the return translated into the reference currency, so it
already contains FX; the local quote, against which a local strike is compared, is recovered by
removing the FX response. A LOCAL response describes the quote in its own currency, and FX is
applied once at the payoff level. Several actual contracts can share one response, for example
all listed maturities mapped to one continuous-futures proxy, while keeping their own spot
levels and strikes.

> **Pitfall.** Declaring a fit on reference-currency returns as LOCAL counts the currency twice;
> declaring a fit on local returns as REFERENCE removes the currency from a foreign asset. For a
> quote in the reference currency the two bases coincide, so the error only appears on foreign
> holdings.

### Holding valuation

#### Funded holdings

A funded holding has exactly one `DELTA_1` leg. Its observed mark is the authoritative size; the
leg quantity is audit information whose sign must agree with the mark. With $g_h(z)$ the
translated log return of the identity above,

$$
\mathrm{PnL}_h(z)=\mathrm{MTM}_h\big(\exp(g_h(z))-1\big),
\qquad
\mathrm{MTM}_h(z)=\mathrm{MTM}_h\exp(g_h(z)).
$$

This is the exponential valuation of [Factor stress testing](stress_testing.md) with the
holding's log-return loadings $\beta_h=B^{\top}(M^{q}_{i\cdot}+M^{x}_{c\cdot})^{\top}$; qis
evaluates it with `expm1`. A zero mark carries zero funded exposure.

#### Calls, puts and futures

A derivative holding is a set of signed legs $p$, each on a quote $i=i(p)$ in currency $c=c(p)$,
with local intrinsic payoffs

$$
\pi^{\mathrm{call}}_p(S)=(S-K_p)^{+},\qquad
\pi^{\mathrm{put}}_p(S)=(K_p-S)^{+},\qquad
\pi^{\mathrm{fut}}_p(S)=S-S_{i,0}.
$$

**Definition (anchored derivative value).** The model payoff, P&L and stressed value of holding
$h$ are

$$
\begin{aligned}
\Pi_h(z)&=\sum_{p\in h}X_{c(p)}(z)\,Q_p\,\pi_p\big(S_{i(p)}(z)\big),\\
\mathrm{PnL}_h(z)&=\Pi_h(z)-\Pi_h(0),\\
\mathrm{MTM}_h(z)&=\mathrm{MTM}_h+\mathrm{PnL}_h(z)=b_h+\Pi_h(z),
\end{aligned}
$$

with constant basis offset $b_h=\mathrm{MTM}_h-\Pi_h(0)$. A composite `HoldingPayoff` supplies
$\Pi_h(z)$ directly in reference currency and is anchored the same way.

**Proposition (mark anchoring).** Every holding satisfies $\mathrm{MTM}_h(0)=\mathrm{MTM}_h$,
and the basis offset is unaffected by any shock: $\mathrm{MTM}_h(z)-\Pi_h(z)=b_h$ for all $z$.

**Proof.** For a derivative, $\mathrm{PnL}_h(0)=\Pi_h(0)-\Pi_h(0)=0$, and the second statement is
the definition rearranged. For a funded holding, $\exp(0)-1=0$ and $b_h=0$. $\square$

The offset collects time value, accrued amounts and any difference between the source mark and
the intrinsic proxy, including a negative mark on a written option. The stressed value is
therefore the source mark plus the change in expiry value. It is not the change in a
[Black and Scholes (1973)](https://doi.org/10.1086/260062) price: time value, volatility and
theta are frozen. Full time-value repricing is developed in
[Stress testing with options](stress_testing_with_options.md).

A future usually has a zero mark and $b_h=0$. Its P&L,
$X_c(z)\,Q_p\big(S_i(z)-S_{i,0}\big)$, is the variation on the stressed quote converted at the
stressed FX rate; the settlement reference $S_{i,0}$ is never rebased.

#### Portfolio return and attribution

The portfolio return of scenario $z$ is $R(z)=V^{-1}\sum_h\mathrm{PnL}_h(z)$. Factor attribution
follows two rules. A funded holding with a nonzero mark uses the exact proportional allocation
of [Factor stress testing](stress_testing.md): its factor $f$ component is
$\mathrm{MTM}_h\,\beta_{hf}z_f\,q(g_h)$, with $q(g)=(\exp(g)-1)/g$ and $q(0)=1$. Every other
holding contributes the first-order term $E_{hf}z_f$ from its current Jacobian (defined below).
A final column, `Nonlinear payoff adjustment`, is total P&L minus the factor components, so each
scenario reconciles exactly. For an all-funded book the adjustment is zero; for a book with
options it measures curvature and strike crossings that the current Jacobian misses.

### Scenario construction

#### Anchors, free factors and completion

A request is a table of anchors: one row per scenario, one column per factor or declared family,
with an empty cell for an unspecified factor. An empty cell leaves the factor free; an explicit
zero pins it. Every row must supply at least one finite anchor. Keys that are neither fitted
factors nor declared families are rejected.

Completion follows `ScenarioMode`. `INDEPENDENT` sets every free factor to zero.
`CONDITIONAL` sets the free factors to the Gaussian conditional mean
$z_F=\Sigma_{FA}\Sigma_{AA}^{-1}z_A$ of [Kupiec (1998)](https://doi.org/10.3905/jod.1998.408008),
computed by `qis.conditional_factor_shock` with a linear solve; the conditioning formulas,
their proof from Anderson (2003, Section 2.5) and the validation rules are in
[Factor stress testing](stress_testing.md). A row that already supplies every factor needs no
solve. The supplied anchors are written back exactly after the solve, so an anchor never drifts
by rounding. A per-row `scenario_modes` entry fixes that row's completion, even when a comparison
batch overrides the default mode.

#### Factor families: split, then convert

A `FactorGroupSpec` declares a family $G$ with members $f\in G$ and weights $\xi_f$ (equal by
default, otherwise nonnegative and summing to one, never renormalised). A family bump $x$ expands
to member anchors

$$
z_f=
\begin{cases}
\log(1+\xi_f\,x), & \text{SIMPLE convention},\\
\xi_f\,x, & \text{LOG convention},
\end{cases}
\qquad f\in G .
$$

With an equal split the SIMPLE rule gives $z_f=\log(1+x/n)$. The whole family enters the anchored
set $A$ jointly, so conditional completion keeps every member fixed and solves for the remaining
factors together. A family instruction that overlaps a member instruction in the same row is
rejected, and each expanded simple bump must exceed $-1$.

> **Pitfall.** A $-10\%$ SIMPLE bump of a two-member family gives each member $-5\%$, that is
> $\log 0.95=-0.05129$; it is neither $\log 0.9$ per member nor
> $\tfrac12\log 0.9=-0.05268$. A response loading one on both members therefore moves by
> $(1-0.05)^2-1=-9.75\%$, not $-10\%$.

#### Historical replay

A historical panel supplies realised factor log-return vectors $z_t$, at most one per calendar
month, with exactly the model factors as columns. Rows with a missing factor, and rows dated after
the valuation date, are excluded and recorded in a coverage table. Each eligible vector is applied
to today's holdings and loadings, and months are ranked by exact portfolio P&L
$\sum_h\mathrm{PnL}_h(z_t)$. The replay is a descriptive stress, not the portfolio's realised
history, and carries no probability. The covariance used elsewhere is the model-date snapshot;
the replay does not re-estimate it month by month.

### Local risk

#### The response Jacobian

**Definition.** The dollar sensitivity of holding $h$ to response $j$ at scenario $z$ is

$$
J_{hj}(z)=\frac{\partial\,\mathrm{MTM}_h(z)}{\partial y_j},
$$

the reference-currency change per unit log move of response $j$ with all other responses fixed.
The current Jacobian is $J(0)$.

**Proposition (Jacobian of the primitives).** Let $\delta_p(S)=\pi_p'(S)$ be the quote slope of a
leg: $\mathbf{1}\{S>K_p\}$ for a call, $\mathbf{1}\{S>K_p\}-1$ for a put and $1$ for a future.
Away from strikes, and with quotes and FX evaluated at $z$,

$$
\begin{aligned}
J_{hj}&=\mathrm{MTM}_h\exp(g_h)\,\big(M^{q}_{ij}+M^{x}_{cj}\big)
&&\text{(funded)},\\
J_{hj}&=\sum_{p\in h}X_{c}\,Q_p\Big(\delta_p(S_i)\,S_i\,M^{q}_{ij}+\pi_p(S_i)\,M^{x}_{cj}\Big)
&&\text{(vanilla legs, } i=i(p),\ c=c(p)\text{)}.
\end{aligned}
$$

**Proof.** From the definition of quote and FX shocks, $\partial S_i/\partial y_j=S_iM^{q}_{ij}$
and $\partial X_c/\partial y_j=X_cM^{x}_{cj}$. Differentiate $\mathrm{MTM}_h\exp(g_h)$ for a
funded holding, and $X_c\,Q_p\,\pi_p(S_i)$ by the product and chain rules for each leg. $\square$

The first term of a leg is its dollar delta; the second is the currency exposure of its intrinsic
value. A future has zero intrinsic value at $z=0$ and so no current FX exposure, but a nonzero
dollar delta $Q_pS_{i,0}$ although its mark is zero. `InstrumentPortfolio.response_jacobian`
evaluates $J(z)$ at any complete vector with the original marks, strikes, futures references and
FX baselines. A composite supplies `response_jacobian(context)` at $z=0$ and, for stressed points,
`scenario_response_jacobian(context)`; without the latter, stressed risk fails explicitly instead
of reusing a stale sensitivity.

> **Insight.** A mark measures what a position is worth, not what it is exposed to. In the
> worked example the zero-mark bond future carries $-20\%$ of the denominator on each credit
> factor, and a put marked at USD 6,000 carries $-11\%$ on its equity response. A derivative's
> exposure scales with its notional and delta, not with its mark.

#### Kink policies

At $S_i=K_p$ the intrinsic payoff has a kink and $\delta_p$ is undefined. A holding's
`KinkPolicy` sets the call slope at the strike to $\eta=0$ (`LEFT`), $\eta=1$ (`RIGHT`) or
$\eta=\tfrac12$ (`MIDPOINT`, the default), and the put slope to $\eta-1$, for all legs of that
holding. The common choice preserves the intrinsic put–call identity
$(S-K)^{+}-(K-S)^{+}=S-K$, whose slope is one on both sides. For a continuing accumulator proxy,
$Q$ calls minus $LQ$ puts at one strike, the left and right slopes are $LQ$ and $Q$; `RIGHT`
reports the favourable above-strike rate. For a decumulator proxy, $Q$ puts minus $LQ$ calls,
the slopes are $-Q$ (left) and $-LQ$ (right), and `LEFT` reports the favourable side. The policy
binds only when a quote equals a strike exactly, which is common for remaining-quantity proxies
struck at the current fixing and for trades struck at spot.

#### Exposures and model variance

**Definition.** Aggregating by shared response,

$$
\begin{aligned}
d_j&=\sum_h J_{hj},\qquad \omega_j=\frac{d_j}{V},\qquad
E_{hf}=\sum_j J_{hj}B_{jf},\qquad e=B^{\top}\omega=\frac{1}{V}\sum_h E_{h\cdot},\\
v&=e^{\top}\Sigma\,e+\sum_j\omega_j^2\,\sigma^2_{\varepsilon,j}
 =v_{\mathrm{sys}}+v_{\varepsilon},\qquad \sigma=\sqrt{v}.
\end{aligned}
$$

Residual risk is counted once per response on the *net* sensitivity. A stock and a written call
on the same response share one residual term,
$(J_{\mathrm{stock}}+J_{\mathrm{call}})^2\sigma^2_{\varepsilon,j}/V^2$, not one term per leg.
The supplied-covariance total is
$\sigma_{\Omega}=\sqrt{\omega^{\top}\Omega\,\omega}$ (`annual_total_vol`), and its horizon value
$\sigma_{\Omega}\sqrt{\tau}$ is `local_vol_horizon`; $\sigma$ is `annual_factor_model_vol`.

#### Euler volatility contributions

**Definition.** For $\sigma>0$, the Euler contributions of factor $f$, of holding $h$ through
factor $f$, and of residual risk are

$$
\mathrm{RC}_f=\frac{e_f\,(\Sigma e)_f}{\sigma},\qquad
\mathrm{RC}_{hf}=\frac{E_{hf}}{V}\,\frac{(\Sigma e)_f}{\sigma},\qquad
\mathrm{RC}_{\varepsilon}=\frac{v_{\varepsilon}}{\sigma}
=\sum_j\frac{\omega_j^2\,\sigma^2_{\varepsilon,j}}{\sigma}.
$$

At $\sigma=0$ every contribution is defined as zero.

**Proposition (Euler additivity).** If $\sigma>0$, then
$\sum_f\mathrm{RC}_f+\mathrm{RC}_{\varepsilon}=\sigma$ and $\sum_h\mathrm{RC}_{hf}=\mathrm{RC}_f$
for every factor $f$.

**Proof.** $\sum_f e_f(\Sigma e)_f=e^{\top}\Sigma e=v_{\mathrm{sys}}$, so the left side of the
first statement is $(v_{\mathrm{sys}}+v_{\varepsilon})/\sigma=\sigma$. The second follows from
$\sum_h E_{hf}=V e_f$. $\square$

Viewed as a function of the factor exposures $e$ and the residual weights $\omega$,
$\sigma$ is positively homogeneous of degree one, and
$\mathrm{RC}_f=e_f\,\partial\sigma/\partial e_f$, while each residual term is
$\omega_j\,\partial\sigma/\partial\omega_j$ through the residual channel. The decomposition is
therefore the Euler allocation of [Tasche (2008)](https://arxiv.org/abs/0708.2542) and the
marginal-contribution view of Litterman (1996), applied to the factor and residual channels as
separate coordinates; the general theory is in
[Portfolio risk and Euler contributions](risk_contributions.md).
Contributions are signed: a hedge that lowers volatility has a negative contribution.
Standalone systematic and residual volatilities do not add; only the Euler terms do.

#### Family, holding and cluster aggregation

A family's Euler contribution is the unweighted sum $\mathrm{RC}_G=\sum_{f\in G}\mathrm{RC}_f$. The
scenario weights $\xi_f$ play no part, because the question is how much of $\sigma$ the family's
factors carry. Aggregation requires disjoint families; if declared families overlap there is no
unique additive partition, and the exhibits fall back to atomic factors. Family exposures are
reported two ways: `exposure_sum` $=\sum_{f\in G}E_f$ and `split_bump_exposure`
$=\sum_{f\in G}\xi_fE_f$, where $E_f=Ve_f$ is the portfolio dollar exposure to factor $f$.

**Proposition (split-bump sensitivity).** Under independent completion, with $z(x)$ the family
expansion of a total bump $x$ and every other factor at zero,

$$
\frac{d}{dx}\,\Big(\sum_h\mathrm{PnL}_h\big(z(x)\big)\Big)\bigg|_{x=0}=\sum_{f\in G}\xi_f\,E_f
$$

for both conventions, whenever the payoffs are differentiable at $z=0$.

**Proof.** $dz_f/dx=\xi_f$ at $x=0$ under both $\log(1+\xi_fx)$ and $\xi_fx$. The chain rule
through $y=Bz$ gives $\partial\sum_h\mathrm{PnL}_h/\partial z_f=\sum_{h,j}J_{hj}B_{jf}=E_f$ at
$z=0$. $\square$

The split-bump figure is a local sensitivity per unit of total bump; finite scenario P&L always
comes from full revaluation. With a kink at $z=0$ the derivative is one-sided and the kink policy
decides it.

Clusters group holdings by the fitted cluster labels of the responses they reference, through
their legs or through a nonzero Jacobian entry. A holding in exactly one cluster belongs to it; a
holding spanning several goes to an explicit multi-cluster bucket; a holding with an unlabelled
response is unassigned. Cluster P&L, exposures and systematic Euler terms are sums of holding
terms. The residual term of response $j$ is allocated to holdings in proportion to their signed
share of its net sensitivity:

$$
\mathrm{RC}_{\varepsilon,h}=\sum_{j:\,d_j\neq0}\frac{J_{hj}}{d_j}\,
\frac{\omega_j^2\,\sigma^2_{\varepsilon,j}}{\sigma}.
$$

**Identity (cluster reconciliation).** Systematic and residual cluster contributions sum to
$\sigma$.

**Proof.** $\sum_h J_{hj}/d_j=1$ whenever $d_j\neq0$, and a response with $d_j=0$ has zero
residual term, so $\sum_h\mathrm{RC}_{\varepsilon,h}=\mathrm{RC}_{\varepsilon}$. Adding
$\sum_{h,f}\mathrm{RC}_{hf}=\sum_f\mathrm{RC}_f$ and applying Euler additivity gives $\sigma$.
$\square$

RiskModel's per-response residual contributions divide by $\sigma_{\Omega}$; before this
allocation qis rescales them by $\sigma_{\Omega}/\sigma$, so every cluster exhibit reconciles to
the factor-model volatility.

<a id="conditional-shock-illustration-536"></a>

### Conditional shocks at fixed and volatility-sized anchors

Two report exhibits show the co-moves the covariance implies, one anchor at a time, using the
single-anchor case of $z_F=\Sigma_{FA}\Sigma_{AA}^{-1}z_A$. For each atomic factor $a$ and
simple anchor $x_a$,

$$
z_a=\log(1+x_a),\qquad
z_f=\frac{\Sigma_{fa}}{\Sigma_{aa}}\,z_a\quad(f\neq a),\qquad
\text{displayed entry}=\exp(z_f)-1 .
$$

The first exhibit uses $x_a=\pm10\%$; the second uses annual-volatility-sized anchors
$x_a=\pm\sqrt{\Sigma_{aa}}$, so an annual volatility of 13.3% gives simple anchors of $\pm13.3\%$.
These are simple returns converted with $\log(1+\cdot)$, not $\exp(\pm\sqrt{\Sigma_{aa}})-1$, and
they are not scaled to a month. Rows are affected factors, columns anchored factors, and the
diagonal is the anchor itself. No family splitting is used. An anchor with zero variance, or a
downside anchor at or below $-100\%$, is marked unavailable rather than clipped.

**Proposition (sign asymmetry).** With conditional regression coefficient
$\beta_{fa}=\Sigma_{fa}/\Sigma_{aa}$ and $0<x<1$, the displayed responses to $+x$ and $-x$
satisfy

$$
\big[(1+x)^{\beta_{fa}}-1\big]+\big[(1-x)^{\beta_{fa}}-1\big]
=\beta_{fa}(\beta_{fa}-1)\,x^2+O(x^4),
$$

and the sum is nonzero unless $\beta_{fa}\in\{0,1\}$, so the two tables are not negatives of
each other.

**Proof.** $\exp(\beta_{fa}\log(1+x))=(1+x)^{\beta_{fa}}$. On $\zeta>0$ the map
$\zeta\mapsto\zeta^{\beta_{fa}}$ is strictly concave for $0<\beta_{fa}<1$ and strictly convex for
$\beta_{fa}<0$ or $\beta_{fa}>1$, so by Jensen's inequality
$(1+x)^{\beta_{fa}}+(1-x)^{\beta_{fa}}\neq2$ in those cases. Expanding $(1\pm x)^{\beta_{fa}}$ in
powers of $x$, the odd terms cancel and the quadratic term is $2\binom{\beta_{fa}}{2}x^2$.
$\square$

A negative coefficient produces an opposite-sign co-move. The tables are conditional mean
returns, not covariance matrices. The uncertainty that remains is the conditional covariance
$C$ of [Factor stress testing](stress_testing.md): the Schur complement
$C_{FF}=\Sigma_{FF}-\Sigma_{FA}\Sigma_{AA}^{-1}\Sigma_{AF}$ of
[Kim and Finger (2000)](https://www.risk.net/journal-risk/2161074/stress-test-incorporate-correlation-breakdown),
embedded in full factor order with zero rows and columns for the anchors. For a Gaussian model it
depends on the anchored set, not on the size or sign of the anchors.

### Scenario-local conditional bands

**Definition (scenario-local band).** Let a grid row request anchors that expand to the set
$A(x)$ and resolve to the complete vector $z(x)$, and let $C=C_{A(x)}$. With sensitivities
re-evaluated at the scenario point and the denominator held fixed,

$$
\begin{aligned}
\omega(x)&=\frac{1}{V}\sum_h J_{h\cdot}\big(z(x)\big),\qquad e(x)=B^{\top}\omega(x),\\
v(x)&=e(x)^{\top}C\,e(x)+\sum_j\omega_j(x)^2\,\sigma^2_{\varepsilon,j},\\
\text{band}(x)&=R\big(z(x)\big)\pm\kappa\sqrt{\tau\,v(x)},\qquad \kappa\in\{1,2\}.
\end{aligned}
$$

The exported quantile bounds use $\Phi^{-1}\big((1+\gamma)/2\big)$ in place of $\kappa$, with
$\gamma$ the configured `confidence`. The horizon Euler contributions are
$\sqrt{\tau}\,e_f(x)(Ce(x))_f/\sqrt{v(x)}$ for each factor and
$\sqrt{\tau}\,v_{\varepsilon}(x)/\sqrt{v(x)}$ for residual risk.

**Proposition (band reconciliation).** At every grid point the horizon Euler contributions sum
to $\sqrt{\tau v(x)}$, and every anchored factor contributes zero.

**Proof.** Apply Euler additivity with $\Sigma$ replaced by $C$ and multiply by $\sqrt{\tau}$. Row
$a$ of $C$ is zero for $a\in A(x)$, so $(Ce)_a=0$. $\square$

The centre $R(z(x))$ is the exact revaluation. The conditional covariance is computed once per
anchored set, while the exposures move with the payoffs along the grid. This is the main
difference from the funded band of [Factor stress testing](stress_testing.md), whose width is
fixed by baseline weights. Marks, strikes, quote baselines and $V$ are never rebased. Bands are
computed only when every row of the grid requests conditional completion. An independent grid
keeps exact payoff points without a band, and `ordinary_asset_bands=False` disables every band.
Do not subtract an anchor's baseline Euler term to approximate conditional risk: condition the
covariance and recompute.

> **Insight.** A band that moves with the payoff is informative in its own right. In the worked
> example the one-month one-sigma half-width falls from 2.24% to 1.50% of the denominator as the
> credit family moves from $-10\%$ to $+10\%$: the conditional equity rally takes the written call
> into the money and its negative delta offsets the funded equity exposure.

### Descriptive curve fits

Each numeric grid is summarised by a quadratic through the origin,
$R(x)\approx\hat\theta_1x+\hat\theta_2x^2$, fitted by ordinary least squares on the grid points.
The legend reports the equation and the uncentred R-squared
$1-\sum_x\big(R(x)-\hat R(x)\big)^2/\sum_xR(x)^2$. Pointwise confidence intervals of the fitted
mean, $\hat R(x)\pm t_{T-2}\,\widehat{\mathrm{se}}(x)$ at central probability $\gamma$ with $T$
grid points, are exported but not plotted. They describe the polynomial approximation of a
deterministic curve, not portfolio risk, and a kink or knockout jump is smoothed by the fit while
the exact points remain authoritative.

## Worked example

The example is small enough to check by hand. Three factors, Equity, Credit and Credit EM, have
annual volatilities of 20%, 10% and 15% and correlations 0.5 (Equity–Credit), 0.6 (Equity–Credit
EM) and 0.8 (Credit–Credit EM). Four fitted responses load on them: `spx` (Equity 1.0, no
residual), `sx5e` (Equity 0.9, residual variance 0.004), `bond` (1.0 on each credit factor,
0.0025) and `eurusd` (Equity 0.1, 0.006). The two credit factors form the equally weighted
family `Credit family`.
The reference currency is USD and the reporting denominator is $V=$ USD 1,000,000.

| Holding | Terms | Response and basis | Mark (USD) |
|---|---|---|---:|
| `fund` | Funded, 120 units of SPX at 5,000 | `spx`, REFERENCE | 600,000 |
| `call` | Short 4 SPX calls, multiplier 10, strike 5,000 (at the money) | `spx`, REFERENCE | −15,000 |
| `put` | Long 2 SX5E puts, multiplier 10, strike EUR 5,200; SX5E at EUR 5,000; EURUSD 1.10 | `sx5e`, LOCAL | 6,000 |
| `future` | Short 2 bond futures, multiplier 1,000, quote 100 | `bond`, REFERENCE | 0 |

**Valuation.** Take the complete vector with simple moves of −20% (Equity), −5% (Credit) and
−10% (Credit EM), applied as log shocks. SPX falls to 4,000 and the call's intrinsic value stays
at zero. SX5E falls to $5000\times0.8^{0.9}=$ EUR 4,090.2 and EURUSD to
$1.10\times0.8^{0.1}=1.0757$, so the put's intrinsic value rises from USD 4,400 to USD 23,876.
The bond quote falls to $100\times0.95\times0.9=85.5$. Holding P&L is −120,000, 0, +19,476 and
+29,000, a total of USD −71,524, or −7.15% of $V$. At zero shock every holding returns exactly
its mark.

```python
from dataclasses import replace
from math import isclose, log, sqrt

import numpy as np
import pandas as pd
import qis

date = pd.Timestamp('2026-08-31')
factors = ['Equity', 'Credit', 'Credit EM']
sigma = pd.DataFrame([[0.040, 0.010, 0.018],
                      [0.010, 0.010, 0.012],
                      [0.018, 0.012, 0.0225]], index=factors, columns=factors)
responses = ['spx', 'sx5e', 'bond', 'eurusd']
loadings = pd.DataFrame([[1.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.0, 1.0], [0.1, 0.0, 0.0]],
                        index=responses, columns=factors)
residual = pd.Series([0.0, 0.004, 0.0025, 0.006], index=responses)
covar = loadings @ sigma @ loadings.T + pd.DataFrame(
    np.diag(residual), index=responses, columns=responses)
family = qis.FactorGroupSpec('Credit family', ('Credit', 'Credit EM'))
model = qis.RiskModel(covar={date: covar}, factor_loadings={date: loadings},
                      factor_covar={date: sigma}, residual_vars={date: residual},
                      factor_groups={'Credit family': family})

REF, LOCAL = qis.ResponseBasis.REFERENCE, qis.ResponseBasis.LOCAL
quotes = {'SPX': qis.Underlying('SPX', 5000.0, 'USD', 'spx', REF),
          'SX5E': qis.Underlying('SX5E', 5000.0, 'EUR', 'sx5e', LOCAL),
          'BOND': qis.Underlying('BOND', 100.0, 'USD', 'bond', REF)}
Leg, Type = qis.InstrumentLeg, qis.InstrumentType
holdings = (
    qis.PortfolioHolding('fund', 'US equity fund', 600_000.0, (Leg(Type.DELTA_1, 'SPX', 120.0),)),
    qis.PortfolioHolding('call', 'Short SPX call', -15_000.0,
                         (Leg(Type.CALL, 'SPX', -4.0, multiplier=10.0, strike=5000.0),)),
    qis.PortfolioHolding('put', 'SX5E put', 6_000.0,
                         (Leg(Type.PUT, 'SX5E', 2.0, multiplier=10.0, strike=5200.0),)),
    qis.PortfolioHolding('future', 'Short bond future', 0.0,
                         (Leg(Type.FUTURE, 'BOND', -2.0, multiplier=1000.0),)),
)
portfolio = qis.InstrumentPortfolio(
    holdings=holdings, underlyings=quotes, risk_model=model, risk_date=date,
    valuation_date=date, reference_currency='USD', reporting_denominator=1_000_000.0,
    denominator_label='Investment capital',
    fx_rates={'EUR': qis.Underlying('EURUSD', 1.10, 'USD', 'eurusd', REF)})


def levels(z):
    """Independent quote and FX levels: y = B z, LOCAL SX5E, deterministic USD."""
    y = loadings @ z
    return (5000.0 * np.exp(y['spx']), 5000.0 * np.exp(y['sx5e']),
            100.0 * np.exp(y['bond']), 1.10 * np.exp(y['eurusd']))


def hand_pnl(z):
    """Mark-anchored P&L of the four holdings, written out from the payoff definitions."""
    spx, sx5e, bond, eurusd = levels(z)
    return pd.Series({
        'fund': 600_000.0 * (spx / 5000.0 - 1.0),
        'call': -40.0 * max(spx - 5000.0, 0.0),
        'put': 20.0 * (eurusd * max(5200.0 - sx5e, 0.0) - 1.10 * 200.0),
        'future': -2000.0 * (bond - 100.0),
    })


z = pd.Series({'Equity': log(0.8), 'Credit': log(0.95), 'Credit EM': log(0.9)})
pnl = portfolio.get_pnl(z)
np.testing.assert_allclose(pnl[['fund', 'call', 'put', 'future']], hand_pnl(z), atol=1e-8)
np.testing.assert_allclose(pnl, [-120_000.0, 0.0, 19_475.51, 29_000.0], atol=0.01)
assert round(pnl.sum()) == -71_524
np.testing.assert_allclose(portfolio.get_mtm(z * 0.0), [600_000.0, -15_000.0, 6_000.0, 0.0])
```

**Local risk.** The current sensitivities by response are USD 500,000 on `spx` (the fund's
600,000 less 100,000 for the written call, whose midpoint slope at the strike is one half),
−110,000 on `sx5e` (the put's delta $-20\times5000\times1.10$), −200,000 on `bond` from the
zero-mark future, and +4,400 on `eurusd` (the currency exposure of the put's intrinsic value).
The factor betas are 0.4014 (Equity), −0.20 (Credit) and −0.20 (Credit EM). Model volatility is
6.60% a year: 6.49% systematic and 1.22% residual as standalone figures, which do not add. The
Euler contributions are 6.36% (Equity), 0.12% (Credit), −0.10% (Credit EM) and 0.23% (residual),
and sum to 6.60%. Credit EM is a diversifier. The written call's sensitivity is 0, −100,000 or
−200,000 under the `LEFT`, `MIDPOINT` and `RIGHT` kink policies. The family's summed exposure is
USD −400,000, and its sensitivity per unit of split bump is −200,000.

```python
scenarios = qis.StressScenarios(
    pd.DataFrame({'Credit family': [-0.10, 0.10]}, index=['Credit -10%', 'Credit +10%']),
    mode=qis.ScenarioMode.CONDITIONAL, convention=qis.ShockConvention.SIMPLE)
axis = pd.Index([-0.10, 0.0, 0.10], name='Total credit family bump')
grid = qis.StressScenarios(pd.DataFrame({'Credit family': axis.to_numpy()}, index=axis),
                           mode=qis.ScenarioMode.CONDITIONAL,
                           convention=qis.ShockConvention.SIMPLE)
result = qis.run_portfolio_stress_test(portfolio, scenarios, factor_grids={'Credit': grid})

d = result.response_exposures[responses]
np.testing.assert_allclose(d, [500_000.0, -110_000.0, -200_000.0, 4_400.0])
w = d / 1_000_000.0
e = loadings.T @ w
v_sys, v_eps = float(e @ sigma @ e), float(w.pow(2) @ residual)
vol = sqrt(v_sys + v_eps)
rc = e * (sigma @ e) / vol
np.testing.assert_allclose(result.factor_betas, e)
np.testing.assert_allclose(result.risk['annual_factor_model_vol'], vol)
np.testing.assert_allclose(result.risk['annual_total_vol'], vol)
np.testing.assert_allclose(result.report_diagnostics['Factor Euler volatility'].euler_vol, rc)
assert isclose(rc.sum() + v_eps / vol, vol, rel_tol=1e-12)
np.testing.assert_allclose([vol, sqrt(v_sys), sqrt(v_eps), *rc, v_eps / vol],
                           [0.066019, 0.064885, 0.012187, 0.063589, 0.001168, -0.000987,
                            0.002250], atol=5e-7)
holding_rc = result.report_diagnostics['Holding factor Euler volatility']
np.testing.assert_allclose(holding_rc.sum(), rc)
np.testing.assert_allclose(result.factor_group_exposures.loc['Credit family'],
                           [-400_000.0, -200_000.0])

call = portfolio.holdings[1]
for policy, expected in [(qis.KinkPolicy.LEFT, 0.0), (qis.KinkPolicy.MIDPOINT, -100_000.0),
                         (qis.KinkPolicy.RIGHT, -200_000.0)]:
    alone = replace(portfolio, holdings=(replace(call, kink_policy=policy),))
    assert alone.response_jacobian().loc['call', 'spx'] == expected
```

**Scenarios and bands.** A −10% SIMPLE bump of the credit family anchors both members at
$\log0.95=-0.0513$. Conditional completion moves Equity by a log return of −0.0437, a −4.28%
simple return. The book loses 0.19% of $V$. Under a +10% family bump it loses 0.77%: on the way
up the written call, the short future and the fading put give back more than the fund gains.
Anchoring the credit family leaves a
conditional equity variance of 0.02556 (15.99% volatility). The one-month one-sigma band
half-widths at bumps of −10%, 0 and +10% are 2.24%, 1.89% and 1.50%. In the ±10%
conditional-shock tables, an anchor on Credit EM moves Equity by +7.92% and −8.08%. Anchoring
Equity at plus and minus its 20% volatility moves Credit by +4.66% and −5.43%.

```python
requested = result.valuations['requested']
shocks = requested.factor_log_shocks
anchored = ['Credit', 'Credit EM']
coef = np.linalg.solve(sigma.loc[anchored, anchored], sigma.loc[anchored, 'Equity'])
for label, bump in [('Credit -10%', -0.10), ('Credit +10%', 0.10)]:
    np.testing.assert_allclose(shocks.loc[label, anchored], log(1.0 + bump / 2.0))
    np.testing.assert_allclose(shocks.loc[label, 'Equity'], coef.sum() * log(1.0 + bump / 2.0))
    np.testing.assert_allclose(requested.pnl.loc[label, ['fund', 'call', 'put', 'future']],
                               hand_pnl(shocks.loc[label]), atol=1e-8)
np.testing.assert_allclose(result.summaries['requested'].portfolio_return,
                           [-0.001948, -0.007717], atol=5e-7)
np.testing.assert_allclose(result.attribution['requested'].sum(axis=1), requested.portfolio_pnl)


def hand_dollars(z):
    """Scenario-local sensitivities d_j(z); the call slope is one half exactly at its strike."""
    spx, sx5e, bond, eurusd = levels(z)
    call_slope = 1.0 if spx > 5000.0 else 0.5 if spx == 5000.0 else 0.0
    return pd.Series({'spx': 600_000.0 * spx / 5000.0 - 40.0 * spx * call_slope,
                      'sx5e': -20.0 * eurusd * sx5e * (sx5e < 5200.0),
                      'bond': -2000.0 * bond,
                      'eurusd': 20.0 * eurusd * max(5200.0 - sx5e, 0.0)})


c_equity = sigma.loc['Equity', 'Equity'] - sigma.loc['Equity', anchored] @ coef
assert isclose(c_equity, 0.04 - 0.013 / 0.9, rel_tol=1e-12)
bands = result.grid_summaries['Credit']
euler = result.report_diagnostics['Grid conditional factor Euler'].loc['Credit']
for x in axis:
    w_x = hand_dollars(result.grids['Credit'].factor_log_shocks.loc[x]) / 1_000_000.0
    v_x = (loadings.T @ w_x)['Equity'] ** 2 * c_equity + float(w_x.pow(2) @ residual)
    np.testing.assert_allclose(bands.loc[x, 'conditional_vol_horizon'], sqrt(v_x / 12.0))
    np.testing.assert_allclose(bands.loc[x, 'upper_2sigma'] - bands.loc[x, 'portfolio_return'],
                               2.0 * sqrt(v_x / 12.0))
    np.testing.assert_allclose(euler.loc[x].drop('Total').sum(), sqrt(v_x / 12.0))
    assert euler.loc[x, 'Credit'] == 0.0 and euler.loc[x, 'Credit EM'] == 0.0
np.testing.assert_allclose(bands.conditional_vol_horizon, [0.022407, 0.018857, 0.014974],
                           atol=5e-7)

tables = result.report_diagnostics
np.testing.assert_allclose(tables['Conditional factor shocks +10%'].loc['Equity', 'Credit EM'],
                           1.1 ** 0.8 - 1.0, rtol=1e-12)
np.testing.assert_allclose(tables['Conditional factor shocks -10%'].loc['Equity', 'Credit EM'],
                           0.9 ** 0.8 - 1.0, rtol=1e-12)
np.testing.assert_allclose(tables['Conditional factor shocks +1sigma'].loc['Credit', 'Equity'],
                           1.2 ** 0.25 - 1.0, rtol=1e-12)
np.testing.assert_allclose(tables['Conditional factor shocks -1sigma'].loc['Credit', 'Equity'],
                           0.8 ** 0.25 - 1.0, rtol=1e-12)
```

The checks are independent of the stress engine where it matters: quotes, FX, payoffs, the
conditional regression coefficients, the Schur complement, the scenario-local sensitivities and
the Euler sums are all recomputed from the definitions with numpy. The inputs are fixed teaching
numbers, not a fitted model.

## Implementation in qis

### Entry points and formulas

| Quantity | Formula | qis entry point |
|---|---|---|
| Fitted factor model | $B$, $\Sigma$, $\sigma^2_{\varepsilon,j}$, $\Omega$ | `qis.RiskModel` (exact date keys) |
| Factor family | $G$, $\xi_f$ | `qis.FactorGroupSpec` in `RiskModel.factor_groups` |
| Actual quote and response | $S_{i,0}$, $\rho(i)$, $c(i)$ | `qis.Underlying` |
| Currency basis | REFERENCE or LOCAL rows of $M^{q}$ | `qis.ResponseBasis` |
| Primitive leg | $Q_p$, $K_p$, $\pi_p$ | `qis.InstrumentLeg`, `qis.InstrumentType` (`DELTA_1`, `CALL`, `PUT`, `FUTURE`) |
| Holding and mark | $\mathrm{MTM}_h$, legs or composite | `qis.PortfolioHolding` |
| Kink slope | $\eta\in\{0,1,\tfrac12\}$ | `qis.KinkPolicy` (`LEFT`, `RIGHT`, `MIDPOINT`) |
| Composite payoff | $\Pi_h(z)$ and $J_{h\cdot}$ | `qis.HoldingPayoff` (`evaluate`, `response_jacobian`, optional `scenario_response_jacobian`) |
| Market view for composites | $S_i(z)$, $X_c(z)$, $M^{q}$, $M^{x}$, $y$ | `qis.PayoffContext` (read-only copies) |
| Snapshot | registries, dates, $V$ | `qis.InstrumentPortfolio` |
| Stressed value and P&L | $\mathrm{MTM}_h(z)$, $\mathrm{PnL}_h(z)$ | `InstrumentPortfolio.get_mtm`, `get_pnl`, `evaluate` returning `qis.PortfolioValuationResult` (`mtm`, `pnl`, `audit`) |
| Basis offset | $b_h$ | `PortfolioValuationResult.audit` (`observed_mtm`, `model_baseline`, `basis_offset`) |
| Family expansion | $\log(1+\xi_fx)$ or $\xi_fx$ | `qis.StressScenarios.expanded_anchors`, `qis.ShockConvention` |
| Completion | $z_F=0$ or $\Sigma_{FA}\Sigma_{AA}^{-1}z_A$ | `StressScenarios.resolve`, `qis.ScenarioMode`, `qis.conditional_factor_shock` |
| Level target to anchor | $\log(P_1/P_0)$ | `qis.price_target_log_shock` |
| Response Jacobian | $J_{hj}(z)$ | `InstrumentPortfolio.response_jacobian(delta_f)`; `PortfolioStressResult.response_jacobian` |
| Exposures | $d_j$, $E_{hf}$, $E_f$, $e_f$ | `response_exposures`, `holding_factor_exposures`, `factor_exposures`, `factor_betas` |
| Family exposures | $\sum_{f\in G}E_f$, $\sum_{f\in G}\xi_fE_f$ | `factor_group_exposures` (`RiskModel.compute_factor_group_exposures_at_date`) |
| Model and supplied risk | $\sigma$, $\sqrt{v_{\mathrm{sys}}}$, $\sqrt{v_{\varepsilon}}$, $\sigma_{\Omega}$ | `PortfolioStressResult.risk` (`RiskModel.compute_tre_decomposition_at_date`, `compute_tre_at_date`) |
| Euler contributions | $\mathrm{RC}_f$, $\mathrm{RC}_{\varepsilon}$ | `report_diagnostics["Factor Euler volatility"]`, `["Annualised portfolio risk"]` (`qis.compute_portfolio_risk_contributions`) |
| Holding and family Euler | $\mathrm{RC}_{hf}$, $\mathrm{RC}_G$ | `report_diagnostics["Holding factor Euler volatility"]`, `["Family Euler volatility"]`, `["Reported factor groups"]` |
| Response Euler | $\omega_j(\Omega\omega)_j/\sigma_{\Omega}$ | `response_risk_contributions` (`RiskModel.compute_marginal_tre_at_date`) |
| Attribution | funded $q(g)$ allocation, $E_{hf}z_f$, adjustment | `PortfolioStressResult.attribution` |
| Historical replay | $\sum_h\mathrm{PnL}_h(z_t)$ | `historical`, `historical_ranking`, `historical_coverage` |
| Conditional covariance | $C$, zero on anchored rows and columns | `qis.conditional_factor_covariance`, called once per anchored set |
| Conditional-shock tables | $\exp(z_f)-1$ | `report_diagnostics["Conditional factor shocks -10%"]`, `+10%`, `-1sigma`, `+1sigma` |
| Scenario-local bands | $R\pm\kappa\sqrt{\tau v(x)}$ | `grid_summaries[...]` (`lower_1sigma` to `upper_2sigma`, `lower_bound`, `upper_bound`, `band_half_width`) |
| Band audit | horizon Euler, $d_j(x)$ | `report_diagnostics["Grid conditional factor Euler"]`, `["Grid conditional family Euler"]`, `["Grid scenario response exposures"]` |
| Curve fit | $\hat\theta_1$, $\hat\theta_2$, uncentred R-squared | `report_diagnostics["Grid polynomial regressions"]`, `["Grid regression confidence bands"]` |
| Settings | $\tau$, $\gamma$ | `qis.StressTestConfig` |
| Orchestration | all of the above | `qis.run_portfolio_stress_test` returning `qis.PortfolioStressResult` |
| Report | pages, CSV, workbook, manifest | `qis.generate_portfolio_stress_report`, `qis.StressReportConfig`, `qis.StressReportArtifacts` |

`run_portfolio_stress_test` evaluates three batches from one request: `requested` (the request's
own completion), `independent`, and, unless `include_conditional_comparison=False`,
`conditional`. It then evaluates every named grid, attaches bands to grids whose rows are all
conditional, replays history and builds the diagnostics. It performs no estimation, data access,
plotting or file output. See the
{doc}`run_portfolio_stress_test API <api/generated/qis.run_portfolio_stress_test>` and the
{doc}`InstrumentPortfolio API <api/generated/qis.InstrumentPortfolio>` for signatures.

The implementation is in
[instruments.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/instruments.py)
(terms, payoffs, kink slopes, `PayoffContext`),
[portfolio.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/portfolio.py)
(anchored valuation and the Jacobian),
[scenarios.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/scenarios.py)
(family expansion and completion),
[analytics.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/analytics.py)
(orchestration, attribution, replay) and
[reporting.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/reporting.py).
Three internal modules are not public API:
[_valuation.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/_valuation.py)
builds the quote and FX maps $M^{q}$ and $M^{x}$,
[_bands.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/_bands.py)
computes the scenario-local bands, and
[_clusters.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/stress/_clusters.py)
computes the cluster exhibits. Risk arithmetic is delegated to
[risk_model.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/risk_model.py)
and [contributions.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/contributions.py);
conditioning is in
[stress_testing.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/stress_testing.py).

### Inputs, objects and outputs

| Step | Public object or function | Caller supplies | QIS produces |
|---|---|---|---|
| 1 | `RiskModel` | Dated response covariance, factor loadings, factor covariance, residual variances and optional `FactorGroupSpec` definitions | One assigned model shared by holdings and scenario calculations |
| 2 | `Underlying` | Actual quote ID, positive local spot, currency, fitted response ID and `ResponseBasis` | A distinct valuation quote connected to a possibly shared risk response |
| 3 | `PortfolioHolding` | Original ID/name, observed mark, signed `InstrumentLeg` terms or `HoldingPayoff` | One attributed holding, regardless of its number of synthetic legs |
| 4 | `InstrumentPortfolio` | Holdings, model/position dates, quote and FX registries, positive reporting denominator | `get_mtm`, `get_pnl`, batch `evaluate` and current `response_jacobian` |
| 5 | `StressScenarios` | Factor/family anchors, simple/log convention and completion policy | Complete factor log-shock vectors; independent or jointly conditional |
| 6 | `run_portfolio_stress_test` | Portfolio, requests, optional monthly history and named grids | Detached `PortfolioStressResult`: full valuations, exposures, local risk, attribution and audit tables |
| 7 | `generate_portfolio_stress_report` | Completed result and `StressReportConfig` | Twelve analysis PDF pages, optional coverage, final notation guide, all numerical tables and artifact hashes |

FX quotes are `Underlying` objects quoted in the reference currency per unit of local currency
(for example `fx_rates["EUR"]` at 1.20 USD per EUR) with a REFERENCE response; the reference
currency itself is not supplied. The packaged conventions note, shipped inside the installed
package, is also available as
[plain source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/portfolio_stress.md).

### Runnable examples

Both examples run on the core installation and the repository's synthetic data generator.
They require no credentials, market-data service, estimator package or private consumer code.
Their seven-factor names are illustrative; the inputs are not a production MATF calibration.

| Example | What it demonstrates |
|---|---|
| [`examples/portfolios/instrument_portfolio_stress.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/instrument_portfolio_stress.py) | Funded and mixed portfolios; all four primitive types; continuing accumulator/decumulator legs; EUR local quotes with USD fitted responses; Credit and Carry families; monthly replay; four conditional grids; standard reporting. |
| [`examples/portfolios/composite_payoff_stress.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/composite_payoff_stress.py) | A terminal-knockout wrapper implementing `HoldingPayoff`, retaining source marks, vanilla valuation and shared-response risk; independent terminal-P&L and finite-difference checks. |

From the source checkout, using its configured Python interpreter:

~~~console
python -m examples.portfolios.instrument_portfolio_stress
python -m examples.portfolios.composite_payoff_stress
~~~

By default the examples compute and verify results and write no files. To generate reports,
supply fresh output directories outside the source checkout:

~~~console
python -m examples.portfolios.instrument_portfolio_stress --case all --output-dir /path/to/new/instrument_reports
python -m examples.portfolios.composite_payoff_stress --output-dir /path/to/new/composite_report
~~~

`--case all` creates `funded/` and `mixed/`; `--case funded` and `--case mixed` select one.
The custom-payoff example writes directly to its supplied directory. On Windows, replace the
example paths with quoted absolute paths on the local C drive and use the repository's external
interpreter. An existing target is rejected before report output is written.

The composite example replaces one remaining-quantity accumulator with a terminal knockout at
EUR 115 and moves its strike to EUR 95, with spot at EUR 100. Away from the strike and barrier,
a central finite difference of public portfolio P&L agrees with the analytic factor sensitivity.
At or beyond the barrier it declares zero remaining payoff and zero local sensitivity. This
boundary convention does not make a discontinuous barrier differentiable and is not a path
simulation; its limitations are carried into the position audit and the report appendix.

### Report layout

`generate_portfolio_stress_report` renders a completed result; it never reprices a payoff, refits
a model or obtains prices. The PDF has twelve analysis pages, an optional parser-owned coverage
page, and a final notation guide: thirteen pages without the appendix and fourteen with it.
Titles use `StressReportConfig.model_name`.

| Page | Subject | Content |
|---|---|---|
| 1 | Requested stress scenarios | The `requested` batch: portfolio and top holding contributions as fractions of $V$ |
| 2 | Requested scenarios with the model's co-moves | The `conditional` comparison batch at the model date |
| 3 | Worst historical scenario months | The `historical_count` worst complete months replayed on today's holdings |
| 4 | Exposures and risk | Factor betas, the annualised risk table and family Euler contributions |
| 5 | Largest factor exposures: asset risk contributors | Six largest absolute family or factor Euler terms, each with its ten largest holding contributions |
| 6 | Sensitivity to largest factor exposures | Up to six grids: exact points, through-zero quadratic, one- and two-sigma shading |
| 7 | Asset cluster dendrograms | Caller-supplied linkages, cutoffs and memberships |
| 8 | Cluster contributions | Cluster stress P&L, factor exposures and signed Euler risk |
| 9 | Loadings and explanatory power | Response betas, caller-supplied R-squared, unit response risk, Rest and Portfolio rows |
| 10 | Correlation and scenario construction | Dated correlation and volatility matrix and target mappings |
| 11 | Conditional shocks at 10% | The $\pm10\%$ tables, conditional covariance and the band formula |
| 12 | Conditional shocks at one sigma | The $\pm\sqrt{\Sigma_{aa}}$ tables |
| 13 (optional) | Parser appendix | `appendix_table`, at most 24 rows and ten columns, with `appendix_notes` |
| Last | Notation and guide to the analysis | Definitions of the denominator, sensitivities, covariance and Euler terms, and a guide to each exhibit |

Reading notes:

- **Current betas and dollar exposures** use today's payoff Jacobian, aggregated by shared
  response. Options crossing strikes or barriers can lose heavily despite a small current beta.
- **Requested and conditional pages** value the full payoff. Summed family exposures and summed
  Euler contributions answer aggregation questions; they do not use the scenario split weights.
- **Historical months** are ranked by exact portfolio P&L on today's holdings and are not the
  portfolio's realised investment history.
- **Page 5** selects factors or families by absolute Euler contribution, which need not be the
  largest betas or dollar exposures, then ranks their holding contributions. The PDF shows a
  subset; the exported tables are complete and additive.
- **Sensitivity grids** fit a through-zero quadratic for every portfolio, including derivatives,
  and show the equation and uncentred R-squared. Exact payoff points remain authoritative. Blue
  shading shows scenario-local conditional one- and two-sigma bands; the OLS mean-fit intervals
  are exported diagnostics, not shading.
- **Loadings and fit** distinguish unit response risk from portfolio exposure. Portfolio
  R-squared is an absolute-exposure-weighted average of caller-supplied response R-squared, not a
  portfolio regression; the Rest of assets row uses full-denominator weights. Absent diagnostics
  are labelled unavailable, never invented.
- **Denominator.** Percentages use the application's explicit reporting denominator. Derivative
  report captions label it as such; funded-asset reports keep NAV terminology.

The cluster page groups holdings by the caller's fitted response memberships, with cadence
prefixes such as ME-1 and QE-1. Its top heatmap shows P&L of the first twelve scenarios of the
conditional-comparison batch by cluster, divided by $V$; the bottom left shows cluster factor
exposures $\sum_{h}E_{hf}/V$; the bottom right stacks systematic and residual Euler contributions
to model volatility, which reconcile to $\sigma$. Each table appends an additive portfolio row.
The display keeps at most eight groups, reserving explicit unassigned and multi-cluster buckets
and combining the smallest regular clusters as Other clusters, ordered by gross mark. A
contributor panel names, for each displayed row, the three holdings with the largest absolute
P&L in that row's worst conditional-comparison scenario. The factor-risk bars show the five largest
absolute atomic factor Euler terms with fixed colours; their annotations are subtotals over those
five factors only. `StressReportConfig.cluster_labels` maps cadence-prefixed IDs to descriptive
labels, which `qis.plot_clusters` also accepts. qis never estimates a tree or a label.

Display names should be unique aliases of at most 20 characters, supplied as
`PortfolioHolding.metadata["short_name"]` and `StressReportConfig.response_diagnostics["name"]`;
full names and IDs remain in the exports.

### Exports and audit

Every table is exported to CSV and, with `write_workbook=True`, to a formatted workbook with a
linked contents sheet. `manifest.json` records conventions, the table mapping, display limits and
SHA-256 hashes of every artefact; an existing output directory is rejected. The numerical tables
retain every scenario, holding and grid point without PDF rounding, including the four
`Conditional factor shocks` tables, `Positions and payoff audit` (marks, intrinsic baselines,
basis offsets, coverage and boundary policy) and `Vanilla leg terms`. Additional plain
DataFrames can be attached to a copied `report_diagnostics` mapping with `dataclasses.replace`.

### Verification

The offline example harness executes both examples, and the package suite covers the engine:

~~~console
python -m pytest src/qis/tests/test_examples.py -k "instrument_portfolio_stress or composite_payoff_stress"
python -m pytest --pyargs qis.portfolio.stress.tests
~~~

The checks cover zero-shock mark identity, full attribution reconciliation, the Credit split,
Euler additivity, nonzero exposure on a zero-mark future, exact local FX conversion, the custom
terminal payoff and finite-difference factor sensitivities, label validation, ambiguous factor
instructions, shared residuals and kink policies.

#### Full source: funded and mixed portfolios

~~~{literalinclude} ../examples/portfolios/instrument_portfolio_stress.py
:language: python
~~~

#### Full source: a custom payoff

~~~{literalinclude} ../examples/portfolios/composite_payoff_stress.py
:language: python
~~~

## Interpretation and limitations

- **Intrinsic, not priced.** Derivatives move by the change in intrinsic value; the basis offset
  carrying time value is frozen. There is no volatility, theta, skew or barrier path. A written
  option far from its strike shows no stress P&L even when its time value would change.
- **Local risk is delta risk.** Exposures, Euler contributions and bands use the first derivative
  of the payoff. They omit curvature and strike or knockout crossings. A zero local delta, such as
  an out-of-the-money option, can give zero band width while nonlinear risk sits nearby.
- **Bands are pointwise and local.** One and two conditional standard deviations correspond to
  about 68% and 95% only under the local Gaussian approximation. They are not nonlinear
  confidence intervals, not simultaneous over the grid, and exclude parameter, covariance-regime
  and tail uncertainty. Additive lower bounds are not clipped.
- **Frozen snapshot.** Holdings, marks and the covariance are fixed at the model date. There is no
  rebalancing, margining, liquidation, collateral haircut or settlement path; the stressed value
  is not executable proceeds or lending value.
- **Point in time.** The model date must be an exact key no later than the valuation date, and
  replayed months after the valuation date are excluded. The loadings and covariance may still
  have been estimated on a sample that contains the replayed months, so the replay is
  descriptive rather than an out-of-sample test.
- **Model consistency.** `annual_total_vol` uses the supplied $\Omega$; the Euler exhibits use
  the factor model. They differ when
  $\Omega\neq B\Sigma B^{\top}+\operatorname{diag}(\sigma^2_{\varepsilon})$, and qis shows
  both instead of reconciling them.
- **Positive quotes only.** Shocks are multiplicative, so a nonpositive quote, such as a negative
  futures price or a spread quoted around zero, fails explicitly; additive quote models are
  outside this version.
- **Denominator and families.** $V$ is a declared capital amount, not automatically NAV. Family
  weights are the caller's economic choice, and overlapping families revert to atomic factors in
  every additive exhibit.

## See also

- [Factor stress testing](stress_testing.md): conditional anchors, the Schur complement and the
  funded prediction band.
- [Stress testing with options](stress_testing_with_options.md): full time-value option
  repricing.
- [Portfolio risk and Euler contributions](risk_contributions.md)
- [Factor risk models](factor_risk_models.md)
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [FX hedging and market data](fx_hedging_and_market_data.md)
- [Factsheets and reporting](factsheets_and_reporting.md)
- [Notation and conventions](notation_and_conventions.md)
- [Packaged instrument-portfolio conventions note](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/portfolio_stress.md)

~~~{toctree}
:hidden:

_included/portfolio_stress
~~~

## References

1. Kupiec, P. H. (1998). Stress testing in a value at risk framework. *Journal of Derivatives*, 6(1), 7–24. [DOI: 10.3905/jod.1998.408008](https://doi.org/10.3905/jod.1998.408008). Fixed anchors with free factors completed by the covariance regression.
2. Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis*, 3rd edition. Wiley. Conditional distributions of the multivariate normal (Section 2.5).
3. Kim, J., and Finger, C. C. (2000). A stress test to incorporate correlation breakdown. *Journal of Risk*, 2(3). [Publisher page](https://www.risk.net/journal-risk/2161074/stress-test-incorporate-correlation-breakdown). Conditional covariance of the free factors.
4. Black, F., and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654. [DOI: 10.1086/260062](https://doi.org/10.1086/260062). The time-value pricing that the intrinsic proxy deliberately freezes.
5. Tasche, D. (2008). Capital allocation to business units and sub-portfolios: the Euler principle. Working paper. [arXiv:0708.2542](https://arxiv.org/abs/0708.2542). The Euler allocation of a homogeneous risk measure.
6. Litterman, R. (1996). Hot Spots and Hedges. *Goldman Sachs Risk Management Series*. Marginal risk contributions and hedges as negative contributors.
7. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
