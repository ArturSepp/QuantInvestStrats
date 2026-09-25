---
myst:
  html_meta:
    description: >-
      Information coefficients, the IC information ratio and pooled predictive regressions of
      lagged signals on cross-sectionally normalised forward returns, as implemented in
      qis.estimate_signal_diagnostics, qis.compute_ic_timeseries and qis.estimate_ic_ir.
---

# Signal diagnostics: information coefficient and information ratio

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A signal diagnostic asks whether a score known at one date ranks the returns that follow it. The
information coefficient (IC) is the cross-sectional correlation between the lagged score and the
subsequent return at one date; the IC information ratio is the time-series mean of the IC divided
by its standard deviation, and measures how consistently the ranking works. qis estimates both,
together with a pooled predictive regression, from point-in-time and non-overlapping
(signal, forward return) pairs.

## Overview

| Question | qis entry point | Output |
|---|---|---|
| Does the lagged signal predict the cross-section of forward returns? | `qis.estimate_signal_diagnostics` | `pooled_universe`: `n`, `beta`, `se`, `t_stat`, `IC_pearson`, `IC_spearman` per horizon |
| In which segment does it work? | `estimate_signal_diagnostics(group_data=...)` | `per_group`: the same columns per (horizon, group) |
| How dispersed is the predictive slope across assets? | `qis.compute_per_asset_betas` | One time-series $\beta$ per (horizon, asset) |
| Is the IC stable from date to date? | `qis.compute_ic_timeseries`, `qis.estimate_ic_ir` | Per-date IC series; mean, standard deviation, IC ratio, t-statistic and hit rate |
| What does it look like? | `qis.plot_signal_diagnostics` and four related plots | Conditional-return and $\beta$ boxplots |

The computation has three layers. **Pairing** builds, for each asset and horizon, pairs of the
signal known at $t-1$ and the return that follows, on the asset's own sampling grid and without
overlapping windows. **Normalisation** demeans the forward returns across names at each date and
divides them by their cross-sectional standard deviation. **Estimation** summarises the pairs in
two ways: a single pooled regression over all pairs, and a time series of one IC per date.

The two summaries answer different questions. The pooled regression treats every
(asset, date) pair as an independent observation. The IC series collapses each date into one
observation, in the spirit of [Fama and MacBeth (1973)](https://doi.org/10.1086/260061), so
its t-statistic reflects the date-to-date variability of the signal's efficacy. A signal is
credible when both agree.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns by default (`is_log_returns=True`): $h$ native log returns are summed; with `False`, simple returns are compounded. Returns are used as supplied, total or excess |
| Sampling grid | Each asset's native grid, the pandas frequency key of `asset_returns_dict`; integer $h$ counts native periods and pairs are taken every $h$-th date; a string horizon resamples all NAVs to that frequency |
| Annualisation | None for $\hat\beta$, the ICs and the pooled t-statistic; `IC_IR_an` multiplies the IC ratio by $\sqrt{\widehat{\mathrm{AN}}}$, with $\widehat{\mathrm{AN}}$ from `periods_per_year` or 365.25 over the median day gap of the IC dates |
| Mean adjustment | Forward returns are demeaned across names at each date and divided by their cross-sectional $s_t$ (`ddof=1`); signals are not transformed; the pooled regression has no intercept; $s(\mathrm{IC})$ uses `ddof=1` |
| Timing | The signal last observed in the native period ending at $t-1$ is paired with the return over $(t-1,t-1+h]$; qis applies exactly one native-period lag |
| Output units | $\hat\beta$ in cross-sectional standard deviations of forward return per unit of signal; ICs, t-statistics and IC ratios dimensionless; hit rate a fraction |
| qis default | `estimate_signal_diagnostics(horizons=(1, 3, 6), fit_intercept=False, is_log_returns=True, is_vol_normalised=True, min_obs_per_date=5, min_obs_per_group=10)`; `estimate_ic_ir(method='spearman', return_col='r_norm_univ', periods_per_year=None, min_obs_per_date=5)` |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| `asset_returns_dict` | Returns per native frequency, e.g. `{'ME': monthly, 'QE': quarterly}` | Each asset in one frame; the index must carry that frequency's period-end labels |
| `signal` | Panel of scores, dates by assets | Any frequency; resampled to each native grid with the last value per period |
| $z_{i,t}$ | Signal of asset $i$, last value observed in the native period ending at $t$ | Signal units; not normalised by qis |
| $h$ | Forward horizon | Integer count of the asset's native periods, or a pandas frequency string |
| $y_{i,t}$ | Forward return paired with $z_{i,t-1}$ | Over $(t-1,t-1+h]$; column `r` of the pairs frame |
| $\mathcal{A}_t$, $n_t$ | Names with a finite pair at regression date $t$, and their number | All frequency frames pooled; $n_t\ge$ `min_obs_per_date` |
| $\bar y_t$, $s_t(y)$ | Cross-sectional mean and standard deviation of $y_{\cdot,t}$ | Over $\mathcal{A}_t$, `ddof=1` |
| $\bar z_t$, $s_t(z)$ | Cross-sectional mean and standard deviation of $z_{\cdot,t-1}$ | Over $\mathcal{A}_t$, `ddof=1` |
| $x_{i,t}$, $x^{g}_{i,t}$ | Forward return normalised across all names, or within group $g$ | Columns `r_norm_univ` and `r_norm_group` |
| $n$ | Number of pooled pairs, $n=\sum_t n_t$ | Column `n` |
| $T$ | Number of dates in an IC series | Column `n_dates` |
| $\hat\beta$, $\varepsilon_{i,t}$, $\hat\sigma^2_{\varepsilon}$ | Pooled slope, residual and residual variance | No intercept by default |
| $\mathcal{T}_{\beta}$, $\mathcal{T}_{\mathrm{IC}}$ | t-statistics of the pooled slope and of the mean IC | Calligraphic, so that $t$ stays a date |
| $\mathrm{IC}^{\mathrm{P}}_t$, $\mathrm{IC}^{\mathrm{S}}_t$ | Pearson and Spearman IC at date $t$ | In $[-1,1]$ |
| $\overline{\mathrm{IC}}$, $s(\mathrm{IC})$ | Time-series mean and standard deviation of the per-date IC | Over $T$ dates, `ddof=1` |
| $\mathrm{IR}_{\mathrm{IC}}$ | IC information ratio | Per IC period; column `IC_IR` |
| $\widehat{\mathrm{AN}}$ | Inferred periods per year of the IC dates | 365.25 divided by the median gap in days |
| $\mathrm{HR}$ | Hit rate | Fraction of dates with $\mathrm{IC}_t>0$ |
| $\mathrm{BR}$ | Breadth: independent bets per year | Grinold (1989) |
| $\Phi$ | Standard normal distribution function | |
| $w_{i,t}$ | Signal standardised across names at date $t$ | Unit `ddof=1` dispersion |
| $F$, $e_j$ | String-horizon frequency and its period ends | Pandas frequency, e.g. `'YE'` |
| $t_0$ | First date common to the return index and the resampled signal | Fixes the sampling phase |
| $\eta_k$, $\sigma_{\eta}$, $Y_t$ | Uncorrelated increments, their volatility, and their overlapping $h$-sums | Overlap proposition only |
| $a$, $b$, $\pi$, $C$ | Centred score and return vectors, a random permutation, their correlation | Null-dispersion proof only |
| $c$ | Constant added to every signal value | Level-shift pitfall only |

The inputs must satisfy three conditions. First, the index of each frame must equal the
period-end labels that `signal.resample(key)` produces: month-end data stored under `'ME'` must
sit on calendar month-ends. A business-month-end index under the `'ME'` key keeps only the
months whose last business day is the calendar month-end and silently drops the others; store
such data under `'BME'`. Second, the signal value dated $t$ must use only information available
at $t$; qis lags it once and cannot detect a mis-dated input. Third, the default
`is_log_returns=True` is the opposite of the `qis.to_returns` and
`qis.compute_asset_returns_dict` defaults, which produce simple returns; pass the flag that
matches the data.

## Methodology

### Pairing signals with forward returns

#### Integer horizons in native cadence

**Definition (pair).** Let $t$ run over the native grid of asset $i$, the index of its frame in
`asset_returns_dict`. The signal is resampled to that grid with the last finite value of each
period and shifted by one period. The pair dated $t$ is $(z_{i,t-1},y_{i,t})$ with

$$
y_{i,t}=\sum_{k=0}^{h-1}\ell_{i,t+k}\quad\text{(default)},
\qquad
y_{i,t}=\prod_{k=0}^{h-1}\big(1+r_{i,t+k}\big)-1\quad\text{(simple returns)} .
$$

The return $\ell_{i,t}$ covers $(t-1,t]$, so the signal known at $t-1$ predicts the return over
$(t-1,t-1+h]$. A window with any missing native return gives a missing $y_{i,t}$, and a pair is
kept only when both of its values are finite. For $h>1$ the window extends $h-1$ periods beyond
the date label $t$.

The horizon is counted in each asset's own periods: $h=1$ is one month for an asset stored under
`'ME'` and one quarter for an asset stored under `'QE'`. A quarterly asset is never forced onto a
monthly grid, which would create zero-then-jump returns. At a quarter-end, the quarterly
asset's one-quarter return and the monthly assets' one-month returns enter the same
cross-section.

#### Non-overlapping sampling

The regression dates are every $h$-th date of the grid, $t_0,t_0+h,t_0+2h,\dots$, where $t_0$ is
the first date common to the return index and the resampled signal index. Consecutive windows
$(t-1,t-1+h]$ and $(t-1+h,t-1+2h]$ are then adjacent and disjoint. Sampling every date instead
would make neighbouring forward returns share $h-1$ native returns.

**Proposition (overlapping windows).** Let $\eta_k$ be uncorrelated increments with common
variance $\sigma^2_{\eta}$, and let $Y_t=\sum_{k=t}^{t+h-1}\eta_k$ be observed at every date
$t=1,\dots,T$. Then, as $T/h\to\infty$,

$$
T\operatorname{Var}\big(\bar Y\big)\to h^2\sigma^2_{\eta},
\qquad\text{while}\qquad
\operatorname{Var}(Y_t)=h\,\sigma^2_{\eta} .
$$

The textbook standard error $s(Y)/\sqrt{T}$ therefore understates the true one by the factor
$\sqrt h$, and a t-statistic built from it is inflated by $\sqrt h$.

**Proof.** $\operatorname{Cov}(Y_t,Y_{t+k})=(h-k)\,\sigma^2_{\eta}$ for $0\le k<h$ and zero
beyond. Hence

$$
T\operatorname{Var}(\bar Y)=\sum_{\lvert k\rvert<h}\Big(1-\frac{\lvert k\rvert}{T}\Big)(h-\lvert k\rvert)\,\sigma^2_{\eta}
\;\to\;\Big(h+2\sum_{k=1}^{h-1}(h-k)\Big)\sigma^2_{\eta}=h^2\sigma^2_{\eta} .
$$

The limit uses $2\sum_{k=1}^{h-1}(h-k)=h(h-1)$. $\square$

With every $h$-th date, the $T/h$ windows are uncorrelated and the variance of their mean is
$h\sigma^2_{\eta}/(T/h)=h^2\sigma^2_{\eta}/T$: the same precision, now estimated correctly by
the independent-observation formula. For a mean, overlap adds no information and only
corrupts the standard error. A persistent signal passes the same overlap to the regression
products $z_{i,t-1}y_{i,t}$. The alternative that keeps all windows is a HAC standard error
([Newey and West, 1987](https://www.nber.org/papers/t0055)); see
[regression and HAC inference](regression_and_hac.md).

The phase $t_0$ is fixed by the first common date. For $h>1$ the other $h-1$ phases are equally
valid subsamples and give different estimates; comparing them is a cheap robustness check.

#### String horizons

**Definition (string horizon).** For a pandas frequency $F$ with period ends
$\dots<e_{j-1}<e_j$, qis rebuilds a level from each asset's native returns,
$P_{i,t}=\exp\big(\sum_{k\le t}\ell_{i,k}\big)$ or $P_{i,t}=\prod_{k\le t}(1+r_{i,k})$,
concatenates the assets, forward-fills, keeps the last level in each $F$ period, forward-fills
again, and pairs

$$
y_{i,e_j}=\log\frac{P_{i,e_j}}{P_{i,e_{j-1}}}\quad\text{or}\quad\frac{P_{i,e_j}}{P_{i,e_{j-1}}}-1,
\qquad
z_{i,e_{j-1}}=\text{last signal value in }(e_{j-2},e_{j-1}] .
$$

The pair is dated $e_j$, the end of its window, and each asset contributes one non-overlapping
pair per $F$ period. A string horizon overrides the native cadence: `'YE'` gives every asset
calendar-year returns. For month-end assets, `'YE'` has the same pair content as $h=12$ when the
integer phase starts at a January month-end; only the date label differs (December against
January). A string frequency finer than an asset's native grid reintroduces the
zero-then-jump returns that integer horizons avoid.

### Cross-sectional normalisation

**Definition (normalised forward return).** At regression date $t$, let $\mathcal{A}_t$ hold the
names, from all frequency frames, with a finite pair dated $t$, and $n_t=\lvert\mathcal{A}_t\rvert$.
If $n_t\ge$ `min_obs_per_date` (default 5) and $s_t(y)>0$,

$$
x_{i,t}=\frac{y_{i,t}-\bar y_t}{s_t(y)},
\qquad
\bar y_t=\frac{1}{n_t}\sum_{i\in\mathcal{A}_t}y_{i,t},
\qquad
s_t(y)^2=\frac{1}{n_t-1}\sum_{i\in\mathcal{A}_t}\big(y_{i,t}-\bar y_t\big)^2 ;
$$

otherwise every pair of that date is dropped. With `is_vol_normalised=False`, $s_t(y)$ is
replaced by 1 and the returns are only demeaned. Only the returns are normalised; the signal
enters the regression exactly as supplied.

**Identity (normalised moments).** On every retained date $\sum_{i\in\mathcal{A}_t}x_{i,t}=0$, and
with `is_vol_normalised=True` also $\sum_{i\in\mathcal{A}_t}x_{i,t}^2=n_t-1$.

**Proof.** Subtracting $\bar y_t$ centres the values; dividing by $s_t(y)$ makes their sum of
squares $(n_t-1)s_t(y)^2/s_t(y)^2$. $\square$

The group version $x^{g}_{i,t}$ applies the same formula over the names of group $g$ in
$\mathcal{A}_t$. It needs at least two such names and a positive group standard deviation, and it
exists only on dates retained by the universe rule. Otherwise, and for names without a group
label, it is missing.

> **Insight.** Demeaning at each date removes everything common to all names on that date: the
> market return, a common cash return, a common currency translation. For log returns and for
> one-period simple returns the removal is exact, so total and excess returns give the same
> diagnostics. Dividing by $s_t(y)$ puts calm and turbulent months on the same footing, so a
> crisis month with wide dispersion does not dominate the pooled fit.

### Pooled predictive regression

**Definition (pooled regression).** Over all retained pairs of one horizon, with $n\ge5$ finite
pairs and $\sum z^2>0$,

$$
\begin{aligned}
\hat\beta&=\frac{\sum_{t}\sum_{i\in\mathcal{A}_t}z_{i,t-1}\,x_{i,t}}{\sum_{t}\sum_{i\in\mathcal{A}_t}z_{i,t-1}^2},
\qquad
\varepsilon_{i,t}=x_{i,t}-\hat\beta\,z_{i,t-1},\\
\hat\sigma^2_{\varepsilon}&=\frac{1}{n-1}\sum_{t}\sum_{i\in\mathcal{A}_t}\varepsilon_{i,t}^2,
\qquad
\operatorname{se}(\hat\beta)=\sqrt{\frac{\hat\sigma^2_{\varepsilon}}{\sum_{t}\sum_{i\in\mathcal{A}_t}z_{i,t-1}^2}},
\qquad
\mathcal{T}_{\beta}=\frac{\hat\beta}{\operatorname{se}(\hat\beta)} .
\end{aligned}
$$

This is least squares through the origin with the classical standard error: homoskedastic,
independent pairs, $n-1$ residual degrees of freedom. Fewer than five pairs, or a zero signal,
give a row of missing values.

**Identity (slope and per-date ICs).** With `is_vol_normalised=True`, and with
$\mathrm{IC}^{\mathrm{P}}_t$ the Pearson correlation of $z_{\cdot,t-1}$ and $y_{\cdot,t}$ over
$\mathcal{A}_t$,

$$
\hat\beta=\frac{\sum_t (n_t-1)\,s_t(z)\,\mathrm{IC}^{\mathrm{P}}_t}{\sum_t\big[(n_t-1)\,s_t(z)^2+n_t\,\bar z_t^{\,2}\big]} .
$$

**Proof.** Because $\sum_i x_{i,t}=0$, $\sum_i z_{i,t-1}x_{i,t}=\sum_i(z_{i,t-1}-\bar z_t)x_{i,t}$,
which equals $(n_t-1)s_t(z)s_t(x)\,\mathrm{IC}^{\mathrm{P}}_t$ with $s_t(x)=1$; the correlation of
$z$ with $x$ equals that with $y$ because $x$ is a positive affine map of $y$. The denominator
splits as $\sum_i z_{i,t-1}^2=(n_t-1)s_t(z)^2+n_t\bar z_t^{\,2}$. Sum over $t$. $\square$

Two cases matter. If the signal is standardised across names at each date ($\bar z_t=0$,
$s_t(z)=1$), then $\hat\beta=\sum_t(n_t-1)\,\mathrm{IC}^{\mathrm{P}}_t/\sum_t(n_t-1)$ is a
breadth-weighted mean of the per-date Pearson ICs, and it also equals the pooled Pearson IC,
since both variables then have pooled mean zero and sum of squares $n-T$. If the signal has a
cross-sectional level $\bar z_t\ne0$, the level inflates the denominator and attenuates
$\hat\beta$, while no IC changes.

**Identity (pooled t-statistic for a standardised signal).** If $\bar z_t=0$ and $s_t(z)=1$ on every
date, then

$$
\mathcal{T}_{\beta}=\frac{\hat\beta\,\sqrt{n-1}}{\sqrt{1-\hat\beta^2}} .
$$

**Proof.** Here $\sum z^2=\sum x^2=n-T$ and $\sum zx=\hat\beta(n-T)$, so
$\sum\varepsilon^2=\sum x^2-\hat\beta^2\sum z^2=(n-T)(1-\hat\beta^2)$. Then
$\operatorname{se}(\hat\beta)^2=(1-\hat\beta^2)/(n-1)$. $\square$

The pooled t-statistic is thus a function of the average IC and the pair count alone. It cannot
tell a signal that works a little every month from one that works strongly in a few months and
fails in the rest; the IC series below can.

#### Per-group and per-asset fits

With `group_data`, the same estimator is fitted, for every group label, to the pairs
$(z_{i,t-1},x^{g}_{i,t})$ with finite $x^{g}_{i,t}$, and reported when at least
`min_obs_per_group` (default 10) pairs remain. Groups appear in `group_order`, or in their order
of first appearance in `group_data`.

`qis.compute_per_asset_betas` fits the same estimator to one asset's time series of
$(z_{i,t-1},x_{i,t})$ pairs, on the universe-normalised returns, and keeps assets with at least
`min_obs_per_asset` (default 12) pairs. Its $\beta_i$ measures whether the asset's relative
performance follows its own signal through time. Because there is no intercept, a signal that
is persistently high for one asset contributes through its time-series mean as well.

#### Fitting an intercept

With `fit_intercept=True`, qis calls `scipy.stats.linregress` on the pooled pairs:

$$
\hat\beta_{\alpha}=\frac{\sum_{t,i}(z_{i,t-1}-\bar z)(x_{i,t}-\bar x)}{\sum_{t,i}(z_{i,t-1}-\bar z)^2},
\qquad
\hat\alpha=\bar x-\hat\beta_{\alpha}\bar z=-\hat\beta_{\alpha}\,\bar z ,
$$

where $\bar z$ and $\bar x$ are pooled means over all pairs and $\bar x=0$ by the normalised-moments
identity. The standard error uses $n-2$ degrees of freedom, `IC_pearson` is the linregress
correlation (the pooled Pearson IC), and $\hat\alpha$ is not reported. The docstring says that
cross-sectional demeaning of the left-hand side makes $\alpha=0$ by construction. That holds only
when the pooled mean of the signal is zero. Otherwise $\hat\alpha=-\hat\beta_{\alpha}\bar z\ne0$
and the two fits give different slopes. The intercept removes a constant signal level, but not
a level that varies by date.

### Information coefficients

**Definition (information coefficient).** At regression date $t$,

$$
\mathrm{IC}^{\mathrm{P}}_t=\frac{\sum_{i\in\mathcal{A}_t}(z_{i,t-1}-\bar z_t)\,x_{i,t}}{\sqrt{\sum_{i\in\mathcal{A}_t}(z_{i,t-1}-\bar z_t)^2\,\sum_{i\in\mathcal{A}_t}x_{i,t}^2}},
$$

and $\mathrm{IC}^{\mathrm{S}}_t$ is the same correlation computed on the ranks of $z_{\cdot,t-1}$
and $x_{\cdot,t}$, the rank correlation of
[Spearman (1904)](https://doi.org/10.2307/1412159). Ties receive average ranks (`scipy`).
The rank IC is insensitive to outlying returns and to monotone transformations of the signal.

**Identity (invariance to normalisation).** The per-date Pearson and Spearman ICs computed on
$x_{\cdot,t}$ equal those computed on the raw forward returns $y_{\cdot,t}$.

**Proof.** At a retained date $x_{i,t}$ is a strictly increasing affine function of $y_{i,t}$.
Pearson correlation is invariant under positive affine maps of either argument, and ranks under
strictly increasing maps. $\square$

The per-date IC therefore depends neither on `is_vol_normalised` nor on market or cash
components. The pooled `IC_pearson` and `IC_spearman` of `pooled_universe` are different
objects: correlations over all $n$ pairs at once, not averages of per-date ICs. A signal level
that drifts through time adds variance to the pooled $z$ but no covariance with $x$, whose mean is
zero at every date, so it dilutes the pooled ICs; the per-date ICs are immune to it.

**Identity (IC as a portfolio return).** Let $w_{i,t}=(z_{i,t-1}-\bar z_t)/s_t(z)$ be the signal
standardised across names. Then

$$
\sum_{i\in\mathcal{A}_t}w_{i,t}\,x_{i,t}=(n_t-1)\,\mathrm{IC}^{\mathrm{P}}_t ,
$$

and $\mathrm{IC}^{\mathrm{P}}_t$ is also the least-squares slope, with intercept, of $x_{\cdot,t}$
on $w_{\cdot,t}$ at date $t$.

**Proof.** $\sum_i w_{i,t}=0$ and $\sum_i w_{i,t}^2=\sum_i x_{i,t}^2=n_t-1$, so the Pearson
correlation of $w$ and $x$ is $\sum_i w_{i,t}x_{i,t}/(n_t-1)$, and standardising $z$ does not
change the correlation. The slope is the ratio $\sum_i w_{i,t}x_{i,t}/\sum_i w_{i,t}^2$, the same
number. $\square$

The IC series is thus, up to the factor $n_t-1$, the return on normalised returns of a
dollar-neutral portfolio whose weights have unit cross-sectional dispersion. It is also the
series of per-date cross-sectional slopes of Fama and MacBeth (1973).

**Proposition (null dispersion of a per-date IC).** Fix a date with $n_t\ge3$ names and
non-constant scores and returns. Suppose that, given the scores, every assignment of the realised
returns to the names is equally likely: no predictive content, exchangeable names. Then both the
Pearson and the Spearman IC satisfy

$$
\mathbb{E}\big[\mathrm{IC}_t\big]=0,
\qquad
\operatorname{Var}\big(\mathrm{IC}_t\big)=\frac{1}{n_t-1} .
$$

**Proof.** Both statistics have the form $C=\sum_i a_i b_{\pi(i)}/\sqrt{\sum a^2\sum b^2}$ with
fixed centred vectors $a$ (scores or their ranks) and $b$ (returns or their ranks) and a uniform
random permutation $\pi$. Since $\mathbb{E}[b_{\pi(i)}]=0$, $\mathbb{E}[C]=0$. For $i\ne j$,
$\mathbb{E}[b_{\pi(i)}b_{\pi(j)}]=-\sum b^2/(n_t(n_t-1))$, and
$\mathbb{E}[b_{\pi(i)}^2]=\sum b^2/n_t$. With $\sum_{i\ne j}a_ia_j=-\sum a^2$,

$$
\operatorname{Var}\Big(\sum_i a_ib_{\pi(i)}\Big)
=\frac{\sum b^2}{n_t}\Big(\sum a^2+\frac{\sum a^2}{n_t-1}\Big)
=\frac{\sum a^2\sum b^2}{n_t-1} .
$$

Dividing by $\sum a^2\sum b^2$ gives the variance of $C$. $\square$

Pure noise gives a per-date IC with standard deviation 0.50 on 5 names, the default minimum,
0.23 on 20 names and 0.10 on 101 names. Two consequences follow for the pooled regression. Under
this null, independently across dates, a standardised signal gives
$\operatorname{Var}(\hat\beta)=1/(n-T)$, whereas the implemented standard error is close to
$1/\sqrt{n-1}$: $\mathcal{T}_{\beta}$ is overstated by about $\sqrt{n_t/(n_t-1)}$, a factor of
1.12 on 5 names and 1.03 on 20. The pooled residual variance does not charge for the mean and
scale estimated at each date. Second, residual correlation across names (industries, styles,
regions) raises $\operatorname{Var}(\mathrm{IC}_t)$ above $1/(n_t-1)$, which the pooled
t-statistic cannot see at all.

### The IC information ratio

**Definition (IC information ratio).** `qis.compute_ic_timeseries` computes one IC per date
(Spearman by default), skipping dates with fewer than `min_obs_per_date` names or a constant
signal or return. For the resulting $T$ values, `qis.estimate_ic_ir` reports

$$
\begin{aligned}
\overline{\mathrm{IC}}&=\frac{1}{T}\sum_{t}\mathrm{IC}_t,
\qquad
\mathrm{IR}_{\mathrm{IC}}=\frac{\overline{\mathrm{IC}}}{s(\mathrm{IC})},
\qquad
\mathrm{IR}^{\mathrm{ann}}_{\mathrm{IC}}=\mathrm{IR}_{\mathrm{IC}}\sqrt{\widehat{\mathrm{AN}}},\\
\mathcal{T}_{\mathrm{IC}}&=\mathrm{IR}_{\mathrm{IC}}\sqrt{T},
\qquad
\mathrm{HR}=\frac{1}{T}\sum_{t}\mathbf{1}\big\{\mathrm{IC}_t>0\big\},
\end{aligned}
$$

in the columns `mean_IC`, `IC_IR`, `IC_IR_an`, `t_stat` and `hit_rate`, together with
`n_dates` $=T$ and `std_IC` $=s(\mathrm{IC})$ with `ddof=1`. The ratio is missing when
$s(\mathrm{IC})=0$ or $T=1$. A date with an IC of exactly zero counts as a miss.

**Proposition (IC t-statistic).** $\mathcal{T}_{\mathrm{IC}}$ is the one-sample t-statistic of the
hypothesis $\mathbb{E}[\mathrm{IC}_t]=0$. If the $\mathrm{IC}_t$ are independent and identically
normally distributed with mean zero, it has a Student t distribution with $T-1$ degrees of freedom.

**Proof.** $\mathrm{IR}_{\mathrm{IC}}\sqrt{T}=\overline{\mathrm{IC}}/\big(s(\mathrm{IC})/\sqrt{T}\big)$,
the sample mean divided by its estimated standard error; the distribution is the textbook
one-sample result. $\square$

With `method='pearson'`, the IC-as-portfolio identity makes $\mathcal{T}_{\mathrm{IC}}$ exactly the
Fama and MacBeth (1973) t-statistic of the per-date slope of normalised returns on the
standardised signal. It is robust to any correlation across names within a date, because each
date contributes one number. It is not robust to serial correlation of the IC. Non-overlapping
windows remove the mechanical part; a persistent signal regime can leave some behind, which a HAC
standard error on the IC series handles. If $\mathrm{IC}_t$ is normal,
$\mathrm{HR}=\Phi(\mathrm{IR}_{\mathrm{IC}})$ in expectation; a hit rate well below that value
points to a skewed IC: many small wins and a few large losses.

#### Annualisation of the IC ratio

$\widehat{\mathrm{AN}}$ is `periods_per_year` if given, and otherwise 365.25 divided by the
median gap in days between consecutive IC dates. The inference adapts to the horizon, because
non-overlapping $h$-period pairs are $h$ periods apart, but it is not the calendar convention of
`qis.get_annualization_factor`. For samples of a year or more:

| IC dates | Median gap (days) | $\widehat{\mathrm{AN}}$ | $\mathrm{AN}$ convention |
|---|---:|---:|---:|
| Month-ends, $h=1$ | 31 | 11.78 | 12 |
| Month-ends, $h=3$ | 92 | 3.97 | 4 |
| Quarter-ends, $h=1$ | 91.5–92 | 3.97–3.99 | 4 |
| Weekly | 7 | 52.18 | 52 |
| Business days | 1 | 365.25 | 252 |

The month-end difference is small: $\sqrt{11.78/12}=0.991$. The business-day case overstates
the annualised ratio by $\sqrt{365.25/252}=1.20$. A `periods_per_year` argument applies to every
horizon in the call, so `periods_per_year=12` is wrong for the $h=3$ row of the same table; call
`qis.estimate_ic_ir` once per horizon grid when overriding.

#### Relation to the fundamental law of active management

[Grinold (1989)](https://doi.org/10.3905/jpm.1989.409211) states that the information ratio of
an optimally constructed active portfolio is approximately

$$
\mathrm{IR}\approx\mathrm{IC}\,\sqrt{\mathrm{BR}},
$$

where $\mathrm{BR}$ is the number of independent forecasts per year. The law assumes forecasts of
equal skill on independent residual returns, active weights proportional to the risk-adjusted
forecasts with no constraints or costs, and a small IC. Grinold and Kahn (2000) develop it and
its refinements; it is an approximation, not an identity.

The per-date IC gives it a direct empirical form. By the IC-as-portfolio identity, the Pearson
$\mathrm{IR}_{\mathrm{IC}}$ is the information ratio of the unit-dispersion signal portfolio on
normalised returns, exactly when $n_t$ is constant. If the true IC is constant and the only
variation is sampling noise, the null-dispersion proposition gives, to first order in the IC,
$s(\mathrm{IC})\approx1/\sqrt{n_t-1}$ for the Pearson and the rank IC alike, hence

$$
\mathrm{IR}_{\mathrm{IC}}\approx\overline{\mathrm{IC}}\,\sqrt{n_t-1},
\qquad
\mathrm{IR}^{\mathrm{ann}}_{\mathrm{IC}}\approx\overline{\mathrm{IC}}\,\sqrt{(n_t-1)\,\mathrm{AN}} ,
$$

the fundamental law with $\mathrm{BR}=(n_t-1)\,\mathrm{AN}$; demeaning uses one degree of freedom
per date. Read backwards, $1/s(\mathrm{IC})^2$ is an effective breadth per period. It falls below
$n_t-1$ when names are correlated after normalisation or when the true IC varies through time.

The ratio of the two t-statistics measures the same thing. For a standardised signal,
$\mathcal{T}_{\beta}\approx\overline{\mathrm{IC}}\sqrt{n}$ and
$\mathcal{T}_{\mathrm{IC}}=\overline{\mathrm{IC}}\sqrt{T}/s(\mathrm{IC})$, so
$(\mathcal{T}_{\beta}/\mathcal{T}_{\mathrm{IC}})^2\approx n_t\,s(\mathrm{IC})^2$. This is the
design effect of cluster sampling (Kish, 1965), with dates as clusters. Values well above one
mean that the pooled t-statistic overstates the evidence.

## Worked example

The panel has 20 names and 61 month-ends from January 2020 to January 2025. The signal is standard
normal and standardised across names at each date. Next month's log return is a common market
return plus 5% times the sum of $0.1\,z_{i,t-1}$ and independent standard normal noise. The
population Pearson IC is $0.1/\sqrt{1.01}=0.0995$ and the population rank IC is about 0.095. The
inputs are synthetic, generated with a fixed seed.

```python
import numpy as np
import pandas as pd
from scipy import stats

import qis


def assert_quoted(values, quoted, half_unit):
    """Check each value against the rounded number quoted in the text."""
    gap = np.abs(np.asarray(values, dtype=float) - np.asarray(quoted, dtype=float))
    assert np.all(gap <= np.asarray(half_unit, dtype=float)), gap


rng = np.random.default_rng(20260725)
n_names, n_months = 20, 61
dates = pd.date_range('2020-01-31', periods=n_months, freq='ME')
names = [f'A{i:02d}' for i in range(n_names)]

# Scores standardised across names at every date: mean 0, ddof=1 standard deviation 1.
raw = rng.standard_normal((n_months, n_names))
score = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, ddof=1, keepdims=True)
# Log return over (t-1, t]: market + 5% x (0.1 x score at t-1 + noise).
lagged = np.vstack([np.zeros((1, n_names)), score[:-1]])
market = rng.normal(0.005, 0.04, size=(n_months, 1))
log_returns = market + 0.05 * (0.1 * lagged + rng.standard_normal((n_months, n_names)))

signal = pd.DataFrame(score, index=dates, columns=names)
returns = pd.DataFrame(log_returns, index=dates, columns=names)
groups = pd.Series(['G1'] * 10 + ['G2'] * 10, index=names)
result = qis.estimate_signal_diagnostics(asset_returns_dict={'ME': returns}, signal=signal,
                                         group_data=groups, horizons=(1, 3))

assert result.horizon_labels == ['1', '3']
assert [len(result.pairs[h]) for h in ('1', '3')] == [1200, 380]
assert result.pairs['3']['date'].nunique() == 19
assert (result.start_date, result.end_date) == (pd.Timestamp('2020-02-29'),
                                                pd.Timestamp('2025-01-31'))
```

The first month has no lagged signal, so horizon 1 has 60 regression dates and 1,200 pairs. At
horizon 3, every third month-end is sampled and the last window is incomplete, leaving 19 dates
and 380 pairs. The next block rebuilds the pooled regression with numpy from the same pairs:
$z$ is the score at $t-1$, and $x$ is next month's return normalised across names.

```python
z = score[:-1]
y = log_returns[1:]
x = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
n = z.size
beta = (z * x).sum() / (z * z).sum()
sigma2 = ((x - beta * z) ** 2).sum() / (n - 1)
se = np.sqrt(sigma2 / (z * z).sum())

pooled = result.pooled_universe.loc['1']
np.testing.assert_allclose(pooled[['n', 'beta', 'se', 't_stat']].to_numpy(dtype=float),
                           [n, beta, se, beta / se], rtol=1e-10)
np.testing.assert_allclose(pooled['IC_pearson'], np.corrcoef(z.ravel(), x.ravel())[0, 1],
                           rtol=1e-10)
np.testing.assert_allclose(pooled['IC_spearman'], stats.spearmanr(z.ravel(), x.ravel())[0],
                           rtol=1e-10)
assert_quoted([beta, se, beta / se, pooled['IC_spearman']], [0.0831, 0.0288, 2.89, 0.0613],
              [5e-5, 5e-5, 5e-3, 5e-5])

# Standardised signal: beta is the pooled and the mean per-date Pearson IC, and t depends on
# beta and n only.
ic_pearson = np.array([stats.pearsonr(z[k], y[k])[0] for k in range(len(z))])
np.testing.assert_allclose([pooled['IC_pearson'], ic_pearson.mean()], [beta, beta], rtol=1e-10)
np.testing.assert_allclose(beta / se, beta * np.sqrt(n - 1) / np.sqrt(1.0 - beta ** 2),
                           rtol=1e-10)

# Within-group normalisation for G1, and a per-asset time-series beta for A00.
x_g1 = (y[:, :10] - y[:, :10].mean(axis=1, keepdims=True)) / y[:, :10].std(
    axis=1, ddof=1, keepdims=True)
beta_g1 = (z[:, :10] * x_g1).sum() / (z[:, :10] ** 2).sum()
np.testing.assert_allclose(result.per_group.loc[('1', 'G1'), 'beta'], beta_g1, rtol=1e-10)
assert_quoted(result.per_group.xs('1')['beta'], [0.041, 0.106], 1e-3)
per_asset = qis.compute_per_asset_betas(result)
a00 = per_asset[(per_asset['horizon'] == '1') & (per_asset['asset'] == 'A00')].iloc[0]
np.testing.assert_allclose(a00['beta'], (z[:, 0] * x[:, 0]).sum() / (z[:, 0] ** 2).sum(),
                           rtol=1e-10)
```

The pooled slope is $\hat\beta=0.0831$ with standard error 0.0288 and $\mathcal{T}_{\beta}=2.89$.
Because the signal is standardised, $\hat\beta$ equals both the pooled Pearson IC and the mean of
the 60 per-date Pearson ICs, and $\mathcal{T}_{\beta}$ follows from $\hat\beta$ and $n=1200$
alone. The pooled rank IC is 0.0613. The two groups share one data-generating process, yet their
slopes are 0.041 and 0.106: with 600 pairs each, the standard error of each is about 0.04, so a
difference of this size between segments is not evidence of segment-specific skill.

```python
ic_rank = np.array([stats.spearmanr(z[k], y[k])[0] for k in range(len(z))])
ic_series = qis.compute_ic_timeseries(result)['1']
np.testing.assert_allclose(ic_series['IC'].to_numpy(dtype=float), ic_rank, rtol=1e-10)
assert (ic_series['n'] == n_names).all()

table = qis.estimate_ic_ir(result)
mean_ic, std_ic, n_dates = ic_rank.mean(), ic_rank.std(ddof=1), len(ic_rank)
an_hat = 365.25 / 31.0  # the median gap between month-ends is 31 days
expected = [n_dates, mean_ic, std_ic, mean_ic / std_ic, mean_ic / std_ic * np.sqrt(an_hat),
            mean_ic / std_ic * np.sqrt(n_dates), np.mean(ic_rank > 0.0)]
np.testing.assert_allclose(table.loc['1'].to_numpy(dtype=float), expected, rtol=1e-10)
np.testing.assert_allclose(table.loc['1', 't_stat'], stats.ttest_1samp(ic_rank, 0.0).statistic,
                           rtol=1e-10)
np.testing.assert_allclose(table.loc['3', 'IC_IR_an'],
                           table.loc['3', 'IC_IR'] * np.sqrt(365.25 / 92.0), rtol=1e-10)
assert_quoted(expected[1:], [0.0467, 0.2176, 0.215, 0.737, 1.66, 35 / 60],
              [5e-5, 5e-5, 5e-4, 5e-4, 5e-3, 1e-12])

# Pearson IC ratio: Fama-MacBeth view, effective breadth and design effect.
pearson = qis.estimate_ic_ir(result, method='pearson', periods_per_year=12.0).loc['1']
np.testing.assert_allclose([pearson['mean_IC'], pearson['std_IC']],
                           [beta, ic_pearson.std(ddof=1)], rtol=1e-10)
breadth = 1.0 / pearson['std_IC'] ** 2
design_effect = (pooled['t_stat'] / pearson['t_stat']) ** 2
assert_quoted([pearson['IC_IR'], pearson['t_stat'], breadth, design_effect],
              [0.376, 2.91, 20.5, 0.98], [5e-4, 5e-3, 5e-2, 5e-3])
```

The rank IC averages 0.0467 with a standard deviation of 0.2176 across the 60 months, so
$\mathrm{IR}_{\mathrm{IC}}=0.215$ per month. Annualised with the inferred
$\widehat{\mathrm{AN}}=365.25/31=11.78$ it is 0.737; with 12 it would be 0.744. The t-statistic
is 1.66, identical to `scipy.stats.ttest_1samp`, and the IC is positive in 35 of 60 months
($\mathrm{HR}=0.583$, against $\Phi(0.215)=0.585$). This sample's rank IC lies 1.7 standard errors
below its population value of 0.095. A genuine IC of 0.1 on 20 names over five years is
therefore only marginally detectable.

The Pearson IC series has mean 0.0831, equal to $\hat\beta$, and $\mathcal{T}_{\mathrm{IC}}=2.91$.
Its standard deviation, 0.221, is close to the pure-noise value $1/\sqrt{19}=0.229$, so the
effective breadth $1/s(\mathrm{IC})^2=20.5$ per month matches $n_t-1=19$. The design effect
$(\mathcal{T}_{\beta}/\mathcal{T}_{\mathrm{IC}})^2=0.98$ confirms that the names are independent
in this panel, so the pooled and the Fama–MacBeth views agree. On real data they usually do not.

```python
# A constant added to the signal changes no IC but shrinks the no-intercept slope by 1140/2340.
shifted = qis.estimate_signal_diagnostics(asset_returns_dict={'ME': returns}, signal=signal + 1.0,
                                          horizons=(1,)).pooled_universe.loc['1']
np.testing.assert_allclose(shifted['beta'], beta * (n - n_dates) / (n - n_dates + n), rtol=1e-10)
np.testing.assert_allclose(shifted[['IC_pearson', 'IC_spearman']].to_numpy(dtype=float),
                           pooled[['IC_pearson', 'IC_spearman']].to_numpy(dtype=float),
                           rtol=1e-10)
assert_quoted([shifted['beta'], shifted['t_stat']], [0.0405, 2.01], [5e-5, 5e-3])

# With an intercept the slope is recovered, but the intercept is -beta, not zero.
with_alpha = qis.estimate_signal_diagnostics(asset_returns_dict={'ME': returns},
                                             signal=signal + 1.0, horizons=(1,),
                                             fit_intercept=True).pooled_universe.loc['1']
np.testing.assert_allclose(with_alpha['beta'], beta, rtol=1e-10)
np.testing.assert_allclose(stats.linregress((z + 1.0).ravel(), x.ravel()).intercept, -beta,
                           rtol=1e-8)

# Look-ahead: a signal stamped one period before its information date.
leaky = qis.estimate_ic_ir(qis.estimate_signal_diagnostics(
    asset_returns_dict={'ME': returns}, signal=returns.shift(-1), horizons=(1,))).loc['1']
np.testing.assert_allclose([leaky['mean_IC'], leaky['hit_rate']], [1.0, 1.0], rtol=1e-12)
assert np.isnan(leaky['IC_IR'])  # zero IC dispersion leaves the ratio undefined

# Double lag: a signal already shifted by the caller is lagged again by qis.
stale = qis.estimate_ic_ir(qis.estimate_signal_diagnostics(
    asset_returns_dict={'ME': returns}, signal=signal.shift(1), horizons=(1,))).loc['1']
assert stale['n_dates'] == 59
assert_quoted([stale['mean_IC'], stale['t_stat']], [0.026, 0.81], [5e-4, 5e-3])
```

Adding 1 to every score leaves every IC unchanged but multiplies $\hat\beta$ by
$(n-T)/(n-T+n)=1140/2340$, to 0.0405, and cuts $\mathcal{T}_{\beta}$ to 2.01. Fitting an
intercept restores the slope of 0.0831, with an intercept of $-0.0831$ rather than zero. A
return series passed as the signal and shifted back one month is look-ahead by construction: the
mean rank IC is exactly 1, every month is a hit, and the ratio is undefined. Passing a signal that
is already lagged makes qis lag it a second time; the mean IC falls to 0.026 with a t-statistic
of 0.81, and the genuine signal is lost.

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| Forward return, integer horizon | $y_{i,t}$: $h$ native log returns summed, or simple returns compounded; every $h$-th date | `qis.estimate_signal_diagnostics(horizons=(1, 3, 6), is_log_returns=True)` |
| Forward return, string horizon | Rebuilt NAV, forward-filled, last value per $F$ period | `horizons=('YE',)`, mixable with integers |
| Lagged signal | Last value per native period, shifted one period | Built in; no argument |
| Normalised return | $x_{i,t}$, date dropped if $n_t<5$ or $s_t(y)\le0$ | `pairs[h]['r_norm_univ']`; `is_vol_normalised`, `min_obs_per_date` |
| Group-normalised return | $x^{g}_{i,t}$, at least two names per group and date | `pairs[h]['r_norm_group']`; `group_data` |
| Pooled slope | $\hat\beta$, $\operatorname{se}(\hat\beta)$, $\mathcal{T}_{\beta}$, $n$ | `pooled_universe` columns `beta`, `se`, `t_stat`, `n` |
| Pooled ICs | Pearson and Spearman correlation over all pairs | `pooled_universe` columns `IC_pearson`, `IC_spearman` |
| Slope with intercept | $\hat\beta_{\alpha}$ from `scipy.stats.linregress` | `fit_intercept=True` |
| Per-group slope | Pooled estimator on $(z,x^{g})$, at least 10 pairs | `per_group`, index (`horizon`, `group`); `min_obs_per_group`, `group_order` |
| Per-asset slope | Pooled estimator on one asset's time series, at least 12 pairs | `qis.compute_per_asset_betas(result, min_obs_per_asset=12, fit_intercept=False)` |
| Per-date IC | $\mathrm{IC}^{\mathrm{S}}_t$ or $\mathrm{IC}^{\mathrm{P}}_t$ with at least 5 names | `qis.compute_ic_timeseries(result, method='spearman', return_col='r_norm_univ', min_obs_per_date=5)` |
| IC summary | $T$, $\overline{\mathrm{IC}}$, $s(\mathrm{IC})$, $\mathrm{IR}_{\mathrm{IC}}$, $\mathrm{IR}^{\mathrm{ann}}_{\mathrm{IC}}$, $\mathcal{T}_{\mathrm{IC}}$, $\mathrm{HR}$ | `qis.estimate_ic_ir(result, method='spearman', return_col='r_norm_univ', periods_per_year=None, min_obs_per_date=5)` |
| Periods per year | 365.25 over the median day gap | Internal `_infer_periods_per_year` |
| Result container | Tables, pairs and labels | `qis.SignalDiagnosticsResult` |
| Column names | `n`, `beta`, `se`, `t_stat`, `IC_pearson`, `IC_spearman` | `qis.SignalDiagnosticsColumns` members `N`, `BETA`, `SE`, `T_STAT`, `IC_PEARSON`, `IC_SPEARMAN` |
| Conditional-return boxplot | $x$ by quantile bucket of $z$; title with $\hat\beta$, $\mathcal{T}_{\beta}$, pooled Pearson IC, $n$ | `qis.plot_signal_diagnostics_boxplot(result, horizon, num_buckets=10)` |
| Per-group $\beta$ boxplot | Per-asset no-intercept $\beta$ by group; per-group $\mathcal{T}_{\beta}$ annotated | `qis.plot_signal_diagnostics_group_boxplot(result, horizon, min_obs_per_asset=12)` |
| Composite figure | Boxplots per horizon, group row when groups exist | `qis.plot_signal_diagnostics(result, num_buckets=10, min_obs_per_asset=12)` |
| Estimate and plot | `estimate_signal_diagnostics`, then the composite figure | `qis.plot_signal_diagnostics_for_returns(asset_returns_dict, signal, horizons=(1, 2, 3, 6))` |
| Per-asset $\beta$ by horizon | `compute_per_asset_betas`, boxes coloured by native cadence | `qis.plot_signal_diagnostics_beta_boxplot(asset_returns_dict, signal, horizons=(1, 2, 3, 6), hue='asset_freq')` |

The estimators are in
[signal_diagnostics.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/signal_diagnostics.py)
and the plots in
[signal_diagnostics_plot.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/plots/derived/signal_diagnostics_plot.py).
API pages: {doc}`estimate_signal_diagnostics <api/generated/qis.estimate_signal_diagnostics>`,
{doc}`compute_ic_timeseries <api/generated/qis.compute_ic_timeseries>`,
{doc}`estimate_ic_ir <api/generated/qis.estimate_ic_ir>`,
{doc}`compute_per_asset_betas <api/generated/qis.compute_per_asset_betas>` and
{doc}`SignalDiagnosticsResult <api/generated/qis.SignalDiagnosticsResult>`.

**Result contract.** `qis.SignalDiagnosticsResult` holds:

- `pooled_universe`: one row per horizon, indexed by the label (`'1'`, `'3'`, `'YE'`), with the
  six `SignalDiagnosticsColumns`; a horizon without enough pairs gives a row of missing values.
- `per_group`: rows indexed by (`horizon`, `group`); empty, with that index, when `group_data` is
  `None`.
- `pairs`: per horizon label, a long frame with columns `date`, `asset`, `asset_freq`, `group`,
  `z`, `r`, `r_norm_univ` and `r_norm_group`, ordered by date and then asset. `r` is the raw
  forward return $y_{i,t}$.
- `horizon_labels`, `group_order`, and `start_date`/`end_date`, the first and last pair dates
  over all horizons.

**Validation.** A non-Series `group_data` raises `TypeError`. An empty `asset_returns_dict`, a
signal with no column in common with the returns, and a horizon that is neither a positive
integer nor a string raise `ValueError`. Assets absent from the signal are dropped without a
warning. An asset present in several frequency frames is assigned to the first one in dict order
(internal `_asset_to_freq_map`), also without a warning.

**Plot conventions.** Significance stars in plot titles mark $\lvert\mathcal{T}_{\beta}\rvert$ at
1.65, 1.96 and 2.58, the two-sided normal 10%, 5% and 1% levels, and inherit the pooled
t-statistic's independence assumption. The conditional-return boxplot reduces the number of
quantile buckets, down to three, when tied signal values produce duplicate edges, and falls back
to a single box below that. The two compute-and-plot wrappers default to `horizons=(1, 2, 3, 6)`,
not the estimator's `(1, 3, 6)`, and they use the estimator's defaults for `fit_intercept`,
`is_vol_normalised`, `min_obs_per_date` and `min_obs_per_group`.

## Interpretation and limitations

- **Pooled or per date.** $\hat\beta$ and the per-date ICs measure the same association with
  different weights, but $\mathcal{T}_{\beta}$ assumes independent pairs. Report
  $\mathcal{T}_{\mathrm{IC}}$ as the headline significance and the design effect
  $(\mathcal{T}_{\beta}/\mathcal{T}_{\mathrm{IC}})^2$ as a diagnostic.
- **Units of $\hat\beta$.** $\hat\beta$ is in cross-sectional standard deviations per unit of
  signal and depends on the signal's scale and level. Only for a signal standardised across names
  is it a breadth-weighted mean IC; otherwise compare signals by their ICs.
- **Mixed cadence.** At dates shared by monthly and quarterly assets, one-month and one-quarter
  returns are normalised together. Their dispersions differ, so the longer-horizon names tend to
  sit in the tails of the cross-section. Use `group_data` by cadence, or a string horizon, when
  that matters.
- **Pooled ICs.** `IC_pearson` and `IC_spearman` correlate across dates as well as names, are
  diluted by a signal level that drifts through time, and are not the means of the per-date ICs;
  read `mean_IC` from `qis.estimate_ic_ir` for the latter.
- **Association, not a backtest.** The diagnostics ignore costs, turnover, capacity and position
  limits. A significant IC is a necessary condition for a profitable strategy, not a sufficient
  one; see [portfolio backtesting](portfolio_backtesting.md).

### Pitfalls

> **Pitfall.** qis lags the signal by exactly one native period and trusts its date stamps. A
> signal computed with data published after its date, standardised with full-sample moments, or
> stamped at the start rather than the end of its period leaks future returns, as the look-ahead
> check above shows with a mean IC of 1. Conversely, a caller who lags the signal before passing
> it loses a period of information.

> **Pitfall.** Overlapping forward windows inflate t-statistics by about $\sqrt h$. qis samples
> integer horizons every $h$-th date for this reason. Do not rebuild the pairs with a rolling
> forward return on every date and reuse the qis t-statistics; use a HAC standard error if every
> window must be kept.

- **Small cross-sections.** Under the null, a per-date IC has standard deviation
  $1/\sqrt{n_t-1}$: 0.50 at the default minimum of 5 names. Dates this small add mostly noise to
  the IC series. The pooled $\mathcal{T}_{\beta}$ is additionally overstated by about
  $\sqrt{n_t/(n_t-1)}$, because the per-date mean and scale are not charged as degrees of freedom.
  Raise `min_obs_per_date` for universes that are small on some dates.
- **Survivorship.** The diagnostics use exactly the names in the frames. A universe built from
  today's constituents tests the signal only on survivors. For integer horizons, a window that
  contains a missing return is dropped, so a name's final, often large and negative, return
  before delisting is lost unless it is recorded. For string horizons, the rebuilt NAV is
  forward-filled: a name before its first return or after its last one receives a return of
  exactly zero for each whole period, and partial periods at the edges enter as full-period
  returns. Set the signal to missing for such periods to keep them out of the pairs.
- **Annualisation.** `IC_IR_an` uses the calendar spacing of the IC dates: 11.78 rather than 12 on
  month-ends and 365.25 rather than 252 on business days. Compare annualised ratios only across
  identical grids, or pass `periods_per_year` per grid.
- **Many signals.** Screening many signals and horizons and reporting the best inflates the
  headline t-statistic. The IC t-statistic has no multiple-testing adjustment.

## See also

- [Notation and conventions](notation_and_conventions.md)
- [Regression and HAC inference](regression_and_hac.md)
- [Serial dependence and autocorrelation](serial_dependence.md)
- [Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md)
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Portfolio breadth and allocation efficiency](portfolio_breadth.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Returns, NAVs, excess returns, fees and leverage](returns_and_navs.md)
- [Bibliography](bibliography.md)

## References

1. Grinold, R. C. (1989). The Fundamental Law of Active Management. *The Journal of Portfolio Management*, 15(3), 30–37. [DOI: 10.3905/jpm.1989.409211](https://doi.org/10.3905/jpm.1989.409211). States that the information ratio is approximately the IC times the square root of breadth.
2. Grinold, R. C., and Kahn, R. N. (2000). *Active Portfolio Management*, 2nd edition. McGraw-Hill. Develops the information coefficient, breadth and the fundamental law in a portfolio-construction framework.
3. Spearman, C. (1904). The Proof and Measurement of Association between Two Things. *The American Journal of Psychology*, 15(1), 72–101. [DOI: 10.2307/1412159](https://doi.org/10.2307/1412159). Introduces the rank correlation used for the default IC.
4. Fama, E. F., and MacBeth, J. D. (1973). Risk, Return, and Equilibrium: Empirical Tests. *Journal of Political Economy*, 81(3), 607–636. [DOI: 10.1086/260061](https://doi.org/10.1086/260061). Introduces inference from the time series of per-date cross-sectional slopes.
5. Kish, L. (1965). *Survey Sampling*. Wiley. Defines the design effect of clustered samples used to compare the pooled and per-date t-statistics.
6. Newey, W. K., and West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. [Working paper and published-version record](https://www.nber.org/papers/t0055). Provides the HAC alternative to non-overlapping sampling.
7. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
