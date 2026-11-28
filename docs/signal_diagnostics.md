---
myst:
  html_meta:
    description: >-
      Information coefficients, the IC information ratio and pooled predictive regressions of
      lagged signals on cross-sectionally normalised forward returns, as implemented in
      qis.estimate_signal_diagnostics, qis.compute_ic_timeseries and qis.estimate_ic_ir.
---

# Signal diagnostics: information coefficient and information ratio

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-25](https://github.com/ArturSepp/QuantInvestStrats/commit/ddee40a0f97780a6bfe3a8623dbd65e113805e90)*

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
| Does the score predict raw or factor-adjusted returns over the same holding period? | `compute_forward_return_targets`, `compute_matched_signal_ic`, `compute_signal_ic_summary` | Matched ICs, their paired changes and pointwise intervals by horizon. See [Raw and factor-adjusted forward returns](#raw-and-factor-adjusted-forward-returns). |

The native-cadence computation in `estimate_signal_diagnostics` has three layers. **Pairing** builds, for each asset and horizon, pairs of the
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
| Sampling grid | For `estimate_signal_diagnostics`, each asset's native grid is the pandas frequency key of `asset_returns_dict`; integer $h$ counts native periods and pairs are taken every $h$-th date; a string horizon resamples all NAVs to that frequency |
| Annualisation | None for $\hat\beta$, the ICs and the pooled t-statistic; `IC_IR_an` multiplies the IC ratio by $\sqrt{\mathrm{af}_h}$, the IC periods per year: $\mathrm{af}/h$ for integer $h$, with $\mathrm{af}$ from `qis.get_annualization_factor` of the finest native key (or `periods_per_year`), and `get_annualization_factor(label)` for a string horizon |
| Mean adjustment | Forward returns are demeaned across names at each date and divided by their cross-sectional $s_t$ (`ddof=1`); signals are not transformed; the pooled regression has no intercept, and its residual variance charges one degree of freedom per date for the demeaning; $s(\mathrm{IC})$ uses `ddof=1` |
| Timing | The signal last observed in the native period ending at $t-1$ is paired with the return over $(t-1,t-1+h]$; qis applies exactly one native-period lag |
| Output units | $\hat\beta$ in cross-sectional standard deviations of forward return per unit of signal; ICs, t-statistics and IC ratios dimensionless; hit rate a fraction |
| qis default | `estimate_signal_diagnostics(horizons=(1, 3, 6), fit_intercept=False, is_log_returns=True, is_vol_normalised=True, min_obs_per_date=5, min_obs_per_group=10)`; `estimate_ic_ir(method='spearman', return_col='r_norm_univ', periods_per_year=None, min_obs_per_date=5)` |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| `asset_returns_dict` | Returns per native frequency, e.g. `{'ME': monthly, 'QE': quarterly}` | Each asset in one frame; keys are period-end frequencies, and dates inside a period are aligned to its label |
| `signal` | Panel of scores, dates by assets | Any frequency; resampled to each native grid with the last value per period |
| $z_{i,t}$ | Signal of asset $i$, last value observed in the native period ending at $t$ | Signal units; not normalised by qis |
| $h$ | Forward horizon | Integer count of the asset's native periods, or a pandas frequency string |
| $y_{i,t}$ | Forward return paired with $z_{i,t-1}$ | Over $(t-1,t-1+h]$; column `r` of the pairs frame |
| $\mathcal{A}_t$, $n_t$ | Names with a finite pair at regression date $t$, and their number | All frequency frames pooled; $n_t\ge$ `min_obs_per_date` |
| $\bar y_t$, $s_t(y)$ | Cross-sectional mean and standard deviation of $y_{\cdot,t}$ | Over $\mathcal{A}_t$, `ddof=1` |
| $\bar z_t$, $s_t(z)$ | Cross-sectional mean and standard deviation of $z_{\cdot,t-1}$ | Over $\mathcal{A}_t$, `ddof=1` |
| $x_{i,t}$, $x^{g}_{i,t}$ | Forward return normalised across all names, or within group $g$ | Columns `r_norm_univ` and `r_norm_group` |
| $n$ | Number of pooled pairs, $n=\sum_t n_t$ | Column `n` |
| $T$ | Number of regression dates of a horizon; of dates in its IC series | Column `n_dates` |
| $\hat\beta$, $\varepsilon_{i,t}$, $\hat\sigma^2_{\varepsilon}$ | Pooled slope, residual and residual variance | No intercept by default |
| $\mathcal{T}_{\beta}$, $\mathcal{T}_{\mathrm{IC}}$ | t-statistics of the pooled slope and of the mean IC | Calligraphic, so that $t$ stays a date |
| $\mathrm{IC}^{\mathrm{P}}_t$, $\mathrm{IC}^{\mathrm{S}}_t$ | Pearson and Spearman IC at date $t$ | In $[-1,1]$ |
| $\overline{\mathrm{IC}}$, $s(\mathrm{IC})$ | Time-series mean and standard deviation of the per-date IC | Over $T$ dates, `ddof=1` |
| $\mathrm{IR}_{\mathrm{IC}}$ | IC information ratio | Per IC period; column `IC_IR` |
| $\mathrm{af}_h$ | Periods per year of the IC series of horizon $h$ | $\mathrm{af}/h$, or the string horizon's own factor |
| $\mathrm{HR}$ | Hit rate | Fraction of dates with $\mathrm{IC}_t>0$ |
| $\mathrm{BR}$ | Breadth: independent bets per year | Grinold (1989) |
| $\Phi$ | Standard normal distribution function | |
| $w_{i,t}$ | Signal standardised across names at date $t$ | Unit `ddof=1` dispersion |
| $F$, $e_j$ | String-horizon frequency and its period ends | Pandas frequency, e.g. `'YE'` |
| $t_0$ | First date common to the return index and the resampled signal | Fixes the sampling phase |
| $\eta_k$, $\sigma_{\eta}$, $Y_t$ | Uncorrelated increments, their volatility, and their overlapping $h$-sums | Overlap proposition only |
| $a$, $b$, $\pi$, $C$ | Centred score and return vectors, a random permutation, their correlation | Null-dispersion proof only |
| $c$ | Constant added to every signal value | Level-shift pitfall only |

The inputs must satisfy three conditions. First, each key must be a period-end frequency, and
each frame must hold at most one date per period of its key. Dates inside a period but off its
label, such as business month-ends stored under `'ME'`, are aligned to the label that
`signal.resample(key)` produces, and the signal paired with the next return is the last value
observed at or before the return date; pairs are dated at the labels, so business-month-end and
calendar-month-end frames share one cross-section. A frame with several dates in one period of
its key is finer than the key: qis warns and pairs only the dates on the labels. Second, the
signal value dated $t$ must use only information available at $t$; qis lags it once and cannot
detect a mis-dated input. Third, the default `is_log_returns=True` is the opposite of the
`qis.to_returns` and `qis.compute_asset_returns_dict` defaults, which produce simple returns;
pass the flag that matches the data. The default is kept for compatibility with existing callers,
whose per-frequency pipelines deliver log returns, and the docstring states it.

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
cross-section. The [monthly and quarterly worked example](#worked-example-with-monthly-and-quarterly-assets)
traces the same opening scores under native and decision-calendar horizons.

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
$\dots<e_{j-1}<e_j$, let $G_{i,j}$ be the dates of asset $i$'s frame in $(e_{j-1},e_j]$. qis
compounds the native returns within each period and pairs

$$
y_{i,e_j}=\sum_{t\in G_{i,j}}\ell_{i,t}\quad\text{or}\quad\prod_{t\in G_{i,j}}\big(1+r_{i,t}\big)-1,
\qquad
z_{i,e_{j-1}}=\text{last signal value in }(e_{j-2},e_{j-1}] ,
$$

provided $G_{i,j}$ is not empty, asset $i$ has a finite return at every date of $G_{i,j}$, and
the frame covers the period: at the frame's first and last periods, its first and last dates
must fall in the first and last native periods of the key inside them. Otherwise $y_{i,e_j}$ is
missing. This is the rule of integer horizons, where a window with a missing native return is
dropped. Periods before an asset's first return, after its last one, partially covered at the
sample edges, or containing a missing return produce no pair, rather than a zero or a
partial-period return. A date absent from the whole frame, such as an exchange holiday in a
business-day frame, does not make a period incomplete; at the frame's edges, where a holiday
cannot be told from a late start, the period is dropped.

The pair is dated $e_j$, the end of its window, and each asset contributes one non-overlapping
pair per complete $F$ period. A string horizon overrides the native cadence: `'YE'` gives every
asset calendar-year returns. For month-end assets, `'YE'` has the same pair content as $h=12$
when the integer phase starts at a January month-end, ragged starts and ends included; only the
date label differs (December against January). A string frequency finer than an asset's native
grid leaves the periods without a native print empty and attributes each native return to the
period of its end date, so a quarterly return enters the cross-section of a single month.

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
\hat\sigma^2_{\varepsilon}&=\frac{1}{n-T-1}\sum_{t}\sum_{i\in\mathcal{A}_t}\varepsilon_{i,t}^2,
\qquad
\operatorname{se}(\hat\beta)=\sqrt{\frac{\hat\sigma^2_{\varepsilon}}{\sum_{t}\sum_{i\in\mathcal{A}_t}z_{i,t-1}^2}},
\qquad
\mathcal{T}_{\beta}=\frac{\hat\beta}{\operatorname{se}(\hat\beta)} ,
\end{aligned}
$$

where $T$ is the number of regression dates. This is least squares through the origin with a
classical standard error for homoskedastic, independent pairs. The residual degrees of freedom
are $n-T-1$: one for the slope and one for each date, whose cross-sectional mean the
normalisation removed. The proposition on residual degrees of freedom below shows that this
count makes the squared standard error unbiased under the null. Fewer than five pairs, or a zero
signal, give a row of missing values.

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
\mathcal{T}_{\beta}=\frac{\hat\beta\,\sqrt{n-T-1}}{\sqrt{1-\hat\beta^2}} .
$$

**Proof.** Here $\sum z^2=\sum x^2=n-T$ and $\sum zx=\hat\beta(n-T)$, so
$\sum\varepsilon^2=\sum x^2-\hat\beta^2\sum z^2=(n-T)(1-\hat\beta^2)$. Then
$\operatorname{se}(\hat\beta)^2=(1-\hat\beta^2)/(n-T-1)$. $\square$

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
identity. The standard error is that of linregress with the residual variance rescaled from
$n-2$ to $n-T-1$ degrees of freedom: the date means removed by the normalisation act as date
fixed effects, whose span contains the constant, so the intercept costs no further degree of
freedom. `IC_pearson` is the linregress correlation (the pooled Pearson IC), and $\hat\alpha$ is
not reported. Cross-sectional demeaning of the left-hand side does not make the intercept zero:
$\hat\alpha=0$ only when the pooled mean of the signal is zero. Otherwise
$\hat\alpha=-\hat\beta_{\alpha}\bar z\ne0$ and the two fits give different slopes. The intercept
removes a constant signal level, but not a level that varies by date.

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
0.23 on 20 names and 0.10 on 101 names. The proposition also fixes the degrees of freedom of the
pooled regression.

**Proposition (residual degrees of freedom).** Suppose the null of the previous proposition holds
at every date, independently across dates, that `is_vol_normalised=True`, and that the signal is
standardised across names at each date. Then

$$
\mathbb{E}\big[\hat\beta\big]=0,
\qquad
\operatorname{Var}\big(\hat\beta\big)=\frac{1}{n-T},
\qquad
\mathbb{E}\Big[\sum_{t}\sum_{i\in\mathcal{A}_t}\varepsilon_{i,t}^2\Big]=n-T-1,
$$

so the squared standard error with $n-T-1$ degrees of freedom is unbiased for
$\operatorname{Var}(\hat\beta)$.

**Proof.** With a standardised signal, $\sum_i z_{i,t-1}x_{i,t}=(n_t-1)\,\mathrm{IC}^{\mathrm{P}}_t$
and $\sum z^2=n-T$. By the previous proposition the numerator terms have mean zero and variances
$n_t-1$, independent across dates, so $\operatorname{Var}(\hat\beta)=\sum_t(n_t-1)/(n-T)^2=1/(n-T)$.
From the proof of the t-statistic identity,
$\sum\varepsilon^2=(n-T)-(n-T)\hat\beta^2$, whose expectation is $n-T-1$. Dividing by $n-T-1$ and
by $\sum z^2=n-T$ gives $\mathbb{E}[\operatorname{se}(\hat\beta)^2]=1/(n-T)$. $\square$

Up to qis 5.30.3 the residual variance was divided by $n-1$, which does not charge the per-date
means; under this null that overstated $\mathcal{T}_{\beta}$ by about
$\sqrt{(n-1)/(n-T-1)}\approx\sqrt{n_t/(n_t-1)}$, a factor of 1.12 on 5 names and 1.03 on 20. The
proposition has a limit: residual correlation across names (industries, styles, regions) raises
$\operatorname{Var}(\mathrm{IC}_t)$ above $1/(n_t-1)$, which the pooled t-statistic cannot see at
all.

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
\mathrm{IR}^{\mathrm{ann}}_{\mathrm{IC}}=\mathrm{IR}_{\mathrm{IC}}\sqrt{\mathrm{af}_h},\\
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

An integer horizon $h$ samples every $h$-th native period, so its IC series has
$\mathrm{af}_h=\mathrm{af}/h$ periods per year. $\mathrm{af}$ is `periods_per_year` if given and
otherwise `qis.get_annualization_factor` of the finest native key among the pairs'
`asset_freq`, which sets the IC dates when cadences are mixed. A string horizon is its own grid:
$\mathrm{af}_h$ is `qis.get_annualization_factor` of the label, and `periods_per_year` does not
apply to it. The factors are those of every other annualised statistic in qis, whatever the
sample length:

| IC dates | Native key | $h$ | $\mathrm{af}_h$ |
|---|---|---:|---:|
| Month-ends | `'ME'` | 1 | 12 |
| Every third month-end | `'ME'` | 3 | 4 |
| Quarter-ends | `'QE'` | 1 | 4 |
| Weeks | `'W-FRI'` | 1 | 52 |
| Business days | `'B'` | 1 | 252 |
| Year-ends | string horizon `'YE'` | | 1 |

Up to qis 5.30.3, $\mathrm{af}_h$ was 365.25 divided by the median day gap of the IC dates, which
gave 11.78 on month-ends and 365.25 on business days, overstating the business-day annualised
ratio by $\sqrt{365.25/252}=1.20$, and a `periods_per_year` argument was applied unscaled to every
horizon.

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
\mathrm{IR}^{\mathrm{ann}}_{\mathrm{IC}}\approx\overline{\mathrm{IC}}\,\sqrt{(n_t-1)\,\mathrm{af}} ,
$$

the fundamental law with $\mathrm{BR}=(n_t-1)\,\mathrm{af}$; demeaning uses one degree of freedom
per date. Read backwards, $1/s(\mathrm{IC})^2$ is an effective breadth per period. It falls below
$n_t-1$ when names are correlated after normalisation or when the true IC varies through time.

The ratio of the two t-statistics measures the same thing. For a standardised signal and a
constant $n_t$, $\mathcal{T}_{\beta}\approx\overline{\mathrm{IC}}\sqrt{n-T}$ by the t-statistic
identity and $\mathcal{T}_{\mathrm{IC}}=\overline{\mathrm{IC}}\sqrt{T}/s(\mathrm{IC})$, so
$(\mathcal{T}_{\beta}/\mathcal{T}_{\mathrm{IC}})^2\approx(n_t-1)\,s(\mathrm{IC})^2$, the ratio of
$n_t-1$ to the effective breadth. This is the design effect of cluster sampling (Kish, 1965), with
dates as clusters. It is close to one when the names are independent, and values well above one
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
$z$ is the score at $t-1$, and $x$ is next month's return normalised across names. The residual
variance has $n-T-1$ degrees of freedom, with $T=60$ regression dates.

```python
z = score[:-1]
y = log_returns[1:]
x = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
n, n_reg_dates = z.size, z.shape[0]
beta = (z * x).sum() / (z * z).sum()
sigma2 = ((x - beta * z) ** 2).sum() / (n - n_reg_dates - 1)
se = np.sqrt(sigma2 / (z * z).sum())

pooled = result.pooled_universe.loc['1']
np.testing.assert_allclose(pooled[['n', 'beta', 'se', 't_stat']].to_numpy(dtype=float),
                           [n, beta, se, beta / se], rtol=1e-10)
np.testing.assert_allclose(pooled['IC_pearson'], np.corrcoef(z.ravel(), x.ravel())[0, 1],
                           rtol=1e-10)
np.testing.assert_allclose(pooled['IC_spearman'], stats.spearmanr(z.ravel(), x.ravel())[0],
                           rtol=1e-10)
assert_quoted([beta, se, beta / se, pooled['IC_spearman']], [0.0831, 0.0295, 2.81, 0.0613],
              [5e-5, 5e-5, 5e-3, 5e-5])
# n - 1 degrees of freedom, as up to qis 5.30.3, overstate the t-statistic
se_n1 = np.sqrt(((x - beta * z) ** 2).sum() / (n - 1) / (z * z).sum())
assert_quoted(beta / se_n1, 2.89, 5e-3)

# Standardised signal: beta is the pooled and the mean per-date Pearson IC, and t depends on
# beta, n and T only.
ic_pearson = np.array([stats.pearsonr(z[k], y[k])[0] for k in range(len(z))])
np.testing.assert_allclose([pooled['IC_pearson'], ic_pearson.mean()], [beta, beta], rtol=1e-10)
np.testing.assert_allclose(beta / se,
                           beta * np.sqrt(n - n_reg_dates - 1) / np.sqrt(1.0 - beta ** 2),
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

The pooled slope is $\hat\beta=0.0831$ with standard error 0.0295 and $\mathcal{T}_{\beta}=2.81$.
Because the signal is standardised, $\hat\beta$ equals both the pooled Pearson IC and the mean of
the 60 per-date Pearson ICs, and $\mathcal{T}_{\beta}$ follows from $\hat\beta$, $n=1200$ and
$T=60$ alone. With $n-1$ degrees of freedom, as up to qis 5.30.3, it would be 2.89. The pooled
rank IC is 0.0613. The two groups share one data-generating process, yet their
slopes are 0.041 and 0.106: with 600 pairs each, the standard error of each is about 0.04, so a
difference of this size between segments is not evidence of segment-specific skill.

```python
ic_rank = np.array([stats.spearmanr(z[k], y[k])[0] for k in range(len(z))])
ic_series = qis.compute_ic_timeseries(result)['1']
np.testing.assert_allclose(ic_series['IC'].to_numpy(dtype=float), ic_rank, rtol=1e-10)
assert (ic_series['n'] == n_names).all()

table = qis.estimate_ic_ir(result)
mean_ic, std_ic, n_dates = ic_rank.mean(), ic_rank.std(ddof=1), len(ic_rank)
an_monthly = 12.0  # qis.get_annualization_factor('ME')
expected = [n_dates, mean_ic, std_ic, mean_ic / std_ic, mean_ic / std_ic * np.sqrt(an_monthly),
            mean_ic / std_ic * np.sqrt(n_dates), np.mean(ic_rank > 0.0)]
np.testing.assert_allclose(table.loc['1'].to_numpy(dtype=float), expected, rtol=1e-10)
np.testing.assert_allclose(table.loc['1', 't_stat'], stats.ttest_1samp(ic_rank, 0.0).statistic,
                           rtol=1e-10)
# horizon 3 samples every third month-end: 12 / 3 = 4 IC periods per year
np.testing.assert_allclose(table.loc['3', 'IC_IR_an'],
                           table.loc['3', 'IC_IR'] * np.sqrt(an_monthly / 3.0), rtol=1e-10)
assert_quoted(expected[1:], [0.0467, 0.2176, 0.215, 0.744, 1.66, 35 / 60],
              [5e-5, 5e-5, 5e-4, 5e-4, 5e-3, 1e-12])
assert_quoted(mean_ic / std_ic * np.sqrt(365.25 / 31.0), 0.737, 5e-4)  # calendar ratio, <= 5.30.3

# Pearson IC ratio: Fama-MacBeth view, effective breadth and design effect.
pearson = qis.estimate_ic_ir(result, method='pearson', periods_per_year=12.0).loc['1']
np.testing.assert_allclose([pearson['mean_IC'], pearson['std_IC']],
                           [beta, ic_pearson.std(ddof=1)], rtol=1e-10)
breadth = 1.0 / pearson['std_IC'] ** 2
design_effect = (pooled['t_stat'] / pearson['t_stat']) ** 2
assert_quoted([pearson['IC_IR'], pearson['t_stat'], breadth, design_effect],
              [0.376, 2.91, 20.5, 0.93], [5e-4, 5e-3, 5e-2, 5e-3])
np.testing.assert_allclose(design_effect, (n_names - 1) / breadth, atol=0.01)
```

The rank IC averages 0.0467 with a standard deviation of 0.2176 across the 60 months, so
$\mathrm{IR}_{\mathrm{IC}}=0.215$ per month. Annualised with $\mathrm{af}_1=12$, the qis factor of
month-ends, it is 0.744; the calendar ratio $365.25/31=11.78$ used up to qis 5.30.3 gave 0.737.
At horizon 3 the factor is $12/3=4$. The t-statistic
is 1.66, identical to `scipy.stats.ttest_1samp`, and the IC is positive in 35 of 60 months
($\mathrm{HR}=0.583$, against $\Phi(0.215)=0.585$). This sample's rank IC lies 1.7 standard errors
below its population value of 0.095. A genuine IC of 0.1 on 20 names over five years is
therefore only marginally detectable.

The Pearson IC series has mean 0.0831, equal to $\hat\beta$, and $\mathcal{T}_{\mathrm{IC}}=2.91$.
Its standard deviation, 0.221, is close to the pure-noise value $1/\sqrt{19}=0.229$, so the
effective breadth $1/s(\mathrm{IC})^2=20.5$ per month matches $n_t-1=19$. The design effect
$(\mathcal{T}_{\beta}/\mathcal{T}_{\mathrm{IC}})^2=0.93$, close to $19/20.5$, confirms that the
names are independent in this panel, so the pooled and the Fama–MacBeth views agree. On real data
they usually do not.

```python
# A constant added to the signal changes no IC but shrinks the no-intercept slope by 1140/2340.
shifted = qis.estimate_signal_diagnostics(asset_returns_dict={'ME': returns}, signal=signal + 1.0,
                                          horizons=(1,)).pooled_universe.loc['1']
np.testing.assert_allclose(shifted['beta'], beta * (n - n_dates) / (n - n_dates + n), rtol=1e-10)
np.testing.assert_allclose(shifted[['IC_pearson', 'IC_spearman']].to_numpy(dtype=float),
                           pooled[['IC_pearson', 'IC_spearman']].to_numpy(dtype=float),
                           rtol=1e-10)
assert_quoted([shifted['beta'], shifted['t_stat']], [0.0405, 1.96], [5e-5, 5e-3])

# With an intercept the slope is recovered, but the intercept is -beta, not zero.
with_alpha = qis.estimate_signal_diagnostics(asset_returns_dict={'ME': returns},
                                             signal=signal + 1.0, horizons=(1,),
                                             fit_intercept=True).pooled_universe.loc['1']
np.testing.assert_allclose(with_alpha[['beta', 't_stat']].to_numpy(dtype=float),
                           [beta, pooled['t_stat']], rtol=1e-10)
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
$(n-T)/(n-T+n)=1140/2340$, to 0.0405, and cuts $\mathcal{T}_{\beta}$ to 1.96. Fitting an
intercept restores the slope of 0.0831 and $\mathcal{T}_{\beta}=2.81$, with an intercept of
$-0.0831$ rather than zero. A
return series passed as the signal and shifted back one month is look-ahead by construction: the
mean rank IC is exactly 1, every month is a hit, and the ratio is undefined. Passing a signal that
is already lagged makes qis lag it a second time; the mean IC falls to 0.026 with a t-statistic
of 0.81, and the genuine signal is lost.

### Worked example with monthly and quarterly assets

On 31 March 2025 the momentum scores are $+0.8$ for monthly asset M and $-0.4$ for quarterly
asset Q. The frames in `asset_returns_dict` contain **end-labelled log returns**. The April,
May and June rows of `asset_returns_dict['ME']['M']` are $+2\%, -1\%, +3\%$. The June,
September and December rows of `asset_returns_dict['QE']['Q']` are $+6\%, -2\%, +1\%$.
The June row for Q therefore measures the entire return from 31 March to 30 June.

**Native horizons.** For the opening scores on 31 March, `estimate_signal_diagnostics`
produces the following pairs. The pair's `date` is $t$ in $(z_{i,t-1},y_{i,t})$, the date
of the first return row used. The last column shows when the full outcome is known.

| Asset | Native $h$ | Score `z` | Return rows | Forward log return `r` | Pair `date` | Outcome known |
|---|---:|---:|---|---:|---|---|
| M | 1 month | +0.8 | Apr | +2% | 30 Apr | 30 Apr |
| Q | 1 quarter | −0.4 | Jun | +6% | 30 Jun | 30 Jun |
| M | 3 months | +0.8 | Apr, May, Jun | +4% | 30 Apr | 30 Jun |
| Q | 3 quarters | −0.4 | Jun, Sep, Dec | +5% | 30 Jun | 31 Dec |

The history starts in July 2024, so the every-third-date sample includes April 2025 on the
monthly grid and June 2025 on the quarterly grid. The example retains single-asset dates to
inspect the raw pairs. An IC requires a cross-section of assets.

```python
import numpy as np
import pandas as pd
import qis
from qis.perfstats.forward_return_targets import compute_forward_return_targets

months = pd.date_range('2024-07-31', '2025-12-31', freq='ME')
quarters = months[months.is_quarter_end]
decisions = pd.date_range('2025-03-31', periods=4, freq='ME')
mar, apr, may, jun = decisions
asset_returns_dict = {'ME': pd.DataFrame(np.nan, index=months, columns=['M']),
                      'QE': pd.DataFrame(np.nan, index=quarters, columns=['Q'])}
asset_returns_dict['ME'].loc[apr:jun, 'M'] = [0.02, -0.01, 0.03]
asset_returns_dict['QE'].loc[jun:, 'Q'] = [0.06, -0.02, 0.01]
score = pd.DataFrame(np.nan, index=months, columns=['M', 'Q'])
score.loc[mar] = [0.8, -0.4]
score.loc[may, 'M'] = 0.1

native = qis.estimate_signal_diagnostics(asset_returns_dict=asset_returns_dict, signal=score,
                                         horizons=(1, 3), is_log_returns=True,
                                         min_obs_per_date=1, is_vol_normalised=False)
for h, expected in [('1', [0.02, 0.06]), ('3', [0.04, 0.05])]:
    pairs = native.pairs[h].set_index(['date', 'asset'])
    march_scores = pairs.loc[[(apr, 'M'), (jun, 'Q')]]
    np.testing.assert_allclose(march_scores['z'], [0.8, -0.4])
    np.testing.assert_allclose(march_scores['r'], expected)
```

The native one-period cross-section on 30 June also contains M. Its pair is the **31 May**
score $+0.1$ and June return $+3\%$, alongside Q's **31 March** score and quarter return.

**Decision-calendar horizons.** With monthly decisions starting on 31 March, $h$ counts
months for both assets. `compute_forward_return_targets` sums the native increments over
`(formation_date, realisation_date]`. Join the score available at formation by date and asset.

| Decision $h$ | Formation | Realisation | M (score +0.8) | Q (score −0.4) |
|---|---|---|---|---|
| 1 month | 31 Mar | 30 Apr | +2% (Apr) | Missing (`native_endpoints`), since 30 Apr is off Q's grid |
| 3 months | 31 Mar | 30 Jun | +4% (Apr, May, Jun) | +6% (Jun) |

```python
targets = compute_forward_return_targets(decisions, asset_returns_dict, horizons=(1, 3))
opening_scores = score.rename_axis(index='formation_date', columns='asset').stack().rename('z')
paired = targets.join(opening_scores, on=['formation_date', 'asset'])
march = paired.loc[paired['formation_date'].eq(mar)].set_index(['horizon', 'asset'])
np.testing.assert_allclose(march.loc[3, 'z'], [0.8, -0.4])
np.testing.assert_allclose(march.loc[3, 'raw'], [0.04, 0.06])
assert march.loc[3, 'realisation_date'].eq(jun).all()
assert np.isclose(march.at[(1, 'M'), 'raw'], 0.02)
assert pd.isna(march.at[(1, 'Q'), 'raw'])
assert march.at[(1, 'Q'), 'raw_status'] == 'native_endpoints'
```

The one-month call warns that Q has no aligned window. At three decision steps, both
assets have observed quarter-end marks and enter the same March-to-June return comparison.

Run the [asset membership example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/signal_ic_asset_membership.py)
with `python -m examples.perfstats.signal_ic_asset_membership` from the repository root.
It prints the matched names and rank IC for each cross-section, using a minimum of two assets
to illustrate the calculation. Both APIs default to five.

### Raw and factor-adjusted forward returns

For a low-beta score, a positive raw rank IC can arise when low-beta assets lose less in a
falling market. To measure how well the score ranks the part of return left after a specified
hedge, form the raw and adjusted ICs on the same assets at each decision. The adjusted IC
measures residual predictability conditional on the supplied factor model. The paired change
measures how much the hedge changes the ranking relationship.

| Timing choice | Native-cadence `estimate_signal_diagnostics` | Decision-calendar `compute_forward_return_targets` and `compute_matched_signal_ic` |
|---|---|---|
| Score | QIS lags the supplied signal one native period | Supply scores and eligibility known at formation $t$. QIS uses them at $t$. |
| Integer horizon $h$ | Counts native periods per asset | Counts common decision steps. The first decision anchors non-overlapping windows $(t_j,t_{j+h}]$, with $j=0,h,2h,\ldots$. |
| IC date | The integer-horizon label is the first return increment | Each row records both `formation_date` and the actual `realisation_date`. The latter determines inclusion at `as_of`. |

At $h=1$ the native increment label is also its endpoint. At $h>1$ an as-of analysis needs
the actual end of each holding window. Rebuild windows first if an archived native IC combines
assets with different end dates under one label.

#### Build the forward targets

Supply a sorted, unique decision index and one frame of end-labelled **additive** increments
per native frequency. Log returns can be summed. For simple returns on a NAV, convert to log
returns with `np.log1p` and use factor increments and loadings fitted on that basis. Each asset
belongs to one frequency. Both endpoints must be native marks and every increment inside the
window must be finite. On a monthly decision calendar a quarterly asset contributes only to
aligned windows. A warning identifies a frequency and horizon with no aligned windows.

For an adjustment named `market_adjusted`, QIS sums factor increments over the same window,
takes the latest supplied loading matrix available by formation, and holds the signed exposures
fixed. The explained return is the dot product of those exposures and factor returns. Subtract
it from raw return to get the realised residual, including any intercept. For example, a 3.0%
log return with beta 1.25 and a 2.0% market log return leaves a 0.5% residual.

```python
import numpy as np
import pandas as pd
from qis.perfstats.forward_return_targets import ForwardReturnAdjustment, compute_forward_return_targets
from qis.perfstats.signal_ic import compute_matched_signal_ic, compute_signal_ic_summary

dates = pd.date_range('2024-01-31', periods=2, freq='ME')
betas = pd.DataFrame({'market': [1.25, 0.8, 0.3, 0.0, -0.4]}, index=list('ABCDE'))
assets = pd.DataFrame([[np.nan] * 5, [0.030, 0.020, 0.012, 0.009, 0.003]],
                      index=dates, columns=betas.index)
market = pd.DataFrame({'market': [np.nan, 0.020]}, index=dates)
hedge = ForwardReturnAdjustment(factor_increments_by_freq={'ME': market},
                                loadings_by_date={dates[0]: betas})
targets = compute_forward_return_targets(dates, {'ME': assets}, horizons=(1,),
                                         adjustments={'market_adjusted': hedge})
assert np.isclose(targets.set_index('asset').at['A', 'market_adjusted'], 0.005)
```

Each row identifies the asset, cadence and window. The output includes `raw`, `raw_status`,
and, for each adjustment, its residual, explained return, snapshot date and status. A failed
raw target carries `native_endpoints` or `missing_return`. An aligned adjustment can additionally
report `missing_snapshot`, `missing_factor_return` or `missing_loading`. This distinguishes an
unavailable residual from a realised zero. Model a multifactor hedge by supplying more factor
columns and matching asset-by-factor matrices under a second adjustment name.

#### Two benchmarks with EWMA betas

`compute_forward_return_targets` accepts pre-estimated exposures from any model. For
$D$ decisions, $N_c$ assets at cadence $c$, and $K$ benchmark factors, supply the following.

| Input | Layout | Meaning |
|---|---|---|
| `decision_dates` | Sorted, unique `DatetimeIndex` of length $D$ | An integer horizon counts decision steps. |
| `asset_increments_by_freq[c]` | End dates × $N_c$ asset columns | Additive asset returns on the native grid. Each asset belongs to one cadence. |
| `adjustments[name].factor_increments_by_freq[c]` | End dates × $K$ benchmark columns | Realised benchmark increments on the same native grid and in the units used to fit the exposures. |
| `adjustments[name].loadings_by_date[t]` | Asset rows × $K$ benchmark columns | Matrix of signed exposures available at timestamp $t$. Asset and benchmark labels identify the matching rows and columns. |

For the window $(t,u]$, QIS selects the latest snapshot available by $t$ and freezes
it. It sums the benchmark increments strictly after $t$ through $u$ to obtain the
$K$-vector $f_{t,u}$. For asset $i$, the output row contains `raw`,
`<name>_explained` $= B_{i,t}f_{t,u}$, and `<name>` $= r_{i,t,u}-B_{i,t}f_{t,u}$,
along with the snapshot date and status. A missing required benchmark increment or
asset loading makes the adjusted target unavailable. The signal is joined later by
formation date and asset.

The runnable example uses the fixed-seed QIS synthetic universe with clean price paths.
Both benchmarks and all eight test assets use monthly log price returns with no cash
subtraction or annualisation.

| Role | Synthetic tickers |
|---|---|
| Benchmarks | `SEQ_US` (US equity), `SBD_TSY` (Treasuries) |
| Equity assets | `SEQ_EU` (European equity), `SEQ_EM` (emerging-market equity) |
| Credit assets | `SBD_IG` (investment-grade bonds), `SBD_HY` (high-yield bonds) |
| Commodity assets | `SCM_GLD` (gold), `SCM_BCOM` (broad commodities) |
| Alternative assets | `SAL_HF` (hedge fund), `SAL_PE` (private equity) |

There are 120 month-end returns from January 2015 through December 2024. The joint
through-origin fit uses a 36-month EWMA span and the full 2 × 2 benchmark cross-moment matrix.
`warmup_period=11` masks positions 0 through 11. The first reporting decision is
31 December 2018. The span weights past returns in the **beta estimation**. A
three-month **forward payoff** sums the next three increments with the opening betas.

```python
import numpy as np

from examples.perfstats.ewma_benchmark_residuals import monthly_inputs
from qis import EwmLinearModel, MeanAdjType
from qis.perfstats.forward_return_targets import (
    ForwardReturnAdjustment, compute_forward_return_targets,
)

asset_log, bench_log = monthly_inputs()               # (120, 8), (120, 2)
month_ends = asset_log.loc['2018-12-31':].index
beta_model = EwmLinearModel(x=bench_log, y=asset_log)
ewma_span = 36
beta_model.fit(span=ewma_span, is_x_correlated=True, mean_adj_type=MeanAdjType.NONE,
               warmup_period=11)
# get_loadings_at_date returns benchmarks by assets. The adjustment takes assets by benchmarks.
beta_snapshots = {t: beta_model.get_loadings_at_date(t).T for t in month_ends[:-1]}
joint_hedge = ForwardReturnAdjustment(factor_increments_by_freq={'ME': bench_log},
                                      loadings_by_date=beta_snapshots)
two_factor_targets = compute_forward_return_targets(
    decision_dates=month_ends, asset_increments_by_freq={'ME': asset_log}, horizons=(1, 3),
    adjustments={'two_benchmark': joint_hedge},
)
opening, closing = month_ends[-4], month_ends[-1]
gold = two_factor_targets.query(
    "asset == 'SCM_GLD' and horizon == 3 and formation_date == @opening"
).iloc[0]
moves = bench_log.loc[(bench_log.index > opening) & (bench_log.index <= closing)].sum()
explained = beta_snapshots[opening].loc['SCM_GLD'].dot(moves)
assert gold.two_benchmark_snapshot_date == opening and gold.realisation_date == closing
np.testing.assert_allclose(gold.two_benchmark, gold.raw - explained)
np.testing.assert_allclose([gold.raw, explained, gold.two_benchmark],
                           [-0.05859, -0.02908, -0.02951], rtol=0, atol=0.000005)
```

At the 30 September 2024 close, synthetic gold’s equity beta was 0.12234 and its
Treasury beta was −0.34220. Through 31 December, the benchmarks returned −19.27%
and +1.61% in log units. Equity contributed −2.357 percentage points and Treasuries
contributed −0.551 points. Gold’s raw log return of −5.859% therefore splits into
−2.908% explained and **−2.951% residual** (rounded).

Run `python -m examples.perfstats.ewma_benchmark_residuals` for the two opening betas,
separate benchmark payoffs and final residual for every asset in the September–December
2024 window. [Runnable example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/ewma_benchmark_residuals.py)
· [EWMA loading estimation](factor_risk_models.md#estimating-loadings-with-ewm-regressions)

#### Compare the same assets

```python
scores = pd.DataFrame({'formation_date': dates[0], 'asset': betas.index,
                       'eligible': True, 'low_beta': -betas.market.to_numpy()})
panel = targets.merge(scores, on=['formation_date', 'asset'], validate='many_to_one')
dated_ic = compute_matched_signal_ic(panel, signals=['low_beta'],
                                     targets=['raw', 'market_adjusted'], reference_target='raw')
common = dated_ic.loc[dated_ic['sample'].eq('common')].set_index('target')
assert common.n_assets.eq(5).all()
assert common.at['market_adjusted', 'ic_minus_reference'] > 0
```

Supply one observation per group, horizon, formation date, realisation date and asset. Pass
`group_by=['fund']` at both IC stages for separate fund cross-sections. Each sample requires
eligibility and finite values for **every requested score**. The `common` sample also requires
all targets. Each `raw_vs_<target>` sample requires only that pair of targets, while
`available_raw` gives coverage for the reference alone. All sample labels are present even when
memberships coincide. Spearman correlation uses average ranks for ties. Set `method='pearson'`
for linear correlation. The default minimum is five assets. Inspect `n_assets` and `status`
to distinguish `insufficient_assets`, `constant_score` and `constant_target` from a measured
IC of zero. The paired difference requires both same-date ICs to be defined.

#### Weight dated ICs and interpret the intervals

`compute_signal_ic_summary` also accepts saved ICs. Its required columns are `horizon`,
`sample`, `signal`, `target`, `realisation_date` and `ic`. Include `ic_minus_reference` for
paired inference and `n_assets` for breadth. Optional `method` and `reference_target` identify
the correlation and comparison. The cutoff includes windows realised on or before `as_of`.

| Time-weighting policy | Arguments | Selected dates within each group |
|---|---|---|
| Full history (default) | `ewma_span=None, rolling_span=None` | Equal weights on every finite observation through the cutoff |
| Rolling | `rolling_span=K` | Equal weights on the last $K$ valid observations, with $K\ge2$. Estimates start once the window fills. |
| EWMA | `ewma_span=S` | Weights proportional to $\lambda^{\mathrm{age}}$, with $\lambda=(S-1)/(S+1)$ and $S>1$. Age counts valid observations. |

Choose one policy per call. For dated ICs $[0.25,-0.50,0.75]$, the full-history mean is $1/6$,
the rolling-two mean is $1/8$, and EWMA span three gives weights $[1,2,4]/7$ and mean $9/28$.

```python
saved = pd.DataFrame({'realisation_date': pd.date_range('2024-02-29', periods=3, freq='ME'),
                      'horizon': 1, 'sample': 'common', 'signal': 'momentum',
                      'target': 'raw', 'ic': [0.25, -0.50, 0.75]})
report = compute_signal_ic_summary(saved, ewma_span=3, warmup_periods=12)
np.testing.assert_allclose(report.dated.weight, np.array([1, 2, 4]) / 7)
np.testing.assert_allclose(report.summary[['mean_ic', 'effective_n']].iloc[0], [9/28, 7/3])
assert report.summary.inference.iat[0] == 'insufficient_dates'
```

`SignalICReport.dated` contains normalized `weight` and `paired_weight`. Each series selects
its own finite dates. In `summary`, `effective_n` and `paired_effective_n` are the respective
Kish counts $1/\sum_t w_t^2$. The IC-IR is the weighted mean IC divided by its weighted sample
standard deviation, using the correction $1-\sum_t w_t^2$. The ratio is in the observed
horizon's units. The hit rate weights positive ICs. The reported breadth, `mean_n_assets`, is
the arithmetic mean number of assets across selected IC dates.

`warmup_periods=12` requires 12 effective dates for pointwise HAC inference. Means remain
available during warm-up under full-history and EWMA weighting. Setting it to zero still
requires two observations, effective count above one and variation. A rolling window shorter
than the warm-up, or an EWMA span at or below it, cannot qualify. Inspect `inference` and
`paired_inference` separately. `hac_lags=3` uses up to three Bartlett lags, capped by the
number of selected dates minus one. `confidence_level=0.95` gives two-sided pointwise bounds.

The same EWMA span implies a longer calendar memory at longer non-overlapping horizons. With
monthly decisions, span 24 has a half-life of about 8.3 valid observations, or roughly 50 months
at $h=6$. For a common calendar decay on regular, complete windows, use $\lambda_h=\lambda_1^h$
and $S_h=(1+\lambda_h)/(1-\lambda_h)$ in separate horizon summaries.

#### Real ETF comparison with SPY and TLT

Run the Yahoo Finance example from the QIS checkout with the `qis[data]` extra installed.

```bash
python -m examples.perfstats.yahoo_etf_signal_ic
python -m examples.perfstats.yahoo_etf_signal_factsheet
```

The second command writes the eleven-page validation PDF and a ZIP of the underlying
CSV tables to the configured QIS output directory. To replay a saved snapshot or
choose an output directory, call the examples from Python:

<!-- docs-test: skip -->

```python
from examples.perfstats.yahoo_etf_signal_ic import run_example as print_comparison
from examples.perfstats.yahoo_etf_signal_factsheet import run_example as write_factsheet

print_comparison(prices_csv='adjusted_closes.csv')
write_factsheet(output_dir='reports', prices_csv='adjusted_closes.csv')
```

The example downloads daily distribution- and split-adjusted closes via `yfinance` and
takes the last observed trading session of each month from December 2007 through December
2025. A replay CSV has trading dates in the
first column and the ten ticker symbols below as column headers. The script reports the
first missing or invalid month-end price before fitting the model. [Runnable ETF example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/yahoo_etf_signal_ic.py)

| Role in comparison | ETFs |
|---|---|
| Benchmarks used to estimate exposures | SPY (US large-cap equity), TLT (long US Treasuries) |
| Ranked equity ETFs | IWM (US small caps), EFA (developed markets outside the US and Canada), EEM (emerging markets) |
| Ranked credit ETFs | LQD (US dollar investment-grade corporate bonds), HYG (US dollar high-yield corporate bonds) |
| Other ranked ETFs | GLD (gold), DBC (broad commodity futures), VNQ (US real estate) |

At each formation close $t$, QIS fits both benchmark betas jointly through the origin
on monthly log returns with `ewma_span=36`. The **low-beta** score is the negative signed
SPY coefficient, conditional on TLT. Lower equity exposure receives a higher score.
**12–1 momentum** is $\log(P_{t-1}/P_{t-12})$ on adjusted closes, omitting the latest
month. The first decision is 31 December 2010, after 36 monthly returns. Each horizon
uses non-overlapping windows from that decision through 31 December 2025.

**Raw** is the ETF’s forward log total return. **Residual** deducts the realised SPY
and TLT log returns multiplied by the respective betas frozen at formation. For each
signal, the two Spearman ICs use the same eligible ETFs and the same valid dates.
The comparison pools the eight ranked ETFs across asset classes. The mean IC uses
`ewma_span=36`, `rolling_span=None`, with one decay step per **valid IC observation**.
Thus the three-month series decays once per realised quarter and the twelve-month
series once per realised year.

| Horizon (months) | Valid dates | Matched ETFs per date | Low-beta raw IC | Low-beta residual IC | Momentum raw IC | Momentum residual IC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 180 | 8 | 0.0224 | 0.2960 | 0.0287 | 0.1282 |
| 3 | 60 | 8 | −0.0009 | 0.3611 | 0.0584 | 0.0792 |
| 6 | 30 | 8 | 0.0007 | 0.4044 | −0.0311 | −0.0016 |
| 12 | 15 | 8 | −0.1020 | 0.4531 | 0.0582 | 0.0346 |

All eight ETFs entered every window in this Yahoo price history. The script prints
matched-window counts for each ticker and horizon so readers can check coverage on
their own download. In this basket, adjustment increased low-beta IC at every horizon.
Momentum had its largest residual IC at one month. The twelve-month mean uses only 15
cross-sections.

#### Beta reuse and rank attribution

Pages 3 and 4 use the same 1-, 3-, 6- and 12-month holding windows as the headline table.

**Does sharing a fitted beta change the low-beta IC?** At each formation close, fit two
joint through-origin SPY/TLT regressions, one on alternate historical months (A) and one
on the complementary months (B). With monthly span $s=36$, the monthly decay is
$\lambda=(s-1)/(s+1)$. Each fold advances two months per observation and uses
$\lambda^2$. Only returns available through formation enter a fit. Freeze the loadings
over the forward window and use the realised SPY and TLT log returns over that window.

For orientation A, the score is the negative SPY loading from A. The same-fit residual
subtracts A’s SPY loading times the SPY return. The disjoint-fit residual substitutes
B’s SPY loading. Both residuals also subtract the full-history TLT loading times the
TLT return. Repeat with the A and B roles reversed. The joint-hedge control also
switches the TLT coefficient between the same and opposite folds. All ICs are Spearman
correlations on the common eligible ETFs. Average the two orientations **within each
date**, requiring both complete pairs, then take the EWMA mean over valid dates.
The reuse effect is same-fit minus disjoint-fit. Its pointwise 95% interval uses the
dated paired differences, Bartlett HAC with up to three lags and `warmup_periods=12`.
The two folds use disjoint historical observations and track the same economic exposures.
They also use fewer observations per fit than the headline model.

**How much IC comes from persistent ranks?** For each signal and target, use the
headline common-sample dates and their exact QIS summary weights $w_t$ (summing to one).
At each date rank the score and outcome across the same ETFs, centre each rank vector
on its cross-sectional mean and divide by its Euclidean norm. Call the resulting
vectors $u_t$ and $v_t$. Their dot product is the dated Spearman IC. The weighted
mean has the exact decomposition

$$
\sum_t w_t u_t^\top v_t
= \bar u^\top \bar v
+ \sum_t w_t (u_t-\bar u)^\top(v_t-\bar v),
\qquad \bar u=\sum_t w_t u_t,\quad \bar v=\sum_t w_t v_t.
$$

The first term measures alignment of persistent ETF score and outcome ranks. The
second measures co-movement of each ETF’s ranks around its own weighted means. The
asset-level tables sum to the two terms. With eight ETFs and no ties, each centred
rank vector has squared norm 42. Average ranks and date-specific norms handle ties.
The default uses a fixed, complete basket. For changing universes pass
`varying_membership=True`. Rank only finite score/return pairs at each date.
For asset $i$, let $p_i$ be the sum of selected date weights when it is present.
Compute its mean unit ranks conditional on those dates. Its persistent contribution
is $p_i$ times the product of those means. Its within contribution is the sum of
the products of deviations from those means, weighted by the original date weights.
Both terms together recover the original IC on the changing universe. These historical
components describe the realised IC. The within-ETF contribution is in IC units and
does not measure the return of a timing strategy.

Use `compute_beta_reuse_ic` and `decompose_rank_ic` from
`qis.perfstats.signal_ic_attribution`. The former accepts aligned native asset/factor
returns, decision dates, the score factor, horizons, a beta span and the IC weighting
settings. It returns a `SignalICReport` and the directional pairs. The latter accepts
two date-by-asset panels and the dated IC weights from that report or another
`compute_signal_ic_summary` result. It returns each asset’s mean ranks and its persistent
and within-asset contributions. For the ETF export, `beta_reuse_folds.csv`,
`beta_reuse_dated.csv` and `beta_reuse_summary.csv` trace the test from each orientation
to its interval. `rank_attribution.csv` and `rank_attribution_by_etf.csv` contain the
two components for both low beta and momentum against both targets.

#### Synthetic low-beta and momentum report

The offline example uses quarterly decisions, eight monthly and two quarterly assets, and
fixed, signed market and bond loadings. The negative market drift favours low beta in raw
returns. Each quarter's residual contains a component proportional to the preceding quarter's
return rank. With `ewma_span=24`, the one-quarter common sample gives the following results.

| Formation score | Raw mean IC | Two-factor mean IC (95% interval) | Adjusted minus raw (95% paired interval) |
|---|---:|---:|---:|
| Low beta | 0.340 | 0.193 [0.011, 0.374] | −0.148 [−0.243, −0.052] |
| Prior-quarter momentum | 0.432 | 0.465 [0.342, 0.589] | 0.033 [−0.072, 0.139] |

```python
from examples.perfstats.forward_signal_report import synthetic_inputs, report_for_lookback

decisions, observed, adjustments, quarterly, low_beta = synthetic_inputs()
forward = compute_forward_return_targets(decisions, observed, adjustments=adjustments)
summary = report_for_lookback(forward, quarterly, low_beta, lookback=1)
one = summary.query("horizon == 1 and sample == 'common'").set_index(['signal', 'target'])
for signal, raw, adjusted in [
    ('low_beta', 0.340, [0.193, 0.011, 0.374, -0.148, -0.243, -0.052]),
    ('momentum', 0.432, [0.465, 0.342, 0.589, 0.033, -0.072, 0.139]),
]:
    fields = ['mean_ic', 'ic_lower', 'ic_upper', 'mean_ic_minus_reference',
              'difference_lower', 'difference_upper']
    np.testing.assert_allclose(one.loc[(signal, 'raw'), 'mean_ic'], raw, atol=0.0005)
    np.testing.assert_allclose(one.loc[(signal, 'factor_adjusted'), fields].astype(float),
                               adjusted, atol=0.0005)
```

Removing the two realised factor contributions weakens the low-beta ranking. Momentum still
ranks the residual returns, but its paired interval includes zero, so the evidence for a change
in IC is inconclusive. A missing bond loading reduces average common breadth from 10 to 9.79
assets. At six quarters the effective count drops to 11.10 and the default warm-up withholds
both intervals. Searching the example's lookbacks or horizons and then reporting the strongest
cell introduces selection bias. Pointwise bounds describe each prespecified cell. Validate a
selected specification on a subsequent, held-out period.

Run the [saved-IC weighting example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/signal_ic_summary.py)
and the [forward signal report](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/forward_signal_report.py)
from the repository root with `python -m examples.perfstats.signal_ic_summary` and
`python -m examples.perfstats.forward_signal_report`.

Run `python -m examples.perfstats.signal_validation_reports` to display the matched IC
reports and native regression diagnostics. Its `Locals.EXPORT` case saves a PDF and a CSV
archive containing the underlying targets, dated ICs, weights, regression pairs and
endpoint estimates. The
[examples index](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/README.md#perfstats--performance-metrics)
lists the reports by question and page.

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| Forward return, integer horizon | $y_{i,t}$: $h$ native log returns summed, or simple returns compounded; every $h$-th date | `qis.estimate_signal_diagnostics(horizons=(1, 3, 6), is_log_returns=True)` |
| Forward return, string horizon | Native returns compounded within each $F$ period; complete periods only | `horizons=('YE',)`, mixable with integers |
| Lagged signal | Last value per native period, shifted one period; dates off the key's labels aligned point in time | Built in; no argument |
| Normalised return | $x_{i,t}$, date dropped if $n_t<5$ or $s_t(y)\le0$ | `pairs[h]['r_norm_univ']`; `is_vol_normalised`, `min_obs_per_date` |
| Group-normalised return | $x^{g}_{i,t}$, at least two names per group and date | `pairs[h]['r_norm_group']`; `group_data` |
| Pooled slope | $\hat\beta$, $\operatorname{se}(\hat\beta)$ with $n-T-1$ degrees of freedom, $\mathcal{T}_{\beta}$, $n$ | `pooled_universe` columns `beta`, `se`, `t_stat`, `n` |
| Pooled ICs | Pearson and Spearman correlation over all pairs | `pooled_universe` columns `IC_pearson`, `IC_spearman` |
| Slope with intercept | $\hat\beta_{\alpha}$ from `scipy.stats.linregress`, standard error with $n-T-1$ degrees of freedom | `fit_intercept=True`; recorded in `result.fit_intercept` |
| Per-group slope | Pooled estimator on $(z,x^{g})$, at least 10 pairs | `per_group`, index (`horizon`, `group`); `min_obs_per_group`, `group_order` |
| Per-asset slope | Pooled estimator on one asset's time series, at least 12 pairs, classical $n-1$ (or $n-2$) degrees of freedom | `qis.compute_per_asset_betas(result, min_obs_per_asset=12, fit_intercept=False)` |
| Per-date IC | $\mathrm{IC}^{\mathrm{S}}_t$ or $\mathrm{IC}^{\mathrm{P}}_t$ with at least 5 names; any other `method` raises `ValueError` | `qis.compute_ic_timeseries(result, method='spearman', return_col='r_norm_univ', min_obs_per_date=5)` |
| IC summary | $T$, $\overline{\mathrm{IC}}$, $s(\mathrm{IC})$, $\mathrm{IR}_{\mathrm{IC}}$, $\mathrm{IR}^{\mathrm{ann}}_{\mathrm{IC}}$, $\mathcal{T}_{\mathrm{IC}}$, $\mathrm{HR}$ | `qis.estimate_ic_ir(result, method='spearman', return_col='r_norm_univ', periods_per_year=None, min_obs_per_date=5)` |
| Periods per year | $\mathrm{af}_h=\mathrm{af}/h$, or the string horizon's factor | `qis.get_annualization_factor`; internal `_ic_periods_per_year` |
| Result container | Tables, pairs and labels | `qis.SignalDiagnosticsResult` |
| Column names | `n`, `beta`, `se`, `t_stat`, `IC_pearson`, `IC_spearman` | `qis.SignalDiagnosticsColumns` members `N`, `BETA`, `SE`, `T_STAT`, `IC_PEARSON`, `IC_SPEARMAN` |
| Conditional-return boxplot | $x$ by quantile bucket of $z$; title with $\hat\beta$, $\mathcal{T}_{\beta}$, pooled Pearson IC, $n$ | `qis.plot_signal_diagnostics_boxplot(result, horizon, num_buckets=10)` |
| Per-group $\beta$ boxplot | Per-asset no-intercept $\beta$ by group; per-group $\mathcal{T}_{\beta}$ annotated | `qis.plot_signal_diagnostics_group_boxplot(result, horizon, min_obs_per_asset=12)` |
| Composite figure | Boxplots per horizon, group row when groups exist; title names the fitted model | `qis.plot_signal_diagnostics(result, num_buckets=10, min_obs_per_asset=12)` |
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
- `fit_intercept`, the intercept choice of the pooled and per-group fits.

**Validation.** A non-Series `group_data` raises `TypeError`. An empty `asset_returns_dict`, a
signal with no column in common with the returns, and a horizon that is neither a positive
integer nor a string raise `ValueError`, as does an IC `method` other than `'spearman'` or
`'pearson'`. Assets absent from the signal are dropped with a `UserWarning` that names them. An
asset present in several frequency frames is assigned to the first one in dict order (internal
`_asset_to_freq_map`), with a `UserWarning`, and its other frames are ignored by every horizon. A
frame with more than one date in some period of its key also warns.

**Plot conventions.** Significance stars in plot titles mark $\lvert\mathcal{T}_{\beta}\rvert$ at
1.65, 1.96 and 2.58, the two-sided normal 10%, 5% and 1% levels, and inherit the pooled
t-statistic's independence assumption. The default title of the composite figure names the
regression recorded in `result.fit_intercept`, with or without intercept. The conditional-return
boxplot reduces the number of
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

> **Pitfall.** `estimate_signal_diagnostics` lags the signal by exactly one native period and trusts its date stamps. A
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
  the IC series, and each costs the pooled regression a degree of freedom for its mean. Raise
  `min_obs_per_date` for universes that are small on some dates.
- **Survivorship.** The diagnostics use exactly the names in the frames. A universe built from
  today's constituents tests the signal only on survivors. A window or a string-horizon period
  that contains a missing return is dropped, so a name's final, often large and negative, return
  before delisting is lost unless it is recorded, and the period in which it delists never enters.
- **Annualisation.** `IC_IR_an` uses the qis factor of the IC grid, $\mathrm{af}/h$: 12 on
  month-ends, 252 on business days, 4 for $h=3$ on month-ends. A `periods_per_year` override is
  the factor of the native grid and is divided by $h$ in the same way.
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
