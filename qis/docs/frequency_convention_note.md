# Performance Statistics Are Frequency-Relative: A Reporting Convention for Internally Consistent Factsheets

*[author / affiliation / date — placeholder]*

## Abstract

Annualized volatility, the Sharpe ratio, skewness, beta and correlation are not properties of a
return series. They are properties of a return series *sampled at a stated frequency*. A track
record reported as "12% annualized volatility, Sharpe 1.1" is underspecified until one knows whether
those numbers were estimated from daily, weekly, monthly or quarterly observations — and for
serially dependent returns the answer materially changes the figures. Yet performance factsheets
routinely either hard-code a single frequency or, more often, mix several frequencies within one
document without labelling them, which makes cross-strategy and cross-mandate comparison quietly
unreliable. This note argues for treating the reporting frequency as an explicit, first-class
choice: every statistic in a report is calibrated to a chosen frequency, or labelled with the
frequency at which it was computed; an up-sampling guard forbids reporting at a finer resolution
than the data supports; and trailing windows widen as the frequency coarsens so that estimates
remain estimable. We set out the convention and its correctness properties, and describe its
implementation in the open-source `qis` library.

## 1. Statistics are frequency-relative

Let $r_t$ denote periodic returns observed at frequency $f$ with $m_f$ periods per year
(approximately 260 for business days, 52 weekly, 12 monthly, 4 quarterly). The annualized
volatility reported from this series is

$$\hat\sigma_{\text{ann}}(f) = \hat\sigma_f \sqrt{m_f},$$

where $\hat\sigma_f$ is the sample standard deviation of per-period returns. The question is whether
$\hat\sigma_{\text{ann}}$ depends on the choice of $f$. It does, in two distinct ways.

**Under serial independence the point estimate is frequency-invariant, but its precision is not.**
If returns are i.i.d., variance is additive across periods, so $\sigma_f^2 \propto 1/m_f$ and the
product $\sigma_f\sqrt{m_f}$ is constant in expectation. Coarsening the frequency does not change
what we are estimating — but it sharply reduces the number of observations behind the estimate
(roughly $m_f \times$ years), so the same ten-year track record yields a far noisier annualized
volatility at quarterly sampling than at daily. Frequency choice is, at minimum, an
estimation-variance choice.

**Under serial dependence the point estimate itself diverges across frequencies.** For weakly
stationary returns with autocorrelations $\rho_j$, the variance of $k$-period returns is not $k$
times the one-period variance. Writing the variance ratio

$$\mathrm{VR}(k) = \frac{\mathrm{Var}\!\left(\sum_{i=1}^{k} r_i\right)}{k\,\mathrm{Var}(r)}
= 1 + 2\sum_{j=1}^{k-1}\left(1 - \frac{j}{k}\right)\rho_j,$$

the annualized volatility estimated at the coarser ($k$-period) frequency equals the
finer-frequency annualized volatility scaled by $\sqrt{\mathrm{VR}(k)}$. Positive autocorrelation —
trend, momentum, or the stale-pricing/smoothing that pervades illiquid and privately marked assets
— gives $\mathrm{VR}(k) > 1$ and inflates low-frequency volatility relative to high; negative
autocorrelation (mean reversion) gives $\mathrm{VR}(k) < 1$ and deflates it. The same NAV path
therefore produces a *term structure* of annualized volatility across daily, weekly, monthly and
quarterly sampling, and the gradient of that term structure is informative about the return process
rather than being a nuisance to be averaged away.

The case where this matters most is smoothing. Funds holding illiquid or model-marked positions
report a series whose observed returns are a moving average of true economic returns; the induced
positive serial correlation depresses short-horizon volatility and distorts the variance term
structure (Getmansky, Lo and Makarov, 2004). A monthly-sampled volatility of such a series can
understate the economic volatility substantially, and the discrepancy is precisely a
frequency-of-sampling artefact. The Sharpe ratio inherits the whole problem: the familiar
$\sqrt{T}$ annualization of a Sharpe estimate is invalid under autocorrelation, and the correct
scaling depends on the same $\mathrm{VR}(k)$ object (Lo, 2002). None of these statistics has a
frequency-free value to report.

## 2. The silent-mixing failure mode

If frequency dependence were merely acknowledged, the problem would be manageable. The practical
failure is that factsheets leave the frequency implicit and, worse, inconsistent across panels.
A common layout shows drawdowns computed on the daily NAV, a Sharpe ratio annualized from monthly
returns, regression betas estimated on a quarterly grid, and regime-conditional statistics on yet
another grid — with none of these cadences stated. A reader cannot reconstruct what is comparable
to what. Two managers each reporting "Sharpe 1.2" may have computed it from daily and from monthly
returns respectively, and under any serial dependence those numbers are not the same quantity.

The point is not that mixing frequencies is always wrong. Some statistics are intrinsically tied to
a particular grid — a drawdown is a property of the realized path and should be measured on the
native, highest-resolution series, not down-sampled. The defect is leaving the choice implicit. A
statistic computed at a frequency the reader cannot identify is a statistic the reader cannot use.

## 3. Two axes: reporting frequency and horizon

A report is specified by two separate choices that together fix every window and grid in it.

The first axis is the **reporting frequency** — daily, weekly, monthly or quarterly — which fixes
the sampling grid on which returns are taken and the annualization factor applied to per-period
quantities. The second axis is the **horizon**, the distinction between a long track record and a
short one. The horizon sets the *lengths* of rolling windows and selects a coarser regime grid for
long histories and a finer one for short. The two axes are orthogonal: a fifteen-year book reported
monthly and an eighteen-month book reported daily call for different window lengths even for the
nominally identical "rolling volatility," and conflating frequency with horizon is a frequent source
of miscalibrated windows.

## 4. The convention: one stated frequency, applied by lookup

The convention is simple to state. Within a report, every statistic is computed at the chosen
reporting frequency, or — for the path-native statistics of Section 8 — at a frequency explicitly
labelled on its own panel. All rolling-window lengths, the regression grid, the regime-classification
grid and the annualization factor are *derived* from the pair (frequency, horizon) by table lookup,
not chosen per statistic at the analyst's discretion. Removing that discretion is the point: it is
what makes two reports produced by two people on two books mechanically comparable.

The calibration we use is the following, with window lengths expressed in periods of the sampling
grid and the horizon axis shown as *long · short*:

| Reporting frequency | Sampling grid | Vol / Sharpe window | Beta window | Regime grid | Periods/yr |
|---|---|---|---|---|---|
| Daily     | business day | 260 · 260 | 780 · 260 | quarterly · monthly | ≈260 |
| Weekly    | W-WED        | 156 · 52  | 156 · 52  | quarterly · monthly | 52   |
| Monthly   | month-end    | 36 · 12   | 36 · 12   | quarterly · monthly | 12   |
| Quarterly | quarter-end  | 12 · 4    | 12 · 4    | quarterly · monthly | 4    |

Three calibration choices are worth noting. The volatility and Sharpe windows are deliberately
identical, because a rolling Sharpe is a rolling mean over a rolling volatility and reporting them on
different windows invites visual misattribution. The beta window is longer than the volatility
window (most visibly at daily frequency), because covariation with a benchmark is a slower-moving
quantity than own-volatility and a short beta window is dominated by noise. The regime grid is
always coarser than the reporting grid, so that each regime bucket contains enough observations for
the conditional statistic to mean anything.

## 5. The up-sampling guard as a correctness invariant

A report may always be produced at a frequency *coarser* than the data; it may never be produced at
a frequency *finer* than the data. Computing daily statistics from a monthly-marked NAV is not a
degraded estimate — it is a fabricated one, because the intra-month path required to define daily
returns does not exist in the data and can only be invented by interpolation. Coarsening, by
contrast, is well defined: it is subsampling or aggregation of observations that are actually there.

We therefore treat the direction of the frequency change as a correctness property and enforce it: a
validation step compares the requested reporting frequency against the native sampling of the input
and refuses any request to report finer than the data supports. This is not an ergonomic guard rail
to be relaxed under pressure; it is the boundary between summarizing a track record and manufacturing
one.

## 6. What changes with frequency, and what does not

Cumulative or total return is frequency-invariant: compounding telescopes, so the growth of a unit
of capital over a fixed interval is the same whether measured through daily or quarterly steps
(absent data gaps). Every dispersion- and dependence-based statistic, however, is frequency-dependent
— volatility and Sharpe as shown in Section 1, and beta and correlation through the same variance-
and covariance-ratio mechanics.

Skewness is the most acute case, and it makes the cleanest worked example. Under aggregation the
central limit theorem washes higher moments out: summing more increments pushes the distribution of
the aggregate toward Gaussian. For i.i.d. returns the third central moment is additive while the
standard deviation grows as $\sqrt{k}$, so the skewness of a $k$-period return decays as

$$\mathrm{skew}(S_k) = \frac{\gamma_1}{\sqrt{k}}, \qquad
\text{excess kurtosis}(S_k) = \frac{\kappa}{k}.$$

A monthly skewness is therefore roughly a daily skewness divided by $\sqrt{21}$ — a different number
by construction, not by estimation noise. Reporting "skewness $= -0.4$" without a frequency is close
to meaningless, and computing skewness on a fixed grid while the rest of the report sits on another
grid is internally inconsistent in exactly the way the convention is meant to forbid. The convention
samples skewness on the reporting grid, so the moment shares the cadence of every other statistic on
the page; a report at monthly frequency shows monthly skewness, and one at quarterly frequency shows
quarterly.

## 7. Estimation under coarse sampling: trailing-window adaptation

Coarsening the frequency reduces the observation count inside every window, and below a floor the
window stops being an estimator. A "trailing one-year correlation" reported at quarterly frequency
rests on four observations; it is not a correlation estimate in any useful sense. The convention
therefore widens trailing and recent windows as the frequency coarsens, enforcing a minimum
observation count $N_{\min}$:

$$\text{trailing years} = \max\!\left(1,\ \left\lceil \frac{N_{\min}}{m_f} \right\rceil\right).$$

At daily, weekly and monthly frequencies a one-year window already clears the floor, so nothing
changes; at quarterly frequency a nominal one-year trailing window widens to three years to reach a
workable sample. The widening is a deliberate bias–variance trade: a longer window is less responsive
to recent change, but below the observation floor the alternative is not a more responsive estimate,
it is noise, so the floor binds first.

## 8. Per-panel labelling discipline

The convention is credible only if it is legible, so every panel states the frequency at which it
was computed. Cumulative and rolling statistics carry the reporting frequency; turnover and cost
panels carry the frequency at which they were sampled; regime panels name the regime grid. The
path-native statistics — drawdowns and time-under-water — are explicitly labelled with the *native*
NAV grid and are deliberately not down-sampled, because a drawdown is a property of the realized path
and subsampling it to the reporting frequency would step over intra-period troughs and understate the
true maximum loss. The labelling is what allows a reader to audit the convention from the page itself,
and it is what distinguishes a deliberate mix of frequencies (drawdowns on the native grid, dispersion
statistics on the reporting grid) from the silent mix criticized in Section 2.

## 9. Implementation

The convention is implemented in the open-source `qis` library. A small enumeration
(`ReportingFrequency`) carries the four frequencies; a configuration layer
(`make_factsheet_config` / `fetch_default_report_kwargs`) performs the table lookup over the
(frequency, horizon) pair and emits the calibrated windows, grids and annualization factor; and a
single entry point (`qis.factsheet`) produces a report from prices or returns with the reporting
frequency as an argument. The up-sampling guard is enforced at the validation step, and the
calibration, the guard, the moment behaviour and the per-panel labels are all locked by a regression
test suite so that the convention cannot silently drift. Full details are in the library's reporting
documentation.

## 10. Discussion and limitations

The immediate payoff is comparability. When two mandates — or the same mandate reported in two base
currencies — are produced under one convention, their volatilities, Sharpe ratios and higher moments
are the same kind of object and can be placed side by side. The secondary payoff is honesty: a report
shows only the resolution the track record can support, and states the resolution it used.

The convention standardizes and labels; it does not dissolve the underlying estimation problem.
The choice of horizon and the definition of regimes remain matters of judgment that the calibration
table fixes by fiat rather than derives. Annualization itself assumes that scaling a per-period
quantity to an annual one is meaningful, which Section 1 shows is exactly where serial dependence
bites — so a fully rigorous treatment of a smoothed or strongly autocorrelated series would report
the variance term structure, or a serial-correlation-adjusted Sharpe, rather than a single annualized
number. The convention's contribution is narrower and, we think, more broadly useful: it makes the
frequency of every reported statistic an explicit, consistent and auditable choice, and it refuses to
report a resolution the data does not contain. That is a small, enforceable discipline, and it removes
a common and nearly invisible source of error in performance reporting.

## References

- Getmansky, M., Lo, A. W., and Makarov, I. (2004). An econometric model of serial correlation and
  illiquidity in hedge fund returns. *Journal of Financial Economics*, 74(3).
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4).
- Lo, A. W., and MacKinlay, A. C. (1988). Stock market prices do not follow random walks: evidence
  from a simple specification test. *Review of Financial Studies*, 1(1).
