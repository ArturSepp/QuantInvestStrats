# Sharpe Ratio Conventions: Arithmetic vs. Per-Annum (Compound) Returns

**Status**: decision record and implementation brief, 12 July 2026.
**Origin**: review of Proposition `prop:decomposition` in *The Convexity Premium of
Portfolio Overlays* (JOIM draft), which uses arithmetic mean returns, against the
qis default of p.a. (compound) returns.
**Decisions**: the paper keeps the arithmetic convention with a stated
reconciliation (Section 6). qis keeps the p.a. default and gains an explicit
`SharpeConvention` switch with labeled output (Section 7). Implementation of the
qis changes is scoped to a separate work stream.

---

## 1. The three Sharpe objects

Let `P_0, ..., P_M` be a NAV sampled at `M` periods over `T` years, with
annualization factor `a = M/T` (`a = 4` quarterly, `12` monthly, `260` daily).
Simple periodic returns are `r_m = P_m/P_{m-1} - 1`, log returns are
`l_m = ln(P_m/P_{m-1})`, and all returns are in excess of cash. Three Sharpe
definitions are in institutional use:

**(A) Arithmetic Sharpe** (periodic mean, square-root-of-time annualization):

    SR_arith = sqrt(a) * mean(r_m) / std(r_m)
             = [a * mean(r_m)] / [sqrt(a) * std(r_m)]

**(L) Log Sharpe** (same construction on log returns):

    SR_log = sqrt(a) * mean(l_m) / std(l_m)

**(P) Per-annum (compound / geometric) Sharpe** (the CAGR over annualized vol):

    C_pa   = (P_M / P_0)^(1/T) - 1
    SR_pa  = C_pa / [sqrt(a) * std(r_m)]

Relations. `C_pa = exp(a * mean(l_m)) - 1`, so (P) is the exponential map of
(L)'s numerator; empirically `SR_pa ≈ SR_log` to about 0.01 for the series in
this project. The wedge to the arithmetic object is the volatility drag:
`mean(l_m) ≈ mean(r_m) - var(r_m)/2`, giving to first order

    SR_pa - SR_arith ≈ -(sigma_ann / 2) * (1 - h),    h = exp-convexity clawback,

so the p.a. Sharpe sits below the arithmetic one by roughly half the annualized
volatility, damped for high-Sharpe series. At 10% vol the wedge is 0.03-0.05;
at 15-18% vol (unhedged gold or duration futures) it reaches 0.08-0.09.

## 2. Empirical wedge on the paper's exhibits

Long-run panel, quarterly excess returns, 1975Q1-2026Q2, per-series windows,
one regime partition classified on the benchmark (from the session's
verification run):

| series                | ann vol | SR_arith | SR_log | SR_pa  | wedge  | CP_arith | CP_log |
|-----------------------|---------|----------|--------|--------|--------|----------|--------|
| Balanced              | 0.107   | 0.543    | 0.488  | 0.500  | -0.044 | -0.042   | -0.054 |
| Fast LS(5,21)         | 0.111   | -0.086   | -0.142 | -0.140 | -0.054 | 0.128    | 0.120  |
| Medium-Fast LS(10,63) | 0.099   | 0.618    | 0.576  | 0.582  | -0.036 | 0.144    | 0.138  |
| Medium LS(20,126)     | 0.101   | 0.749    | 0.707  | 0.718  | -0.031 | 0.104    | 0.099  |
| Slow LS(20,252)       | 0.106   | 0.770    | 0.724  | 0.739  | -0.031 | 0.059    | 0.054  |
| Gold                  | 0.163   | 0.142    | 0.061  | 0.062  | -0.081 | 0.018    | 0.015  |
| Long Duration         | 0.182   | 0.251    | 0.165  | 0.164  | -0.087 | 0.044    | 0.033  |
| 100/100 Stack         | 0.145   | 0.930    | 0.876  | 0.901  | -0.028 | 0.041    | 0.032  |

Findings. (i) Median |SR wedge| is 0.04; rankings are identical under all three
conventions. (ii) `SR_log ≈ SR_pa` throughout, so the p.a. Sharpe is numerically
the log-return Sharpe. (iii) The convexity premium is convention-robust: median
|CP_log - CP_arith| = 0.007 against bootstrap standard errors of ~0.04, because
CP is a difference of same-asset conditional quantities and the drag cancels.
The 50-year OOS row moves 0.96/0.84 (arith) to 0.94/0.81 (p.a.) with the
dynamic-static gap unchanged.

## 3. Convention survey: who uses what

### 3.1 Arithmetic camp (inference, academia, mutual-fund reporting)

- **Sharpe (1994)** and Sharpe's methodological notes: the historic
  excess-return Sharpe ratio is the mean of periodic excess returns over their
  standard deviation, annualized by sqrt(periods/year), explicitly ignoring
  compounding.
- **The Sharpe-inference literature**: Lo (2002), Bailey and Lopez de Prado
  (2012, PSR/MinTRL), Bailey and Lopez de Prado (2014, DSR), and the JPM 2026
  reporting standard (Lopez de Prado and co-authors) all define SR = mu/sigma on
  periodic excess returns; the 2026 standard additionally computes Sharpe ratios
  at observation frequency without annualizing, since inference does not need
  it, and recommends against ranking on annualized point estimates at all.
- **Morningstar** (headline statistic): monthly arithmetic mean excess return
  over the monthly standard deviation of excess returns, annualized by sqrt(12).
- **AQR** journal publications (e.g., *Do Hedge Funds Hedge?*): annualized
  excess return as 12x the monthly mean, over annualized standard deviation.
- **J.P. Morgan AM** glossary: average sub-period excess return over the
  standard deviation of sub-period excess returns, simple returns, sqrt-scaled.
- **HFRX** index methodology: average monthly return difference over its
  standard deviation in constituent scoring.
- **Python quant stack** (empyrical/pyfolio lineage and retail platforms such
  as Composer): arithmetic mean scaled by sqrt(252), with the volatility drag
  explicitly cited as the reason for not using the geometric mean.

### 3.2 Per-annum camp (CTA/managed futures, performance-measurement tradition)

- **BarclayHedge** (the CTA reference database): Sharpe equals compound annual
  rate of return minus the T-bill rate, divided by the annualized standard
  deviation of monthly returns. Managed-futures formularies and manager tear
  sheets follow this definition, consistent with Calmar (CAGR over drawdown).
- **PerformanceAnalytics (R)**, the Bacon-tradition standard library:
  `SharpeRatio.annualized(..., geometric = TRUE)` by default, i.e. the
  geometrically chained annualized return over sigma*sqrt(scale), with
  `geometric = FALSE` available.
- **Morningstar Direct** custom statistics: a named *Geometric Sharpe Ratio*
  (geometric excess return numerator, compounded annualization) alongside the
  *Arithmetic Sharpe Ratio*.
- **qis** (current default): p.a. returns throughout PerfStat and factsheets,
  including the regime-conditional Sharpe illustrations.

### 3.3 The fault line and the rationales

The split is not noise; each camp's convention is the correct object for its
question.

Arithmetic serves *inference and decomposition*: it is the plug-in estimator of
the population mu/sigma with a known sampling distribution (Lo 2002; Eq. 3 of
the JPM 2026 standard under non-normal AR(1) returns); linear identities hold
exactly (regime decomposition, performance attribution); and portfolio means
aggregate linearly across constituents, which geometric means do not.

Per-annum serves *reporting and allocator experience*: the CAGR is the return a
buy-and-hold investor actually earned; it matches the headline number of every
track record; and it is conservative, charging the sigma^2/2 volatility drag of
the path so a fat-tailed series never looks better than the investor's
experience. Databases pin one definition down precisely because the choice
moves the number and is otherwise gameable.

## 4. Why the paper cannot simply switch: the exactness constraint

Proposition `prop:decomposition` (SR = SR_Bear + SR_Normal + SR_Bull with
`SR_s = sqrt(a) p_s m_s / sigma`) is linearity of expectation and holds only
for arithmetic means of periodic returns. Under compound p.a. returns the
additivity fails. Two exact repairs exist, both in Sepp (2020, *Smart
Diversification Analytics*):

1. **Log space**: the log p.a. return decomposes exactly,
   `L = sum_g a f_g rbar_g` with `rbar_g` the regime-conditional mean log
   return. A log-Sharpe framework (SR_log with kappa and the 16% rule
   unchanged, since log returns are the natural Gaussian objects) transplants
   the whole apparatus, and SR_log ≈ SR_pa numerically.
2. **Compound space with adjustment**: regime-conditional p.a. returns
   `C_g = exp(a f_g rbar_g) - 1` do not sum to `C_pa`; adding the equal-split
   constant

       c = (1/G) [ (exp(a rbar) - 1) - sum_g (exp(a f_g rbar_g) - 1) ]
         ≈ (1/2G) sum_g a f_g (rbar - rbar_g)^2        (cross-sectional variance)

   restores `C_pa = sum_g (C_g + c)` exactly. This is the correct way to report
   regime-conditional p.a. returns that tie to the headline CAGR.

## 5. Alignment with the current inference standard

The JPM 2026 reporting standard (PSR, MinTRL, power, pFDR/oFDR, DSR, SFDR) is
built entirely on the arithmetic plug-in estimator and its generalized sampling
variance under skewness, kurtosis, and AR(1) serial correlation. Any exhibit
that carries standard errors, significance stars, or bootstrap inference should
live in the same units as the machinery a referee or allocator would use to
check it. Two cheap borrowings for the paper: (i) fund-level Sharpe sampling
errors under their closed form are roughly +/-0.2-0.3 at T = 108 monthly
observations with hedge-fund-typical skew and autocorrelation, worth one
sentence; (ii) the four systems form a one-parameter theory-constrained family,
which is their own argument for low selection-inflation across the span dial,
worth one footnote if multiple testing is raised.

## 6. Decision for the paper (Option A)

All Sharpe ratios in the manuscript are annualized arithmetic means of periodic
excess returns over annualized volatility. Rationale: (i) the regime
decomposition and the Gaussian null are exact in this metric; (ii) it is the
convention of the Sharpe-inference literature whose standard errors the tables
carry; (iii) it is consistent with Paper A's analytical Sharpe via Phi_nu. A
convention paragraph states this once, and a footnote reconciles to the p.a.
convention of practitioner databases and qis factsheets: p.a. Sharpes are lower
by approximately sigma/2 (0.03-0.05 at 10% vol), rankings and premia are
unchanged (Section 2), and Sepp (2020) provides the exact bridge.

## 7. Decision for qis: keep p.a. default, make the convention explicit

qis is a reporting library in the BarclayHedge / PerformanceAnalytics
tradition; its audience reconciles against tear sheets. Do not silently change
the meaning of existing outputs. Instead:

1. **`SharpeConvention` enum** (house style: enum-driven modes):

       class SharpeConvention(Enum):
           PA = 1          # compound annual excess return / annualized vol (current default)
           ARITHMETIC = 2  # sqrt(a) * mean / std of periodic excess returns
           LOG = 3         # sqrt(a) * mean / std of periodic log excess returns

2. **Touchpoints**: `PerfParams` gains `sharpe_convention: SharpeConvention =
   SharpeConvention.PA`; `PerfStat`-driven tables, `compute_ra_returns`, and
   the factsheet layers read it. Default unchanged (PA), so all existing
   factsheets are byte-stable.
3. **Self-documenting labels**: column and legend text carries the convention,
   e.g. `Sharpe (p.a.)` vs `Sharpe (arith)`, mirroring
   PerformanceAnalytics' explicit `geometric` flag and Morningstar Direct's
   named Arithmetic/Geometric variants.
4. **Inference path**: bootstrap standard errors, significance tests, and
   regime-conditional decompositions route through `ARITHMETIC` (or `LOG`),
   where the additive identities are exact. Regime-conditional p.a. returns,
   when displayed, apply the Sepp (2020) c-adjustment of Section 4 so the
   regime bars tie to the headline CAGR.
5. **Regression guard**: a unit test pinning one reference series to all three
   conventions with the identity `SR_pa ≈ SR_log` and the wedge
   `SR_arith - SR_pa ≈ sigma/2` within tolerance, so future edits cannot
   silently swap conventions.

## 8. Regime-conditional Sharpe ratios under each convention

`PerfParams.sharpe_convention` selects how `compute_regimes_pa_perf_table` (and thus
`plot_regime_data`) computes the regime-conditional Sharpe ratios, with regime
probabilities p_s and conditional means m_s of the sampled periodic returns r:

- **PA (default).** Compound per-annum regime returns, patched so the regime
  p.a. returns sum to the total p.a. return, divided by the annualized vol. This
  keeps all pre-existing outputs byte-stable and matches the reporting default of
  Section 3.2.
- **ARITHMETIC.** sr_s = sqrt(af) * p_s * m_s / std(r) on simple returns. By
  linearity of the mean, sum_s sr_s equals the total arithmetic Sharpe ratio
  exactly, with numerator and denominator paired on the identical periodic
  series. This is the additive decomposition of the regime-attribution
  literature (the exactness constraint of Section 4).
- **LOG.** The same construction on log(1+r); exactly additive to the log Sharpe
  ratio of Section 1.

The additive conventions are the natural choice for regime attribution: the
stacked regime bars of `plot_regime_data` then total to the headline Sharpe
ratio. `PerfParams.copy()` carries `sharpe_convention` through, so a copied
params object keeps the selected convention.

## 9. References

- Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of Business* 39(1).
- Sharpe, W. F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*
  21(1), 49-58. See also Sharpe's notes on Morningstar's measures:
  https://web.stanford.edu/~wfsharpe/art/stars/stars3.htm
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts
  Journal* 58(4).
- Bailey, D. H., and M. Lopez de Prado (2012). The Sharpe Ratio Efficient
  Frontier. *Journal of Risk* 15(2). (PSR, MinTRL.)
- Bailey, D. H., and M. Lopez de Prado (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management* 40(5).
- Lopez de Prado, M., and co-authors (2026). Sharpe Ratio Inference: A New
  Standard for Decision Making and Reporting. *Journal of Portfolio
  Management* 52(6). https://pm-research.com/content/iijpormgmt/52/6
- Sepp, A. (2020). Smart Diversification Analytics. Working note, 24 September
  2020. (P.a. returns, exact log-space regime decomposition, c-adjustment.)
- Morningstar. Standard Deviation and Sharpe Ratio (methodology):
  https://www.morningstar.com.au/investing/standard-deviation-and-sharpe-ratio
- Morningstar Direct. Custom Calculations glossary (Arithmetic and Geometric
  Sharpe Ratio):
  https://gladmainnew.morningstar.com/directhelp/Glossary/Custom_Statistics/Sharpe_Ratio.htm
- BarclayHedge. Sharpe Ratio definition:
  https://www.barclayhedge.com/sharpe-ratio/
- Peterson, B. G., and P. Carl. PerformanceAnalytics: SharpeRatio.annualized
  (geometric = TRUE default):
  https://www.rdocumentation.org/packages/PerformanceAnalytics/versions/2.0.8/topics/SharpeRatio.annualized
- Asness, C., R. Krail, and J. Liew (2001). Do Hedge Funds Hedge? *Journal of
  Portfolio Management* 28(1). (Arithmetic annualization stated in exhibits.)
- J.P. Morgan Asset Management. Investment Glossary (Sharpe ratio):
  https://am.jpmorgan.com/hk/en/asset-management/adv/tools-resources/investment-glossary/
- HFR. HFRX Hedge Fund Indices: Defined Formulaic Methodology:
  https://www.hfr.com/pdf/HFRX_formulaic_methodology.pdf
- Bacon, C. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed., Wiley. (The performance-measurement tradition behind
  PerformanceAnalytics.)
