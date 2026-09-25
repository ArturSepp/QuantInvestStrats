---
myst:
  html_meta:
    description: >-
      Every qis.PerfStat performance-table column defined exactly, with the formula, sampling
      grid, return basis and units used by compute_ra_perf_table, compute_desc_table and the
      benchmark and regime tables.
---

# The performance-statistic catalogue

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A performance statistic in qis is a column of a table indexed by asset, and each column is named
by a member of `qis.PerfStat`. This chapter defines all 61 members: the formula the code
computes, the grid on which its inputs are sampled, the return basis, and the units. Where a
published measure such as the Sortino ratio, the Calmar ratio or sample skewness is defined
differently in the literature, the chapter states the implemented definition and the difference.

## Overview

Four functions produce the columns:

1. `qis.compute_ra_perf_table` builds the risk-adjusted table: dates, counts, total and per-annum
   returns, the Sharpe family, volatility, downside volatility, Sortino, Calmar, drawdowns,
   skewness, kurtosis and extreme returns.
2. `qis.compute_desc_table` and `qis.compute_desc_freq_table` describe any panel: mean, standard
   deviation, t-statistic, quantiles, share of positive values, last value and its rank, and a
   normality test.
3. `qis.compute_ra_perf_table_with_benchmark` adds the regression columns: alpha, annualised
   alpha, beta, $R^2$ and the p-value of alpha.
4. The regime classifiers, through `qis.compute_bnb_regimes_pa_perf_table`, add the bear, normal
   and bull columns.

One object, `qis.PerfParams`, assigns a sampling grid to each group of statistics. That design
gives the reader three things to watch. A single table mixes grids: by default the maximum
drawdown is daily while the volatility it is divided by is monthly. Visible return columns use
each asset's native first and last observations, while ratio numerators use complete sampling
boundaries. And calling `compute_ra_perf_table` without `perf_params` infers the grid from the
index, so a single missing row can change the reported volatility.

This chapter defines each column compactly. The Sharpe columns are treated in depth in the
[Sharpe-ratio chapter](performance_analytics_and_sharpe.md), drawdown episodes in
[Drawdowns and time under water](drawdowns.md), the regression columns in
[Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md), and the
regime columns in [Regime-conditional performance](regime_conditional_performance.md).

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple returns for total, per-annum, arithmetic and extreme-return columns; `PerfParams.return_type` returns (log by default) for `VOL`, `DOWNSIDE_VOL`, `AVG_LOG_RETURN`, `SKEWNESS` and `KURTOSIS`; returns in excess of `rates_data` for every excess column |
| Sampling grid | Native observations for visible return columns; `freq_vol` for volatility, the Sharpe and Sortino numerators and the arithmetic family; `freq_drawdown` for drawdowns, `WORST` and `BEST`; `freq_skewness` for moments; `freq_reg` for regressions |
| Annualisation | $\mathrm{AN}$ inferred from the sampled `freq_vol` index: $\sqrt{\mathrm{AN}}$ for volatilities, $\mathrm{AN}$ for arithmetic means; 365.25-day years for per-annum returns; moments, drawdowns and extreme returns are not annualised |
| Mean adjustment | `VOL`: sample mean removed, `ddof=1`; `DOWNSIDE_VOL`: mean of the negative returns removed, `ddof=1`; table moments: bias-corrected $G_1$, $G_2$; descriptive-table and rolling moments: uncorrected $g_1$, $g_2$ |
| Timing | Full-sample and descriptive: every column uses the whole history, so none is point in time; the cash rate is lagged one observation of the rate series |
| Output units | Decimal fractions for returns, volatilities and drawdowns; dimensionless ratios, moments and p-values; dates, prices, counts and years where labelled |
| qis default | `PerfParams()`: `freq_vol='ME'`, `freq_skewness='ME'`, `freq_drawdown='D'`, `freq_reg='QE'`, `freq_excess_return='ME'`, `return_type=ReturnTypes.LOG`, `sharpe_convention=SharpeConvention.PA`, `rates_data=None`; `compute_ra_perf_table(prices, perf_params=None)` infers `freq` from the index instead |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $t_0$, $t_1$ | First and last native observation of an asset | Each asset's own support |
| $b_0$, $b_1$ | First and last complete `freq_vol` boundary inside $[t_0,t_1]$ | Partial periods at both ends are excluded |
| $Y$, $Y_{\mathrm{smp}}$ | Elapsed years from $t_0$ to $t_1$, and from $b_0$ to $b_1$ | Calendar days divided by 365.25 |
| $\mathrm{TR}$, $R_{\mathrm{pa}}$, $\tilde R_{\mathrm{pa}}$ | Native total, per-annum and excess per-annum return | Decimal |
| $R^{\mathrm{smp}}_{\mathrm{pa}}$, $\tilde R^{\mathrm{smp}}_{\mathrm{pa}}$ | Per-annum and excess per-annum return from $b_0$ to $b_1$ | Decimal; the ratio numerators |
| $v_k$ | `freq_vol` returns in the `return_type` basis, $k=1,\ldots,T$ | Log returns by default |
| $r_k$, $\tilde r_k$ | `freq_vol` simple returns and simple excess returns | Decimal per period |
| $r^{\mathrm{dd}}_t$ | Simple returns on the `freq_drawdown` grid | Daily by default |
| $x_k$ | A generic sample: `freq_skewness` returns in the risk table, one supplied column in the descriptive table | Units of the input |
| $T$ | Number of observations of the sample at hand; `NUM_OBS` for $v_k$ | Count |
| $\mathrm{AN}$, $\mathrm{AN}_{\mathrm{reg}}$, $\mathrm{AN}_{\mathrm{c}}$ | Periods per year of `freq_vol`, of `freq_reg` and of a regime classifier's grid | 12 for `ME`, 4 for `QE` |
| $\sigma_v$ | `VOL`, equal to $\sqrt{\mathrm{AN}}\,s(v)$ | Annualised decimal |
| $T_{-}$, $\bar v_{-}$ | Number and mean of the negative $v_k$ | Count; decimal per period |
| $\sigma^{-}$ | `DOWNSIDE_VOL` as implemented | Annualised decimal |
| $\theta$, $\delta_{\theta}$ | Minimum acceptable return; target downside deviation | Decimal per period; annualised decimal |
| $m_j$ | $j$-th central sample moment with divisor $T$ | Units of $x$ to the power $j$ |
| $g_1$, $g_2$; $G_1$, $G_2$ | Uncorrected and bias-corrected skewness and excess kurtosis | Dimensionless |
| $\mathrm{MDD}$ | `MAX_DD`, the minimum of $D_t$ on the `freq_drawdown` grid | Decimal, at most zero |
| $K^2$ | D'Agostino–Pearson omnibus normality statistic | $\chi^2_2$ under normality |
| $\hat q_p$ | Empirical $p$-quantile with linear interpolation | Units of $x$ |
| $y_t$, $d_t$ | Annual cash rate from `rates_data`; calendar date of observation $t$ | Decimal per year; date |
| $\omega$, $p_{\omega}$ | Benchmark regime (bear, normal, bull) and its share of periods | Label; fraction |
| $R^2$ | Coefficient of determination of the benchmark regression | Fraction |

Inputs are a `pandas.DataFrame` (or, for `compute_ra_perf_table`, a `pandas.Series`) of positive
prices with a sorted `DatetimeIndex`. Each column is evaluated on its own support: leading
missing values are dropped, interior gaps are forward-filled by the sampling, and no column is
extended beyond its own last observation because another column continues. All statistics are
full-sample and descriptive. A table computed on a whole history uses information that was not
available at earlier dates, so it describes a history and must not be used inside a backtest.

## Methodology

### The PerfParams contract

`qis.PerfParams` is a dataclass with nine fields. The table lists each field with its default in
`qis.PerfParams()`, the functions that read it, and the columns it governs.

| Field | Default | Read by | Governs |
|---|---|---|---|
| `freq` | `None`, stored as `'ME'` | Factsheet panels only, not the table functions | A shortcut that overwrites the fields below |
| `freq_vol` | `'ME'` | `compute_risk_table`, `compute_ra_perf_table` | `VOL`, `DOWNSIDE_VOL`, `NUM_OBS`, the arithmetic family, the Sharpe and Sortino numerators, and $\mathrm{AN}$ |
| `freq_skewness` | `'ME'` | `compute_risk_table` | `SKEWNESS` and `KURTOSIS` |
| `freq_drawdown` | `'D'` | `compute_risk_table` | `MAX_DD`, `CURRENT_DD`, `WORST`, `BEST` and the numerator of `MAX_DD_VOL` |
| `freq_reg` | `'QE'` | `compute_ra_perf_table_with_benchmark` | `ALPHA`, `ALPHA_AN`, `BETA`, `R2`, `ALPHA_PVALUE` and $\mathrm{AN}_{\mathrm{reg}}$ |
| `freq_excess_return` | `'ME'` | No calculation | Nothing: excess returns are formed on the grid of the column that uses them |
| `return_type` | `ReturnTypes.LOG` | `compute_risk_table` | The basis of $v_k$ and $x_k$: `VOL`, `DOWNSIDE_VOL`, `AVG_LOG_RETURN`, `SKEWNESS`, `KURTOSIS` |
| `sharpe_convention` | `SharpeConvention.PA` | Regime tables only | `BEAR_SHARPE`, `NORMAL_SHARPE`, `BULL_SHARPE` |
| `rates_data` | `None` | Performance, risk and benchmark tables | Every excess column, the Sortino and Calmar numerators, and excess regressions |

**Definition (the `freq` shortcut).** `PerfParams(freq=f)` sets `freq_vol`, `freq_reg` and
`freq_excess_return` to `f`. It sets `freq_drawdown` to `f` only when `freq_drawdown` is empty,
which it is not by default, and it never sets `freq_skewness`. Two consequences follow.
`PerfParams(freq='QE')` computes volatility on quarters but skewness on months, and
`PerfParams(freq='ME')` moves the benchmark regression from the default quarterly grid to the
monthly one.

`return_type` accepts the five `qis.ReturnTypes` members, but only `LOG` and `RELATIVE` are
meaningful for a price table. It does not affect the arithmetic Sharpe family, which always uses
simple returns, nor `WORST`, `BEST` and the drawdowns, which always use levels or simple returns.
Under `RELATIVE` the column labelled `AvgLogReturn` holds the mean simple return.

**Definition (native endpoints and sampled boundaries).** For an asset observed from $t_0$ to
$t_1$, the sampled support is the set of complete `freq_vol` boundaries inside $[t_0,t_1]$: the
first boundary $b_0$ on or after $t_0$ and the last boundary $b_1$ on or before $t_1$, each
carrying the last available price. The visible return columns `TOTAL_RETURN`, `NAV1`,
`NUM_YEARS`, `PA_RETURN`, `PA_EXCESS_RETURN`, `AN_LOG_RETURN` and `AN_LOG_EXCESS_RETURN`, and the
date and price columns, use the native endpoints $t_0$ and $t_1$. The ratio numerators of
`SHARPE_RF0`, `SHARPE_EXCESS`, `SHARPE_LOG_AN`, `SHARPE_LOG_EXCESS` and `SORTINO_RATIO` use the
sampled boundaries $b_0$ and $b_1$, the same support as their volatility denominators. The
numerator of `CALMAR_RATIO` uses the native endpoints.

Consequently, `PA_RETURN` divided by `VOL` need not reproduce `SHARPE_RF0`. The two agree only
when $t_0$ and $t_1$ fall on the `freq_vol` grid. In the worked example below the history starts
on 2 January, the first month-end boundary is 31 January, and the two quotients differ by 16%.

> **Pitfall.** `compute_ra_perf_table(prices)` without `perf_params` builds
> `PerfParams(freq=pd.infer_freq(prices.index))`. A regular business-day index is inferred as
> `B`, so `VOL` becomes a daily volatility annualised with $\sqrt{252}$, `NUM_OBS` counts days and
> the regression grid becomes daily, while skewness stays monthly and drawdowns daily. Remove a
> single holiday row and `pd.infer_freq` returns `None`, the defaults apply, and `VOL` is monthly
> again. An index with fewer than three dates raises `ValueError`. `compute_risk_table` and
> `compute_performance_table` fall back to `PerfParams()` without inference, and
> `get_ra_perf_columns` and `compute_ra_perf_table_with_benchmark` inherit the inference. Always
> pass an explicit `PerfParams`.

### The catalogue

The four tables below list every `PerfStat` member. The label is `PerfStat.X.to_str()`, the
string used as the column name. The grid and basis column uses these words: *native* for the
asset's own observations, `freq_vol`, `freq_drawdown`, `freq_skewness` and `freq_reg` for the
`PerfParams` grids, *simple* and *excess* for simple and cash-excess returns, and
*`return_type`* for the basis selected by `PerfParams.return_type`.

#### Risk-adjusted table: `compute_ra_perf_table`

| `PerfStat` | Label | Formula | Grid and basis | Units |
|---|---|---|---|---|
| `START_DATE` | `Start date` | $t_0$ | native | date |
| `END_DATE` | `End date` | $t_1$ | native | date |
| `START_PRICE` | `Start` | $P_{t_0}$ | native | price |
| `END_PRICE` | `End` | $P_{t_1}$ | native | price |
| `NUM_OBS` | `Num Obs` | $T$, complete boundaries minus one | `freq_vol` | count |
| `TOTAL_RETURN` | `Total` | $\mathrm{TR}=P_{t_1}/P_{t_0}-1$ | native, simple | decimal |
| `NAV1` | `1$ Invested` | $1+\mathrm{TR}$ | native, simple | wealth per unit |
| `NUM_YEARS` | `Num Years` | $Y=(t_1-t_0)/365.25$ | native | years |
| `PA_RETURN` | `P.a. return` | $R_{\mathrm{pa}}=(1+\mathrm{TR})^{1/Y}-1$ if $Y>1$, else $\mathrm{TR}$ | native, simple | decimal p.a. |
| `PA_EXCESS_RETURN` | `P.a. excess return` | $\tilde R_{\mathrm{pa}}$: $R_{\mathrm{pa}}$ of $\prod(1+\tilde r_t)$ | native, excess | decimal p.a. |
| `AN_LOG_RETURN` | `An. log return` | $\log(1+R_{\mathrm{pa}})$ | native | decimal p.a. |
| `AN_LOG_EXCESS_RETURN` | `An. log return ex` | $\log(1+\tilde R_{\mathrm{pa}})$ | native, excess | decimal p.a. |
| `AVG_LOG_RETURN` | `AvgLogReturn` | $\bar v$ | `freq_vol`, `return_type` | decimal per period |
| `AVG_ARITH_RETURN` | `Avg Arith Return` | $\bar r$ | `freq_vol`, simple | decimal per period |
| `AVG_ARITH_EXCESS_RETURN` | `Avg Arith Ex return` | $\bar{\tilde r}$ | `freq_vol`, excess | decimal per period |
| `AN_ARITH_RETURN` | `An. arith return` | $\mathrm{AN}\,\bar r$ | `freq_vol`, simple | decimal p.a. |
| `AN_ARITH_EXCESS_RETURN` | `An. arith excess return` | $\mathrm{AN}\,\bar{\tilde r}$ | `freq_vol`, excess | decimal p.a. |
| `SHARPE_RF0` | `Sharpe (rf=0)` | $R^{\mathrm{smp}}_{\mathrm{pa}}/\sigma_v$ | `freq_vol` | ratio |
| `SHARPE_EXCESS` | `Ex Sharpe` | $\tilde R^{\mathrm{smp}}_{\mathrm{pa}}/\sigma_v$ | `freq_vol`, excess | ratio |
| `SHARPE_LOG_AN` | `Log Sharpe` | $\log(1+R^{\mathrm{smp}}_{\mathrm{pa}})/\sigma_v$ | `freq_vol` | ratio |
| `SHARPE_LOG_EXCESS` | `Log Ex Sharpe` | $\log(1+\tilde R^{\mathrm{smp}}_{\mathrm{pa}})/\sigma_v$ | `freq_vol`, excess | ratio |
| `SHARPE_ARITH` | `Sharpe Arith` | $\sqrt{\mathrm{AN}}\,\bar r/s(r)$ | `freq_vol`, simple | ratio |
| `SHARPE_ARITH_EXCESS` | `Ex Sharpe Arith` | $\sqrt{\mathrm{AN}}\,\bar{\tilde r}/s(\tilde r)$ | `freq_vol`, excess | ratio |
| `VOL` | `Vol` | $\sigma_v=\sqrt{\mathrm{AN}}\,s(v)$ | `freq_vol`, `return_type` | decimal p.a. |
| `DOWNSIDE_VOL` | `DownVol` | $\sigma^{-}=\sqrt{\mathrm{AN}}\,s(v\mid v<0)$; zero if $T_{-}<2$ | `freq_vol`, `return_type` | decimal p.a. |
| `SORTINO_RATIO` | `Sortino` | $\tilde R^{\mathrm{smp}}_{\mathrm{pa}}/\sigma^{-}$ | `freq_vol`, excess numerator | ratio |
| `CALMAR_RATIO` | `Calmar` | $\tilde R_{\mathrm{pa}}/\lvert\mathrm{MDD}\rvert$ | native numerator, `freq_drawdown` | ratio |
| `MAX_DD` | `Max DD` | $\mathrm{MDD}=\min_t D_t$ | `freq_drawdown` | decimal, at most 0 |
| `CURRENT_DD` | `Current DD` | $D_t$ at the last observation | `freq_drawdown` | decimal, at most 0 |
| `MAX_DD_VOL` | `Max DD/Vol` | $\mathrm{MDD}/\sigma_v$; zero if $\sigma_v=0$ | `freq_drawdown` over `freq_vol` | ratio |
| `SKEWNESS` | `Skewness` | $G_1$; missing if $T\le 2$ | `freq_skewness`, `return_type` | dimensionless |
| `KURTOSIS` | `Kurtosis` | $G_2$ (excess); missing if $T\le 3$ | `freq_skewness`, `return_type` | dimensionless |
| `WORST` | `Worst` | $\min_t r^{\mathrm{dd}}_t$ | `freq_drawdown`, simple | decimal per period |
| `BEST` | `Best` | $\max_t r^{\mathrm{dd}}_t$ | `freq_drawdown`, simple | decimal per period |

#### Descriptive tables: `compute_desc_table` and `compute_desc_freq_table`

Here $x_1,\ldots,x_T$ are the observed values of one column of whatever panel is supplied; no
resampling or annualisation happens unless requested. `compute_desc_table` returns formatted
strings; `compute_desc_freq_table` returns numbers.

| `PerfStat` | Label | Formula | Grid and basis | Units |
|---|---|---|---|---|
| `AVG` | `Avg` | $\bar x$ | supplied | units of $x$ |
| `STD` | `Std` | $s(x)$ | supplied | units of $x$ |
| `STD_AN` | `Std An` | $\sqrt{\mathrm{AN}}\,s(x)$, $\mathrm{AN}$ inferred from the index | supplied | units of $x$ p.a. |
| `T_STAT` | `T-stat` | $\bar x\,\sqrt{T}/s(x)$ | supplied | dimensionless |
| `MEDIAN` | `Median` | $\hat q_{0.5}$ | supplied | units of $x$ |
| `MIN` | `Min` | $\min_k x_k$ | supplied | units of $x$ |
| `MAX` | `Max` | $\max_k x_k$ | supplied | units of $x$ |
| `QUANT_M_1STD` | `-1std` | $\hat q_{0.16}$ | supplied | units of $x$ |
| `QUANT_P1_STD` | `+1std` | $\hat q_{0.84}$ | supplied | units of $x$ |
| `POSITIVE` | `Positive` | $\#\{x_k>0\}/T$ | supplied | fraction |
| `LAST` | `Last` | $x_T$ | supplied | units of $x$ |
| `RANK` | `Rank` | $(\#\{x_k<x_T\}+\#\{x_k\le x_T\}+1)/(2T)$ | supplied | fraction |
| `NORMTEST` | `P-val` | $P(\chi^2_2>K^2)=e^{-K^2/2}$; needs $T\ge 20$ | supplied | probability |

The descriptive table also reports skewness and kurtosis, as the uncorrected $g_1$ and $g_2$,
under the short labels `Skew` and `Kurt` taken from `PerfStat.SKEWNESS` and `PerfStat.KURTOSIS`.

#### Benchmark table: `compute_ra_perf_table_with_benchmark`

The regression is $r_{i,k}=\alpha+\beta\,r_{b,k}+\varepsilon_k$ of asset $i$ on benchmark $b$,
fitted by ordinary least squares on their joint `freq_reg` support.

| `PerfStat` | Label | Formula | Grid and basis | Units |
|---|---|---|---|---|
| `ALPHA` | `Alpha` | $\hat\alpha$ | `freq_reg`, simple or log, excess with `rates_data` | decimal per period |
| `ALPHA_AN` | `An Alpha` | $\mathrm{AN}_{\mathrm{reg}}\,\hat\alpha$ | `freq_reg` | decimal p.a. |
| `BETA` | `Beta` | $\hat\beta$ | `freq_reg` | dimensionless |
| `R2` | `R2` | $R^2$ | `freq_reg` | fraction |
| `ALPHA_PVALUE` | `p-Alpha` | Two-sided OLS t-test p-value of $\hat\alpha$; 1 for the benchmark row | `freq_reg` | probability |

#### Regime table: `compute_bnb_regimes_pa_perf_table`

The classifier buckets benchmark returns on its own grid (default `QE`, simple returns) at the
16% and 84% quantiles into bear, normal and bull periods. $\bar r_{\mid\omega}$ is the mean
periodic return of the asset in regime $\omega$.

| `PerfStat` | Label | Formula | Grid and basis | Units |
|---|---|---|---|---|
| `BEAR_AVG`, `NORMAL_AVG`, `BULL_AVG` | `Bear Avg`, `Normal Avg`, `Bull Avg` | $\bar r_{\mid\omega}$ | classifier grid, simple | decimal per period |
| `BEAR_PA`, `NORMAL_PA`, `BULL_PA` | `Bear P.a.`, `Normal P.a.`, `Bull P.a.` | $e^{\mathrm{AN}_{\mathrm{c}}\,p_{\omega}\,\bar r_{\mid\omega}}-1$, shifted pro rata to sum to $R_{\mathrm{pa}}$ | classifier grid | decimal p.a. |
| `BEAR_SHARPE`, `NORMAL_SHARPE`, `BULL_SHARPE` | `Bear-Sharpe`, `Normal-Sharpe`, `Bull-Sharpe` | regime p.a. over $\sigma_v$ under `SharpeConvention.PA` | classifier grid, `freq_vol` | ratio |

### Return, date and count columns

The date, price and count columns are read off the data. `NUM_OBS` counts the returns on the
`freq_vol` grid, which is the number of complete boundaries minus one; it is not the number of
native observations.

**Definition (per-annum and excess per-annum return).** With $Y$ elapsed years,

$$
R_{\mathrm{pa}}=
\begin{cases}
(1+\mathrm{TR})^{1/Y}-1, & Y>1,\\
\mathrm{TR}, & Y\le 1.
\end{cases}
$$

With a cash series, the per-period cash return is $r^{f}_t=y_{t-1}\,(d_t-d_{t-1})/365$. Here
$y_{t-1}$ is the `rates_data` quote one observation *of the rate series* before the last quote on
or before $d_t$. When the rate series is on the return grid this is the rate at the start of the
period; a daily rate series applied to month-end returns uses the quote of the business day
before the month-end for the whole month. The excess
per-annum return compounds $\tilde r_t=r_t-r^{f}_t$ and applies the same rule:
$\tilde R_{\mathrm{pa}}=\big(\prod_t(1+\tilde r_t)\big)^{1/Y}-1$ for $Y>1$. Without a cash series
every excess column equals its zero-rate counterpart. The log columns are
$\log(1+R_{\mathrm{pa}})$ and $\log(1+\tilde R_{\mathrm{pa}})$, which equal the annualised mean
log return by the identity in [Notation and conventions](notation_and_conventions.md). The
derivation of excess returns and NAVs is in
[Returns, NAVs, excess returns, fees and leverage](returns_and_navs.md).

The arithmetic columns are periodic means of simple returns on the `freq_vol` grid, and their
`AN_` versions multiply by $\mathrm{AN}$. They sit above the compound columns by roughly half the
annualised variance.

### The Sharpe columns

Six columns report three conventions, each with and without cash. Four of them divide a
compound numerator on the sampled boundaries by the table volatility $\sigma_v$; the arithmetic
pair divides the mean simple return by its own standard deviation, so numerator and denominator
share one series. `PerfParams.sharpe_convention` does not change any of the six; it selects the
convention of the regime Sharpe columns. The formulas are in the catalogue; the estimators, their
sampling error and the reconciliation of the three conventions follow
[Sharpe (1994)](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm) and are treated in the
[Sharpe-ratio chapter](performance_analytics_and_sharpe.md).

### Volatility and downside risk

**Definition (volatility).** `VOL` is $\sigma_v=\sqrt{\mathrm{AN}}\,s(v)$: the sample standard
deviation of the `freq_vol` returns in the `return_type` basis, with `ddof=1`, annualised with
the factor inferred from the sampled index.

**Definition (downside volatility as implemented).** Let $\mathcal{N}=\{k: v_k<0\}$ index the
$T_{-}$ negative returns and $\bar v_{-}$ their mean. Then

$$
\sigma^{-}=\sqrt{\mathrm{AN}}\,\sqrt{\frac{1}{T_{-}-1}\sum_{k\in\mathcal{N}}\big(v_k-\bar v_{-}\big)^2},
\qquad \sigma^{-}=0 \text{ if } T_{-}<2 .
$$

This is the standard deviation of the losses about their own mean. The downside risk of
Sortino and van der Meer (1991) and the target downside deviation
of Sortino and Price (1994) measure something else: shortfall below a minimum acceptable return
$\theta$, averaged over all observations. Its sample analogue, in the form given by Bacon (2008),
is

$$
\delta_{\theta}=\sqrt{\mathrm{AN}}\,\sqrt{\frac{1}{T}\sum_{k=1}^{T}\min\big(v_k-\theta,\,0\big)^2}.
$$

**Proposition (what the implemented downside volatility leaves out).** For $\theta=0$,

$$
\delta_{0}^{2}=\frac{T_{-}-1}{T}\,\big(\sigma^{-}\big)^{2}+\frac{T_{-}}{T}\,\mathrm{AN}\,\bar v_{-}^{2}.
$$

**Proof.** Only negative returns contribute to $\delta_0$. Split their second moment about zero
into dispersion and level:
$\sum_{k\in\mathcal{N}}v_k^2=\sum_{k\in\mathcal{N}}(v_k-\bar v_{-})^2+T_{-}\bar v_{-}^2
=(T_{-}-1)(\sigma^{-})^2/\mathrm{AN}+T_{-}\bar v_{-}^2$. Divide by $T$ and multiply by
$\mathrm{AN}$. $\square$

The implemented $\sigma^{-}$ keeps the first term, rescaled, and drops the second: it ignores how
large the losses are on average and how often they occur. A strategy that loses exactly 1% in
every down month has $\sigma^{-}=0$. In the worked example the dropped level term is 69% of
$\delta_0^2$ for the equity asset.

**Definition (Sortino ratio as implemented).** `SORTINO_RATIO` is
$\tilde R^{\mathrm{smp}}_{\mathrm{pa}}/\sigma^{-}$: the sampled excess per-annum return over the
implemented downside volatility. Sortino and Price (1994) define the ratio as the excess of the
return over $\theta$ divided by $\delta_{\theta}$, so that numerator and denominator refer to the
same target. In qis the numerator is in excess of cash, while the denominator uses a zero
threshold on total returns in the `return_type` basis, centred on the mean loss.

Two degenerate cases need care. A history with fewer than two negative `freq_vol` returns has
$\sigma^{-}=0$, and `SORTINO_RATIO` is then reported as infinite. A monotonically rising NAV has
$\mathrm{MDD}=0$, and `CALMAR_RATIO` is then reported as minus infinity, because the code divides
minus the numerator by a zero drawdown. Both are undefined, not extreme, values.

### Drawdowns and the Calmar ratio

The drawdown on the `freq_drawdown` grid is $D_t=P_t/\max_{t'\le t}P_{t'}-1$. `MAX_DD` is its
minimum over the full history, `CURRENT_DD` its value at the asset's last observation, and
`MAX_DD_VOL` is $\mathrm{MDD}/\sigma_v$, set to zero when $\sigma_v=0$. Episodes, durations and
time under water are in [Drawdowns and time under water](drawdowns.md).

**Proposition (sub-sampling cannot deepen the maximum drawdown).** Let grid $G'$ observe a
subset of the prices observed on grid $G$. Then $\mathrm{MDD}_{G'}\ge\mathrm{MDD}_{G}$.

**Proof.** $\mathrm{MDD}_G=\min\{P_t/P_{t'}-1: t'\le t,\ t,t'\in G\}$, because the running peak
is the maximum over earlier prices. Restricting the minimum to pairs in $G'$ can only raise it.
$\square$

Month-end prices are forward-filled daily prices, so the monthly drawdown is never deeper than
the daily one, and a business-day history re-sampled to calendar days (`D`) has the same drawdown
as on business days. The proposition matters because of the default grids.

> **Pitfall.** With `PerfParams()` the numerator of `MAX_DD_VOL` is a daily maximum drawdown and
> its denominator a monthly volatility. The column mixes a daily path statistic with a monthly
> risk scale and, by the proposition, is never shallower than the monthly-grid ratio. In the
> worked example the bond asset reports −1.96, against −1.70 with a monthly drawdown. `WORST` and
> `BEST` also come from the drawdown grid: by default they are the worst and best *day*, not
> month. Pass `freq_drawdown` equal to `freq_vol` when a single-grid table is required.

> **Insight.** `MAX_DD_VOL` is not free of the horizon. For a driftless Brownian motion in log
> price, the expected maximum drawdown over $Y$ years is $\sqrt{\pi/2}\,\sigma\sqrt{Y}$
> ([Magdon-Ismail et al., 2004](https://doi.org/10.1239/jap/1077134674)), so the ratio grows in
> magnitude like $1.25\sqrt{Y}$. Compare it only across histories of equal length.

**Definition (Calmar ratio as implemented).** `CALMAR_RATIO` is
$\tilde R_{\mathrm{pa}}/\lvert\mathrm{MDD}\rvert$: the native excess per-annum return over the
full-history maximum drawdown on the `freq_drawdown` grid. Young (1991) defines the ratio on a
trailing 36-month window: the compound annual return over the window divided by the maximum
drawdown within it, conventionally measured on month-end NAVs. The qis column differs in three
ways. Its window is the full history, which can only deepen the drawdown; its drawdown is daily
by default, which by the proposition above can only deepen it further; and its numerator is in
excess of cash when `rates_data` is given. A full-history Calmar ratio is therefore smaller in
magnitude than a 36-month one on the same fund, and the two are not interchangeable.

### Higher moments and extreme returns

For a sample $x_1,\ldots,x_T$ let $m_j=T^{-1}\sum_k(x_k-\bar x)^j$. The uncorrected skewness and
excess kurtosis are $g_1=m_3/m_2^{3/2}$ and $g_2=m_4/m_2^2-3$.

**Identity (bias-corrected moments).** The adjusted Fisher–Pearson skewness $G_1$ and the
bias-corrected excess kurtosis $G_2$ compared by Joanes and Gill (1998) are

$$
G_1=g_1\,\frac{\sqrt{T(T-1)}}{T-2},
\qquad
G_2=\frac{T-1}{(T-2)(T-3)}\Big[(T+1)\,g_2+6\Big].
$$

**Proof.** $G_1=k_3/k_2^{3/2}$ and $G_2=k_4/k_2^2$ in terms of the k-statistics
$k_2=Tm_2/(T-1)$, $k_3=T^2m_3/\big((T-1)(T-2)\big)$ and
$k_4=T^2\big[(T+1)m_4-3(T-1)m_2^2\big]/\big((T-1)(T-2)(T-3)\big)$. Substituting,
$G_1=g_1\,T^2(T-1)^{3/2}/\big((T-1)(T-2)\,T^{3/2}\big)$, which simplifies to the stated form. For
$G_2$, dividing $k_4$ by $k_2^2=T^2m_2^2/(T-1)^2$ gives
$(T-1)\big[(T+1)(g_2+3)-3(T-1)\big]/\big((T-2)(T-3)\big)$, and
$(T+1)(g_2+3)-3(T-1)=(T+1)g_2+6$. $\square$

`SKEWNESS` and `KURTOSIS` in the risk table are $G_1$ and $G_2$, computed by
`scipy.stats.skew` and `scipy.stats.kurtosis` with `bias=False` on the `freq_skewness` returns in
the `return_type` basis. They are missing for $T\le 2$ and $T\le 3$ respectively. The descriptive
table (`Skew`, `Kurt`) and the rolling skewness of `qis.compute_rolling_perf_stat` with
`RollingPerfStat.SKEW` use SciPy's default `bias=True`, that is $g_1$ and $g_2$. At $T=143$ the
correction multiplies $g_1$ by 1.011; on a 36-month rolling window it multiplies it by 1.044, so a
rolling value and a table value of the same window differ by 4% before any sampling error.

> **Insight.** The correction is small next to the sampling error. Under normality the standard
> errors of $G_1$ and $G_2$ are close to $\sqrt{6/T}$ and $\sqrt{24/T}$: 0.20 and 0.41 for twelve
> years of months. The worked example's skewness of 0.29 and excess kurtosis of −0.29 come from
> Gaussian paths, and neither is distinguishable from zero.

`WORST` and `BEST` are the minimum and maximum simple returns $r^{\mathrm{dd}}_t$ on the `freq_drawdown` grid.
When that grid is calendar days and the input is business days, weekend rows carry forward-filled
prices and zero returns, so `WORST` is at most zero and `BEST` at least zero.

### Descriptive-table columns

`compute_desc_table` works on whatever it is given, typically a panel of returns on one grid. It
drops missing values per column, rejects infinite values, and formats every number as a string.
`AVG` is the mean, `STD` the sample standard deviation with `ddof=1`, and `STD_AN` replaces `STD`
when `annualize_vol=True`, scaling by $\sqrt{\mathrm{AN}}$ with $\mathrm{AN}$ inferred from the
index (252 with a warning on an irregular index). `MEDIAN`, `QUANT_M_1STD` and `QUANT_P1_STD` are
the 50%, 16% and 84% empirical quantiles with NumPy's linear interpolation; the labels `-1std` and
`+1std` recall that these are the one-sigma quantiles of a normal distribution. `POSITIVE` counts
zeros as non-positive and excludes missing values from the denominator. `LAST` is the last
observed value and `RANK` its percentile rank within the column, computed by
`scipy.stats.percentileofscore` with `kind='rank'`.

**Definition (t-statistic).** `T_STAT` is $\bar x\,\sqrt{T}/s(x)$, missing when $T<2$ or
$s(x)=0$. It is not annualised and does not depend on `annualize_vol`.

**Identity (t-statistic and arithmetic Sharpe).** If $x_k=r_k$ are the simple `freq_vol`
returns, then `T_STAT` equals `SHARPE_ARITH` times $\sqrt{T/\mathrm{AN}}$, which is close to
$\sqrt{Y_{\mathrm{smp}}}$.

**Proof.** `SHARPE_ARITH` is $\sqrt{\mathrm{AN}}\,\bar r/s(r)$; multiply by
$\sqrt{T/\mathrm{AN}}$. On a regular grid $T/\mathrm{AN}$ is the sampled number of years. $\square$

A Sharpe ratio of 0.5 therefore needs 16 years of data to reach a t-statistic of 2, before any
correction for serial correlation.

**Definition (normality test).** `NORMTEST` is the p-value of the omnibus test of
D'Agostino and Pearson (1973), computed by `scipy.stats.normaltest`. The statistic
$K^2=Z_1^2+Z_2^2$ adds the squared normalised skewness and kurtosis statistics of
`scipy.stats.skewtest` and `scipy.stats.kurtosistest`, and is $\chi^2_2$ under normality. qis
requires at least 20 observations, the size from which SciPy documents the kurtosis test as valid
(SciPy itself returns a value from 8 observations), and prints the p-value with two decimals
under the label `P-val`.

**Identity (p-value of a two-degree chi-square).** $P(\chi^2_2>K^2)=e^{-K^2/2}$.

**Proof.** The $\chi^2_2$ density is $\tfrac12e^{-z/2}$ on $z\ge 0$; integrate from $K^2$ to
infinity. $\square$

`compute_desc_freq_table(df, freq='YE', agg_func=np.sum)` first aggregates each column to one
value per period with `agg_func`, drops periods with any missing value, and reports `AVG`, `STD`,
`-1std`, `Median` and `+1std` of the period values as numbers. With simple returns the default sum
is not the compounded period return; pass log returns, whose sum is the period log return, or a
compounding `agg_func`.

### Benchmark and regime columns

The benchmark table regresses each asset on the benchmark by ordinary least squares on the
`freq_reg` grid, with simple returns by default (log returns with `is_log_returns=True`) and
excess returns for both sides when `rates_data` is given. `ALPHA` is the periodic intercept of
Jensen (1968), `ALPHA_AN` multiplies it linearly by $\mathrm{AN}_{\mathrm{reg}}$, and
`ALPHA_PVALUE` is the conventional two-sided p-value of the OLS t-test, which assumes serially
uncorrelated, homoskedastic residuals. The benchmark's own row has its p-value set to 1. Tracking
error, information ratio and robust inference are in
[Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md).

The regime columns condition on benchmark regimes. Under the default `SharpeConvention.PA` the
regime Sharpe is the adjusted regime per-annum return divided by $\sigma_v$; under
`ARITHMETIC` and `LOG` it is $\mathrm{AN}_{\mathrm{c}}\,p_{\omega}$ times the conditional mean,
divided by the annualised standard deviation of the same returns, which adds up exactly to the
full-sample Sharpe ratio of that convention. The regime table labels its average columns
`Bear Average`, `Normal Average` and `Bull Average`, while `PerfStat.BEAR_AVG.to_str()` is
`Bear Avg`; the per-annum and Sharpe labels match their members. The decomposition is in
[Regime-conditional performance](regime_conditional_performance.md).

## Worked example

The example uses the frozen synthetic universe with its reporting quirks disabled, so the two
assets are clean Gaussian paths: `SEQ_US`, an equity with 17% volatility, and `SBD_TSY`, a
government bond with 6%. The sample runs from 2 January 2014 to 31 December 2025: 3,130 business
days, 144 month-end boundaries from 31 January 2014, and $T=143$ monthly returns. Cash is a flat
1% a year quoted from the business day before the first price. Every checked number is
recomputed with NumPy or SciPy from the month-end or daily prices.

| Statistic | `SEQ_US` | `SBD_TSY` | Independent calculation |
|---|---:|---:|---|
| `VOL` | 17.10% | 5.59% | $\sqrt{12}\,s(\ell)$ of 143 month-end log returns |
| `DOWNSIDE_VOL` | 8.85% | 2.97% | 74 and 63 negative months |
| Target downside deviation $\delta_0$ | 11.33% | 3.20% | Not a qis column |
| $\tilde R^{\mathrm{smp}}_{\mathrm{pa}}$ | 0.50% | 2.81% | Month-end excess NAV over 11.92 years |
| `SORTINO_RATIO` | 0.056 | 0.946 | Ratio of the two lines above |
| `SKEWNESS` ($G_1$) | 0.287 | 0.244 | $g_1$ = 0.284 and 0.241 |
| `KURTOSIS` ($G_2$) | −0.288 | 0.285 | $g_2$ = −0.320 and 0.234 |
| `MAX_DD` | −46.37% | −10.95% | Daily running peak |
| `CALMAR_RATIO` | 0.0064 | 0.270 | Native excess p.a. of 0.29% and 2.96% |
| `POSITIVE` (months) | 48.3% | 55.9% | Share of positive simple returns |
| `T_STAT` | 0.59 | 2.41 | `SHARPE_ARITH` of 0.172 and 0.700 times $\sqrt{143/12}$ |
| `NORMTEST` | 0.28 | 0.34 | $K^2$ of 2.52 and 2.17 |

The first block recomputes the risk-table columns. It also shows the native-endpoint rule: the
visible `PA_RETURN` of `SEQ_US` is 1.30% from 2 January, while the numerator of `SHARPE_RF0` is
1.51% from 31 January, so `PA_RETURN` over `VOL` gives 0.076 against a reported 0.088.

```python
import numpy as np
import pandas as pd
from scipy import stats

import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(start='2014-01-02', end='2025-12-31', seed=20260725,
                                       apply_quirks=False)
prices = universe.prices[['SEQ_US', 'SBD_TSY']]
cash = pd.Series(0.01, index=pd.bdate_range('2013-12-31', '2025-12-31'), name='cash')
params = qis.PerfParams(freq='ME', freq_drawdown='D', return_type=qis.ReturnTypes.LOG,
                        rates_data=cash)
table = qis.compute_ra_perf_table(prices=prices, perf_params=params)
Stat = qis.PerfStat


def column(stat: qis.PerfStat) -> pd.Series:
    """One statistic for both assets, selected by its PerfStat label."""
    return table[stat.to_str()]


# Independent inputs: complete month-end boundaries, their returns and the cash accrual.
month_end = prices.resample('ME').last()
log_m = np.log(month_end).diff().dropna()
simple_m = month_end.pct_change().dropna()
T, AN = len(log_m), 12.0
years_smp = (month_end.index[-1] - month_end.index[0]).days / 365.25
accrual_m = 0.01 * month_end.index.to_series().diff().dt.days.iloc[1:].to_numpy() / 365.0
pa_smp = (month_end.iloc[-1] / month_end.iloc[0]) ** (1.0 / years_smp) - 1.0
pa_excess_smp = (1.0 + simple_m.sub(accrual_m, axis=0)).prod() ** (1.0 / years_smp) - 1.0
assert T == 143 and (column(Stat.NUM_OBS) == T).all()

# Native endpoints for visible returns, sampled boundaries for ratio numerators.
vol = np.sqrt(AN) * log_m.std(ddof=1)
np.testing.assert_allclose(column(Stat.VOL), vol, rtol=1e-12)
np.testing.assert_allclose(column(Stat.SHARPE_RF0) * vol, pa_smp, rtol=1e-10)
np.testing.assert_allclose(pa_smp['SEQ_US'], 0.0151, atol=5e-5)
np.testing.assert_allclose(column(Stat.PA_RETURN)['SEQ_US'], 0.0130, atol=5e-5)
np.testing.assert_allclose(column(Stat.PA_RETURN)['SEQ_US'] / vol['SEQ_US'], 0.076, atol=5e-4)

# Downside volatility: standard deviation of the losses about their own mean.
down_vol = log_m.apply(lambda x: np.sqrt(AN) * x[x < 0.0].std(ddof=1))
np.testing.assert_allclose(column(Stat.DOWNSIDE_VOL), down_vol, rtol=1e-12)
np.testing.assert_allclose(column(Stat.SORTINO_RATIO), pa_excess_smp / down_vol, rtol=1e-10)
np.testing.assert_allclose(vol, [0.1710, 0.0559], atol=5e-5)
np.testing.assert_allclose(down_vol, [0.0885, 0.0297], atol=5e-5)
np.testing.assert_allclose(pa_excess_smp, [0.0050, 0.0281], atol=5e-5)
np.testing.assert_allclose(column(Stat.SORTINO_RATIO), [0.056, 0.946], atol=5e-4)

# The target downside deviation and the decomposition proposition.
losses = log_m.where(log_m < 0.0)
n_neg, mean_neg = losses.count(), losses.mean()
target_dd = np.sqrt(AN * (np.minimum(log_m, 0.0) ** 2).mean())
level_term = n_neg / T * AN * mean_neg ** 2
np.testing.assert_allclose(target_dd ** 2, (n_neg - 1) / T * down_vol ** 2 + level_term,
                           rtol=1e-12)
assert list(n_neg) == [74, 63]
np.testing.assert_allclose(target_dd, [0.1133, 0.0320], atol=5e-5)
np.testing.assert_allclose(level_term['SEQ_US'] / target_dd['SEQ_US'] ** 2, 0.69, atol=5e-3)


# Skewness and kurtosis: bias-corrected G1 and G2 from central moments.
def central_moment(x: pd.Series, j: int) -> float:
    """Central sample moment with divisor T."""
    return float(((x - x.mean()) ** j).mean())


g1 = log_m.apply(lambda x: central_moment(x, 3) / central_moment(x, 2) ** 1.5)
g2 = log_m.apply(lambda x: central_moment(x, 4) / central_moment(x, 2) ** 2 - 3.0)
G1 = g1 * np.sqrt(T * (T - 1)) / (T - 2)
G2 = ((T + 1) * g2 + 6.0) * (T - 1) / ((T - 2) * (T - 3))
np.testing.assert_allclose(column(Stat.SKEWNESS), G1, rtol=1e-10)
np.testing.assert_allclose(column(Stat.KURTOSIS), G2, rtol=1e-10)
np.testing.assert_allclose(G1, stats.skew(log_m, bias=False), rtol=1e-10)
np.testing.assert_allclose(G1, [0.287, 0.244], atol=5e-4)
np.testing.assert_allclose(G2, [-0.288, 0.285], atol=5e-4)
np.testing.assert_allclose(g1, [0.284, 0.241], atol=5e-4)
np.testing.assert_allclose(g2, [-0.320, 0.234], atol=5e-4)

# Drawdown, Calmar and worst return on the daily grid; Calmar has a native numerator.
daily = prices.pct_change().iloc[1:]
years_native = (prices.index[-1] - prices.index[0]).days / 365.25
accrual_d = 0.01 * prices.index.to_series().diff().dt.days.iloc[1:].to_numpy() / 365.0
pa_excess_native = (1.0 + daily.sub(accrual_d, axis=0)).prod() ** (1.0 / years_native) - 1.0
max_dd_daily = (prices / prices.cummax() - 1.0).min()
np.testing.assert_allclose(column(Stat.MAX_DD), max_dd_daily, rtol=1e-12)
np.testing.assert_allclose(column(Stat.PA_EXCESS_RETURN), pa_excess_native, rtol=1e-10)
np.testing.assert_allclose(column(Stat.CALMAR_RATIO), pa_excess_native / -max_dd_daily,
                           rtol=1e-10)
np.testing.assert_allclose(column(Stat.WORST), daily.min(), rtol=1e-12)
np.testing.assert_allclose(max_dd_daily, [-0.4637, -0.1095], atol=5e-5)
np.testing.assert_allclose(pa_excess_native, [0.0029, 0.0296], atol=5e-5)
np.testing.assert_allclose(column(Stat.CALMAR_RATIO), [0.0064, 0.2703], atol=1e-4)
np.testing.assert_allclose(daily.min()['SEQ_US'], -0.0360, atol=5e-5)
np.testing.assert_allclose(simple_m.min()['SEQ_US'], -0.0905, atol=5e-5)
```

The worst *day* of `SEQ_US`, −3.60%, is what `WORST` reports; its worst month was −9.05%. The
second block checks the descriptive table on the same month-end returns: the positive share, the
t-statistic and its link to the arithmetic Sharpe ratio, the uncorrected moments, and the
normality test.

```python
monthly = qis.to_returns(prices=prices, freq='ME', drop_first=True)
np.testing.assert_allclose(monthly, simple_m, rtol=1e-12)
desc = qis.compute_desc_table(df=monthly, desc_table_type=qis.DescTableType.WITH_POSITIVE_PROB,
                              is_add_tstat=True, norm_variable_display_type='{:.6f}')
positive = (simple_m > 0.0).mean()
assert list(desc[Stat.POSITIVE.to_str()]) == ['48.3%', '55.9%']
assert list(desc[Stat.POSITIVE.to_str()]) == [f'{share:.1%}' for share in positive]

t_stat = simple_m.mean() * np.sqrt(T) / simple_m.std(ddof=1)
np.testing.assert_allclose(desc[Stat.T_STAT.to_str()].astype(float), t_stat, atol=1e-6)
np.testing.assert_allclose(t_stat, column(Stat.SHARPE_ARITH) * np.sqrt(T / AN), rtol=1e-10)
np.testing.assert_allclose(t_stat, [0.592, 2.415], atol=5e-4)
np.testing.assert_allclose(column(Stat.SHARPE_ARITH), [0.172, 0.700], atol=5e-4)

# The descriptive table reports the uncorrected moments g1 and g2 as 'Skew' and 'Kurt'.
moments = qis.compute_desc_table(df=log_m, desc_table_type=qis.DescTableType.WITH_NORMAL_PVAL,
                                 norm_variable_display_type='{:.6f}')
np.testing.assert_allclose(moments['Skew'].astype(float), g1, atol=1e-6)
np.testing.assert_allclose(moments['Kurt'].astype(float), g2, atol=1e-6)

k_squared, p_value = stats.normaltest(log_m)
z_skew, z_kurt = stats.skewtest(log_m).statistic, stats.kurtosistest(log_m).statistic
np.testing.assert_allclose(k_squared, z_skew ** 2 + z_kurt ** 2, rtol=1e-12)
np.testing.assert_allclose(p_value, np.exp(-k_squared / 2.0), rtol=1e-12)
np.testing.assert_allclose(k_squared, [2.52, 2.17], atol=5e-3)
assert list(moments[Stat.NORMTEST.to_str()]) == ['0.28', '0.34']
```

Neither asset rejects normality, as it should not: the paths are Gaussian by construction. The
third block reproduces the three pitfalls of the methodology section. Without `perf_params` the
business-day index is inferred as `B` and `VOL` becomes 16.56% and 5.97%, daily volatilities
annualised with $\sqrt{252}$, instead of 17.10% and 5.59%; dropping the single row of 4 July 2014
makes the index irregular and restores the monthly numbers. The `freq` shortcut leaves skewness on
months. And `MAX_DD_VOL` divides the daily drawdown by the monthly volatility.

```python
inferred = qis.compute_ra_perf_table(prices=prices)  # perf_params=None: freq is inferred as 'B'
daily_log = np.log(prices).diff().iloc[1:]
np.testing.assert_allclose(inferred[Stat.VOL.to_str()],
                           np.sqrt(252.0) * daily_log.std(ddof=1), rtol=1e-12)
assert (inferred[Stat.NUM_OBS.to_str()] == 3129).all()
np.testing.assert_allclose(inferred[Stat.VOL.to_str()], [0.1656, 0.0597], atol=5e-5)

irregular = prices.drop(pd.Timestamp('2014-07-04'))  # one holiday row removed
assert pd.infer_freq(irregular.index) is None
fallback = qis.compute_ra_perf_table(prices=irregular)
np.testing.assert_allclose(fallback[Stat.VOL.to_str()], vol, rtol=1e-12)

assert qis.PerfParams(freq='QE').freq_skewness == 'ME'
assert qis.PerfParams().freq_reg == 'QE' and qis.PerfParams(freq='ME').freq_reg == 'ME'
assert qis.PerfParams(freq='ME').freq_drawdown == 'D'

max_dd_monthly = (month_end / month_end.cummax() - 1.0).min()
assert (max_dd_monthly >= max_dd_daily).all()
np.testing.assert_allclose(column(Stat.MAX_DD_VOL), max_dd_daily / vol, rtol=1e-12)
np.testing.assert_allclose(max_dd_daily / vol, [-2.71, -1.96], atol=5e-3)
np.testing.assert_allclose(max_dd_monthly / vol, [-2.64, -1.70], atol=5e-3)
```

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| Grids, return basis and cash | The contract above | `qis.PerfParams` |
| Column names and display formats | `to_str()`, `ValueType` | `qis.PerfStat`, `qis.ColVar`, `qis.ValueType` |
| Return basis of $v_k$ and $x_k$ | $\ell_t$ or $r_t$ | `qis.ReturnTypes` (`LOG`, `RELATIVE`) |
| Regime Sharpe convention | PA, arithmetic or log | `qis.SharpeConvention` |
| Risk-adjusted table | All risk-adjusted columns | `qis.compute_ra_perf_table(prices, perf_params=None)` |
| Visible return columns | $\mathrm{TR}$, $R_{\mathrm{pa}}$, $\tilde R_{\mathrm{pa}}$, $Y$ | `qis.compute_performance_table(prices, perf_params)` |
| Risk columns | $\sigma_v$, $\sigma^{-}$, $\mathrm{MDD}$, $G_1$, $G_2$, $\min r^{\mathrm{dd}}_t$, arithmetic family | `qis.compute_risk_table(prices, perf_params=None)` |
| Downside volatility | $\sqrt{\mathrm{AN}}\,s(v\mid v<0)$ | internal `_safe_downside_vol` in `qis/perfstats/perf_stats.py` |
| Maximum and current drawdown | $\min_t D_t$, last $D_t$ | `qis.compute_max_current_drawdown` |
| Benchmark columns | $\hat\alpha$, $\mathrm{AN}_{\mathrm{reg}}\hat\alpha$, $\hat\beta$, $R^2$, p-value | `qis.compute_ra_perf_table_with_benchmark(prices, benchmark, benchmark_price, perf_params)` |
| Regime columns | $\bar r_{\mid\omega}$, regime p.a., regime Sharpe | `qis.compute_bnb_regimes_pa_perf_table` |
| Formatted preset table | Columns of a preset, as strings | `qis.get_ra_perf_columns`, `qis.plot_ra_perf_table`, `qis.plot_ra_perf_table_benchmark` |
| Descriptive table | $\bar x$, $s(x)$, t-statistic, quantiles, positive share, rank, $K^2$ p-value | `qis.compute_desc_table(df, desc_table_type, annualize_vol, is_add_tstat)` |
| Period aggregates | Mean, standard deviation and quantiles of period sums | `qis.compute_desc_freq_table(df, freq='YE', agg_func=np.sum)` |

The implementations are in
[perf_stats.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/perf_stats.py),
[config.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/config.py),
[desc_table.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/desc_table.py),
[returns.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py)
and [regime_classifier.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/regime_classifier.py).

### Descriptive-table modes

`compute_desc_table` always starts from `Avg` and `Std` (or `Std An`), adds `T-stat` when
`is_add_tstat=True`, and then applies the mode.

| `DescTableType` | Columns added or removed |
|---|---|
| `NONE` | Not implemented; raises `TypeError` |
| `SHORT` (default) | No further columns |
| `AVG_WITH_POSITIVE_PROB` | Removes `Avg` and `Std`; adds `Positive` |
| `WITH_POSITIVE_PROB` | Adds `Positive` |
| `WITH_KURTOSIS` | Adds `Skew`, `Kurt` |
| `WITH_NORMAL_PVAL` | Adds `Skew`, `Kurt`, `P-val` |
| `WITH_SCORE` | Adds `Last`, `Rank` |
| `EXTENSIVE` | Adds `Skew`, `Kurt`, `Min`, `-1std`, `Median`, `+1std`, `Max` |
| `SKEW_KURTOSIS` | Removes `Avg` and `Std`; adds `Skew`, `Kurt` |
| `WITH_MEDIAN` | Adds `Median`, `Skew`, `Kurt` |

### Column presets

Reporting functions select columns through preset tuples of `PerfStat` members. The twelve
exported presets are listed below, in column order. `get_ra_perf_columns` and
`plot_ra_perf_table` draw from `compute_ra_perf_table` and silently omit a requested column that
table does not produce, so a preset with regression or regime columns needs the producer named
in the last column.

| Preset | Columns | Producer |
|---|---|---|
| `STANDARD_TABLE_COLUMNS` | `START_DATE`, `END_DATE`, `PA_RETURN`, `VOL`, `SHARPE_RF0`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS`, `KURTOSIS` | Risk-adjusted table; default of `get_ra_perf_columns` |
| `RA_TABLE_COLUMNS` | `STANDARD_TABLE_COLUMNS` followed by `WORST`, `BEST` | Risk-adjusted table |
| `RA_TABLE_COMPACT_COLUMNS` | `PA_RETURN`, `VOL`, `SHARPE_RF0`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS`, `KURTOSIS` | Risk-adjusted table |
| `COMPACT_TABLE_COLUMNS` | `TOTAL_RETURN`, `PA_RETURN`, `VOL`, `SHARPE_RF0`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS` | Risk-adjusted table |
| `FULL_TABLE_COLUMNS` | `START_DATE`, `END_DATE`, `NUM_OBS`, `AVG_LOG_RETURN`, `PA_RETURN`, `VOL`, `SHARPE_RF0`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS`, `KURTOSIS`, `WORST`, `BEST` | Risk-adjusted table |
| `EXTENDED_TABLE_COLUMNS` | `START_DATE`, `END_DATE`, `START_PRICE`, `END_PRICE`, `TOTAL_RETURN`, `PA_RETURN`, `VOL`, `SHARPE_RF0`, `SHARPE_EXCESS`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS`, `KURTOSIS` | Risk-adjusted table |
| `LN_TABLE_COLUMNS` | `START_DATE`, `END_DATE`, `TOTAL_RETURN`, `PA_RETURN`, `AN_LOG_RETURN`, `VOL`, `SHARPE_RF0`, `SHARPE_LOG_AN`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS`, `KURTOSIS` | Risk-adjusted table |
| `LN_BENCHMARK_TABLE_COLUMNS` | `LN_TABLE_COLUMNS` followed by `ALPHA`, `BETA`, `R2` | Benchmark table |
| `LN_BENCHMARK_TABLE_COLUMNS_SHORT` | `TOTAL_RETURN`, `PA_RETURN`, `AN_LOG_RETURN`, `VOL`, `SHARPE_RF0`, `SHARPE_LOG_AN`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS`, `ALPHA`, `BETA`, `R2` | Benchmark table |
| `BENCHMARK_TABLE_COLUMNS` | `PA_RETURN`, `VOL`, `SHARPE_RF0`, `MAX_DD`, `SKEWNESS`, `ALPHA_AN`, `BETA`, `R2`, `ALPHA_PVALUE` | Benchmark table; default of `plot_ra_perf_table_benchmark` |
| `BENCHMARK_TABLE_COLUMNS2` | `TOTAL_RETURN`, `PA_RETURN`, `VOL`, `SHARPE_EXCESS`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS`, `ALPHA_AN`, `BETA`, `R2` | Benchmark table |
| `SD_PERF_COLUMNS` | `START_DATE`, `END_DATE`, `PA_RETURN`, `VOL`, `SHARPE_RF0`, `BEAR_SHARPE`, `NORMAL_SHARPE`, `BULL_SHARPE`, `MAX_DD`, `MAX_DD_VOL`, `SKEWNESS` | Regime table |

Every preset except `BENCHMARK_TABLE_COLUMNS` contains the mixed-grid `MAX_DD_VOL`, and none
contains `DOWNSIDE_VOL`, `SORTINO_RATIO`, `CALMAR_RATIO`, `CURRENT_DD` or the arithmetic Sharpe
pair; pass an explicit tuple of members to report them. The block below checks these two claims
against the exported presets.

```python
presets = [name for name in qis.__all__ if '_COLUMNS' in name]
assert len(presets) == 12
assert qis.RA_TABLE_COLUMNS == qis.STANDARD_TABLE_COLUMNS + (Stat.WORST, Stat.BEST)
assert qis.LN_BENCHMARK_TABLE_COLUMNS == qis.LN_TABLE_COLUMNS + (Stat.ALPHA, Stat.BETA, Stat.R2)
assert [name for name in presets if Stat.MAX_DD_VOL not in getattr(qis, name)] == [
    'BENCHMARK_TABLE_COLUMNS']
absent = (Stat.DOWNSIDE_VOL, Stat.SORTINO_RATIO, Stat.CALMAR_RATIO, Stat.CURRENT_DD,
          Stat.SHARPE_ARITH, Stat.SHARPE_ARITH_EXCESS)
assert not any(stat in getattr(qis, name) for name in presets for stat in absent)
assert len(qis.PerfStat) == 61

custom = (Stat.PA_EXCESS_RETURN, Stat.VOL, Stat.DOWNSIDE_VOL, Stat.SORTINO_RATIO,
          Stat.MAX_DD, Stat.CALMAR_RATIO)
formatted = qis.get_ra_perf_columns(prices=prices, perf_params=params, perf_columns=custom)
assert list(formatted.columns) == [stat.to_str() for stat in custom]
```

Two naming details matter when code selects columns. `PerfStat.X.to_str()` returns the display
label, and so does `PerfStat.X.name`, because the `ColVar` field `name` shadows the enumeration's
member name; use `PerfStat.X._name_` or `PerfStat['VOL']` for the member name. `ALPHA` and
`ALPHA_AN` share the wrapped label `Alpha`, so a wrapped header containing both is ambiguous.

## Interpretation and limitations

- Every column is a full-sample, descriptive statistic. None is point in time, and none should
  enter a backtest decision at an earlier date.
- The default table mixes grids. `MAX_DD`, `CURRENT_DD`, `WORST` and `BEST` are daily, `VOL`,
  the Sharpe family and the Sortino ratio monthly, and `MAX_DD_VOL` divides one by the other. The
  regression grid is quarterly under `PerfParams()` and monthly under `PerfParams(freq='ME')`.
- `freq_excess_return` is stored, printed and copied but not read by any calculation, and
  `PerfParams.copy` does not carry `freq_skewness`, which returns to `'ME'` in the copy.
- For a sampled history of one year or less the ratio numerators are total, not annualised,
  returns, while `VOL` is annualised; the resulting Sharpe and Sortino ratios are not comparable
  with those of longer histories.
- Supply `rates_data` starting at least one observation before the first price. In the current
  implementation a rate series that starts on the first price date leaves the first native
  excess return missing, and the native excess per-annum return then compounds from the second
  observation while still dividing by the full $Y$.
- Degenerate inputs give values that are undefined rather than extreme: an infinite Sortino
  ratio, a minus-infinite Calmar ratio, a zero `MAX_DD_VOL` for zero volatility, and missing
  moments for fewer than three or four observations.
- Skewness, kurtosis and the normality test assume independent observations. Under serial
  correlation, as in smoothed private-asset returns, they are less precise than their nominal
  standard errors suggest; see [Serial dependence and autocorrelation](serial_dependence.md).
- The bias-corrected moments of the risk table and the uncorrected moments of the descriptive
  table and rolling skewness are different estimators; compare like with like.

## See also

- [Notation and conventions](notation_and_conventions.md)
- [Returns, NAVs, excess returns, fees and leverage](returns_and_navs.md)
- [Sharpe ratios: conventions and inference](performance_analytics_and_sharpe.md)
- [Drawdowns and time under water](drawdowns.md)
- [Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md)
- [Regime-conditional performance](regime_conditional_performance.md)
- [Reporting frequency and annualisation](frequency_convention_note.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Factsheets and reporting](factsheets_and_reporting.md)
- {doc}`PerfParams API <api/generated/qis.PerfParams>`,
  {doc}`PerfStat API <api/generated/qis.PerfStat>`,
  {doc}`compute_ra_perf_table API <api/generated/qis.compute_ra_perf_table>` and
  {doc}`compute_desc_table API <api/generated/qis.compute_desc_table>`
- [Bibliography](bibliography.md)

## References

1. Sortino, F. A., and van der Meer, R. (1991). Downside Risk. *The Journal of Portfolio Management*, 17(4), 27–31. Downside risk measured against a minimum acceptable return.
2. Sortino, F. A., and Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. *The Journal of Investing*, 3(3), 59–64. The Sortino ratio and the target downside deviation contrasted with `DOWNSIDE_VOL`.
3. Young, T. W. (1991). Calmar Ratio: A Smoother Tool. *Futures*, 20(1), 40. The 36-month Calmar convention contrasted with the full-history column.
4. Joanes, D. N., and Gill, C. A. (1998). Comparing Measures of Sample Skewness and Kurtosis. *Journal of the Royal Statistical Society: Series D (The Statistician)*, 47(1), 183–189. [DOI: 10.1111/1467-9884.00122](https://doi.org/10.1111/1467-9884.00122). The $g_1$, $G_1$, $g_2$ and $G_2$ estimators and their relation.
5. D'Agostino, R., and Pearson, E. S. (1973). Tests for Departure from Normality. Empirical Results for the Distributions of b2 and √b1. *Biometrika*, 60(3), 613–622. [DOI: 10.1093/biomet/60.3.613](https://doi.org/10.1093/biomet/60.3.613). The omnibus normality test behind `NORMTEST`.
6. Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd edition. Wiley. Practitioner definitions of downside deviation, the Sortino ratio and the Calmar ratio.
7. Sharpe, W. F. (1994). The Sharpe Ratio. *The Journal of Portfolio Management*, 21(1), 49–58. [Author's copy](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm). The historical Sharpe ratio behind the arithmetic columns.
8. Jensen, M. C. (1968). The Performance of Mutual Funds in the Period 1945–1964. *The Journal of Finance*, 23(2), 389–416. [DOI: 10.1111/j.1540-6261.1968.tb00815.x](https://doi.org/10.1111/j.1540-6261.1968.tb00815.x). The regression alpha reported by `ALPHA`.
9. Magdon-Ismail, M., Atiya, A. F., Pratap, A., and Abu-Mostafa, Y. S. (2004). On the Maximum Drawdown of a Brownian Motion. *Journal of Applied Probability*, 41(1), 147–161. [DOI: 10.1239/jap/1077134674](https://doi.org/10.1239/jap/1077134674). The square-root growth of the expected maximum drawdown with the horizon.
10. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
