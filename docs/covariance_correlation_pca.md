---
myst:
  html_meta:
    description: >-
      Sample, pairwise-complete and exponentially weighted covariance and correlation estimators,
      principal components, eigen-portfolios and eigenvalue clipping, as implemented in qis.
---

# Covariance, correlation and principal components

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

The covariance matrix $\Sigma$ of $n$ asset returns collects their variances and covariances; the
correlation matrix $\rho$ is its scale-free form. Principal component analysis diagonalises either
matrix and ranks orthogonal return directions by the variance they carry. This chapter defines the
estimators qis implements (full-sample, pairwise-complete and exponentially weighted) and the
spectral tools built on them, and states for each which observations, which mean and which dates
enter an estimate.

## Overview

Three questions separate the estimators, and the qis functions answer them differently:

1. **Which observations define an entry?** One common sample for every pair, each pair's own
   overlap, or exponentially decaying weights on all past rows.
2. **Which mean is removed?** The sample mean, each series' own mean, an EWM mean, or none.
3. **Is the matrix positive semidefinite (PSD)?** Only a PSD matrix is a valid covariance: every
   portfolio variance $w^{\top}\Sigma w$ must be non-negative.

| Estimator | qis entry point | Mean removed | PSD | Point in time |
|---|---|---|---|---|
| Common-sample covariance or correlation | `compute_masked_covar_corr` on a panel without NaN | Sample mean | Yes | No: full sample |
| Pairwise-complete covariance or correlation | `compute_masked_covar_corr` on a panel with NaN | Own-series mean (covariance), overlap mean (correlation) | Not guaranteed | No: full sample |
| EWM covariance on rebalancing dates | `estimate_rolling_ewma_covar` | EWM mean, including the current return | Yes | Yes |
| Uncentred EWM correlation paths | `compute_ewm_corr_df`, `compute_ewm_corr_single`, `compute_data_pca_r2` | None | Yes, on gap-free input | Yes |

The spectral layer works on any symmetric matrix: `apply_pca` and `compute_pca_r2` for eigenvalues
and explained-variance shares, `compute_eigen_portfolio_weights` for unit-variance principal
portfolios, and `matrix_regularization` for eigenvalue clipping.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Caller's returns for `compute_masked_covar_corr` and the EWM correlation functions; `estimate_rolling_ewma_covar` forms log returns from prices |
| Sampling grid | Native rows of the input; `estimate_rolling_ewma_covar` samples prices at `returns_freq` and reports on `rebalancing_freq` dates |
| Annualisation | Per period unless stated: a covariance annualises by $\mathrm{AN}$, a volatility by $\sqrt{\mathrm{AN}}$, a correlation not at all; `estimate_rolling_ewma_covar` multiplies by $\mathrm{AN}$ inferred from its return grid |
| Mean adjustment | Sample mean with `ddof=1` (common sample); own-series mean (pairwise covariance); overlap means (pairwise correlation); EWM mean including the current return (`estimate_rolling_ewma_covar`); none (EWM correlations) |
| Timing | Full-sample and pairwise matrices use the whole sample and are descriptive; an EWM matrix dated $t$ uses returns up to and including $t$ and serves a decision at $t$ applied over $(t,t+1]$ |
| Output units | Covariance in squared return units per period, or per year after $\mathrm{AN}$; correlations and variance shares dimensionless; eigen-portfolios scaled to unit variance in the units of $\Sigma$ |
| qis default | `estimate_rolling_ewma_covar(returns_freq='W-WED', rebalancing_freq='QE', span=52, demean=True, apply_an_factor=True)`; `compute_ewm_corr_df(ewm_lambda=0.94)`; `matrix_regularization(cut=1e-5)` |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $n$ | Number of assets | Columns of the return panel |
| $x_t$, $x_{i,t}$ | Column vector of the $n$ returns at $t$; its $i$-th entry | Simple or log, as supplied |
| $\bar x$, $\bar x_i$ | Sample mean vector; mean of asset $i$ over all its observations | Per period |
| $d$ | Degrees-of-freedom correction `ddof` | $d=1$ by default, $d=0$ with `bias=True` |
| $\hat\Sigma$, $\hat\Sigma_t$ | Covariance estimate; EWM estimate at $t$ | Squared return units per period |
| $\sigma_i$, $\Delta$ | Volatility $\sqrt{\Sigma_{ii}}$; the diagonal matrix $\operatorname{diag}(\sigma_1,\ldots,\sigma_n)$ | $\Delta$ is local and unrelated to the drawdown $D_t$ |
| $\rho$, $\rho_{ij}$ | Correlation matrix and entry | Dimensionless |
| $\Gamma$, $c$ | Positive diagonal matrix and positive scalar in a rescaling | Dimensionless |
| $\eta$ | Variance tolerance of the normalisation kernel | $100$ times machine epsilon times the largest finite $\lvert\Sigma_{kl}\rvert$ |
| $O_i$, $O_{ij}$, $n_{ij}$ | Dates on which asset $i$ is observed; overlap $O_i\cap O_j$; its size | Pairwise estimators |
| $\bar x^{(ij)}_i$ | Mean of asset $i$ over the overlap $O_{ij}$ | Per period |
| $m_t$, $e_t$ | EWM mean and EWM-demeaned return $x_t-m_t$ | Per period |
| $S_t$ | Uncentred EWM second-moment matrix | Squared return units per period |
| $K$ | Number of returns since the zero seed | Warm-up count |
| $\mathcal{T}_{\mathrm{reb}}$ | Rebalancing dates on the return grid | Dates |
| $a_t$, $b_{k,t}$, $O_k$ | Pivot return, $k$-th comparison return, dates where $b_k$ is observed | `corr_to_pivot_row` |
| $\nu_j$, $v_j$ | $j$-th eigenvalue in descending order; unit eigenvector | $\nu$ because $\lambda$ is the EWM decay |
| $Q$ | Orthogonal matrix with columns $v_j$ | $Q^{\top}Q=I$ |
| $\pi_j$ | Explained-variance share $\nu_j/\sum_k\nu_k$ | Dimensionless |
| $f_{j,t}$ | Return of the $j$-th principal component | Unit of $v_j^{\top}\Delta^{-1}x_t$ |
| $w^{(j)}$ | $j$-th eigen-portfolio weight vector | Unit variance in the units of $\Sigma$ |
| $M$, $Z$ | A symmetric matrix to be repaired; a PSD candidate | Units of the matrix |
| $\kappa$ | Eigenvalue cut of `matrix_regularization` | Absolute, in the units of the matrix |
| $q$, $\nu_{\pm}$ | Aspect ratio $n/T$; Marchenko–Pastur edges | Correlation eigenvalue units |
| $N_{\mathrm{eff}}$ | Effective number of observations of EWM weights | Equals the span $N$ |

The functions assume nothing about the return distribution. The interpretation of a sample or EWM
matrix as an estimate of a population $\Sigma$ assumes returns that are stationary over the window
the weights cover.

## Methodology

### Sample covariance and correlation

**Definition.** For a complete panel of $T$ rows with sample mean
$\bar x=\frac{1}{T}\sum_{t=1}^{T}x_t$, the sample covariance is

$$
\hat\Sigma=\frac{1}{T-d}\sum_{t=1}^{T}(x_t-\bar x)(x_t-\bar x)^{\top},
$$

with $d=1$ (unbiased, the default) or $d=0$ (`bias=True`). The correlation matrix is

$$
\rho=\Delta^{-1}\Sigma\,\Delta^{-1},
\qquad
\rho_{ij}=\frac{\Sigma_{ij}}{\sigma_i\sigma_j},
\qquad
\Delta=\operatorname{diag}(\sigma_1,\ldots,\sigma_n),\quad \sigma_i=\sqrt{\Sigma_{ii}} .
$$

**Proposition (a common sample gives a PSD matrix).** $\hat\Sigma$ built from one common sample
is PSD, and so is its correlation matrix.

**Proof.** For any $w$,
$w^{\top}\hat\Sigma w=\frac{1}{T-d}\sum_t\big(w^{\top}(x_t-\bar x)\big)^2\ge0$. The correlation is
a congruence: $w^{\top}\rho w=u^{\top}\hat\Sigma u$ with $u=\Delta^{-1}w$. $\square$

**Identity (scale invariance of correlation).** For $c>0$ and a positive diagonal $\Gamma$, the
matrices $\Sigma$ and $c\,\Gamma\Sigma\Gamma$ have the same correlation matrix. In particular the
annualised covariance $\mathrm{AN}\,\Sigma$, the per-period $\Sigma$, and the $d=0$ and $d=1$
estimates all share one correlation matrix.

**Proof.** The rescaling maps $\Sigma_{ij}\mapsto c\gamma_i\gamma_j\Sigma_{ij}$ and
$\sigma_i\mapsto\sqrt{c}\,\gamma_i\sigma_i$, so the factor $c\gamma_i\gamma_j$ cancels in
$\Sigma_{ij}/(\sigma_i\sigma_j)$. $\square$

A covariance is a second moment and annualises like a variance:
$\Sigma^{\mathrm{ann}}=\mathrm{AN}\,\Sigma$ and
$\sigma^{\mathrm{ann}}_i=\sqrt{\mathrm{AN}}\,\sigma_i$. The invariance is algebraic. Whether the
annualised matrix describes annual returns depends on serial and cross-serial correlation, which
asynchronous closes and stale prices create; see [serial dependence](serial_dependence.md) and
[reporting frequency and annualisation](frequency_convention_note.md).

**Definition (normalisation kernel).** Since qis 5.30.3 the covariance-to-correlation conversions
share one kernel. With $\eta=100\,\mathrm{eps}\,\max_{k,l}\lvert\Sigma_{kl}\rvert$ over the
finite entries and $\mathrm{eps}\approx2.2\times10^{-16}$, a variance $\Sigma_{ii}>\eta$ is valid;
a variance that is missing, non-finite or within $[-\eta,\eta]$ makes row and column $i$
missing, diagonal included; a variance below $-\eta$ raises `ValueError`. Valid off-diagonal entries are divided by $\sigma_i\sigma_j$ without
clipping to $[-1,1]$, and the valid diagonal is set to one. `covar_to_corr`, the EWM paths with
`is_corr=True`, `compute_eigen_portfolio_weights` and `plot_corr_matrix_from_covar` use it; the
pairwise correlation of `compute_masked_covar_corr` does not.

### Pairwise-complete estimation with missing data

A ragged panel has no common sample. `compute_masked_covar_corr` uses `np.cov` or `np.corrcoef`
when the panel has no NaN, and otherwise estimates each entry from the dates on which both series
are observed. The covariance and correlation paths then use different means.

**Definition (implemented pairwise estimators).** With $O_{ij}=O_i\cap O_j$ and
$n_{ij}=\lvert O_{ij}\rvert$, the covariance path (`is_covar=True`, NumPy masked covariance)
centres each series on its own full-history mean,

$$
\hat\Sigma^{\mathrm{mask}}_{ij}=\frac{1}{n_{ij}-d}\sum_{t\in O_{ij}}(x_{i,t}-\bar x_i)(x_{j,t}-\bar x_j),
\qquad
\bar x_i=\frac{1}{\lvert O_i\rvert}\sum_{t\in O_i}x_{i,t},
$$

while the correlation path (`is_covar=False`, pandas pairwise-complete Pearson) computes every
moment on the overlap:

$$
\hat\rho^{\mathrm{pair}}_{ij}=
\frac{\sum_{t\in O_{ij}}(x_{i,t}-\bar x^{(ij)}_i)(x_{j,t}-\bar x^{(ij)}_j)}
{\Big(\sum_{t\in O_{ij}}(x_{i,t}-\bar x^{(ij)}_i)^2\sum_{t\in O_{ij}}(x_{j,t}-\bar x^{(ij)}_j)^2\Big)^{1/2}} .
$$

By Cauchy–Schwarz on the overlap, $\lvert\hat\rho^{\mathrm{pair}}_{ij}\rvert\le1$. The masked
covariance has no such bound once it is normalised by variances from longer histories, which is
why the correlation path does not normalise it.

**Identity (own-mean versus overlap-mean covariance).** Let $\hat\Sigma^{\mathrm{pair}}_{ij}$ be
the covariance of the overlap sample about its own means, with the same $d$ (pandas
`DataFrame.cov` uses $d=1$). Then

$$
\hat\Sigma^{\mathrm{mask}}_{ij}-\hat\Sigma^{\mathrm{pair}}_{ij}
=\frac{n_{ij}}{n_{ij}-d}\,\big(\bar x^{(ij)}_i-\bar x_i\big)\big(\bar x^{(ij)}_j-\bar x_j\big).
$$

**Proof.** On $O_{ij}$ write
$x_{i,t}-\bar x_i=(x_{i,t}-\bar x^{(ij)}_i)+(\bar x^{(ij)}_i-\bar x_i)$, and likewise for $j$.
Multiply and sum over $O_{ij}$: the two cross terms vanish because deviations from an overlap
mean sum to zero over the overlap, leaving the overlap sum of products plus $n_{ij}$ times the
product of the mean differences. Divide by $n_{ij}-d$. $\square$

The two conventions agree whenever one series of the pair has no observation outside the overlap.

**Proposition (pairwise matrices need not be PSD).** Neither $\hat\Sigma^{\mathrm{mask}}$ nor
$\hat\rho^{\mathrm{pair}}$ is guaranteed PSD. Take three series observed in pairs on three
disjoint blocks of three dates: $A=B=(1,2,3)$ on the first block, $B=C=(1,2,3)$ on the second,
and $A=(1,2,3)$, $C=(3,2,1)$ on the third. Then
$\hat\rho^{\mathrm{pair}}_{AB}=\hat\rho^{\mathrm{pair}}_{BC}=1$,
$\hat\rho^{\mathrm{pair}}_{AC}=-1$, and the weight vector $w=(1,-1,1)^{\top}$ has
$w^{\top}\hat\rho^{\mathrm{pair}}w=-3$. The eigenvalues are $2$, $2$ and $-1$.

**Proof.** Each overlap is an exact increasing or decreasing linear relation, so each pairwise
correlation is $\pm1$. Then
$w^{\top}\hat\rho w=\sum_iw_i^2+2\sum_{i<j}w_iw_j\hat\rho_{ij}=3+2(-1-1-1)=-3<0$, and a negative
quadratic form rules out PSD. Every series has mean 2 over its own six observations and over each
overlap, so $\hat\Sigma^{\mathrm{mask}}$ has variances $0.8$ and covariances $\pm1$, and the same
$w$ gives $2.4-6=-3.6$. $\square$

The PSD proof above needs one Gram matrix of one sample; a pairwise matrix assembles entries from
different samples. The masked covariance in the counterexample even implies a correlation of
$1/0.8=1.25$.

> **Pitfall.** A pairwise-complete matrix can assign negative variance to a portfolio. A
> minimum-variance optimiser finds and levers such a direction, a Cholesky factorisation fails,
> and risk contributions lose their meaning. Check the smallest eigenvalue before using
> `compute_masked_covar_corr` output as a risk model; repair it by clipping and renormalising, or
> estimate on a common sample.

### Exponentially weighted covariance on a rebalancing schedule

`estimate_rolling_ewma_covar` is the backtest-facing estimator. The EWM recursion, its seeds and
its missing-data policies are derived in [exponentially weighted estimators](ewm_estimators.md);
this section states how the function uses them.

1. **Returns.** Prices are sampled at `returns_freq` (the last price on or before each grid date,
   forward-filled) and converted to log returns $\ell_{i,t}$, so $x_t$ is the vector of weekly log
   returns by default. Forward filling turns a gap inside a history into a zero return; a missing
   return appears only before an asset's first price.
2. **Demeaning.** With `demean=True`, $e_t=x_t-m_t$ with $m_t=\lambda m_{t-1}+(1-\lambda)x_t$,
   $\lambda=1-2/(N+1)$ and $N$ = `span` in units of `returns_freq`. The mean is seeded with the
   first row of returns, so the first residual is zero; an asset whose returns start later is
   seeded at zero. With `demean=False`, $e_t=x_t$: the second moment about zero.
3. **Recursion.** Starting from $\hat\Sigma_0=0$,

   $$
   \hat\Sigma_t=\lambda\hat\Sigma_{t-1}+(1-\lambda)\,e_te_t^{\top},
   $$

   with `NanBackfill.ZERO_FILL`: an entry whose update is not finite, because either asset is
   missing at $t$, is reset to zero.
4. **Sampling and annualisation.** The function returns a dictionary from each date in
   $\mathcal{T}_{\mathrm{reb}}$ to $\mathrm{AN}\,\hat\Sigma_t$. A rebalancing date is the first
   return date on or after each scheduled `rebalancing_freq` date
   (`qis.generate_rebalancing_indicators`). $\mathrm{AN}$ comes from
   `qis.infer_annualisation_factor_from_df` on the return grid, 52 for `W-WED`; an irregular grid
   falls back to 252 with a warning. `apply_an_factor=False` returns per-period matrices.

The option `is_apply_vol_normalised_returns=True` rebuilds the matrix as
$\operatorname{diag}(\tilde\sigma_t)\,\tilde\rho_t\operatorname{diag}(\tilde\sigma_t)$, where
$\tilde\sigma_t$ is the EWM volatility of $e_t$ and $\tilde\rho_t$ the EWM correlation of
$e_{i,t}/\tilde\sigma_{i,t}$. That volatility recursion is seeded with the full-sample mean of
$e_t^2$, a look-ahead that decays like $\lambda^t$.

**Identity (the EWM mean includes the current return).**
$e_t=x_t-m_t=\lambda\,(x_t-m_{t-1})$.

**Proof.** Substitute $m_t=\lambda m_{t-1}+(1-\lambda)x_t$ into $x_t-m_t$. $\square$

The demeaning is therefore point in time: $e_t$ uses only returns dated at or before $t$. It also
shrinks every residual by $\lambda$.

**Proposition (steady-state scale of the demeaned estimator).** If the $x_t$ are independent and
identically distributed with mean $\mu$ and covariance $\Sigma$, then once the seeds are forgotten

$$
\mathbb{E}\big[\hat\Sigma_t\big]=\frac{2\lambda^2}{1+\lambda}\,\Sigma=\frac{(N-1)^2}{N(N+1)}\,\Sigma .
$$

**Proof.** $m_{t-1}=(1-\lambda)\sum_{k\ge0}\lambda^kx_{t-1-k}$ is independent of $x_t$, with mean
$\mu$ and covariance $\frac{(1-\lambda)^2}{1-\lambda^2}\Sigma=\frac{1-\lambda}{1+\lambda}\Sigma$.
Hence $x_t-m_{t-1}$ has mean zero and covariance $\frac{2}{1+\lambda}\Sigma$, and by the identity
$\mathbb{E}[e_te_t^{\top}]=\frac{2\lambda^2}{1+\lambda}\Sigma$. The EWM weights sum to one in the
steady state, and $\lambda=(N-1)/(N+1)$ gives the second form. $\square$

For $N=52$ the factor is $0.944$: variances are 5.6% low and volatilities 2.9% low.

> **Insight.** Removing an EWM mean costs more than it saves at typical spans. Without demeaning
> the bias is $\mu\mu^{\top}$, which for weekly returns of an asset with 8% drift and 17%
> volatility is $\mu^2/\sigma^2\approx0.4\%$ of the variance, against the $-5.6\%$ above. The
> demeaning factor is common to all entries, so it cancels in correlations.

**Identity (zero-seed warm-up).** If $\mathbb{E}[e_te_t^{\top}]=S$ for all $t$, then after $K$
returns $\mathbb{E}[\hat\Sigma]=(1-\lambda^{K})\,S$.

**Proof.** The weights on the $K$ outer products are $(1-\lambda)\lambda^{k}$, $k=0,\ldots,K-1$,
and sum to $1-\lambda^{K}$; the zero seed contributes nothing. $\square$

Assets with a common start share the factor, so their correlations are not biased by it. An asset
that starts $K$ returns later has $\hat\Sigma_{ij}$ and $\hat\Sigma_{jj}$ scaled by
$1-\lambda^{K}$ while $\hat\Sigma_{ii}$ is not, which biases $\hat\rho_{ij}$ towards zero by
$\sqrt{1-\lambda^{K}}$.

**Proposition (the EWM covariance is PSD).** Under `ZERO_FILL`, $\hat\Sigma_t$ is PSD at every $t$.

**Proof.** Without missing values $\hat\Sigma_t$ is a positive combination of the PSD matrices
$e_se_s^{\top}$. When some assets are missing at $t$, let $\tilde e_t$ be $e_t$ with those
entries set to zero and $P$ the diagonal projector that zeroes them; the update is
$P\big(\lambda\hat\Sigma_{t-1}+(1-\lambda)\tilde e_t\tilde e_t^{\top}\big)P$, a congruence of a
PSD matrix. $\square$

The default policy of the lower-level kernel `compute_ewm_covar_tensor` is `NanBackfill.FFILL`,
which holds the entries of a missing asset while the others update. Like the pairwise estimator,
it mixes entries of different ages and can lose PSD; it is safe only on gap-free input.

> **Pitfall.** `estimate_rolling_ewma_covar` returns a matrix for every rebalancing date from the
> first one, and the recursion starts from zero. With the default span of 52 weeks the warm-up
> factor $1-\lambda^{K}$ is 0.37 after one quarter and 0.86 after one year. Pass a `time_period`
> that starts at least two spans after the first price (factor 0.98).

### Uncentred EWM correlations

`compute_ewm_corr_df` runs the recursion on raw returns, without removing any mean, from the
seed `covar0` (zero under both `InitType.ZERO` and `InitType.X0`):

$$
S_t=\lambda S_{t-1}+(1-\lambda)\,x_tx_t^{\top},
\qquad
\rho^{\mathrm{u}}_{ij,t}=\frac{S_{ij,t}}{\sqrt{S_{ii,t}S_{jj,t}}} .
$$

The decay is `ewm_lambda=0.94` unless `span` is given. The result is one column per pair, named
`"<column i> - <column j>"`: `CorrMatrixOutput.FULL` returns all pairs with $j<i$, and `TOP_ROW`
the pairs of the first column with every later one. With the zero seed, the first date has
$\rho^{\mathrm{u}}_{ij}=\operatorname{sign}(x_{i,1}x_{j,1})=\pm1$, so the path needs a warm-up
before it means anything. `compute_ewm_corr_single` is the two-column case: it converts `span`
to $\lambda$ and returns the one series.

**Identity (uncentred versus centred correlation).** For population moments,

$$
\frac{\mathbb{E}[x_ix_j]}{\sqrt{\mathbb{E}[x_i^2]\,\mathbb{E}[x_j^2]}}
=\frac{\rho_{ij}\sigma_i\sigma_j+\mu_i\mu_j}{\sqrt{(\sigma_i^2+\mu_i^2)(\sigma_j^2+\mu_j^2)}} .
$$

**Proof.** $\mathbb{E}[x_ix_j]=\operatorname{Cov}(x_i,x_j)+\mu_i\mu_j$ and
$\mathbb{E}[x_i^2]=\sigma_i^2+\mu_i^2$. $\square$

The difference is of order $(\mu/\sigma)^2$, about 0.004 for weekly returns with $\mu/\sigma=0.065$.

`corr_to_pivot_row` computes, for a complete pivot return $a_t$ and each comparison column
$b_{k,t}$ that may contain NaN, the uncentred cosine similarity over the jointly observed dates:

$$
\operatorname{cs}_k=\frac{\sum_{t\in O_k}a_tb_{k,t}}{\sqrt{\sum_{t\in O_k}a_t^2\sum_{t\in O_k}b_{k,t}^2}} .
$$

With `is_normalized=False` it returns the raw sum $\sum_{O_k}a_tb_{k,t}$; with
`vol_scalers=[(sigma_a, sigma_k), ...]` it divides that sum by $\lvert O_k\rvert$ times the two
supplied volatilities. It is not a Pearson correlation: no mean is removed. `compute_path_corr`,
by contrast, is the centred Pearson correlation of matching columns of two panels over their
full sample.

### Eigen-decomposition and explained variance

**Definition.** A symmetric matrix, here the correlation matrix, has the spectral decomposition

$$
\rho=Q\operatorname{diag}(\nu_1,\ldots,\nu_n)\,Q^{\top},
\qquad \nu_1\ge\cdots\ge\nu_n,
\qquad
\pi_j=\frac{\nu_j}{\sum_{k}\nu_k} .
$$

The principal component $f_{j,t}=v_j^{\top}\Delta^{-1}(x_t-\mu)$ has variance $\nu_j$, and
distinct components are uncorrelated. $\pi_j$ is the explained-variance share and
$\sum_{k\le j}\pi_k$ its cumulative form. For a correlation matrix $\sum_k\nu_k=n$.

**Proposition (variance decomposition).** $\sum_k\nu_k$ equals the trace, the total variance of
the standardised returns, and $v_1$ maximises $w^{\top}\rho w$ over unit vectors, with maximum
$\nu_1$.

**Proof.** The trace is invariant under $Q^{\top}(\cdot)Q$. For $\lVert w\rVert=1$ write $w=Qc$
with $\lVert c\rVert=1$; then $w^{\top}\rho w=\sum_j\nu_jc_j^2\le\nu_1$, with equality at
$c=(1,0,\ldots,0)$. $\square$

The shares lie in $[0,1]$ only for a PSD input. For the pairwise counterexample above they are
$2/3$, $2/3$ and $-1/3$.

The participation ratio $1/\sum_j\pi_j^2$ is the effective number of independent correlation
directions: $n$ for the identity, one for a rank-one matrix. `compute_portfolio_breadth` reports
it point in time as the effective number of independent assets; see
[portfolio breadth](portfolio_breadth.md). Jolliffe (2002) treats PCA in full.

`apply_pca` calls `np.linalg.eigh`, which reads only the lower triangle, and reverses its
ascending output. Its default sign convention, `is_max_sign_positive=True`, flips each eigenvector
so that its largest-magnitude loading is positive; this matches the docstring. The sign of an
eigenvector is arbitrary, and the convention pins it only while the largest loading keeps its
identity: a near tie between two loadings can still flip the sign between refits, and an
eigenvector inside a repeated eigenvalue is not unique at all.

`compute_data_pca_r2` applies `compute_pca_r2` through time to the uncentred EWM correlation
tensor (`is_corr=True`) or second-moment tensor (`is_corr=False`) with decay `ewm_lambda`, zero
seed and forward fill. It samples the dates of `time_period.to_pd_datetime_index(freq)`, whole
data span by default, taking the last row on or before each date, so each row is point in time.

> **Insight.** PCA of a covariance matrix ranks directions by variance and is dominated by the
> most volatile assets: in the worked example the first covariance component carries 82% of the
> total variance, against 50% for the first correlation component.
> `compute_eigen_portfolio_weights` decomposes the correlation matrix and brings volatilities back
> only in the scaling, so it ranks directions by shared co-movement.

### Eigen-portfolios

**Definition.** For $\Sigma=\Delta\rho\Delta$ with $\rho=Q\operatorname{diag}(\nu)Q^{\top}$ and
all $\nu_j>0$, the $j$-th eigen-portfolio is

$$
w^{(j)}=\frac{\Delta^{-1}v_j}{\sqrt{\nu_j}},
\qquad
w^{(j)}_i=\frac{(v_j)_i}{\sigma_i\sqrt{\nu_j}} .
$$

**Proposition (unit variance and orthogonality).** $w^{(j)\top}\Sigma\,w^{(k)}=1$ if $j=k$ and
$0$ otherwise.

**Proof.** Because $\Delta^{-1}\Sigma\Delta^{-1}=\rho$,
$w^{(j)\top}\Sigma\,w^{(k)}=v_j^{\top}\rho\,v_k/\sqrt{\nu_j\nu_k}$. With $\rho v_k=\nu_kv_k$
this is $\nu_k\,v_j^{\top}v_k/\sqrt{\nu_j\nu_k}$, which is one for $j=k$ and zero otherwise
because $Q$ is orthogonal. $\square$

The eigen-portfolio returns $w^{(j)\top}x_t$ are uncorrelated with unit variance in the units of
$\Sigma$: 100% annual volatility for an annualised matrix. The weights do not sum to one and carry
the sign convention of `apply_pca`. `compute_eigen_portfolio_weights` returns them as rows, ranked
by descending correlation eigenvalue, and raises `ValueError` when an asset variance is not
materially positive or a correlation eigenvalue is at or below
$100\,\mathrm{eps}\cdot\max(1,\max_k\lvert\nu_k\rvert)$.

### Eigenvalue clipping and the noise band

`matrix_regularization(covar, cut)` diagonalises a symmetric $M=Q\operatorname{diag}(\nu)Q^{\top}$
with `np.linalg.eigh` and sets every eigenvalue at or below $\kappa$ to zero:

$$
M_{\kappa}=Q\operatorname{diag}\big(\nu_j\,\mathbf{1}\{\nu_j>\kappa\}\big)\,Q^{\top} .
$$

With $\kappa=0$ this is the projection $M_{+}$ onto the PSD cone.

**Proposition (nearest PSD matrix).** For symmetric $M$,
$M_{+}=Q\operatorname{diag}(\max(\nu_j,0))Q^{\top}$ minimises the Frobenius distance
$\lVert M-Z\rVert_F$ over PSD $Z$, and the minimum is $\big(\sum_{\nu_j<0}\nu_j^2\big)^{1/2}$.

**Proof.** Write $M=M_{+}-M_{-}$ with $M_{-}=Q\operatorname{diag}(\max(-\nu_j,0))Q^{\top}$; both
parts are PSD and $\operatorname{tr}(M_{+}M_{-})=0$. For PSD $Z$, expanding
$M-Z=(M_{+}-Z)-M_{-}$ gives

$$
\lVert M-Z\rVert_F^2=\lVert M_{+}-Z\rVert_F^2+2\operatorname{tr}(ZM_{-})+\lVert M_{-}\rVert_F^2
\ge\lVert M_{-}\rVert_F^2 ,
$$

because the trace of a product of two PSD matrices is non-negative. Equality holds at $Z=M_{+}$.
$\square$

Three consequences follow. Clipping raises the diagonal, because $M_{+}=M+M_{-}$ and
$M_{-}$ has a non-negative diagonal, so a clipped correlation matrix must be renormalised with
`covar_to_corr`, which preserves PSD by congruence. The result is singular whenever an eigenvalue
was clipped, so it cannot be inverted for mean-variance weights. And the cut $\kappa$ is absolute:
the default `1e-5` is small for an annualised covariance but can remove genuine directions of a
daily covariance, in which an asset with 5% annual volatility has variance
$0.05^2/252\approx0.99\times10^{-5}$, just below the cut.

Clipping only repairs a matrix; it does not filter noise. The noise band is the random-matrix
reference that Laloux et al. (1999) applied to empirical correlation matrices. For $n$ independent
series with $T$ observations each, as $n,T\to\infty$ with $q=n/T\le1$ fixed, the eigenvalues of
the sample correlation matrix fill the Marchenko–Pastur interval

$$
\big[\nu_{-},\nu_{+}\big],
\qquad
\nu_{\pm}=\big(1\pm\sqrt{q}\big)^2 .
$$

Laloux et al. (1999) found most eigenvalues of stock correlation matrices inside this band, so
only those above $\nu_{+}$ are distinguishable from noise. For $n=10$ weekly series over five
years, $T=260$ and the band is $[0.646,1.431]$. An EWM estimator has no fixed $T$; a heuristic
uses its effective number of observations.

**Identity (effective sample size of EWM weights).** The weights $\lambda^{k}$, $k\ge0$, have
Kish effective size

$$
N_{\mathrm{eff}}=\frac{\big(\sum_{k\ge0}\lambda^k\big)^2}{\sum_{k\ge0}\lambda^{2k}}
=\frac{1+\lambda}{1-\lambda}=N .
$$

**Proof.** The sums are $1/(1-\lambda)$ and $1/(1-\lambda^2)$, whose ratio is
$(1+\lambda)/(1-\lambda)$; with $\lambda=1-2/(N+1)$ this is $N$. $\square$

With $n=10$ and $N=52$ the heuristic band is $[0.315,2.069]$. qis implements neither
random-matrix filtering nor shrinkage. Ledoit and Wolf (2004) shrink the sample covariance towards
a structured target with an estimated optimal intensity, which gives an invertible, better
conditioned matrix; it is an alternative to apply outside qis. Factor-structured covariance is
covered in [factor risk models](factor_risk_models.md).

## Worked example

The first four blocks use hand-checkable matrices. The last four use the frozen synthetic
universe, weekly log returns from 2015 to 2020, and check every qis result against a direct NumPy
recursion.

A covariance with annual volatilities of 20%, 10% and 5% and correlations 0.5, −0.2 and 0.3 has
entries $0.2\cdot0.1\cdot0.5=0.01$, $0.2\cdot0.05\cdot(-0.2)=-0.002$ and
$0.1\cdot0.05\cdot0.3=0.0015$ off the diagonal. `covar_to_corr` recovers the correlations, and a
monthly matrix, one twelfth of the annual one, has the same correlations. A zero-variance asset
gets a missing row and column.

```python
import numpy as np
import pandas as pd
import qis

assets = ['Equity', 'Bonds', 'Gold']
vols = np.array([0.20, 0.10, 0.05])
rho = np.array([[1.0, 0.5, -0.2],
                [0.5, 1.0, 0.3],
                [-0.2, 0.3, 1.0]])
covar = pd.DataFrame(np.outer(vols, vols) * rho, index=assets, columns=assets)
np.testing.assert_allclose(covar.to_numpy(), [[0.04, 0.01, -0.002],
                                              [0.01, 0.01, 0.0015],
                                              [-0.002, 0.0015, 0.0025]], atol=1e-15)

corr = qis.covar_to_corr(covar)
assert abs(corr.loc['Equity', 'Gold'] - (-0.002 / (0.20 * 0.05))) < 1e-12
np.testing.assert_allclose(corr.to_numpy(), rho, atol=1e-12)

an = qis.get_annualization_factor('ME')
monthly = covar / an
np.testing.assert_allclose(qis.covar_to_corr(monthly).to_numpy(), rho, atol=1e-12)
np.testing.assert_allclose(np.sqrt(np.diag(an * monthly)), vols, atol=1e-15)

with_cash = covar.reindex(index=assets + ['Cash'], columns=assets + ['Cash'], fill_value=0.0)
corr_cash = qis.covar_to_corr(with_cash)
assert corr_cash['Cash'].isna().all() and corr_cash.loc['Cash'].isna().all()
```

The correlation eigenvalues are 1.5128, 1.1711 and 0.3161. They sum to the trace, 3, and multiply
to the determinant $1+2(0.5)(-0.2)(0.3)-0.25-0.04-0.09=0.56$. The explained-variance shares are
50.4%, 39.0% and 10.5%, and the participation ratio is 2.394. Each eigenvector's largest loading
is positive. The first component of the covariance matrix explains 82.1%.

```python
nu, vecs = qis.apply_pca(cmatrix=corr.to_numpy())
np.testing.assert_allclose(nu, np.sort(np.linalg.eigvalsh(rho))[::-1], atol=1e-12)
assert abs(nu.sum() - 3.0) < 1e-12 and abs(np.prod(nu) - 0.56) < 1e-12
np.testing.assert_allclose(nu, [1.5128, 1.1711, 0.3161], atol=5e-5)
np.testing.assert_allclose(rho @ vecs, vecs * nu, atol=1e-12)
assert all(v[np.argmax(np.abs(v))] > 0.0 for v in vecs.T)

shares = qis.compute_pca_r2(cmatrix=corr.to_numpy())
np.testing.assert_allclose(shares, nu / nu.sum(), atol=1e-15)
np.testing.assert_allclose(shares, [0.504, 0.390, 0.105], atol=5e-4)
np.testing.assert_allclose(qis.compute_pca_r2(cmatrix=corr.to_numpy(), is_cumulative=True),
                           np.cumsum(shares), atol=1e-15)
assert abs(1.0 / np.sum(shares ** 2) - 2.394) < 5e-4

covar_shares = qis.compute_pca_r2(cmatrix=covar.to_numpy())
assert abs(covar_shares[0] - 0.8206) < 5e-5
```

The eigen-portfolios match $\Delta^{-1}v_j/\sqrt{\nu_j}$ computed directly, and their covariance
is the identity. The first one holds 2.644 in equity, 6.003 in bonds and 2.899 in gold for 100%
annual volatility; a 10% volatility target scales it by 0.1.

```python
weights = qis.compute_eigen_portfolio_weights(covar=covar.to_numpy())
expected = (vecs / vols[:, None] / np.sqrt(nu)).T
np.testing.assert_allclose(weights, expected, atol=1e-12)
np.testing.assert_allclose(weights @ covar.to_numpy() @ weights.T, np.eye(3), atol=1e-12)
np.testing.assert_allclose(weights[0], [2.644, 6.003, 2.899], atol=5e-4)
```

The pairwise counterexample reproduces the proposition: the correlation equals pandas' pairwise
result and has eigenvalues $-1$, 2 and 2, and $w=(1,-1,1)$ has variance $-3$. A two-series panel
shows the mean convention. $X=(0,2,4,6,\cdot)$ and $Y=(\cdot,1,3,2,4)$ overlap on three dates with
overlap means 4 and 2 and own means 3 and 2.5. The masked covariance is
$[(-1)(-1.5)+(1)(0.5)+(3)(-0.5)]/2=0.25$, pandas' overlap covariance is 1, and the identity
bridges them: $1+\tfrac32(4-3)(2-2.5)=0.25$. The pairwise correlation is 0.5. Clipping the
negative eigenvalue adds $v_3v_3^{\top}$ with $v_3=(1,-1,1)/\sqrt3$, which lifts the diagonal to
$4/3$; renormalising gives correlations $\pm0.5$ and eigenvalues 0, 1.5 and 1.5.

```python
nan = np.nan
panel = pd.DataFrame({'A': [1, 2, 3, nan, nan, nan, 1, 2, 3],
                      'B': [1, 2, 3, 1, 2, 3, nan, nan, nan],
                      'C': [nan, nan, nan, 1, 2, 3, 3, 2, 1]}, dtype=float)
corr_pw = qis.compute_masked_covar_corr(data=panel, is_covar=False)
pd.testing.assert_frame_equal(corr_pw, panel.corr())
np.testing.assert_allclose(corr_pw.to_numpy(), [[1, 1, -1], [1, 1, 1], [-1, 1, 1]], atol=1e-12)
w = np.array([1.0, -1.0, 1.0])
assert abs(w @ corr_pw.to_numpy() @ w + 3.0) < 1e-12
np.testing.assert_allclose(np.linalg.eigvalsh(corr_pw.to_numpy()), [-1.0, 2.0, 2.0], atol=1e-12)

cov_pw = qis.compute_masked_covar_corr(data=panel, is_covar=True)
pd.testing.assert_frame_equal(cov_pw, panel.cov())
np.testing.assert_allclose(cov_pw.to_numpy(), [[0.8, 1, -1], [1, 0.8, 1], [-1, 1, 0.8]],
                           atol=1e-12)
assert abs(qis.covar_to_corr(cov_pw).loc['A', 'B'] - 1.25) < 1e-12

pair = pd.DataFrame({'X': [0, 2, 4, 6, nan], 'Y': [nan, 1, 3, 2, 4]}, dtype=float)
cov_xy = qis.compute_masked_covar_corr(data=pair)
np.testing.assert_allclose(np.diag(cov_xy), [20 / 3, 5 / 3], atol=1e-12)
assert abs(cov_xy.loc['X', 'Y'] - 0.25) < 1e-12
assert abs(pair.cov().loc['X', 'Y'] - 1.0) < 1e-12
assert abs(pair.cov().loc['X', 'Y'] + 3 / 2 * (4 - 3) * (2 - 2.5) - cov_xy.loc['X', 'Y']) < 1e-12
assert abs(qis.compute_masked_covar_corr(data=pair, is_covar=False).loc['X', 'Y'] - 0.5) < 1e-12

clipped = qis.matrix_regularization(covar=corr_pw.to_numpy())
np.testing.assert_allclose(clipped, np.array([[4, 2, -2], [2, 4, 2], [-2, 2, 4]]) / 3,
                           atol=1e-12)
repaired = qis.covar_to_corr(clipped)
np.testing.assert_allclose(repaired, [[1, 0.5, -0.5], [0.5, 1, 0.5], [-0.5, 0.5, 1]],
                           atol=1e-12)
np.testing.assert_allclose(np.linalg.eigvalsh(repaired), [0.0, 1.5, 1.5], atol=1e-12)
```

The synthetic panel starts with three instruments whose design volatilities are 17%, 6% and 15%.
The EWM mean of `qis.compute_ewm` equals the direct recursion, and the demeaned residual is
exactly $\lambda$ times the deviation from the previous mean, with $\lambda=51/53$ for a span of
52 weeks. The weekly grid gives $\mathrm{AN}=52$.

```python
from qis.datasets import generate_synthetic_prices

prices = generate_synthetic_prices(start='2015-01-01', end='2020-12-31', seed=20260725,
                                   apply_quirks=False)[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq='W-WED')
x = returns.to_numpy()
span = 52
lam = 1.0 - 2.0 / (span + 1.0)
m = np.empty_like(x)
m[0] = x[0]
for t in range(1, len(x)):
    m[t] = lam * m[t - 1] + (1.0 - lam) * x[t]
np.testing.assert_allclose(qis.compute_ewm(x, span=span), m, atol=1e-15)
np.testing.assert_allclose(x[1:] - m[1:], lam * (x[1:] - m[:-1]), atol=1e-15)
assert qis.infer_annualisation_factor_from_df(returns) == 52.0
```

`estimate_rolling_ewma_covar` returns 23 matrices, from 2015-04-01 to 2020-09-30: each is dated on
the first Wednesday on or after a quarter end, such as 2016-01-06 for the 2015 year end. Each
equals 52 times the direct recursion at its date. On 2020-09-30 the annualised volatilities are
17.6%, 5.5% and 13.7%. On 2015-04-01, after only 12 returns, they are 9.0%, 3.3% and 4.5%: the
warm-up factor $1-\lambda^{12}=0.37$ at work.

```python
covars = qis.estimate_rolling_ewma_covar(prices=prices, returns_freq='W-WED',
                                         rebalancing_freq='QE', span=span)
dates = list(covars)
assert len(dates) == 23
assert [d.strftime('%Y-%m-%d') for d in dates[:4]] == ['2015-04-01', '2015-07-01',
                                                        '2015-09-30', '2016-01-06']
state = np.zeros((3, 3))
path = []
for e_t in x - m:
    state = lam * state + (1.0 - lam) * np.outer(e_t, e_t)
    path.append(state)
for date in (dates[0], dates[-1]):
    np.testing.assert_allclose(covars[date].to_numpy(),
                               52.0 * path[returns.index.get_loc(date)], rtol=1e-12)
np.testing.assert_allclose(np.sqrt(np.diag(covars[dates[-1]])), [0.176, 0.055, 0.137], atol=5e-4)
assert returns.index.get_loc(dates[0]) == 11
assert abs(1.0 - lam ** 12 - 0.37) < 1e-3
np.testing.assert_allclose(np.sqrt(np.diag(covars[dates[0]])), [0.090, 0.033, 0.045], atol=5e-4)
```

The uncentred EWM correlation of `compute_ewm_corr_df` equals the direct recursion on raw returns:
−0.218 between the Treasury and US equity series at the last date. Its first row is $\pm1$. Over
the full sample, the uncentred cosine similarities of `corr_to_pivot_row` are −0.227 and 0.098,
against Pearson correlations of −0.238 and 0.096.

```python
corr_path = qis.compute_ewm_corr_df(df=returns, span=span)
assert corr_path.columns.tolist() == ['SBD_TSY - SEQ_US', 'SCM_GLD - SEQ_US',
                                      'SCM_GLD - SBD_TSY']
state = np.zeros((3, 3))
for x_t in x:
    state = lam * state + (1.0 - lam) * np.outer(x_t, x_t)
uncentred = state[1, 0] / np.sqrt(state[0, 0] * state[1, 1])
assert abs(corr_path['SBD_TSY - SEQ_US'].iloc[-1] - uncentred) < 1e-12
assert abs(uncentred + 0.218) < 5e-4
np.testing.assert_allclose(np.abs(corr_path.iloc[0]), 1.0)

cosine = qis.corr_to_pivot_row(pivot=x[:, 0], data=x[:, 1:])
direct = x[:, 0] @ x[:, 1:] / np.sqrt((x[:, 0] @ x[:, 0]) * np.sum(x[:, 1:] ** 2, axis=0))
np.testing.assert_allclose(cosine, direct, atol=1e-14)
np.testing.assert_allclose(cosine, [-0.227, 0.098], atol=5e-4)
np.testing.assert_allclose(np.corrcoef(x.T)[0, 1:], [-0.238, 0.096], atol=5e-4)
```

On all ten instruments, `compute_data_pca_r2` with the span-52 decay gives explained-variance
shares at 2020-09-30 of 42.7%, 18.6% and 12.5% for the first three components, equal to the
eigenvalues of the directly computed uncentred correlation. The eigenvalues are 4.27, 1.86, 1.25,
and so on down to 0.14; the participation ratio is 4.06. Against the heuristic band
$[0.315,2.069]$ for $n=10$ and $N=52$, only the first eigenvalue lies above $\nu_{+}$ and four lie
below $\nu_{-}$. The band assumes unit variance in every noise direction, while the first
component already takes 4.27 of the total of 10. A common refinement scales the band by the
remaining share, $1-\nu_1/n=0.573$, to $[0.181,1.185]$, and then three eigenvalues lie above it.
The count of signal directions depends on that choice, which is why the band is a guide and not
a test.

```python
all_prices = generate_synthetic_prices(start='2015-01-01', end='2020-12-31', seed=20260725,
                                       apply_quirks=False)
all_returns = qis.to_returns(prices=all_prices, is_log_returns=True, drop_first=True,
                             freq='W-WED')
pca_r2 = qis.compute_data_pca_r2(data=all_returns, freq='QE', ewm_lambda=lam)
state = np.zeros((10, 10))
for x_t in all_returns.loc[:'2020-09-30'].to_numpy():
    state = lam * state + (1.0 - lam) * np.outer(x_t, x_t)
direct_corr = state / np.sqrt(np.outer(np.diag(state), np.diag(state)))
nu_all = np.sort(np.linalg.eigvalsh(direct_corr))[::-1]
np.testing.assert_allclose(pca_r2.loc['2020-09-30'].to_numpy(), nu_all / nu_all.sum(), atol=1e-12)
np.testing.assert_allclose(nu_all[:3], [4.27, 1.86, 1.25], atol=5e-3)
assert abs(1.0 / np.sum((nu_all / nu_all.sum()) ** 2) - 4.06) < 5e-3

q = 10 / span
edges = ((1.0 - np.sqrt(q)) ** 2, (1.0 + np.sqrt(q)) ** 2)
np.testing.assert_allclose(edges, [0.315, 2.069], atol=5e-4)
assert np.sum(nu_all > edges[1]) == 1 and np.sum(nu_all < edges[0]) == 4
residual_share = 1.0 - nu_all[0] / 10
scaled_edges = (residual_share * edges[0], residual_share * edges[1])
np.testing.assert_allclose(scaled_edges, [0.181, 1.185], atol=5e-4)
assert abs(residual_share - 0.573) < 5e-4 and np.sum(nu_all > scaled_edges[1]) == 3
```

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| Correlation from covariance | $\Delta^{-1}\Sigma\Delta^{-1}$ with the normalisation kernel | `qis.covar_to_corr(covar)` |
| Common-sample or pairwise covariance | $\hat\Sigma$, or $\hat\Sigma^{\mathrm{mask}}$ with NaN | `qis.compute_masked_covar_corr(data, is_covar=True, bias=False)` |
| Common-sample or pairwise correlation | $\rho$, or $\hat\rho^{\mathrm{pair}}$ with NaN | `qis.compute_masked_covar_corr(data, is_covar=False)` |
| Pearson correlation of matching columns | Centred, full sample | `qis.compute_path_corr(a1, a2)` |
| Rolling EWM covariance | $\mathrm{AN}\,\hat\Sigma_t$ on $\mathcal{T}_{\mathrm{reb}}$ | `qis.estimate_rolling_ewma_covar(prices, time_period, returns_freq, rebalancing_freq, span, is_apply_vol_normalised_returns, demean, apply_an_factor)` |
| Uncentred EWM correlation paths | $\rho^{\mathrm{u}}_{ij,t}$ per pair | `qis.compute_ewm_corr_df(df, corr_matrix_output, span, ewm_lambda, init_value, init_type)`, `qis.CorrMatrixOutput` |
| One uncentred EWM correlation path | $\rho^{\mathrm{u}}_{21,t}$ | `qis.compute_ewm_corr_single(returns, ewm_lambda, span, time_period)` |
| Uncentred cosine similarity to a pivot | $\operatorname{cs}_k$ | `qis.corr_to_pivot_row(pivot, data, is_normalized=True, vol_scalers=None)` |
| Eigenvalues and eigenvectors | $\nu_j$, $v_j$, descending, largest loading positive | `qis.apply_pca(cmatrix, is_max_sign_positive=True)` |
| Explained-variance shares | $\pi_j$ or $\sum_{k\le j}\pi_k$ | `qis.compute_pca_r2(cmatrix, is_cumulative=False)` |
| Shares through time | $\pi_j$ of $\rho^{\mathrm{u}}_t$ on `freq` dates | `qis.compute_data_pca_r2(data, freq='ME', time_period=None, ewm_lambda=0.94, is_corr=True)` |
| Eigen-portfolios | $w^{(j)}=\Delta^{-1}v_j/\sqrt{\nu_j}$, one per row | `qis.compute_eigen_portfolio_weights(covar)` |
| Eigenvalue clipping | $M_{\kappa}$ | `qis.matrix_regularization(covar, cut=1e-5)` |
| Correlation heatmap | Lower triangle $\rho_{ij}$, diagonal $\sigma_i$ | `qis.plot_corr_matrix_from_covar(covar)` |
| EWM kernels | $\hat\Sigma_t$ or $S_t$, final or full path | `qis.compute_ewm_covar`, `qis.compute_ewm_covar_tensor`, `qis.compute_ewm_covar_tensor_vol_norm_returns` |

The estimators are in
[corr_cov_matrix.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/corr_cov_matrix.py),
the spectral functions in
[pca.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/pca.py),
the recursions in
[ewm.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py),
the heatmap in
[plot_correlations.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/plot_correlations.py),
and `covar_to_corr` with the internal normalisation kernel `_covar_to_corr_array` in
[np_ops.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/np_ops.py).

Contract details that the formulas do not show:

- `compute_masked_covar_corr` returns the input's container. On a panel with NaN, a pair with no
  common date gets covariance 0, because the masked-array mask is dropped, but correlation NaN.
  `bias` affects only the covariance.
- `estimate_rolling_ewma_covar` labels each matrix with `prices.columns`. Only the start of
  `time_period` is applied: matrices are returned for every rebalancing date on or after it,
  including dates after its end.
- `CorrMatrixOutput.SUB_TOP` currently returns the same pairs as `FULL`.
- The `eigen_signs` argument of `apply_pca` is documented as one sign per eigenvector, but the
  implementation compares and flips rows of the eigenvector matrix, which are asset coordinates,
  and so breaks the eigenvector property whenever it flips. Use the default convention.
- `plot_corr_matrix_from_covar` shows $\sqrt{\Sigma_{ii}}$ in the units of its input; pass an
  annualised covariance to display annual volatilities.

API reference:
{doc}`covar_to_corr <api/generated/qis.covar_to_corr>`,
{doc}`compute_masked_covar_corr <api/generated/qis.compute_masked_covar_corr>`,
{doc}`estimate_rolling_ewma_covar <api/generated/qis.estimate_rolling_ewma_covar>`,
{doc}`compute_ewm_corr_df <api/generated/qis.compute_ewm_corr_df>`,
{doc}`apply_pca <api/generated/qis.apply_pca>`,
{doc}`compute_pca_r2 <api/generated/qis.compute_pca_r2>`,
{doc}`compute_eigen_portfolio_weights <api/generated/qis.compute_eigen_portfolio_weights>`,
{doc}`matrix_regularization <api/generated/qis.matrix_regularization>`.

## Interpretation and limitations

- Full-sample and pairwise matrices use every date of the sample. They describe a history and
  must not feed a backtest; the EWM estimators are the point-in-time path.
- A pairwise-complete matrix is not guaranteed PSD, and its covariance and correlation outputs
  are mutually inconsistent on ragged data. Check the spectrum before optimising.
- The EWM covariance starts from zero and, when demeaned, is scaled by
  $2\lambda^2/(1+\lambda)$. Both factors cancel in correlations of assets with a common start; a
  late starter's correlations are biased towards zero until $\lambda^{K}$ is small.
- On return panels with gaps, `compute_ewm_corr_df` and `compute_data_pca_r2` hold stale entries
  under the default forward fill and can produce correlations outside $[-1,1]$. Fill or align the
  panel first; `qis.to_returns` forward-fills prices by default.
- `estimate_rolling_ewma_covar` estimates on log returns. Portfolio variance $w^{\top}\Sigma w$
  with capital weights is exact for a covariance of simple returns; with log returns it is an
  approximation whose error is of higher order in the per-period volatility.
- Stale, smoothed or asynchronous prices depress measured volatilities and correlations; see
  [incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md) and
  [private-asset unsmoothing](private_asset_unsmoothing.md).
- The Marchenko–Pastur band is an asymptotic result for independent, identically distributed
  series. Fat tails, volatility clustering and EWM weighting change it, so treat the band as a
  guide, not a test.
- `matrix_regularization` makes a matrix PSD, not invertible and not less noisy; it also changes
  its diagonal.

## See also

- [Notation and conventions](notation_and_conventions.md)
- [Exponentially weighted estimators](ewm_estimators.md)
- [Portfolio breadth and allocation efficiency](portfolio_breadth.md)
- [Portfolio risk and Euler contributions](risk_contributions.md)
- [Factor risk models](factor_risk_models.md)
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Serial dependence and autocorrelation](serial_dependence.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Bibliography](bibliography.md)

## References

1. Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis*, 3rd edition. Wiley. The sample covariance and correlation matrices and their distribution theory.
2. Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd edition. Springer. The eigen-decomposition, explained-variance shares and the choice between covariance and correlation PCA.
3. Laloux, L., Cizeau, P., Bouchaud, J.-P., and Potters, M. (1999). Noise Dressing of Financial Correlation Matrices. *Physical Review Letters*, 83(7), 1467–1470. [DOI: 10.1103/PhysRevLett.83.1467](https://doi.org/10.1103/PhysRevLett.83.1467). The random-matrix noise band applied to empirical correlation matrices.
4. Ledoit, O., and Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *The Journal of Portfolio Management*, 30(4), 110–119. [DOI: 10.3905/jpm.2004.110](https://doi.org/10.3905/jpm.2004.110). Shrinkage as an alternative that qis does not implement.
5. J.P. Morgan and Reuters (1996). *RiskMetrics — Technical Document*, 4th edition. J.P. Morgan. The exponentially weighted covariance and the decay 0.94 used as a default.
6. Kish, L. (1965). *Survey Sampling*. Wiley. The effective sample size of a weighted sample.
7. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
