---
myst:
  html_meta:
    description: >-
      Reproducible option-portfolio stress testing with five Yahoo stocks, ten short
      VOP-priced options, four ETF factors and an EWMA risk model in qis.
---

# Stress testing with options

*[author / affiliation / date — placeholder]*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Option-portfolio stress testing revalues each contract after its underlying moves, rather
than extending today's delta across a large shock. This guide combines observed stock/ETF
history with a synthetic short-option book to illustrate negative convexity: losses accelerate
on the downside while short calls limit participation in an equity rally.

## Overview

The example owns 100-share lots of **AAPL, MSFT, AMZN, GOOGL and NVDA**, and sells one call
and one put position on every stock: **five stocks, five short calls and five short puts**.
Calls cover the held shares; put quantities are about 1.5 times the stock lots. Puts are
additional downside obligations, not protective hedges.

An EWMA model estimates the five stock responses to **SPY, TLT, GLD and USO**. Every stock and
its two options share one fitted underlying response. The model therefore has five responses
and four factors, even though the portfolio has fifteen holdings. The ETFs are risk proxies;
they are not additional positions in the portfolio.

The ordinary `InstrumentLeg` call/put primitives are intrinsic payoffs. To retain time value,
this example supplies a public `HoldingPayoff` implementation priced by
[VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers) (VOP). QIS owns scenario
completion, portfolio aggregation, risk and reporting. VOP is an **example-only dependency**;
importing qis does not import or require an option-pricing library.

The [runnable source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/stress_testing_with_options.py)
and the generated `short_convexity.pdf` show full repricing beside the current-delta approximation.
This is an illustrative book, not a reconstruction of a client portfolio or its contract terms.

## Inputs, notation, and assumptions

| Input or symbol | Meaning | Convention |
|---|---|---|
| $t_0$ | Frozen valuation date | Default 2025-12-31; actual latest complete close at/before the requested cutoff |
| $S_i$ | Current stock quote | Yahoo `Close`, USD per share, separate from adjusted history |
| $y_{it}$ | Stock return for estimation | Weekly W-WED log return of Yahoo `Adj Close` |
| $x_t$ | Four ETF factor returns | Weekly W-WED adjusted log returns, ordered SPY/TLT/GLD/USO |
| $B$ | Stock-by-factor loading matrix | Five rows, four columns; joint EWMA regression |
| $\Sigma_F$, $d_i$ | Factor covariance and stock residual variances | Annualised with 52 weekly observations/year |
| $z$ | Scenario factor vector | Log returns; supplied simple bumps are converted by `log1p` |
| $n_l$, $m$ | Signed option contracts and contract multiplier | Negative contracts for shorts; 100 shares/contract |
| $K_l$, $\tau_l$ | Strike and remaining maturity | USD/share and ACT/365 years at $t_0$ |
| $r$, $q$ | Pricing rate and dividend yield | Assumed continuous 4% and 0%, held fixed in stresses |
| $\sigma_l$ | Pricing volatility | Assumed annual lognormal IV, not a downloaded option quote |
| $N$ | Reporting denominator | Current marked stock value plus signed option marks; positive USD amount |
| $T$ | Conditional-band horizon | One month, $1/12$ year; it does not advance option maturity |

### Data and cache policy

The default download covers 2015-01-01 through 2025-12-31. The code explicitly requests
`auto_adjust=False`, retains both `Close` and `Adj Close`, and passes the next calendar day
as the exclusive end date. These arguments avoid relying on changing provider defaults.
See the [yfinance download contract](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

Only rows with all nine tickers and both price fields are used; missing observations are
not filled with zero returns. The example rejects nonpositive or stale terminal prices and
requires at least three 52-week spans for the fit. Factor history is cut at the valuation
date; complete monthly vectors are used for historical scenario replay.

The first run saves a CSV and request/download metadata with a SHA-256 digest. Subsequent
runs verify and reuse those bytes without another price download. `--refresh` explicitly
requests a new download. Historical Yahoo values can be revised: a fixed cutoff alone does
not guarantee byte-identical future downloads. Keep the cache and provenance to reproduce
the numerical exhibit. No prices, API credentials or operational portfolios are committed.

### Teaching contract construction

For each stock, the example buys the largest number of 100-share lots within USD 2m.
It sells the same number of call contracts, and 1.5 times as many put contracts rounded
down. Strikes are rounded to USD 5 increments. The per-stock call targets are 103%, 105%,
107%, 104% and 108% of spot; put targets are 97%, 95%, 93%, 96% and 92%.

Maturities are the third Fridays two, three, four, five and eight months after the valuation
month. These are **synthetic teaching identifiers**, formatted like
`AAPL US 02/20/26 C280 Equity`; they are not assertions that a matching exchange contract
is listed or tradable. An override of `--as-of` rebuilds strikes and maturities consistently.
There is no exchange-holiday adjustment in the teaching calendar.

A 63-trading-day EWMA realised volatility is annualised with 252 observations/year.
The assumed call IV is the larger of 15% and 1.15 times that realised volatility. The put
IV adds four volatility percentage points. These transparent assumptions introduce a modest
premium and downside skew; neither is calibrated to a historical option surface.

## Methodology

### 1. Estimate the joint EWMA risk model

Write $u_t=(x_t^\top,y_t^\top)^\top$. With span $s=52$, zero initial moments and no mean
subtraction, the EWMA moment recursion is

$$
\lambda=1-\frac{2}{s+1},\qquad
M_t=\lambda M_{t-1}+(1-\lambda)u_tu_t^\top,\qquad M_0=0.
$$

The latest joint regression and annual factor covariance are

$$
B=M_{yx}M_{xx}^{-1},\qquad \Sigma_F=52M_{xx}.
$$

The model uses the complete factor covariance, not four independent univariate regressions.
It fits no intercept: moments are about zero, not de-meaned covariances. This is the explicit
`MeanAdjType.NONE` convention, and no drift or estimated alpha is added to a stress.
The latest $B$ is applied to the estimation sample to calculate residuals
$\epsilon_t=y_t-Bx_t$ and their EWMA second moments:

$$
d_i=52(1-\lambda)\sum_{j=0}^{n-1}\lambda^j\epsilon_{i,n-j}^2,
\qquad
\Sigma_Y=B\Sigma_F B^\top+\operatorname{diag}(d).
$$

`estimate_ewm_factor_model` supplies the loadings; `compute_ewm_covar` supplies factor and
residual moments; the complete snapshot is assigned to `RiskModel`. Residual covariance
between different stocks is discarded. Stocks and options on the **same** stock retain
shared residual risk through their aggregated dollar delta.

The supplied fit diagnostic is the non-centred EWMA $R^2$:

$$
R_i^2=1-\frac{d_i}{52M_{yy,ii}}.
$$

It describes the weighted historical fit of the final loadings, not out-of-sample accuracy.

### 2. Complete correlated factor shocks

For a simple SPY bump $a$, set $z_{\mathrm{SPY}}=\log(1+a)$. In independent scenarios all
other factor shocks are zero. For jointly conditioned scenarios with anchored factors $A$
and remaining factors $F$, use

$$
z_F=\Sigma_{FA}\Sigma_{AA}^{-1}z_A.
$$

An explicit zero is an anchor; an omitted factor is free. QIS performs one joint solve,
not a sum of separately conditioned scenarios. The conditional Gaussian interpretation is
described in [Geyer's partitioned-normal derivation](https://www.stat.umn.edu/geyer/s19/5101/slides/s5.pdf).
The [factor stress-testing guide](stress_testing.md) documents validation and shock conventions.

Stock $i$ then moves to

$$
S_i(z)=S_i(0)\exp(B_i z).
$$

Historical monthly scenarios already contain all four realised factor returns. Replay them
directly against today's holdings and option terms; do not condition them again or reinterpret
the result as realised portfolio performance.

### 3. Price the options with VOP

For each contract define the forward $F=S\exp((r-q)\tau)$ and discount factor
$D=\exp(-r\tau)$. The European Black-Scholes-Merton formulas are

$$
d_1=\frac{\log(F/K)+\tfrac12\sigma^2\tau}{\sigma\sqrt{\tau}},
\qquad d_2=d_1-\sigma\sqrt{\tau},
$$

$$
C=D\left[F\Phi(d_1)-K\Phi(d_2)\right],\qquad
P=D\left[K\Phi(-d_2)-F\Phi(-d_1)\right].
$$

Here $\Phi$ is the standard normal distribution function. VOP's compiled forward-grid
pricer evaluates all shocked spots for one option. Values are per share; signed holding
marks multiply by **contracts times 100**, once. See
[Black and Scholes (1973)](https://www.journals.uchicago.edu/doi/10.1086/260062) and the
[VOP implementation](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/src/vanilla_option_pricers/black_scholes.py).

The option holding change and total portfolio return are

$$
\Delta V_l(z)=n_lm\left[v_l(S_i(z))-v_l(S_i(0))\right],\qquad
R(z)=\frac{\sum_l\Delta V_l(z)}{N}.
$$

Stocks use their signed share counts and spot changes. Option marks are negative for short
positions. The denominator is the sum of the fifteen signed current marks, not gross stock
exposure, option notional or a sum of absolute marks. Premium proceeds are not added a second
time as a separate asset. There is no additional cash/debt position in this teaching book.

Current source marks are set equal to VOP model marks, so the QIS mark-to-model basis offset
is zero. Zero shock gives zero P&L and preserves all fifteen source marks.

### 4. Convert Greeks correctly and measure convexity

VOP returns **discounted forward delta** but **undiscounted forward gamma**. If
$c=\exp((r-q)\tau)$, their spot equivalents are

$$
\Delta_S=c\Delta_F^{\mathrm{VOP}},\qquad
\Gamma_S=Dc^2\Gamma_F^{\mathrm{VOP}}.
$$

Multiplying gamma by the discount factor twice, omitting the forward-to-spot chain rule,
or applying the contract multiplier twice would misstate risk while still producing
plausible-looking prices.

For each holding, the shared-response dollar derivative is

$$
J_{li}=n_lmS_i\Delta_{S,l}.
$$

For a stock holding it is shares times spot. Aggregate $J_i=\sum_lJ_{li}$ **before** computing
risk. In a small underlying simple-return move $h$, the option contribution is approximately

$$
\Delta V_l\approx n_lm\left[\Delta_{S,l}S_i h+\frac{1}{2}\Gamma_{S,l}S_i^2h^2\right].
$$

Long vanilla gamma is positive; the signed gamma of every sold option is negative. The
overlay's quadratic term therefore detracts for both signs of $h$. The separate option-sleeve
panels isolate this effect. The difference between the full portfolio and its frozen
**log-delta** approximation also includes the stock model's exponential return mapping.

### 5. Compute conditional risk bands with stressed deltas

The conditional covariance, padded with zero anchor rows and columns, is

$$
\Sigma_{F\mid A}=\Sigma_{FF}-\Sigma_{FA}\Sigma_{AA}^{-1}\Sigma_{AF}.
$$

At each scenario re-evaluate the option deltas and form
$e(z)=B^\top J(z)/N$. For horizon $T=1/12$ year, QIS computes

$$
v(z)=T\left[e(z)^\top\Sigma_{\mid A}e(z)
+\sum_i\left(\frac{J_i(z)}{N}\right)^2d_i\right].
$$

The standard report displays $R(z)\pm\sqrt{v(z)}$ and $R(z)\pm2\sqrt{v(z)}$.
The exported 95% pointwise bounds use approximately 1.96 standard deviations. These are
scenario-local, linearised Gaussian bands, **not** exact nonlinear P&L quantiles. They do not
include gamma/vega dispersion, volatility-surface shifts, jumps or uncertainty in the fit.
The horizon scales covariance; option TTM and the central valuation date stay fixed.

## Worked example

The frozen 2025-12-31 sample gives net marked value **USD 8,726,094**. The following option
lines illustrate the identifiers and signed quantities; the complete ten-row contract and
pricing table is exported as `option_inventory.csv` and included in the report appendix.

| Synthetic Bloomberg-style identifier | Contracts | IV | VOP value per share |
|---|---:|---:|---:|
| AAPL US 02/20/26 C280 Equity | -73 | 19.38% | USD 5.09 |
| AAPL US 02/20/26 P265 Equity | -109 | 23.38% | USD 5.75 |
| MSFT US 03/20/26 C510 Equity | -41 | 20.62% | USD 9.92 |
| MSFT US 03/20/26 P460 Equity | -61 | 24.62% | USD 10.42 |
| AMZN US 04/17/26 C245 Equity | -86 | 31.93% | USD 11.24 |
| AMZN US 04/17/26 P215 Equity | -129 | 35.93% | USD 9.53 |
| GOOGL US 05/15/26 C325 Equity | -63 | 32.73% | USD 21.65 |
| GOOGL US 05/15/26 P300 Equity | -94 | 36.73% | USD 19.23 |
| NVDA US 08/21/26 C200 Equity | -107 | 39.18% | USD 19.73 |
| NVDA US 08/21/26 P170 Equity | -160 | 43.18% | USD 15.11 |

The main exhibit uses **correlated SPY shocks**. Values below are percentage points of
current net marked value. The other ETF factors follow the latest EWMA conditional co-moves.

| SPY move | Full option repricing | Current log-delta approximation | Stocks | Short calls | Short puts |
|---|---:|---:|---:|---:|---:|
| -30% | -84.15% | -58.05% | -43.69% | +5.96% | -46.42% |
| -20% | -51.54% | -36.31% | -29.98% | +5.75% | -27.31% |
| -10% | -21.70% | -17.15% | -15.41% | +4.48% | -10.77% |
| 0% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| +10% | +11.27% | +15.51% | +16.26% | -9.93% | +4.95% |
| +20% | +15.73% | +29.67% | +33.36% | -24.41% | +6.78% |
| +30% | +17.26% | +42.70% | +51.30% | -41.45% | +7.41% |

At SPY -20%, short puts add a loss of 27.31 percentage points. Gains on sold calls offset
only 5.75 points. At SPY +20%, the calls absorb most of the stock gain. The exact portfolio
curve is therefore strongly asymmetric and bends below the current-delta approximation.
Rounded components may differ from the rounded total by 0.01 percentage point.

These results were computed with VOP 2.2.0 and the QIS checkout on 2026-09-14, using the
Yahoo cache SHA-256
`1e366949dd117c5ef2a62e810811598ada619f33fbf1e5dbbb4add7154e0f98b`.
Generation date and data cutoff are distinct. The run records the actual QIS source hashes
and installed distribution metadata separately, because editable-install metadata can lag
checkout code. Recompute tables when changing the cache, sizing or option assumptions.

## Implementation in qis

Use an environment with the instrument stress-report APIs (`qis>=5.30.1`), the `data` extra,
and the separate VOP package. From a QIS checkout:

~~~console
python -m pip install -e ".[data]" "vanilla-option-pricers==2.2.0"
python -m examples.portfolios.stress_testing_with_options --cache-dir /absolute/cache/options_stress --output-dir /absolute/output/options_stress
~~~

On the maintainer's Windows host, use the existing external environment and C-local paths:

~~~powershell
. '..\ArturSepp\scripts\repo_governance\Enter-AgentRepo.ps1'
& 'C:\Python\QuantInvestStrats312\Scripts\python.exe' -m examples.portfolios.stress_testing_with_options --cache-dir "$env:AGENT_LOCAL_ROOT\data\options_stress_20251231" --output-dir "$env:AGENT_LOCAL_ROOT\runs\options_stress_new"
~~~

Run the same command with a **fresh output directory** to replay the cached input. Omit
`--output-dir` to calculate and print the key results without saving a report. The default
cache is under the user's local application/cache directory, never relative to the checkout.
`--as-of YYYY-MM-DD` selects another frozen date; `--refresh` requests another Yahoo download.
The example raises an error if data acquisition fails; it does not silently substitute
synthetic prices.

| Step | Public API or example function | Output |
|---|---|---|
| Download and freeze | `load_prices` / yfinance | Raw and adjusted daily closes, metadata and hash |
| Estimate | `estimate_ewm_factor_model`, `compute_ewm_covar`, `RiskModel` | Five-stock/four-factor EWMA snapshot and fit diagnostics |
| Price options | `BsmOptionPayoff`, VOP forward-grid prices and Greeks | Signed BSM values and baseline/stressed response Jacobians |
| Build positions | `Underlying`, `PortfolioHolding`, `InstrumentPortfolio` | Fifteen holdings sharing five underlying responses |
| Stress | `StressScenarios`, `run_portfolio_stress_test` | Independent/correlated requests, historical ranking and four grids |
| Report | `generate_portfolio_stress_report`, `StressReportConfig` | Standard PDF, numerical workbook, previews and hashed tables |
| Explain convexity | `plot_line`, `plot_df_table` | Four-panel teaching exhibit and full-precision CSV curves |

The output directory contains:

- `short_convexity.pdf` and `.png`: full repricing versus log delta, stock/call/put contributions,
  short-option contributions by stock, and current signed option Greeks.
- `spy_convexity.csv`, `spy_option_contributions.csv`, `option_inventory.csv` and
  `fit_diagnostics.csv`: full-precision inputs to those exhibits.
- `source_prices.csv` and `example_provenance.json`: observed sample, version/source identity,
  option-example hash, reporting denominator and numerical verification.
- `report/portfolio_stress_report.pdf`, the QIS workbook, individual page previews and
  `report/manifest.json`: the complete standard report and artifact hashes.

The report includes the ten worst complete historical months, conditional covariance pages,
scenario-local bands, and the option-terms appendix. EWMA does not supply fitted clustering
topology, so the standard cluster pages explicitly report it as unavailable. No clusters are
invented solely to fill those pages. TLT, GLD and USO requests are **ETF-return shocks**;
a TLT move does not automatically change the option discount rate.

### Verification

Every run checks the final betas against independent exponentially weighted least squares,
residual variances against explicit weighted residual sums, unit delta/gamma against price
finite differences, and put-call parity at matched terms. It also checks fifteen holdings,
five calls/five puts, negative signed option gamma, zero-shock marks, full factor-delta finite
differences and P&L attribution. The example harness checks parsing and public API usage;
this yfinance example is intentionally excluded from unattended offline execution.

Ordinary source link:
[examples/portfolios/stress_testing_with_options.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/stress_testing_with_options.py).

~~~{literalinclude} ../examples/portfolios/stress_testing_with_options.py
:language: python
~~~

## Interpretation and limitations

- The option positions, maturities, IVs, flat rates and zero dividend yields are teaching
  assumptions. The downloaded series are stock/ETF history, not historical option quotes.
- US equity options are represented by European BSM marks. American exercise, discrete
  dividends, settlement, transaction costs, margin and liquidity are not modelled.
- Contract terms and IV remain fixed across instantaneous shocks. Large downside scenarios
  can be more severe if implied volatility rises; this four-factor model does not estimate
  or condition a volatility-surface factor.
- There are no knockout, autocall, accumulator or path-dependent states in this vanilla
  example. Such contracts need their own `HoldingPayoff` implementation and state assumptions.
- Current EWMA betas are extrapolated across large shocks. Idiosyncratic covariance between
  different stocks is omitted; historical-scenario ranking is a stress diagnostic, not a
  probability forecast or backtest of the short-option strategy.
- Baseline volatility and scenario-local bands remain first-order risk measures. Neither
  negates the negative convexity revealed by full repricing. The Gaussian bands can cross
  -100%; they are not margin limits or exact loss quantiles.

## See also

- [Factor stress testing](stress_testing.md): log-shock conversion and conditional covariance.
- [Instrument portfolios and stress reports](portfolio_stress.md): the public custom-payoff contract.
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md): `RiskModel` conventions.
- [Composite-payoff example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/composite_payoff_stress.py): a separate terminal-knockout teaching wrapper.

## References

1. Black, F., and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
   *Journal of Political Economy*, 81(3), 637-654.
   [DOI: 10.1086/260062](https://www.journals.uchicago.edu/doi/10.1086/260062).
2. Geyer, C. J. (2019). *Stat 5101 Lecture Slides: Deck 5*, University of Minnesota,
   slides 136-140. [Conditional multivariate normal distributions](https://www.stat.umn.edu/geyer/s19/5101/slides/s5.pdf).
3. Sepp, A. [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers),
   software and forward-price/Greek implementation. Example verified with version 2.2.0.
4. Aroussi, R., and contributors. [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
5. Sepp, A., and qis contributors. [qis software and citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
