---
myst:
  html_meta:
    description: >-
      Brinson allocation, selection and interaction in qis: prior realised weights,
      arithmetic contributions, Frongello linking, costs and NAV reconciliation.
---

# Brinson attribution: contributions, sector returns and linking

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-13](https://github.com/ArturSepp/QuantInvestStrats/commit/29fb6ce856e4480d9643508842a7ef5beff56cf9)*

Brinson attribution decomposes a portfolio's return relative to a benchmark into
allocation, selection and interaction effects. This article describes the BHB
convention and multi-period linking implemented in
[qis](https://github.com/ArturSepp/QuantInvestStrats).
Software reference: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Overview

This file, `docs/brinson_attribution.md`, is the authoritative methodology and
calculation contract for QIS Brinson attribution. Update it when changing the
calculation or its conventions. The numerical implementation is
[`src/qis/portfolio/attribution/brinson.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/brinson.py);
the packaged note points here and does not maintain a second methodology.

QIS decomposes the difference between a strategy and its benchmark into asset
allocation, instrument selection and interaction. The canonical entry point is
`qis.compute_brinson_attribution_table`; `MultiPortfolioData` prepares stored
portfolio histories for the same function. The reporting module re-exports this
function and contains only presentation code.

## Inputs, notation, and assumptions

| Symbol | Meaning and units |
|---|---|
| $p$, $b$ | Strategy portfolio and benchmark, respectively. |
| $i$, $g$, $t$ | Instrument, classification group and native return date. |
| $r_{i,t}$ | Instrument simple return over the period ending at $t$, in decimal units. |
| $w_{i,t-1}$ | Realised portfolio weight immediately before that return. |
| $c_{i,t}$ | Instrument arithmetic contribution to portfolio return, in decimal units. |
| $A_{g,t}$, $S_{g,t}$, $I_{g,t}$ | Group allocation, selection and interaction effects. |
| $P_t$, $B_t$ | Strategy and benchmark wealth starting from one at the report baseline. |
| $a_{g,t}$, $F_{g,t}$ | A raw effect and its cumulative linked value; superscript $e$ identifies allocation, selection or interaction. |

Effects are return contributions, with no annualisation. Multiplying a decimal effect
by 100 expresses it in percentage points. Use the same currency, valuation dates and
group classification for both portfolios.

### Input convention and timing

Use **arithmetic return contributions**, not log returns, unweighted instrument
returns or currency P&L. For instrument i and return date t:

$$
c_{i,t}=w_{i,t-1}r_{i,t}.
$$

The weight is the realised weight held before the return, including drift from
earlier dates. It is not the target or end-of-period weight. If starting from
currency P&L, divide by the preceding portfolio NAV. All four low-level input
frames need identical, ordered, unique return dates. Instruments may differ
between portfolios; their union is retained using canonical identifiers.

For group g, sum its instrument contributions and weights:

$$
c_{p,g,t}=\sum_{i\in g}c_{p,i,t},\qquad
w_{p,g,t}=\sum_{i\in g}w_{p,i,t-1},\qquad
r_{p,g,t}=c_{p,g,t}/w_{p,g,t}.
$$

Apply the same definitions to the benchmark b. Division by group weight is
essential: the grouped contribution already contains that weight.

A missing instrument column or NaN cell is treated as zero, following stored
portfolio history conventions. This is appropriate only for inactive/missing
holdings: supply complete P&L and weights for live positions. Every instrument
requires a classification; infinite inputs and duplicate identifiers are rejected.
Preferred group order is preserved, with additional observed groups appended.

## Methodology

### Single-period BHB effects

For each group:

$$
\begin{aligned}
A_{g,t} &= (w_{p,g,t}-w_{b,g,t})r_{b,g,t}, \\
S_{g,t} &= w_{b,g,t}(r_{p,g,t}-r_{b,g,t}), \\
I_{g,t} &= (w_{p,g,t}-w_{b,g,t})(r_{p,g,t}-r_{b,g,t}).
\end{aligned}
$$

Their sum is the group active contribution, and the sum over groups equals
strategy return minus benchmark return for that period. QIS uses the BHB
allocation definition above, without subtracting the benchmark total return from
each sector return. Do not mix it with the Brinson-Fachler allocation convention.
The BHB terminology follows [Brinson, Hood and Beebower (1986)](https://doi.org/10.2469/faj.v42.n4.39).

By default, `is_exclude_interaction_term=True` adds interaction to selection.
Set it to False to return interaction separately.

For zero group weight, the corresponding group return is defined as zero.
Interaction is implemented as the residual of group active contribution less
allocation and selection. This preserves P&L from off-benchmark holdings,
cost-only entries and zero-net-weight groups; for such groups the split is a
convention rather than an identifiable within-group return. In particular,
an absent benchmark group has zero allocation and its active contribution goes
to interaction, or selection when interaction is merged.

### Multiple periods: Frongello linking

Let $a_{g,t}$ be any raw allocation, selection or interaction effect. Define strategy
wealth $P_t = \prod_{s=1}^{t}(1+r_{p,s})$, with $P_0 = 1$, and initialise
$F_{g,0} = 0$. QIS computes the cumulative linked effect by

$$
F_{g,t}=(1+r_{b,t})F_{g,t-1}+P_{t-1}a_{g,t}.
$$

The returned adjusted increment is $F_{g,t} - F_{g,t-1}$. Consequently, ordinary
`cumsum()` of those adjusted increments yields cumulative attribution. Summing over
all groups and effect types $e\in\{A,S,I\}$ gives

$$
\sum_g\sum_{e\in\{A,S,I\}} F^{(e)}_{g,t}
=\prod_{s=1}^{t}(1+r_{p,s})-\prod_{s=1}^{t}(1+r_{b,s}).
$$

This is the difference of two compounded total returns, **not** the relative
wealth ratio $P_t/B_t - 1$. The calculation uses no later observations: extending
the input history does not revise earlier linked effects. Returns at or below
-100% are rejected in linked mode.

The linking method is described by
[Frongello (2002)](https://frongello.com/support/Works/JPMSpring2002.pdf).
The [PortfolioAttribution reference implementation](https://github.com/R-Finance/PortfolioAttribution/blob/master/R/Frongello.R)
provides the recurrence used here.
QIS implements the recurrence through vectorised cumulative products and sums;
tests verify it against a separate recursive implementation.

Linking defaults to `is_linked=True`. Set `is_linked=False` for corrected,
unadjusted arithmetic effects. Their sum answers a different question from
compounded outperformance. For example, +10% in two periods against a flat
benchmark is a 20-point arithmetic sum but a 21-point compounded difference.
Do not compound individual attribution effects with product(1 + effect).

## Worked example

A sector returns 10% in both portfolios, with strategy weight 60% and benchmark
weight 40%; the remainder is zero-return cash. Allocation is
$(0.60 - 0.40) \times 0.10 = 0.02$, or **2 percentage points**, with zero selection
and interaction. Using the benchmark's weighted contribution (4%) as its sector
return incorrectly produces 0.8 points of allocation. A residual interaction
calculation can still force the total to 2 points, so total reconciliation alone
cannot detect that bug.

The complete offline example below checks both the native-period calculation and
its aggregation to monthly and quarterly reports.

## Implementation in qis

### Portfolio wrapper, reporting periods and costs

`MultiPortfolioData.compute_brinson_attribution` uses the following workflow:

1. Select shared NAV observations inside `time_period`. The first observation is
   the NAV baseline. Attribute returns strictly after it through the last shared NAV.
2. Read full-history stored instrument P&L and matching prior realised weights.
   Clip after constructing the lag, preserving the first return after the baseline.
   Interior native return dates must match across the two portfolios.
3. Calculate and link at the native observation frequency.
4. Aggregate adjusted increments to `freq` for display; None retains native dates.

A monthly report of a daily backtest therefore includes every within-month trade.
Changing display frequency to quarterly does not change total attribution or
average weights. Reset the report window and recompute to establish a new zero
baseline; slicing an already linked curve does not restart its accrual.

The wrapper retains **gross instrument P&L by default**. With `is_net=True`,
it deducts each realised instrument trading cost divided by preceding NAV once,
before decomposition and linking. Costs then enter the group returns and
selection/interaction; they are not duplicated as another allocation effect.

Management fees, funding and other cash flows absent from `instrument_pnl` are
not assigned to asset classes. Therefore NAV reconciliation requires those
separate flows to be zero or already included in the supplied instrument P&L.
Do not label gross results as net or promise all-fee NAV reconciliation merely
because `is_net=True`; that option concerns realised trading costs only.
Opening costs already reflected in the first NAV are part of the baseline.

`PortfolioData.get_brinson_inputs(freq=...)` can also aggregate a single portfolio's
contributions geometrically within each bin. Its corresponding averaged weights
are effective exposures, not a claim that weights were constant throughout that bin.
For exact native-date BHB allocation use the standard wrapper, or obtain
`freq=None` inputs, decompose/link, then aggregate the returned increments.

### Returned tables and charts

The five returned DataFrames keep the established tuple order:

| Position | Contents |
| --- | --- |
| 0 | Group summary, followed by Total Sum |
| 1 | Aggregate allocation/selection increments; interaction if requested |
| 2 | Group allocation increments and Total Sum column |
| 3 | Group selection increments and Total Sum column |
| 4 | Group interaction increments and Total Sum column; zeros when merged |

Summary weights are arithmetic means of prior weights over native observations.
In linked mode, each portfolio's `Return Total` column contains instrument/group
contributions multiplied by that portfolio's own preceding cumulative wealth.
These columns sum to each portfolio's compounded return. Their per-group
difference need not equal the Frongello-linked group active effect: portfolio
contribution accumulation and benchmark-relative linking use different factors.
Their **overall totals** do reconcile.

Arithmetic mode labels those columns `Return Sum`. The effect columns always
sum to `Total\nActive`. Existing QIS Brinson plotters apply `cumsum()` to the
returned increments; they must not run linking a second time.

### Migration in 5.30.0

- The existing `qis.compute_brinson_attribution_table` now contains the corrected
  BHB calculation and defaults to linked results. Allocation/selection values
  change because the earlier implementation applied weights to contributions again.
- Default summary headers change from Return Sum to Return Total. For explicitly
  arithmetic reports, pass `is_linked=False`; this retains the sum headers but
  still uses corrected sector formulas.
- The old import through `qis.portfolio.reports.brinson_attribution` remains valid
  and references the same canonical function.
- The temporary local 5.29 entry point
  `qis.portfolio.attribution.brinson.compute_brinson_attribution` is removed.
  Replace it with `qis.compute_brinson_attribution_table`. There is one formula body.
- The wrapper now uses prior weights, stored P&L and native dates. Its historical
  results can change beyond the low-level formula/linking corrections because
  month-end holdings no longer approximate within-month trades.

For a complete executable example, run
`python -m examples.portfolios.brinson_attribution` from a source checkout.
It uses `qis.datasets.synthetic`, has no data/network dependency, prints and
checks the reconciliation, and accepts `--output-dir` for optional PDF/PNG/CSV files.

### Complete offline example

The [runnable offline example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/brinson_attribution.py)
constructs synthetic strategy and benchmark histories and checks attribution
against their compounded NAV difference at native, monthly and quarterly
frequencies.

```{literalinclude} ../examples/portfolios/brinson_attribution.py
:language: python
```

The [calculation tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/tests/brinson_test.py)
are the numerical contract; the example is the reader-facing reproduction path.

## Interpretation and limitations

- Attribution describes realised performance relative to the chosen benchmark and
  grouping. Changing either changes the interpretation of the effects.
- Zero group exposures require the residual convention described above. Their
  allocation/selection split cannot identify an otherwise undefined sector return.
- Gross and trading-cost-adjusted results reconcile only to NAVs with the same
  cash-flow coverage. `is_net=True` does not add missing management or funding fees.
- Multi-period effects depend on the linking method and report baseline. State
  linked versus arithmetic mode when comparing reports.
- A correct aggregate active return is necessary but insufficient: sector returns,
  lagged weights and the effect split also need independent checks.

## See also

- [Portfolio backtesting](portfolio_backtesting.md) for realised holdings and timing.
- [Turnover conventions](turnover_conventions.md) for traded volume and cost conventions.
- [Model-layer attribution](model_layer_attribution.md) for strategy-layer comparisons.
- [Factsheets and reporting](factsheets_and_reporting.md) for report entry points.

## References

- Brinson, G. P., Hood, L. R., and Beebower, G. L. (1986).
  [Determinants of Portfolio Performance](https://doi.org/10.2469/faj.v42.n4.39).
  *Financial Analysts Journal*, 42(4), 39–44.
- Frongello, A. S. B. (2002).
  [Linking Single Period Attribution Results](https://frongello.com/support/Works/JPMSpring2002.pdf).
  *Journal of Performance Measurement*, 6(3), 10–22.
- R-Finance. [Frongello linking implementation](https://github.com/R-Finance/PortfolioAttribution/blob/master/R/Frongello.R).
  Supplemental implementation reference.
- [qis source and project documentation](https://github.com/ArturSepp/QuantInvestStrats).
  Cite the software version used through
  [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
