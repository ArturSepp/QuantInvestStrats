---
myst:
  html_meta:
    description: >-
      Reproduce a seeded qis stationary-bootstrap experiment and understand how circular
      blocks, sample boundaries, random seeds and annualisation affect reported analytics.
---

# Reproducibility: what an unstated convention costs

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-26](https://github.com/ArturSepp/QuantInvestStrats/commit/3633b53d0dd486077aff96f2ca752d23efebb4fe)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Reproducibility means obtaining the same analytical result from identified data, software and
calculation conventions. A method name and a random seed alone are insufficient. This article
uses a fixed synthetic bootstrap experiment to show how changing a sample-boundary rule changes
a reported mean, then specifies the information needed to reproduce it.

## Overview

The stationary bootstrap samples consecutive observations in blocks of random length. qis
uses circular blocks: a block that reaches the last source observation continues from the
first. Its earlier implementation truncated blocks at the source boundary. Both return an
index array of the requested shape, but they draw different observations with different
frequencies.

The circular construction follows [Politis and Romano (1994)](https://doi.org/10.1080/01621459.1994.10476870).
The convention changed in qis 5.1.0; see the
[change history](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CHANGELOG.md).
The comparison below preserves the earlier teaching implementation exactly, including its
loop termination rule. It measures the difference between these two implementations, rather
than isolating a single code change while holding every other detail constant.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Value in this experiment |
|---|---|---|
| $n$ | Number of source observations | 250 ordered periods |
| $M$ | Number of resampled paths | 400 |
| $H$ | Observations in each path | 250 |
| $b$ | Mean geometric block length before any floor | 20 periods |
| $p$ | Probability of ending a geometric block | $1/b=0.05$ |
| $J_{t,m}$ | Zero-based source index at output position $t$ in path $m$ | Integer from 0 to 249 |
| $x_i$ | Synthetic periodic return observation | Decimal arithmetic return |
| $A$ | Linear annualisation factor for the mean | 260 periods per year |
| Index seed | Numba random state used by each sampler | 7 |
| Return seed | NumPy default-generator seed for the source series | 3 |

There are no market observations or calendar dates in this experiment. The 250 periods and
260-period year are deliberately fixed teaching conventions. They do not specify a real
exchange calendar or imply that the series contains 250 monthly observations.

The source series has Gaussian noise with standard deviation 0.01 per period and a deterministic
drift rising from 0.0004 to 0.0020 (4 bp to 20 bp). That changing drift is a deliberate departure
from stationarity: it exposes the effect of uneven sampling. It does **not** establish that a
stationary bootstrap gives valid confidence intervals for a process with changing drift.

## Methodology

### The convention

In the ordinary stationary construction, independently draw a uniform starting index
$U\in\{0,\ldots,n-1\}$ and a block length $L$ with support $1,2,\ldots$:

$$
\Pr(L=\ell)=p(1-p)^{\ell-1},
\qquad E[L]=\frac{1}{p}=b.
$$

The block's source indices are

$$
J_k=(U+k)\bmod n,
\qquad k=0,\ldots,L-1.
$$

Concatenate independently drawn blocks until the output contains $H$ observations, trimming
only the last block to the requested output length. Trimming at the **output** boundary is
necessary; truncating a block at the **source** boundary changes the sampling rule.

The public `generate_bootstrapped_indices` function defaults to `min_block_size=1` for
`BootstrapType.STATIONARY`. For a specified minimum length $L_{\min}$, the sampler uses
$\max(L,L_{\min})$. The resulting lengths no longer follow the original geometric
distribution, and their expected length need not equal `block_size`. Record both parameters.

The historical comparator cuts each source block at $n-1$ and restarts from another uniform
index. Early observations can only be reached by relatively few forward continuations, so
they are underrepresented. Its retained `while next_row < index_length - 1` loop can also
leave the last output position at its initial zero when a fill ends one position early.
The published legacy row includes this behaviour. Neither sampler is changed to make the
illustration cleaner.

### Measuring draw frequencies

Let $C_i$ be the number of times source observation $i$ appears across all $MH$ output positions.
Its relative draw frequency is

$$
f_i=\frac{nC_i}{MH},
\qquad
\frac{1}{n}\sum_{i=0}^{n-1}f_i=1.
$$

A value of 1 means exactly the uniform expected count. The first and last deciles average
$f_i$ over their respective 25 source positions. Circular blocks with independent uniform
starts give equal marginal draw probabilities; a finite run still has Monte Carlo variation.

### Measuring the reported mean

The average resampled mean, source mean and their difference are

$$
\begin{aligned}
\bar x^*
  &=\frac{1}{MH}\sum_{m=1}^{M}\sum_{t=1}^{H}x_{J_{t,m}}
   =\sum_{i=0}^{n-1}\frac{C_i}{MH}x_i,\\
\bar x&=\frac{1}{n}\sum_{i=0}^{n-1}x_i,\\
\delta&=\bar x^*-\bar x.
\end{aligned}
$$

The count-weighted expression independently checks the direct resampled-array calculation.
Here “bias” labels the measured difference $\delta$ for one fixed source and one finite set
of draws; it is not an exact expectation over all random sources and seeds.

The table reports $10^4\delta$ in basis points per period and $100A\delta$ in annualised
percentage points. This is **linear annualisation of a mean-return difference**, not a
compounded annual return, CAGR, Sharpe ratio or probability of profit.

## Worked example

### What it does to the sample

For the fixed 250-period source, 400 paths and mean block length 20:

| Convention | First observation | First decile | Last decile |
|---|---:|---:|---:|
| Historical truncating | **0.110** | 0.526 | 1.073 |
| Circular | 0.978 | 1.007 | 1.020 |

The historical comparator draws the first observation at roughly a ninth of its uniform
expected frequency and the first decile at just over half. The circular row is close to
uniform in this finite experiment; it is not exactly uniform.

### What it does to a reported number

Apply those same index arrays to the changing-drift source:

| Convention | Resampled mean | Bias per period | Bias annualised, $A=260$ |
|---|---:|---:|---:|
| Historical truncating | 13.62 bp | **+0.83 bp** | **+2.15%** |
| Circular | 12.67 bp | −0.12 bp | −0.32% |

The source mean is 12.80 bp per period. The annualised mean difference between the implementations
is about 2.47 percentage points for this fixture. Late observations have higher drift here,
so giving them more weight raises the reported mean. The direction and magnitude depend on
the data: they are not a universal bootstrap adjustment. A constant-valued series has no such
mean difference; constant expected drift with noisy realised observations need not give a
zero difference in a finite run.

All entries are rounded independently from full-precision calculations. Subtracting the
displayed 13.62 and 12.80 does not recover the displayed 0.83 exactly.

From a repository checkout, this block regenerates both rows and checks their means by counting
source indices. It imports the frozen historical comparator only for this demonstration.

~~~python
import numpy as np
import qis
from examples.models.bootstrap_convention import (
    draw_truncating_indices, make_trending_returns,
)

n, paths, length, block, seed = 250, 400, 250, 20, 7
source = make_trending_returns(num_periods=n, seed=3)
legacy = draw_truncating_indices(
    num_data_index=n, num_samples=paths, index_length=length,
    block_size=block, seed=seed,
)
circular = qis.generate_bootstrapped_indices(
    num_data_index=n, bootstrap_type=qis.BootstrapType.STATIONARY,
    num_samples=paths, index_length=length, block_size=block,
    min_block_size=1, seed=seed,
)
for label, indices in [('truncating', legacy), ('circular', circular)]:
    counts = np.bincount(indices.ravel(), minlength=n)
    frequencies = counts * n / indices.size
    direct_mean = source[indices].mean()
    counted_mean = np.dot(counts / indices.size, source)
    np.testing.assert_allclose(direct_mean, counted_mean, atol=1e-15)
    assert np.isclose(frequencies.mean(), 1.0)
    bias = direct_mean - source.mean()
    print(label, frequencies[0], frequencies[:25].mean(), frequencies[-25:].mean())
    print('mean bp:', direct_mean * 1e4,
          'bias bp:', bias * 1e4, 'annualised bias %:', bias * 260 * 100)
~~~

## Implementation in qis

### Reproducing the tables

After the repository's prescribed environment setup, run:

~~~console
python -m examples.models.bootstrap_convention
~~~

The canonical [bootstrap_convention.py example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/models/bootstrap_convention.py)
prints both tables without network access or a data file. It requires a checkout: repository
examples are not installed by `pip install qis`. The current sampler is implemented in
[bootstrap_numba.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/bootstrap/bootstrap_numba.py)
and exposed as `qis.generate_bootstrapped_indices` with `qis.BootstrapType.STATIONARY`.

The [convention regression tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/bootstrap/tests/test_bootstrap_convention.py)
pin the printed quantities to half a unit in their last published digit and check the
article's stated values. The [example tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/test_examples.py)
separately check that the script executes. Execution alone does not establish numerical
correctness. These repository-integrity checks skip when their sources are absent from an
installed wheel.

### A note on version pinning

A reproduction record should identify:

- Input values or their hashes, sample order, missing-data treatment and calendar/frequency.
- Return type, annualisation, rates and any timing or lag convention that affects the result.
- The exact algorithm, block settings, random seeds and random-number implementation.
- The actually imported qis source, its version and commit plus any uncommitted source hashes.
- Python and dependency versions, commands, parameters, result tables and validation checks.

A seed alone does not promise identical output across software versions. For an unchanged
publication, preserve the original code, environment and inputs. For a recomputation using
current qis, report it as a new computation and identify the changed convention. Pinning an
old implementation identifies the result; it does not make the old method preferable.

The [documentation analytics runner](https://github.com/ArturSepp/QuantInvestStrats/blob/main/tools/docs_analytics/README.md)
records these details for the registered image bundle, including source hashes and actual
import location. Generation timestamps and fixed sample dates are separate fields.

## Interpretation and limitations

### What follows for the package

A circular sampler prevents this source-boundary weighting defect. It does not establish
stationarity of the observed data, select a defensible block length, remove look-ahead or
repair missing/stale observations. The original stationary-bootstrap inference applies under
conditions on stationary, weakly dependent data; this changing-drift illustration is a
sampling-convention diagnostic.

State `is_log_returns` explicitly when using `qis.to_returns`, even though the function has a
default. State the frequency and annualisation actually used. qis distinguishes three
[Sharpe conventions](performance_analytics_and_sharpe.md); excess variants require a specified
rate series in `PerfParams.rates_data`. A function name or an output label cannot replace
the recorded settings.

Figures need the same discipline. Include relevant sampling and reporting conventions in
captions, and verify which labels a particular renderer actually displays. Do not assume every
existing panel already carries every convention. A successful regeneration proves that the
script ran; source checks, numerical checks and visual review serve different purposes.

## See also

- [Frequency and annualisation](frequency_convention_note.md)
- [Performance and Sharpe conventions](performance_analytics_and_sharpe.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Model-layer attribution](model_layer_attribution.md)
- [Documentation standard](documentation_standard.md)

## References

- Politis, D. N., and Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American
  Statistical Association*, 89(428), 1303–1313.
  [Publisher record and DOI](https://doi.org/10.1080/01621459.1994.10476870).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
