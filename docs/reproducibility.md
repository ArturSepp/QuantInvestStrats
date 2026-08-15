# What an unstated convention costs

A reported performance number is the output of a method and a set of conventions. The method is
usually named. The conventions usually are not, and two correct implementations that differ in
one of them will disagree without either being wrong.

This page measures one such case end to end, because the argument is easier to accept from a
number than from a principle. Everything below is produced by
`src/qis/examples/models/bootstrap_convention.py`, which runs in about three seconds and needs no
network and no data file.

## The convention

The stationary bootstrap of Politis and Romano resamples a series in blocks whose length is drawn
from a geometric distribution. A block that starts near the end of the sample has to do one of
two things. It can wrap around to the beginning, which is the construction in the original paper,
or it can stop at the last observation.

Both choices produce an index array of the right shape. Both run without error. Neither is
recorded in the output. `qis` used the truncating form until version 5.1.0 and uses the wrapping
form from 5.1.0 onward.

## What it does to the sample

Truncation is not neutral, because a block only ever runs forwards. The earliest observations can
be reached as the start of a block and almost never as the continuation of one, so they are drawn
far less often than the rest.

Measured on a 250-period sample with a mean block length of 20, over 400 draws, where 1.00 means
an observation was drawn exactly as often as uniform sampling would draw it:

| convention | first observation | first decile | last decile |
|---|---:|---:|---:|
| truncating | **0.110** | 0.526 | 1.073 |
| circular | 0.978 | 1.007 | 1.020 |

The first observation of the sample appears at roughly a ninth of its due weight, and the first
tenth of the sample at just over half.

## What it does to a reported number

An uneven draw only matters if the observations differ in what they contribute. On a series with
constant drift it washes out. On a series whose drift changes through the sample, which is the
common case, it becomes a bias in whatever statistic is computed from the resample.

The same two index arrays, applied to a 250-period series whose drift rises from 4 bp to 20 bp:

| convention | resampled mean | bias | bias annualised |
|---|---:|---:|---:|
| truncating | 13.62 bp | **+0.83 bp** | **+2.15%** |
| circular | 12.67 bp | −0.12 bp | −0.32% |

The source series has a mean of 12.80 bp per period. The truncating convention over-weights the
late, higher-drift part of the sample and reports a mean 0.83 bp per period above it, which is
2.15% per year on a 260-period year.

Two researchers running "a stationary bootstrap" on the same data would report annual returns
differing by more than two percentage points, and nothing in either output would say why.

## What follows for the package

Three rules in `qis` come from this class of problem rather than from taste.

The return convention is an argument and not an assumption. `qis.to_returns` takes
`is_log_returns` explicitly, and annualisation follows from the stated frequency rather than from
a default.

Sharpe ratios are named. `qis` distinguishes three conventions, and the excess variants require a
rate series in `PerfParams`, so a Sharpe ratio cannot be computed against an unstated risk-free
assumption.

The reporting frequency appears on every rendered panel, so a reader of a factsheet can see the
convention that produced the numbers without opening the code.

## Reproducing the tables

```bash
python -m qis.examples.models.bootstrap_convention
```

Both tables are printed. The script is executed by the test suite on every run, so the numbers on
this page cannot drift away from the code that produces them.

## A note on version pinning

Because 5.1.0 changed this convention, a resampled result produced with an earlier version does
not reproduce under a later one. Any published result that uses `BootstrapType.STATIONARY` should
state the `qis` version alongside the seed. This is why the package version belongs in a
reproduction record rather than only in the environment file.
