# Reporting-frequency convention

`qis` factsheets treat **reporting frequency as a first-class, propagated, guarded and labelled
axis**. The same book can be reported daily (risk), monthly (clients) or quarterly (investment
committee / board) and every statistic — rolling windows, regression frequency, regime
classification, annualisation — stays internally consistent, with each panel stating the frequency
it was computed at. This page documents how that works and how to use it.

## The problem it solves

Most return-analytics libraries assume you hand them a return series at whatever frequency you
happen to have and compute on it. That leaves two silent failure modes when the *same* book is
reported at more than one cadence:

- **Silent up-sampling.** Asking for daily-style statistics from monthly data fabricates
  observations that do not exist and produces confident-looking nonsense.
- **Inconsistent windows and labels.** A "1-year rolling" window means 260 points on daily data
  and 12 on monthly data; a Sharpe, a beta and a regime split computed at different, unstated
  frequencies are not comparable, and a max-drawdown read off a resampled series silently differs
  from the one on the native price path.

The convention below makes the reporting frequency an explicit input that drives everything
coherently, refuses to up-sample, and labels every panel so the provenance is never ambiguous.

## The two axes

Factsheet settings are driven by two independent axes:

1. **Data sampling frequency** — `ReportingFrequency` ∈ {`DAILY`, `WEEKLY`, `MONTHLY`,
   `QUARTERLY`}. Each maps to a pandas resampling grid (`B`, `W-WED`, `ME`, `QE`); every `freq_*`
   field uses that grid, and rolling windows / EWM spans are annualised from the grid's
   periods-per-year (260 / 52 / 12 / 4).
2. **Reported time span** — long vs short horizon. This sets window *lengths*, the returns-heatmap
   and x-axis frequency, and the regime-classification frequency:
   - **long** — vol / var / sharpe 3y (daily uses 1y, since a 3y window is too slow on daily
     returns), beta 3y, turnover / cost 1y; heatmap and x-axis `YE`, regimes `QE`.
   - **short** — *all* rolling windows collapse to 1y (so a ~3y report still populates);
     turnover / cost 1y; heatmap and x-axis `ME`, regimes `ME`.

   Long vs short is selected automatically from the reporting span against a threshold
   (`long_threshold_years`, default 5 years).

## Resulting presets

Counts are in periods of the base grid; `LONG | SHORT` where they differ:

| Frequency | grid | vol / var / sharpe | factor beta | turnover / cost | regime |
|---|---|---|---|---|---|
| DAILY | `B` | 260 \| 260 | 780 \| 260 | 260 | `QE` \| `ME` |
| WEEKLY | `W-WED` | 156 \| 52 | 156 \| 52 | 52 | `QE` \| `ME` |
| MONTHLY | `ME` | 36 \| 12 | 36 \| 12 | 12 | `QE` \| `ME` |
| QUARTERLY | `QE` | 12 \| 4 | 12 \| 4 | 4 | `QE` \| `ME` |

`vol_rolling_window` and `var_span` are EWM spans; `sharpe_rolling_window` is a rolling window;
`turnover` / `cost` periods are rolling-sum windows. Note that regime classification stays on its
own horizon frequency (`QE` long, `ME` short) regardless of the reporting frequency — a quarterly
report and a daily report of the same long book are both split on the same `QE` regimes, so the
conditioning is comparable across cadences.

## The up-sampling guard

Before rendering, every factsheet validates the input series against the requested reporting
frequency with `validate_reporting_frequency(data, reporting_frequency)`. Reporting at a frequency
**finer** than the data's native sampling raises a `ValueError`; equal or coarser is fine. Monthly
data reported monthly or quarterly is accepted; monthly data reported weekly or daily is refused
rather than silently up-sampled. The multi-asset report applies the guard to both the asset
universe and the benchmark prices.

## Per-panel frequency labels

Every panel states the frequency its statistics were computed at, so native versus resampled
quantities are never conflated:

- Cumulative performance is titled `... ({freq}-freq stats) ...` at the reporting frequency.
- Running drawdowns and time-under-water are labelled with the **native** price-grid frequency
  (e.g. `(B-freq)`), because they are computed on the unresampled path. This is why the
  panel max-drawdown is frequency-invariant while the risk-table max-drawdown (computed on
  resampled returns) coarsens with the reporting frequency — both are correct, and the labels make
  the distinction explicit.
- Rolling Sharpe / Vol, rolling beta, correlation and the return scatter all carry the reporting
  frequency and the window length actually used.

Higher moments respect the reporting frequency too: skewness is computed on the reporting-frequency
returns, so it varies across daily / weekly / monthly / quarterly views of the same book rather
than being frozen at a single frequency.

## Trailing-window correlation at coarse frequencies

The multi-asset report shows a full-history correlation table alongside a recent / trailing-window
one. At coarse reporting frequencies a one-year trailing window has too few observations to be
meaningful (one year of quarterly returns is only ~4 points, which makes the matrix degenerate), so
the trailing window is widened to hold at least `min_trailing_obs` observations (default 12) at the
reporting frequency — i.e. it stays at one year for daily / weekly / monthly and widens to three
years at quarterly.

## Using it

The simplest entry point is the `qis.factsheet` facade, which applies the whole convention for you
and auto-selects the report archetype from the input:

```python
import qis

# one strategy vs a benchmark, monthly reporting, full history -> PDF path
path = qis.factsheet(prices,
                     benchmark_prices=spy,
                     reporting_frequency='monthly',
                     file_name='book')

# the same universe at quarterly cadence
figs = qis.factsheet(prices, benchmark_prices=spy, reporting_frequency='quarterly')
```

For full control, the convention is exposed through the configuration layer and spread into the
generators directly. `fetch_default_report_kwargs` builds the frequency- and horizon-calibrated
kwargs (auto-selecting long vs short from the span):

```python
from qis import ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs

report_kwargs = fetch_default_report_kwargs(time_period=time_period,
                                            reporting_frequency=ReportingFrequency.QUARTERLY)
fig = qis.generate_multi_asset_factsheet(prices=prices, benchmark='SPY',
                                         time_period=time_period, **report_kwargs)
```

`make_factsheet_config(reporting_frequency, is_long_period, **overrides)` returns the underlying
`FactsheetConfig` when you want to pin the horizon or override individual fields. The
`FactsheetConfig` field schema is the public kwargs contract: `fetch_factsheet_config_kwargs()`
spreads `FactsheetConfig._asdict()` into the `generate_*_factsheet(...)` calls, so the field names
stay stable across the generators.

## Worked examples

`examples/factsheets/*_reporting_frequencies.py` render each of the four report types across the
full `DAILY / WEEKLY / MONTHLY / QUARTERLY × {long, short}` grid on one data panel, and are the
reference for the convention end to end.
