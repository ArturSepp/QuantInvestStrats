# Changelog

All notable changes to qis are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [5.6.1] - 2026-08-02

**Two page-geometry limits that used to produce an unreadable panel in silence are now
enforced.** Both are the same arithmetic: a decoration sized in points, laid on a panel sized in
inches, degrades once the series count outgrows the panel. The legend case warns and changes
nothing drawn. The attribution case reduces what is drawn - `plot_performance_attribution` falls
back to the tails of the distribution once the instrument labels would crowd, which is the one
behaviour change in this release. It engages only past the point where the panel had already
stopped naming anything, so a panel that reads today is untouched, and `max_bars=0` restores the
old behaviour unconditionally.

**A factsheet carrying more series than its panel legends hold now warns.** A panel legend is a
fixed height of one row per series, and matplotlib's `constrained_layout` counts the part of it
that spills out of the axes as a layout margin: once the legend outgrows the panel cell the
solver drives the axes height to zero and disables itself for the whole figure, so every panel on
the page reverts to the raw gridspec. One overflowing legend collapses the page, not one panel.
On A4 portrait at `fontsize=5` the capacity is 15 series, the axes start losing height at 16 and
the layout collapses at 19; a 22-asset panel produced an unreadable page with no diagnostic other
than matplotlib's `constrained_layout not applied`, which names neither the cause nor a remedy.
Nothing is dropped and no number changes - pages that render today render identically.

### Added
- `qis.portfolio.reports.config.estimate_legend_capacity(figsize, fontsize, panel_rows,
  gridspec_rows)` returns the number of legend entries a panel carries at full height, and
  `validate_legend_capacity(...)` warns above it, naming the `fontsize` and the `figsize` that
  would fit the requested series count. Both are internal; the calibration constant
  `LEGEND_ROW_HEIGHT_PER_FONTSIZE = 0.01576` in/row/pt is re-measured against matplotlib by
  `qis/portfolio/tests/legend_capacity_test.py`, so a matplotlib change that invalidates it fails
  the suite rather than a report.
- The guard is called by `generate_multi_asset_factsheet` (capacity counted over the asset
  columns plus any benchmark added to the navs), `generate_multi_portfolio_factsheet` and
  `generate_strategy_benchmark_factsheet_plt` (over the portfolios plus any benchmark). The three
  pages share one panel cell, `figsize[1] * 2 / 14`, so they share one capacity.

**An attribution panel wider than its tick labels can name now shows the tails.** A 90-degree
rotated tick label occupies the font's line height horizontally whatever the instrument is
called, so `shorten_instrument_names` never bought room: only a smaller font or a wider axis
does. A half-page panel is 4.09 in and holds 45 labels at `fontsize=5`; an 84-instrument
portfolio drew 84 bars over a solid smear of overlapping names, on the strategy factsheet summary
page and on all ten panels of its attribution page.

### Changed
- `PortfolioData.plot_performance_attribution` takes `max_bars` and `fontsize`. `max_bars=None`,
  the default, keeps every bar until the labels would crowd and then reduces to the tails sorted
  by the attributed value; an explicit count forces the reduction; `max_bars=0` disables it. The
  cut is two-sided for a signed metric - `PNL`, where the losers matter as much as the winners -
  and top-only for a metric that is non-negative by construction, which is `PNL_RISK`, `COSTS`,
  `TURNOVER` and `VOL_ADJUSTED_TURNOVER`: a share of a total has no bottom tail to show.
- The folded remainder is stated in the panel title (`top and bottom 45 of 84: 39 folded away,
  summing to 18.53%`) rather than drawn as one aggregate bar, which would carry a third of the
  total and flatten every instrument the panel exists to show. The `sum=` claim in the title is
  therefore still true and its exception is on the line below it.
- The instrument order is untouched below the capacity, so the asset-class blocks stay visible on
  every panel that reads today. Sorting only happens where the alternative is an unreadable page,
  and it does cost the cross-panel alignment of the attribution page: each panel then sorts by
  its own metric.
- Not applied to `MultiPortfolioData.plot_performance_attribution`, which is a separate
  implementation drawing one column per portfolio on a full-width panel. That geometry holds 92
  labels and is not at risk at the universe sizes this addresses.

### Added
- `qis.plots.utils.estimate_bar_label_capacity(axis_width, fontsize)` and `estimate_axis_width(ax)`,
  the geometry behind the reduction. The constants `BAR_LABEL_WIDTH_PER_FONTSIZE = 0.0138`
  in/label/pt and `AXIS_SHARE_OF_CELL = 0.96` are re-measured against matplotlib by
  `qis/portfolio/tests/attribution_reduction_test.py`.
- `qis.portfolio.portfolio_data.reduce_attribution_to_tails(data, max_bars)` returns the retained
  entries and the folded total, so `kept.sum() + folded` is the original sum by construction.

## [5.6.0] - 2026-08-02

**`weight_implementation_lag` is counted in observations of the price index, not in calendar
days.** It selects the entry price for the units and nothing else: a weight observed at *t* is
traded at the price `weight_implementation_lag` observations later, and prices and instrument
returns are untouched. The calendar-day shift it replaces was resolved onto the price grid by
taking the next available date, so at a lag of two every Thursday/Friday pair landed on the same
Monday - 106 of 523 weight rows on a two-year daily schedule - and because weights are consumed
one row per rebalancing a collapsed date did not skip a row, it shifted every later row. A lag of
one, which is what every `optimalportfolios` call site and every qis example passes, resolves to
the same observation under both readings and no result at lag=1 moves; results at lag>=2 move and
were wrong before. `optimalportfolios` already documented the argument as periods, so the two
packages now agree.

### Fixed
- `backtest_model_portfolio` raises when two weight dates resolve to the same traded date on the
  price index, which happens when the weights frame is denser than the price panel - a
  calendar-daily weights frame against a business-day panel is the common case. Weights are
  consumed one row per rebalancing flag, so the rows after the first collision were applied at
  the wrong dates for the rest of the backtest, and the staleness grew: on a 33-row daily
  schedule at a lag of three, 15 rows were never consumed and the weights traded on 2020-01-31
  were the ones computed on 2020-01-15. Covered by
  `qis/portfolio/tests/test_backtester_weights_consistency.py`.

- `weight_implementation_lag` counts observations of the price index rather than calendar days;
  see the note above. A weight whose traded date would fall past the end of the price history is
  dropped with a warning rather than silently, and every weight date is now traded exactly once.

### Added
- `qis.generate_static_weights_schedule(prices, weights, rebalancing_freq=...)` turns a fixed
  allocation into the rebalancing weight frame `backtest_model_portfolio` consumes, allocating
  over the instruments priced on each rebalancing date. A static vector against a panel whose
  instruments start and stop at different dates leaves the missing instrument's weight in the
  cash balance - the backtester takes weights as given and does not modify them - which is
  rarely the intended allocation. The universe is read at the rebalancing date only, so the
  construction is point in time. Rescaling preserves the total exposure of the specification
  rather than forcing the row to one, so a book that is 90% invested by design stays 90%
  invested; `is_preserve_total_exposure=False` gives the force-to-one form, and
  `is_rescale_to_live_universe=False` gives the cash residual with an explicit `0.0` rather than
  a nan in the reported weights. Core API, `Portfolio and backtesting`.

- `qis.align_weights_to_columns(weights, columns)` is the shared normaliser for the weight
  argument, so `generate_static_weights_schedule` and `backtest_model_portfolio` cannot disagree
  on what a Dict, a pd.Series, a List or an np.ndarray means. Behaviour is unchanged: a Dict or
  pd.Series aligns by name, a List or np.ndarray is positional, and the error contract the
  backtester documented is preserved.

- `backtest_model_portfolio` warns when a weighted instrument has no price on its traded date -
  the leg is not traded and its weight stays in cash - and names the instruments and the first
  date. Checked at the traded date rather than the weight date, so a lag that carries a weight
  past an instrument's last price is caught too. A zero weight against a missing price is
  silent, since holding a column at 0.0 before an instrument starts is deliberate.

- `backtest_model_portfolio` warns when prices carry missing values *inside* an instrument's own
  reported history, and names `prices.asfreq('B', method='ffill')` or `prices.ffill()` as the
  remedy. Units are held through a nan price and `np.nansum` drops the leg from the nav on those
  dates, so a hole removes that leg's whole value from the portfolio with no error. Leading nans
  (not trading yet) and trailing nans (no longer reporting) are legitimate and stay silent.

- `qis/utils/tests/test_static_weights_schedule.py` and
  `qis/portfolio/tests/test_backtester_weights_consistency.py`, including the regression pin that
  a lag of one reproduces the calendar-day schedule for `ME`, `QE`, `W-FRI` and `B` weight
  frequencies.

- `examples/portfolios/static_weight_with_missing_prices.py` and
  `examples/portfolios/lagged_weight_implementation.py`. Both run on the seeded synthetic panel
  with no network and no data file, so `test_examples.py` executes them top to bottom rather than
  only reading them - the first two examples under `examples/portfolios/` that it can. The first
  shows the cash residual, the reallocated schedule and what preserving the total exposure means
  for a book that is 90% invested by design; the second runs one monthly trend book at lags of 0,
  1, 5 and 20 observations, and reports that turnover and realised costs do not move with the lag
  while return and Sharpe do.

## [5.5.0] - 2026-08-01

**Every `axis=1` `pd.concat` in library code states `sort=` explicitly, and the three resampling
entry points sort a panel that reaches them out of order.** pandas 2.2 sorted the union of two
DatetimeIndexes whatever `sort=` said; pandas 3.0 honours `sort=False` and leaves the union in
appearance order, and warns that pandas 4 will stop sorting when the argument is absent. So a
panel joining a benchmark series and a strategy nav on different calendars now arrives unsorted,
and a call that says nothing means one thing today and another after the next major release. No
number moves under pandas 3.0 - the reporting goldens are unchanged - and none moves under
pandas 4 either, which was the point. `load_df_from_csv` also gains `float_precision`, so a CSV
round trip can be exact.

### Fixed
- `df_asfreq` sorts a panel whose index is not in chronological order before resampling it, and
  `compute_periodic_returns` sorts before its ffill/bfill. pandas 3.0 changed
  `pd.concat(axis=1, sort=False)` to leave the union of two non-identical DatetimeIndexes in
  appearance order, so a panel joining a benchmark series and a strategy nav on different
  calendars now arrives unsorted: `df_asfreq` raised `ValueError: index must be monotonic
  increasing or decreasing` from inside `pandas.reindex`, and the fill in
  `compute_periodic_returns` ran in row order and carried the terminal price backwards onto the
  dates a column does not carry, without raising. Covered by
  `qis/utils/tests/test_df_freq_sorting.py` and
  `qis/plots/derived/tests/test_returns_heatmap_sorting.py`.

- `prices_at_freq` sorts the same way on its `freq=None` branch, where the ffill runs in place
  rather than through `df_asfreq`, so `to_returns` without a resample is order-independent too.
  Covered by `qis/perfstats/tests/test_returns_sorting.py`.

- Every `axis=1` `pd.concat` in library code states `sort=` explicitly - 133 call sites in 39
  modules. `sort=True` where the joined index is dates, which is what pandas 2.2 did whatever
  the argument said; `sort=False` where it is instrument or statistic labels, which pandas has
  never sorted. Three sites were joining non-identical DatetimeIndexes and relying on the
  implicit sort that pandas 3.0 deprecates and pandas 4 removes: `compute_fx_optimal_hedge`,
  the FX hedging report, and the multi-frequency nav in `signal_diagnostics`. No number moves
  under pandas 3.0 - the reporting goldens are unchanged - and none moves under pandas 4 either,
  which was the point.

### Added
- `qis/tests/test_concat_sort_convention.py`: an `axis=1` `pd.concat` in library code without an
  explicit `sort=` fails the suite. What the union of two DatetimeIndexes does when the argument
  is absent has changed twice in two major pandas versions, and the difference is a scrambled
  time axis rather than an error.

- `load_df_from_csv(..., float_precision=None)` and
  `load_df_dict_from_csv(..., float_precision=None)`, forwarded to `pd.read_csv`. pandas' default
  C float converter is fast and not correctly rounded: a frame written with `save_df_to_csv` and
  read back differed from the original by up to ~4e-16 per cell on a realistic panel, which makes
  "the file holds what I wrote" impossible to assert. Passing `float_precision='round_trip'`
  returns the value bit for bit. The default is `None`, so nothing changes for an existing caller.
  Both docstrings state why the default converter is not exact and when to pass `'round_trip'`.

## [5.4.0] - 2026-07-29

**`backtest_model_portfolio` accepts `rebalancing_costs` as a panel of dates x tickers, and the
numba kernel beneath it, `backtest_rebalanced_portfolio`, takes a `(t, n)` array.** A float or a
per-instrument `pd.Series` behaves exactly as before, so a caller of the public wrapper sees
nothing change; a date-indexed `Series` now raises rather than being read as per-instrument, and a
direct caller of the kernel has to pass the broadcast array. Everything else in this release is
documentation, tests and packaging metadata, most of it hardening the JOSS submission after an
external review.

### Added
- `backtest_model_portfolio` accepts `rebalancing_costs` as a `pd.DataFrame` of dates x
  tickers: each price date takes the last schedule row at or before it, so a cost schedule
  stated on era boundaries (the `trendfollowing` volume-cost panel is the motivating case)
  applies from each boundary onward. A float or per-instrument `pd.Series` behaves exactly as
  before; a date-indexed Series now raises rather than being misread as per-instrument.
  Covered by `qis/portfolio/tests/test_backtester_costs.py`.

- `qis/tests/test_version_metadata.py`: `pyproject.toml`, `CITATION.cff` and the `@software`
  BibTeX entry in `README.md` must carry the same version, and `date-released` must be an ISO
  date. Nothing had held them together, and in the sibling `optimalportfolios` repository the
  same three read 6.3.0, 6.2.0 and versionless at one commit.
- `qis/tests/test_documentation.py` gains an in-page anchor check: every `#anchor` link resolves
  to a heading or an explicit `<a name>` in the same document. The README table of contents is
  thirteen such links, and a renamed section leaves the entry above it pointing nowhere with
  nothing failing.
- `qis/tests/test_documentation.py` also reads the README's `python` blocks as one script, in
  document order, and requires every bare name a block loads to be a builtin or a name bound in
  a block above it. Static rather than executed: the blocks call `yfinance`, and no test here
  may reach the network.
- `qis/models/bootstrap/tests/test_bootstrap_convention.py` requires `paper.md` to state the
  four values the convention example computes, at the two decimals the manuscript prints. The
  file's failure messages had always claimed the paper quoted them; nothing read it.

### Changed
- `backtest_rebalanced_portfolio` (the numba kernel under `backtest_model_portfolio`) takes
  `rebalancing_costs` as a `(t, n)` array; the wrapper broadcasts the scalar and
  per-instrument forms. Breaking only for direct callers of the kernel, of which the public
  consumers have none.
- Packaging metadata migrated to PEP 639: `license = "MIT"` with `license-files`, and the
  deprecated `License :: OSI Approved :: MIT License` classifier removed. `build-system`
  requires `setuptools>=77.0`, which is where that spelling is supported; this is a
  build-from-source floor and does not affect installing the wheel.
- Every `paper_phrase` in `docs/audit/paper_numbers.json` is a sentence-level fragment rather
  than a bare number. `94` was satisfied by the citation key `politis1994`, so that check would
  have passed on a manuscript that had stopped quoting the count. The two `privateassets`
  counts carry no phrase: the paper cites the package without a figure, so those two checks
  could not fail. `active_months` gains one.
- `paper.md` revised after an external review: the capability table states `vectorbt` and `bt`
  at documented-interface strength, `arch` is credited under it for the paired-resampling
  primitive and the no-counterpart claim is scoped to the integration, the API and example
  inventories give way to the audit record, and the body is 1,745 words against JOSS's 1,750.
  The statement of need also names the two comparisons a research pipeline runs and extends the
  shared-convention argument to agentic AI tooling.
- `README.md`: the table-of-contents entry `Notebooks` is now `Runnable examples`, matching the
  section it has pointed at since the notebooks were removed, and its anchor is renamed with it.
- `qis/docs/gallery.md` and its four screenshots moved to `docs/`. `MANIFEST.in` excludes `*.png`,
  so the gallery shipped inside the wheel with four image links that resolved to nothing for every
  installed user. It is a documentation page rather than a package note, and shipping the images
  instead would have added 1.3 MB to a 730 KB wheel for four screenshots. `AGENTS.md` now states
  the rule that decides which tree a document belongs in.
- `README.md`'s notebooks section points at `qis/examples/` and names the four factsheet scripts.

### Fixed
- The README's example blocks used `PerfStat` twelve times without importing it, so a reader
  pasting them in order got `NameError` at the performance table. The import is added, and the
  new namespace check above fails if it is removed again. Three fences carrying `pip` and `git`
  commands were labelled `python`; they are labelled `bash`.
- `tools/paper_audit.py --check` exits nonzero on any measurement warning and on any difference
  between the generated and the stored metric key sets. It compared only the metrics it had
  measured, so with `docs/audit/consumers.json` absent it compared 16 of 22 and returned 0.

### Removed
- `notebooks/`, six Jupyter notebooks last touched on 2025-07-19. They were the only documented
  surface with no test covering it: `qis/tests/test_examples.py` checks all 58 example scripts,
  and nothing checked a notebook, so drift in one was invisible - the stored outputs kept
  rendering last year's numbers as embedded images. Every notebook duplicated ground already
  covered by a tested script: `multi_assets.py`, `strategy.py`, `strategy_benchmark.py` and
  `multi_strategy.py` under `qis/examples/factsheets/`, `us_election_regimes.py` under
  `qis/examples/regimes/`, and `quickstart.py` under `qis/examples/perfstats/`. Nothing shipped in
  the wheel changes; the notebooks were never in it.

## [5.3.0] - 2026-07-27

**`qis.__all__` now exists and fixes the public surface at 403 names.** Nothing is added to or
removed from the namespace, but `from qis import *` is now defined by an explicit list rather
than by whatever `dir()` returned at the time, and `dir(qis)` no longer answers the question of
what is public: importing a submodule binds its name on the package, so `dir(qis)` grew by one
whenever a process imported `qis.api`. Anything counting the public surface should read
`qis.__all__`.

The seeded data generator moved from `qis.tests.synthetic_data` to `qis.datasets.synthetic`.
The old path still imports the same module, so nothing breaks, but the quickstart no longer tells
a reader to import from a `tests` namespace. Seeds, draw order and every golden pinned to the
generator are unchanged.

### Added
- `qis.datasets`, re-exporting `generate_synthetic_universe`, `generate_synthetic_prices`,
  `SyntheticUniverseData`, `SyntheticInstrument`, `DataQuirk`, `SYNTHETIC_UNIVERSE`,
  `GROUP_ORDER`, `BENCHMARK_TICKER` and `BENCHMARK_WEIGHTS`. The module ships in the wheel; the
  `qis.tests.synthetic_data` path is kept as a compatibility shim.
- `qis.api.PUBLIC_API`, the export list as a literal, so a change to the public surface appears
  in a diff. `tools/sync_public_api.py` regenerates it and `qis/tests/test_core_api.py` fails
  when it disagrees with the namespace.
- `tools/paper_audit.py` and `docs/audit/paper_numbers.json`: every number `paper.md` quotes,
  generated rather than hand-measured. `qis/tests/test_paper_audit.py` fails when the record, the
  repository and the manuscript disagree, and when the manuscript quotes a large number the
  record does not know about.
- `tools/audit_consumers.py` and `docs/audit/consumers.json`: qis usage in its public consumers
  at pinned commits, with the counting rule stated in the script's docstring.
- `qis/models/bootstrap/tests/test_bootstrap_convention.py`, pinning the published values of the
  bootstrap convention example (0.110, 0.526, +2.15%, -0.32% and the rest) rather than only
  asserting that the example exits zero, and checking that `docs/reproducibility.md` still states
  them.
- `qis/tests/test_documentation.py`, asserting that every repository-internal documentation link
  resolves to a file that exists, and that the README's core dependency list is the `dependencies`
  table of `pyproject.toml`.

### Fixed
- `qis/api.py`'s module docstring stated 386 exports, 98 core symbols, 288 non-core, 109
  private-use and 179 uncalled, and said `market_data` had no core symbol, against a `CORE_API`
  holding 116 symbols in 12 groups including 13 market-data and FX names. Every count is removed
  from the prose; the generated record carries them.
- `README.md` listed `yfinance` and `pandas-datareader` as core dependencies; both are in the
  `[data]` extra.
- `README.md`'s ecosystem table omitted `privateassets`, which `paper.md` names as one of the
  three public consumers.
- `README.md` linked to `qis/examples/performances.py` and `qis/examples/notebooks`, neither of
  which has existed since the examples were reorganised, and embedded `perf1`, `perf2` and
  `perf3`, which `.gitignore` excludes by name, so the front page rendered three broken images.
  The links now point at `qis/examples/perfstats/quickstart.py` and `notebooks/`; the three
  embeds are removed and the runner that produces them is named instead.
- `qis/portfolio/backtester.py` had no final newline, which is why it measured 273 lines by
  `wc -l` and 274 by every other count.

### Changed
- `paper.md`: the state-of-the-field section is a capability comparison at checked versions
  rather than a two-class taxonomy, and adds `skfolio`. All ten rows were read from each
  package's current documentation on 2026-07-27. The impact section quotes consumer counts at
  pinned commits and gives no figure for private repositories; every count is taken from the
  generated record; "cannot drift" is replaced by the invariant the tests enforce; the body is
  1,749 words.

### Removed
- `qis/perfstats/ra_returns.py`, an unreferenced duplicate of `qis/models/linear/ra_returns.py`.
  It was added on 2026-04-19 and never imported: not by `qis/perfstats/__init__.py`, not by any
  module or test, and `qis.perfstats.ra_returns` was absent from `sys.modules` after
  `import qis`. Every `qis.<symbol>` in the pair already resolved to the `models.linear` copy, so
  nothing exported changes and the export count stays at 403. The copies had begun to diverge:
  `compute_ewm_long_short_filtered_ra_returns` gained its docstring and its span validation in
  the live module and not in the duplicate, so a deep import of `qis.perfstats.ra_returns`
  returned a version that accepted `vol_span` below 1 and produced NaN through a negative
  variance. Only a direct import of that module path is affected.

## [5.2.1] - 2026-07-27

**`compute_ar_residuals` and `bootstrap_ar_process` change their results on data with gaps, and
raise where they previously returned a number.** Any AR bootstrap run on a series with a missing
observation moves in this release, and a panel with a row incomplete across columns now raises
instead of returning NaN. On gap-free data nothing moves: the AR(1) is still ordinary least
squares on the lag pairs, agreeing with the previous `statsmodels AutoReg` fit to 6.7e-16 across
persistences from -0.6 to 0.95 and lengths 60 to 2000.

**5.2.0 is yanked and this release replaces it.** It was published from an uncommitted working
tree, so no commit set that version and the release could not be reproduced from source. It also
shipped without the second bounds check below.

### Fixed
- `compute_ar_residuals` raised `KeyError: 0` under pandas 3.0. `AutoReg(...).fit().params` is
  indexed by name, so `params[0]` was a label lookup. `bootstrap_ar_process` went down with it.
- `compute_ar_residuals` returned NaN residuals for data with gaps. The fit used `dropna()` while
  the residuals were computed on the original array, so every missing observation left NaN in two
  residual rows, and those entered every draw that sampled them.
- `compute_ar_residuals` fitted steps that spanned a gap as if they were one period apart,
  because `dropna()` makes the observations either side of a gap adjacent. An AR(1) at spacing k
  has persistence theta^k, so the estimate was pulled towards zero.
- `bootstrap_ar_process` drew indices over `len(data.index)` while the residual array has one row
  fewer, so the largest index was one row past the end. `get_bootstrap_ar_data_list` is `@njit`
  with bounds checking off, so the read did not raise: on a 50-point series it returned 5.6e-321
  from adjacent memory and used it as a residual. The draw is now taken over `len(residuals)`.
- `bootstrap_ar_process` rejects a supplied `bootstrapped_indices` that reaches past the residual
  rows. `bootstrap_price_fundamental_data` draws one index set over `len(prices.index)-1` and
  passes it to both the price path and the AR path, which is what keeps the two resampled
  together. Gaps in the fundamental panel shorten the residuals below that length, so the same
  out-of-bounds read returned through that path. Quarterly fundamental panels are where this
  bites.

### Added
- `qis/models/bootstrap/tests/test_bootstrap_ar.py` — 14 tests over the AR residual path. Each of
  the five defects above has a test that fails on it alone.
- `qis/examples/models/ar_bootstrap_gaps.py` — measures what a gap costs an AR(1) fit. Over 20
  gap patterns blanking 30% of a 3000-point series at persistence 0.7, dropping the lag pairs
  that straddle a gap deviates from the gap-free estimate by 0.001 to 0.026, while collapsing the
  gaps first deviates by 0.064 to 0.095. The ranges do not overlap. No network, and the test
  suite executes it.
- `qis/examples/models/bootstrap_convention.py` — measures what the block-resampling convention
  costs. The superseded truncating sampler draws the first observation of a 250-period sample at
  0.11x its uniform weight and the first decile at 0.53x; applied to a series with rising drift
  it reports a mean return 2.15% per year above the source, against 0.32% for the circular
  sampler now used.
- `docs/reproducibility.md` — the same measurement as a documentation page, with what follows
  from it for the return convention, the Sharpe conventions and the reported frequency.

### Changed
- `compute_ar_residuals` requires each row to be complete across all columns, and raises
  `ValueError` below three usable lag pairs. Rows are resampled jointly to preserve the
  cross-section, so a row missing one column cannot be resampled coherently; previously such a
  row produced NaN silently. A constant series now returns a beta of exactly 0.0 and an intercept
  at the level, where `AutoReg` returned 9.999e-05.
- `qis/models/bootstrap/bootstrap_numba.py` no longer imports `statsmodels`. The AR(1) is
  estimated directly; `statsmodels` remains a dependency, used in eight other modules.

## [5.1.0] - 2026-07-26

**`BootstrapType.STATIONARY` produces different draws in this release.** Blocks now wrap around
the end of the sample, which is the correct Politis-Romano construction; see *Changed* below.
Any result produced with `bootstrap_data`, `bootstrap_price_data`, `bootstrap_ar_process` or
`bootstrap_price_fundamental_data` under `STATIONARY` will move. Nothing else changes an
existing number.

### Added
- `BootstrapType.FIXED_BLOCK` — circular block resampling with a block of exactly `block_size`,
  for a block length chosen to match a known cycle rather than drawn.
- `min_block_size` on `generate_bootstrapped_indices`, `bootstrap_data`, `bootstrap_ar_process`,
  `bootstrap_price_data` and `bootstrap_price_fundamental_data`. Floors the drawn block length
  under `BootstrapType.STATIONARY`; set it to the number of periods in the slowest-reporting
  series when the panel mixes frequencies. Default `1`, which is the previous behaviour.
- `theta` on `unsmooth_returns_glm` — supply the Getmansky-Lo-Makarov smoothing weights instead
  of estimating them, for a coefficient that comes from outside the series (a panel estimate
  pooled across vintages, or a value fixed for a production run). A scalar or an array of
  length q; the sample-length guard does not apply, since nothing is fitted. Default `None`,
  which estimates as before.
- `qis/models/bootstrap/tests/test_bootstrap_numba.py` and
  `qis/models/unsmoothing/tests/test_ar_lag_glm.py`.
- `qis/tests/test_examples.py` — every file in `qis/examples/` must parse, every `qis.<name>`
  it references must resolve, every qis module it imports from must exist with the symbol it
  names, and every keyword it passes to a qis callable must be in that callable's signature.
  The nine examples that reach no data vendor are executed in a temporary directory. An
  optional dependency is a skip, never a failure.
- `qis/api.py` — `CORE_API`, the documented core of the public API: 103 symbols grouped by
  capability. Nothing is un-exported; the module records which exports the documentation
  promises to describe. The boundary is measured — a symbol is core when a package that depends
  on qis, or qis's own examples, README or docs, calls it — plus five bootstrap symbols
  promoted by intent. `qis/tests/test_core_api.py` enforces it: a core symbol without an
  `Args:`/`Attributes:` block fails the suite, and the `PENDING_DOCSTRINGS` backlog is a ratchet
  that cannot silently hide finished work.
- `qis/docs/plotting_kwargs.md` — the keyword arguments every `plot_*` function shares (`ax`,
  `title`, `var_format`, `x_date_freq`, `fontsize`, `colors`, `legend_loc`, `y_limits`),
  documented once so individual plot docstrings cover only what is specific to them.
- Six FX analytics exported from `qis`: `compute_fx_optimal_hedge`, `compute_fx_vol_beta`,
  `compute_performance_of_local_ccy_asset_in_reference_ccy`, `compute_multi_asset_fx_hedging`,
  `run_asset_fx_hedging_report` and `plot_multi_asset_fx_hedging_report`, plus
  `compute_local_and_fx_return` and `compute_cash_fx_adjusted_returns`. `market_data` published
  5 of the 11 symbols its consumers use, and the rest were reached by deep import into
  `qis.market_data.fx_hedging` and `qis.market_data.reports.fx_hedging_report`. Market data is
  now a capability group in `CORE_API`. Roadmap item T5.

### Changed
- **`BootstrapType.STATIONARY` blocks now wrap around the end of the sample**, as in
  Politis-Romano (1994). Previously a block was cut short at the last observation, so the
  realised block length was not geometric there and the first observations were drawn far less
  often than the rest: on a 250-period sample with `block_size=20` the first observation
  appeared at 0.11x the uniform rate and the first decile at 0.53x. **This changes every
  `STATIONARY` draw**, including `rosaa/research/analysis/crypto_publication.py`. The effect on
  a long daily sample is confined to the first `block_size` observations; on quarterly or
  monthly panels it is material.
- `generate_bootstrapped_indices` raises with the offending value on an unhandled
  `bootstrap_type`, rather than a bare `not implemented`.
- The API reference is split into **Core API**, grouped by capability, and **Also exported**.
  It is still generated from `dir(qis)` at build time, so it cannot drift from the exports.
- Docstrings with `Args:`/`Attributes:` blocks on `MeanAdjType`, `NanBackfill`,
  `BootstrapOutput` and on every plot function in the documented core: `plot_time_series`,
  `plot_prices`, `plot_prices_with_dd`, `plot_bars`, `plot_scatter`,
  `plot_classification_scatter`, `plot_heatmap`, `plot_qq`, `plot_df_table`,
  `df_boxplot_by_classification_var`, `df_boxplot_by_hue_var`, `set_suptitle` and
  `plot_exposures_strategy_vs_benchmark_stack`. Each documents only its own arguments and
  refers to `qis/docs/plotting_kwargs.md` for the shared ones. Prose docstrings on `PerfStat`
  and `LegendStats`, whose members are compositional and where a per-member block would restate
  the names.
- **Every symbol in `CORE_API` now carries an `Args:`/`Attributes:` block.** The remaining 56
  in this release: the `df_*` aggregation family, the `file_utils` readers and writers, the
  date and annualisation helpers, `get_group_dict` / `split_df_by_groups`, the three
  `df_to_*_allocation` normalisers, `covar_to_corr`, `np_array_to_df_columns`,
  `fit_multivariate_ols`, `compute_masked_covar_corr`, `estimate_rolling_ewma_covar`,
  `compute_ewm_covar_tensor_vol_norm_returns`, `compute_ewm_long_short_filtered_ra_returns`,
  `estimate_hf_ohlc_vol`, `interpolate_infrequent_returns`, `get_ra_perf_columns`,
  `unsmooth_returns_ar1_ewma`, `bootstrap_data`, `bootstrap_price_data`, `EwmLinearModel`,
  `FxRatesData`, `FactorsData`, the FX conversion functions, `fetch_default_report_kwargs`
  and the two factsheet generators. `fetch_default_report_kwargs` moves from numpydoc to the
  house Google style.
- The six remaining numpydoc docstrings in `qis` are converted to Google style:
  `get_nonnan_index`, `FxRatesData.build_cross_fx_cash_nav`, `estimate_dimson_beta`,
  `fetch_factsheet_config_kwargs`, `fetch_default_perf_params` and the FX rates example.
  `qis/tests/test_docstring_convention.py` fails the suite on a numpydoc section heading
  anywhere in the package. `factorlasso` keeps numpydoc; the exception is per-package.

### Fixed
- `unsmooth_returns_ar1_ewma`, `unsmooth_returns_glm` and `compute_ar1_unsmoothed_prices` are
  exported from `qis`. They were documented and referenced as `qis.<name>` by
  `qis/examples/perfstats/unsmoothing_and_delevering.py`, which raised `AttributeError`.
- `bootstrap_price_data` ignored `min_block_size` instead of forwarding it to the sampler.
- `qis/examples/plots/dual_axis_figure.py` wrote its PDF to `qis.get_output_path()`, which
  reads `settings.yaml` and ships as the placeholder `C:\Users\...\`. The example raised
  `FileNotFoundError` on every machine. It now writes to the working directory, as the other
  examples do.

## [5.0.10] - 2026-07-25

### Added
- `qis/tests/synthetic_data.py` — seeded synthetic multi-asset panel for tests, CI and
  documented examples. `generate_synthetic_prices` and `generate_synthetic_universe` draw a
  10-instrument panel carrying ragged starts, missing observations, stale prices, a delisted
  tail, fat tails, appraisal smoothing and a monthly-reported sleeve, with no network and no
  data file. Internal: not exported from `qis/__init__.py`.
- `qis/plots/tests/plot_smoke_test.py` — every `plot_*` exported from `qis` runs on that panel
  and must draw a figure. The parametrisation is read from `dir(qis)` at collection time, so a
  newly exported `plot_*` without a fixture fails the suite rather than going uncovered.
- CI runs `pytest` on the 3.10–3.14 matrix against a core install, repeats it with the
  `[data,io]` extras, and lints the lines a push or pull request changes with a pinned `ruff`.

### Changed
- `[tool.ruff.lint] select` drops `"I"`. The isort rule contradicts the documented import
  convention, which groups stdlib imports under `# packages` after numpy/pandas, so it failed
  every file written to the house style.

### Fixed
- `plot_prices_2ax` passed `trend_line` into `plot_time_series_2ax`, which takes `trend_line1` /
  `trend_line2`, and the stray keyword reached `plot_time_series` alongside `trend_line1`
  (`TypeError`).
- `plot_regime_pdf` called `_asdict()` on `BenchmarkReturnsQuantilesRegime`, which is a class
  and not a NamedTuple (`AttributeError`), and overwrote the caller's `regime_classifier` on the
  preceding line, so that argument was ignored.
- `plot_vbars` indexed a per-row colour array by column (`IndexError` above four columns) and
  left the y locator to matplotlib, so the label count matched only on a frame with exactly
  eight rows (`ValueError`). It also drew every bar in the wrong place on a `DatetimeIndex`: the
  value labels and total markers address rows by integer position, which `barh` honours only for
  non-numeric labels, so the index is now coerced to strings.
- `plot_multivariate_scatter_with_prediction` dereferenced `ax.get_legend()` unguarded and raised
  whenever `hue` was left at its default, since seaborn draws no legend in that case.
- `pytest` at the repository root collected nothing useful: `testpaths` pointed at a top-level
  `tests/` that does not exist, and three modules failed to import. `testpaths` is now `qis`,
  collection uses `--import-mode=importlib`, the `yfinance` import in
  `qis/tests/price_data_test.py` moved inside the branch that needs it, and the parquet and
  feather tests skip rather than fail without the `[io]` extra.

## [5.0.9] - 2026-07-22

### Added
- `SharpeConvention` is exported from `qis`. `compute_regimes_pa_perf_table` and
  `plot_regime_data` accept the convention, so regime tables state which Sharpe object they
  report. `qis/docs/sharpe_conventions.md` extended with the regime decomposition.

## [5.0.8] - 2026-07-16

### Added
- `SharpeConvention` (`PA`, `ARITHMETIC`, `LOG`) on `PerfParams`, defaulting to `PA` so no
  existing statistic changes value, and `compute_regime_sharpe_decomposition`, which is
  exactly additive in the arithmetic convention.

## [5.0.7] - unreleased tag

[TODO: 5.0.7 is on PyPI but no commit in this repository sets that version. Reconstruct from
the uploaded sdist or yank it.]

## [5.0.6] - unreleased tag

[TODO: 5.0.6 is on PyPI but no commit in this repository sets that version. Reconstruct from
the uploaded sdist or yank it.]

## [5.0.5] - 2026-07-13

### Fixed
- Typo fixes and method visibility corrections across `market_data.fx_rates_data`,
  `models.bootstrap.bootstrap_numba`, `portfolio` and the pybloqs factsheet examples.

## [5.0.4] - 2026-07-12

### Fixed
- `qis/portfolio/reports/config.py` did not import `infer_data_frequency_label`, so
  `import qis` failed in 5.0.3. Hotfix release.

## [5.0.3] - 2026-07-12

### Changed
- `yfinance` and `pandas-datareader` moved out of the core dependencies into a
  new `[data]` extra (`pip install qis[data]`). The analytics core no longer
  installs a data-vendor client. The two function-local imports in
  `qis/portfolio/reports/config.py` (the `^IRX` download behind
  `add_rates_data=True`) raise an `ImportError` naming the extra when it is not
  installed. The `[all]` extra includes `data`, so `pip install qis[all]` is
  unchanged.

## [5.0.2] - 2026-07-12

### Added
- Explicit arithmetic Sharpe convention alongside the p.a. (compound) default.
  New `PerfStat` members `AN_ARITH_RETURN`, `AN_ARITH_EXCESS_RETURN`,
  `AVG_ARITH_RETURN`, `AVG_ARITH_EXCESS_RETURN`, `SHARPE_ARITH` and
  `SHARPE_ARITH_EXCESS` report `a * mean(r_m)` and `a * mean(r_m - rf_m)` on
  simple returns at `freq_vol`, with the compounded and log Sharpe ratios
  (`SHARPE_EXCESS`, `SHARPE_LOG_AN`, `SHARPE_LOG_EXCESS`) unchanged. Every
  Sharpe object is now labeled in the output, so the convention is stated rather
  than implied.
- `qis/docs/sharpe_conventions.md` — decision record deriving the three Sharpe
  objects (compound, log, arithmetic) and their reconciliation, and
  `qis/perfstats/tests/sharpe_conventions_test.py` pinning the identities.

### Changed
- The p.a. (compound) Sharpe remains the qis default. No existing statistic
  changes value; the arithmetic convention is additive and opt-in.

## [5.0.0] - 2026-07-12

Breaking release. The public API is reduced from 568 to 373 symbols and the
`qis.utils.df_agg` aggregators are renamed. There are no deprecation shims:
code that used the removed names must be updated at the same time as the
upgrade. Pin `qis <5` to stay on the previous API.

Every removed symbol is still importable by its defining module. Nothing is
deleted; only the top-level `qis` namespace is reduced. Where a name is
listed as removed below, the migration is:

```python
qis.set_spines(ax)                                # 4.x
from qis.plots.utils import set_spines            # 5.0
set_spines(ax)
```

### Breaking: renamed `qis.utils.df_agg` aggregators

`qis.nanmean`, `qis.nanmedian` and `qis.nansum` shadowed the numpy names of
the same spelling while carrying different semantics: they consume a
`pd.DataFrame`, return a `pd.Series`, exclude non-finite entries (`+-inf` is
mapped to nan and skipped), and default to `axis=1`, which is the opposite of
the pandas default. In a module importing both `numpy as np` and `qis`, the
name collision was a trap rather than a convenience. The whole module is
renamed for consistency.

| 4.x | 5.0 |
| --- | --- |
| `qis.nanmean` | `qis.df_nanmean` |
| `qis.nanmedian` | `qis.df_nanmedian` |
| `qis.nansum` | `qis.df_nansum` |
| `qis.nanmean_positive` | `qis.df_nanmean_positive` |
| `qis.nansum_positive` | `qis.df_nansum_positive` |
| `qis.nansum_negative` | `qis.df_nansum_negative` |
| `qis.nanmean_clip` | `qis.df_nanmean_clip` |
| `qis.nansum_clip` | `qis.df_nansum_clip` |
| `qis.nanmean_weighted` | `qis.df_nanmean_weighted` |
| `qis.abssum` | `qis.df_abssum` |
| `qis.abssum_positive` | `qis.df_abssum_positive` |
| `qis.abssum_negative` | `qis.df_abssum_negative` |
| `qis.last_row` | `qis.df_last_row` |
| `qis.sum_weighted` | `qis.series_nansum_weighted` |
| `qis.get_signed_np_data` | `qis.utils.df_agg._get_signed_np_data` (now private) |

`sum_weighted` is renamed rather than prefixed with `df_` because it takes two
`pd.Series` and returns a `float`; its first parameter was also named `df`,
and is now `data`.

### Breaking: reduced public namespace

| subpackage | 4.3.x | 5.0 |
| --- | --- | --- |
| `qis.utils` | 189 | 59 |
| `qis.plots` | 131 | 73 |
| `qis.perfstats` | 75 | 65 |
| `qis.models` | 92 | 92 |
| `qis.portfolio` | 49 | 49 |
| `qis.file_utils` | 27 | 27 |
| **total** | **568** | **373** |

The removed symbols are internal machinery that was published by accident: the
top-level namespace was assembled by `import *` over the subpackages, so
anything a module happened to define became part of the API. Analytics
(`qis.models`, `qis.portfolio`, `qis.perfstats`) is unchanged apart from four
enums, because those are the functions a user of the library calls.

`qis.utils` (130 removed) — numpy helpers (`np_nansum`, `np_shift`,
`repeat_by_rows`, `running_mean`, `to_finite_np`), DataFrame plumbing
(`df_zero_like`, `df_ones_like`, `align_df1_to_df2`, `dfs_to_upper_lower_diag`),
string formatting (`float_to_str`, `str_to_float`, `df_to_numeric`,
`series_to_str`, `date_to_str`), list and dict helpers (`flatten`, `list_diff`,
`list_intersection`, `split_dict`), and date helpers (`is_leap_year`,
`get_weekday`, `months_between`, `min_timestamp`). What survives is the API
proper: `TimePeriod`, `generate_dates_schedule`,
`generate_rebalancing_indicators`, the `df_agg` aggregators, `df_asfreq`,
`get_group_dict`, `split_df_by_groups`, `ColVar`, `ColumnData`, `EnumMap`,
`ValueType`, `update_kwargs`, `covar_to_corr`, `fit_multivariate_ols` and the
annualisation factors.

`qis.plots` (58 removed) — matplotlib axis and legend plumbing (`set_spines`,
`remove_spines`, `set_ax_tick_params`, `set_legend`, `set_title`,
`align_y_limits_axs`, `autolabel`, `rand_cmap`, `subplot_border`), the colour
palette accessors (`get_n_colors`, `get_n_sns_colors`, `get_cmap_colors`), the
table-styling setters in `qis.plots.table` (`set_cells_facecolor`,
`set_row_edge_color`, `set_data_colors`), and five table-computation helpers.
All 63 `plot_*` functions remain public, as do `TrendLine`, `LastLabel`,
`LegendStats` and `PdfType`, which appear in their signatures. `set_suptitle`
remains public.

`qis.perfstats` (10 removed) — the 14 `*_TABLE_COLUMNS` constants, the
`cond_regression` entry points, and the DataFrame operations listed under
*Moved* below.

### Added

- `qis.factsheet` — one-call facade over the four factsheet generators
  (`qis.portfolio.reports.factsheet_facade`). It picks the report archetype
  from the input type, calibrates windows / regressions / regimes /
  annualisation for the requested reporting frequency via
  `fetch_default_report_kwargs`, renders, and optionally writes a PDF. The
  four generators remain available and unchanged for full control. All qis
  imports are deferred into the function bodies, so the module never depends
  on `qis` being fully initialised.
- `qis.df_nanmean_negative` in `qis.utils.df_agg`, completing the
  sum / mean by positive / negative grid. `df_nansum_negative`,
  `df_nansum_positive` and `df_nanmean_positive` already existed;
  the mean of negative entries did not.
- `axis: Literal[0, 1] = 1` argument on `df_nansum_clip`, `df_nanmean_clip`,
  `df_abssum`, `df_abssum_positive`, `df_abssum_negative` and
  `agg_median_mad`. All six hardcoded `axis=1` and could not aggregate along
  the other axis.
- `__all__` in `qis/plots/utils.py`, declaring `TrendLine`, `LastLabel`,
  `LegendStats` and `set_suptitle` as the public surface of that module.

### Changed

- Library modules no longer reach through the `qis` namespace. `qis` imported
  itself — `qis/portfolio/reports/strategy_factsheet.py` called
  `qis.set_spines(...)`, `qis/plots/scatter.py` called `qp.get_n_sns_colors(...)`
  through `import qis.plots as qp`, and `qis/portfolio/backtester.py` called
  `qu.repeat_by_rows(...)` through `import qis.utils as qu`. The top-level
  namespace was therefore not an API decision but an internal calling
  convention that `import *` published. All 68 such call sites across 12 files
  now import from the defining module.
- `qis.utils.df_agg` aggregators share one `_to_agg_series()` helper that
  selects the index from the aggregated axis, replacing the repeated
  `if axis == 0 / else` blocks. `_validate_axis()` rejects values outside
  `{0, 1}`, which numpy would otherwise accept silently (`axis=-1`).
- `compute_df_desc_data` no longer takes a mutable default argument
  (`funcs: Dict = {...}` is now `Optional[Dict] = None`).
- `qis.plots.reports.econ_data_single` is deprecated and emits a
  `DeprecationWarning` on import. `econ_data_report` and `ReportType` are no
  longer exported. Scheduled for removal in 6.0.
- `qis.plots.derived.gantt_data_history` is not imported by
  `qis/plots/__init__.py`. It requires plotly, which is not a qis dependency;
  import it by full path if plotly is installed.

### Moved

Public names are unchanged unless listed under *Breaking* above. Only code
importing these by file path must update.

- `qis/plots/reports/price_history.py` -> `qis/plots/derived/price_history.py`.
- `qis/plots/reports/gantt_data_history.py` -> `qis/plots/derived/gantt_data_history.py`.
- `df_price_ffill_between_nans`, `df_ffill_negatives`,
  `df_fill_first_nan_by_cross_median`, `df_price_fill_first_nan_by_cross_median`
  and `replace_nan_by_median` from `qis.perfstats` to `qis.utils.df_ops`. These
  are pure DataFrame operations and compute nothing about performance.
- `compute_futures_fx_adjusted_returns` and `get_aligned_fx_spots` from
  `qis.perfstats.fx_ops` to `qis.market_data.fx_hedging`, consolidating FX
  handling with `FxRatesData`.
- `get_output_path`, `get_paths` and `get_resource_path` from `qis.file_utils`
  to `qis.local_path`.

### Fixed

- `nansum_negative(df, axis=0)` raised
  `ValueError: Length of values (3) does not match length of index (4)`. The
  function passed `axis` to `np.nansum` but hardcoded `index=df.index`, so the
  `axis=0` result (one entry per column) was given the row index. On a square
  frame it returned the correct numbers under the wrong labels, silently. Now
  `df_nansum_negative`, and correct on both axes.
- `agg_data_by_axis(df, axis=1)` mislabelled its result. It always used
  `index=df.columns`, contradicting its own docstring, so an `axis=1`
  aggregation (one entry per row) carried column labels.
- `qis.compute_desc_table` and `qis.DescTableType` resolved to different
  modules. `compute_desc_table` is defined in both `qis.perfstats.desc_table`
  and `qis.plots.derived.desc_table`, and `DescTableType` in both as well.
  `qis/__init__.py` imported perfstats before plots, so the top-level namespace
  bound `compute_desc_table` from plots and `DescTableType` from perfstats —
  two different Enum classes, for which `==` returns `False`. The plots export
  is removed and the pair now resolves consistently to `qis.perfstats`.
- Three names were exported from two modules each and silently shadowed by
  whichever import ran last: `compute_desc_table` (above),
  `add_bnb_regime_shadows` (`plots.derived.prices` and
  `plots.derived.regime_data`) and `separate_number_from_string`
  (`utils.dates` and `utils.struct_ops`). All deduplicated.
- `nanmean_positive` and `nanmean_negative` leaked
  `RuntimeWarning: Mean of empty slice` when a line contained no entries of the
  requested sign. `nan` is the intended result; the warning is now suppressed
  at the call to `np.nanmean` / `np.nanmedian` rather than propagated to the
  caller.
- Continuation-line alignment in
  `qis/portfolio/reports/overlays_smart_diversification.py`, where wrapped
  keyword arguments were indented 16 columns past the opening parenthesis.

### Removed

- All 15 deprecated `df_agg` aliases. This release is a hard break; there is no
  4.x compatibility layer.
- `qis.examples` exports (`load_usd_assets`, `generate_performance_report`,
  `DEFAULT_RA_TABLE_COLUMNS`) from the public namespace. Examples are
  documentation, not API.

### Migration

For downstream code, the mechanical steps are:

1. Rename the `df_agg` calls per the table above. `qis.nanmean_weighted` is the
   most commonly used and becomes `qis.df_nanmean_weighted`.
2. For any `AttributeError: module 'qis' has no attribute X`, import `X` from
   its defining module. `python -c "import qis.plots.utils as m; print(m.X)"`
   locates it; the module list is in `docs/REMOVED_5_0.md`.
3. Do not import from `qis/examples/` — it is documentation and is
   restructured without notice.

## [4.3.4] - 2026-07-11

Never tagged; ships to users as part of 5.0.3.

### Added
- `qis.perfstats.signal_diagnostics` — cross-sectional predictive regression
  diagnostics for trading signals with per-asset native-cadence handling. For an
  N-asset panel of signal scores and per-frequency return panels,
  `estimate_signal_diagnostics` quantifies cross-sectional predictive content at
  one or more forward horizons via `y_{i,t,t+h} = beta * z_{i,t-1} + eps_{i,t}`
  (through the origin by default), with the horizon expressed in each asset's
  native periods. `compute_per_asset_betas`, `compute_ic_timeseries` and
  `estimate_ic_ir` expose the per-asset betas, the per-date information
  coefficient series and its IC-IR.
- `qis.min_obs_for_ar_unsmoothing` — minimum observation count required for an
  AR(q) unsmoothing fit given `ar_order` and `warmup_period`, validating both
  (`ValueError` naming the offending value).

### Changed
- AR unsmoothing (`qis.models.unsmoothing.ar_lag`) gains opt-in guards for
  short / degenerate columns, selected by enum. Defaults reproduce the previous
  behaviour exactly; `RAISE` reports the offending columns and their observation
  counts instead of silently returning an all-NaN column.

## [4.3.2] - 2026-06-28

### Added
- `qis.estimate_dimson_beta` in `qis.models.unsmoothing.dimson_beta` —
  Dimson (1979) aggregated-coefficient beta to detect return smoothing.
  Regresses each asset on the contemporaneous and lagged market return and
  reports `beta_dimson = sum_k b_k`; the `beta_dimson / b_0` ratio measures
  the contemporaneous understatement and the t-stat on the summed lagged
  slopes tests whether the lag effect is real. Pure numpy/pandas, importable
  standalone.
- `qis.adjust_returns_with_factor_lag` in `qis.models.unsmoothing.factor_lag`
  — factor-lag (Dimson) unsmoothing for illiquid / appraisal-based series.
  Companion to the own-lag AR(q) engine; removes smoothing that manifests as
  a lagged response to a liquid factor, which the own-lag AR cannot see (a
  fund-of-funds with near-zero own autocorrelation but a real lagged-equity
  beta). The correction is mean-preserving and lifts the contemporaneous
  loading to `beta_D`, so a plain contemporaneous regression recovers the
  true loading and the existing HCGL / factor-covariance estimator picks it
  up with no change.
- `qis.adjust_returns_with_joint_unsmoothing` in
  `qis.models.unsmoothing.joint_lag` — single-regression joint own-lag +
  factor-lag unsmoothing, fitting the own-lag coefficient and the
  lagged-factor beta jointly via the rolling EWMA cross-moment estimator.
  Removes the omitted-variable bias and stage-order dependence of running the
  AR engine and the factor-lag engine sequentially.
- Week-of-month / last-week-of-month anchored frequencies (`WOM-*`, `LWOM-*`)
  now resolve to a monthly (12.0) annualisation factor in
  `qis.utils.annualisation`, handled explicitly because the generic frequency
  regex cannot parse the week number in the anchor.

### Changed
- Reorganised unsmoothing into a `qis.models.unsmoothing` subpackage. The
  former `qis/models/unsmoothing.py` (own-lag AR(q) engine,
  `adjust_returns_with_ar`) is now `qis/models/unsmoothing/ar_lag.py`,
  alongside `dimson_beta.py`, `factor_lag.py`, `joint_lag.py` and a `tests/`
  directory. Package-level imports (`qis.adjust_returns_with_ar`, etc.) are
  preserved; only code importing the old module by file path must update.
- `multi_assets_factsheet` regime-Sharpe plotting accepts an optional
  `regime_classifier` argument, falling back to the instance default for a
  per-plot override.

### Fixed
- `RegimeClassifier` degenerate-benchmark guard. A constant / zero-return
  block (e.g. an overlay nav with longer history than the principal,
  back-padded over the union index) collapses interior quantiles, which
  previously surfaced as a bare pandas `Bin edges must be unique`. The new
  check mirrors `pd.qcut` exactly (unique edges <= number of labels), so it
  fires iff qcut would have failed and never on healthy data, and raises a
  descriptive error naming the benchmark, the number of non-empty bands, and
  the remedy (clip inputs to their common live window).
- `qis.plots.lineplot` marker indexing is now cyclic and None-safe
  (`markers[idx % len(markers)] if markers else None`), fixing an IndexError
  when the number of lines exceeds the number of supplied markers.

### Removed
- Internal `qis/market_data/MIGRATION_NOTES.md` scratch file; trimmed the
  `fx_hedging_example.py` example.

## [4.3.0] - 2026-06-19

### Added
- Python 3.14 support.
- `qis.delever_returns`, `qis.lever_returns`, `qis.implied_leverage` in
  `qis.perfstats.returns` for working with levered / unlevered return
  series given leverage and financing rate.
- `qis.unsmooth_returns_ar1_ewma`, `qis.unsmooth_returns_glm`, and
  `qis.compute_ar1_unsmoothed_prices` in `qis.perfstats.unsmoothing` for
  AR(1) EWMA and AR(q) Getmansky-Lo-Makarov unsmoothing of appraisal-based
  NAV series, with severity diagnostics.
- `qis.to_quarterly_returns` in `qis.perfstats.returns` for compounding
  daily / weekly / monthly returns to quarter-end with partial-quarter
  masking.
- Vectorised `qis.compute_risk_table`.
- Reorganised `qis/examples/` into themed sub-packages (`perfstats/`,
  `models/`, `regimes/`, `portfolios/`, `factsheets/`, `plots/`, `utils/`,
  `case_studies/`, `_helpers/`) with a per-folder `README.md` and a
  module-level docstring on every example file.
- New example `qis/examples/perfstats/unsmoothing_and_delevering.py` —
  end-to-end walkthrough of the leverage / unsmoothing functions on a
  bundled OCSL / Oaktree GCF / SPX / US HY / US Agg weekly NAV dataset.
- New example `qis/examples/models/multivariate_ols.py` demonstrating
  `qis.fit_multivariate_ols` directly (separated from the EWM linear-model
  example).
- `bbg-fetch >=2.0.0` listed as optional dependency for examples that
  pull data from a Bloomberg terminal.

### Changed
- Bumped minimum Python from 3.9 to 3.10. (numba 0.61 dropped Python 3.9
  support, and the bump to numba ≥0.63 for Python 3.14 forces the same
  floor here.)
- Bumped minimum numba from 0.60.0 to 0.63.0 (required for Python 3.14
  support; see numba 0.63.0 release notes, Dec 2025).
- Renamed several example files for clarity:
  - `models/ewm_filters.py` → `models/ewm_kernels.py`
  - `models/correlation_matrix.py` → `models/ewm_correlation_table.py`
  - `models/ewma_factor_betas.py` → `models/ewm_linear_model.py`
  - `portfolios/btc_marginal_contribution.py` → `portfolios/balanced_60_40_with_btc.py`
  - `perfstats/perf_excluding_best_worst_days.py` → `perfstats/miss_best_worst_days_impact.py`
- Moved `infrequent_returns_interpolation.py` from `examples/utils/` to
  `examples/perfstats/` (matches the API location:
  `qis.perfstats.timeseries_bfill`).
- `qis.adjust_navs_to_portfolio_pa` renamed to
  `qis.adjust_component_navs_to_portfolio`; the `asset_prices` parameter
  renamed to `component_navs`. The function decomposes a portfolio's
  PA return into its *additive components* (carry types, fundamental
  return sources, gross vs net vs costs), not into asset-level NAVs.
  The original names were misleading. The formula is unchanged: the
  function rescales component NAVs by a time-weighted factor so their
  PA returns sum to the portfolio PA return — useful for stacked-area
  visualisation of return decomposition. Docstring rewritten to
  document the actual invariant.
- `qis.to_portfolio_returns` and `qis.portfolio_returns_to_nav` docstrings
  now explicitly document the NaN convention: a NaN return contributes
  `0` to that period's portfolio PnL (interpreted as "asset held its
  notional but earned 0%"), rather than renormalising the remaining
  weights. Correct convention if NaN means "asset wasn't tradable, held
  cash"; wrong if NaN means "data missing, treat position as continuous".
  No code change; convention was previously undocumented.
- `qis.compute_net_return_ex_perf_man_fees` HWM crystallization block has
  an explanatory comment clarifying the GAV-after-CPF subtraction and
  the resulting audit-trail discontinuity. The numerical output is
  unchanged.
- `qis.utils.df_freq.df_asfreq` explicit-NaN-on-target-date bug. When the
  input DataFrame contained a NaN value on a date that coincided with a
  resample target timestamp (e.g. yfinance returning NaN on the US
  Independence Day Friday that was also the `W-FRI` bucket end),
  `df.reindex(index=freq_index, method='ffill')` returned NaN for that
  bucket — pandas' `reindex(method='ffill')` looks back through input
  *index labels*, not values, so it found the holiday Friday label
  directly and copied its NaN value. The post-reindex ffill could not
  recover the value because nothing earlier in the resampled output
  existed to fill from. Fix is a single pre-reindex `_apply_fill(df, ...)`
  call so the daily series has its NaNs filled before the reindex picks
  bucket anchors, matching `df.resample(freq).last()` on a ffilled series.
  Reported by Ben Richards.
- `qis.to_quarterly_returns` calendar-QE boundary bug. The previous
  implementation used `returns.reindex(q_returns.index).notna()` to detect
  partial trailing quarters, which silently masked the entire output for
  any input whose timestamps did not land on calendar quarter-end dates
  (W-FRI weekly, business-month-end series). The new implementation uses
  a calendar-month coverage check per column: a quarter ending at QE is
  complete iff the input's last non-NaN observation falls in the same
  calendar month as QE.

### Fixed
- `qis.compute_total_return` trailing-NaN handling. Previously the function
  used `prices.iloc[-1]` for the end value, so any series with NaN at the
  end (terminated fund, delisted ETF) silently returned NaN total return.
  Now mirrors the existing leading-NaN treatment via
  `get_last_nonnan_values`, with a matching warning. Fix propagates
  through to `compute_pa_return`, `compute_returns_dict`, and Sharpe /
  alpha computations downstream.
- `qis.compute_excess_returns` look-ahead bias. The function used
  `lag=None` for `multiply_df_by_dt`, applying today's risk-free rate to
  today's funding cost — a small contemporaneous-rate look-ahead.
  `get_excess_returns_nav` already used `lag=1`. Now both functions agree
  on lag=1 (funding cost at t uses the rate set at t-1).
- `qis.prices_at_freq` `ffill_nans=False` ignored when `freq is None`.
  Previously the no-freq branch gated only on `fill_na_method` (default
  `'ffill'`), so callers passing `ffill_nans=False` without also
  overriding `fill_na_method` got ffilled prices anyway — opposite of
  what the parameter name promised. Now `ffill_nans=False` disables fill
  in both branches consistently.
- `qis.df_price_ffill_between_nans` ignored its `method` parameter. The
  body hardcoded `.ffill()` regardless of input, so callers passing
  `method='bfill'` got silent ffill behaviour. Now `method` dispatches
  correctly to ffill / bfill / None.
- `qis.compute_pa_return` returned a 0-d scalar `array(0)` instead of a
  vector of zeros for DataFrame input when `num_years <= 0` (degenerate
  input). `np.zeros_like(n)` where `n` is an `int` returns a 0-d array;
  replaced with `np.zeros(n)`.
- `qis.to_zero_first_nonnan_returns` removed always-true defensive check
  in the `init_period=1` branch. Since `first_date = returns.index[0]`
  and any non-NaN index is by definition >= the first index, the guard
  was dead code. Behaviour is unchanged; code is simpler.

## [2.0.1] - 2023-07-08

### Removed
- `qis.portfolio.optimisation` layer, with core functionality moved to a
  stand-alone Python package
  [bop (Backtesting Optimal Portfolio)](https://pypi.org/project/bop/).
  Removes the cvxpy and sklearn dependencies.

### Added
- Factsheet reporting via [pybloqs](https://github.com/man-group/PyBloqs).
- Four factsheet types with examples in `qis.examples.factsheets`:
  - `multi_asset` — cross-sectional comparison
  - `strategy` — performance / risk / trading stats from `PortfolioData`
  - `strategy_benchmark` — strategy vs benchmark
  - `multi_strategy` — parameter sensitivity sweeps

## [1.0.1] - 2022-12-30

Initial public release.

---

Versions between 1.0.1 ↔ 2.0.0 and 2.0.2 onwards (prior to the next
release) have not been backfilled. Run `git log --tags --oneline` for
release-by-release commit history.
