# Shared plotting arguments

Every `plot_*` function in qis draws onto a matplotlib axis and takes the same set of arguments
for the things that are common to all plots — where to draw, what to title it, how to format
numbers. Those are documented here once. A plot function's own docstring covers only what is
specific to it, and points here for the rest.

The intent is that these mean the same thing everywhere. If a function interprets one of them
differently, that is a defect to report, not a local convention.

## Placement

`ax: plt.Subplot = None`

The axis to draw on. When `None` the function creates its own figure and axis, which is what a
one-off call in a script wants. Pass an axis when composing a multi-panel figure:

```python
fig, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
qis.plot_prices(prices=prices, ax=axs[0])
qis.plot_prices_with_dd(prices=prices, ax=axs[1])
```

Every plot function returns the `plt.Figure` it drew on, so the return value is the figure to
pass to `qis.save_fig` or `qis.save_figs_to_pdf` whether or not you supplied the axis.

## Labelling

`title: str = None` — the axis title. `None` draws no title.

`xlabel: str = None`, `ylabel: str = None` — axis labels. `None` leaves the axis unlabelled
rather than falling back to the column name.

`fontsize: int = 12` — base font size. Tick labels, legend and title scale from it, so this is
the one dial to turn when a figure is going into a slide rather than a page.

`legend_loc: Optional[str] = 'upper left'` — passed to matplotlib. `None` suppresses the legend
entirely, which is what you want when several panels share one.

## Number formatting

`var_format: Optional[str] = '{:,.2f}'` — a Python format string applied to the values in the
legend, the tick labels and any data labels. This is where percent, ratio and price displays are
chosen: `'{:.0%}'` for a percentage, `'{:,.2f}'` for a ratio, `'{:,.0f}'` for a level. `None`
leaves axis ticks to Matplotlib and uses native scalar text for statistic legends, including
descriptive tables where supported.

`xvar_format`, `yvar_format` — the same, per axis, on plots where the two axes carry different
units. A scatter of return against volatility wants `xvar_format='{:.0%}'` and
`yvar_format='{:.2f}'`.

Formatting is deliberately not inferred from the data. A number's units are a property of what
was computed, not of its magnitude, so the caller states them.

## Dates

`x_date_freq: str = 'YE'` — the tick frequency on a time axis, as a pandas offset alias:
`'YE'` year-end, `'QE'` quarter-end, `'ME'` month-end, `'W-WED'` weekly on Wednesdays. `None`
lets matplotlib choose, which is usually worse on a long sample.

Tick *labels* follow the frequency: an annual frequency prints years, a monthly one prints
`Mmm-YY`. Changing the frequency therefore changes the label format too, and that is intended.

## Colour

`colors: List[str] = None` — one colour per series, in column order. `None` takes the default
palette, `qis.get_n_colors(n)`, which is stable for a given `n` so that the same universe gets
the same colours across every panel of a factsheet.

## Limits

`y_limits: Tuple[Optional[float], Optional[float]] = None` — `(low, high)`, either of which may
be `None` to leave that end automatic. Use it to hold several panels on one scale.

`markersize: int` — marker size on scatter and line plots that draw points.

## `**kwargs`

Most plot functions end in `**kwargs` and forward it to the helpers they call —
`qis.get_legend_lines`, the table renderers, and matplotlib itself. Two consequences worth
knowing:

- A keyword these functions do not recognise is **silently ignored** rather than raising. A
  misspelled argument name therefore produces a figure that looks nearly right, which is how
  `trend_line` went unnoticed against `plot_time_series_2ax`'s `trend_line1` / `trend_line2`.
  When a plot ignores something you passed, check the spelling against the signature first.
- Arguments documented here can be passed through a wrapper without being named at each level,
  which is how the factsheet generators pass one `fetch_default_report_kwargs` dict down to
  every panel.
