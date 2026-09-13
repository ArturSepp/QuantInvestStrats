"""Documentation-only layouts; never change qis plotting defaults or calculated data."""

from contextlib import contextmanager
from pathlib import Path

TEXT = '#243746'
MUTED = '#526675'
GRID = '#DCE3E8'


@contextmanager
def render_context():
    """Use a bundled font and restore rcParams after rendering."""
    import matplotlib
    with matplotlib.rc_context({
        'font.family': 'DejaVu Sans', 'text.usetex': False,
        'savefig.dpi': 150, 'figure.dpi': 100,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    }):
        yield


def save_figure(figure, path: Path) -> None:
    """Save a PNG without timestamp metadata."""
    figure.savefig(path, dpi=150, bbox_inches='tight',
                   metadata={'Software': 'qis docs analytics'})


def _frame(figure, title, subtitle, footer):
    """Apply common type and frame styling without changing artist data."""
    from matplotlib.text import Text

    figure.set_layout_engine(None)
    figure.set_facecolor('white')
    for text in list(figure.texts):
        text.remove()
    for ax in figure.axes:
        ax.set_facecolor('white')
        ax.set_title('')
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.spines[['top', 'right']].set_visible(False)
        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_color(GRID)
        for text in ax.findobj(Text):
            text.set_fontsize(12)
            text.set_color(TEXT)
    figure.text(0.09, 0.965, title, fontsize=19, weight='bold', color=TEXT, va='top')
    figure.text(0.09, 0.918, subtitle, fontsize=11.5, color=MUTED, va='top')
    figure.text(0.09, 0.025, footer, fontsize=10.5, color=MUTED, va='bottom')


def _short_legend(ax, *, columns=2, below=False):
    """Keep series identifiers; move calculation conventions into the caption."""
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if handles:
        labels = [label.split(':')[0].split(',')[0].strip() for label in labels]
        options = {'loc': 'upper left', 'ncol': columns, 'fontsize': 11,
                   'frameon': True, 'facecolor': 'white', 'framealpha': 0.9}
        if below:
            options.update(loc='upper center', bbox_to_anchor=(0.5, -0.19))
        ax.legend(handles, labels, **options)


def _compact_table(ax, labels=None):
    """Reflow the first six existing table columns, preserving the displayed values."""
    if len(ax.tables) != 1:
        raise ValueError('Expected one performance table in the selected panel')
    original = ax.tables[0]
    cells = original.get_celld()
    columns = sorted({column for _, column in cells})
    if columns[:6] != list(range(6)):
        raise ValueError('Unexpected performance-table columns')
    headers = [cells[0, column].get_text().get_text() for column in range(6)]
    normalised = [' '.join(value.split()).lower() for value in headers]
    if not (normalised[1:4] == ['total return', 'p.a. return', 'an. vol']
            and 'sharpe' in normalised[4] and 'max' in normalised[5]):
        raise ValueError(f'Unexpected performance-table meanings: {headers}')
    rows = sorted({row for row, _ in cells if row > 0})
    if labels is not None:
        rows = [row for row in rows if cells[row, 0].get_text().get_text() in labels]
        if {cells[row, 0].get_text().get_text() for row in rows} != set(labels):
            raise ValueError('Missing requested portfolio in the performance table')
    values = [[cells[row, column].get_text().get_text() for column in range(6)]
              for row in rows]
    if not values:
        raise ValueError('Empty performance preview')
    original.remove()
    for line in list(ax.lines):
        line.remove()  # separator rules belonged to the original table's row layout
    ax.set_axis_off()
    table = ax.table(
        cellText=values,
        colLabels=['Series', 'Total return', 'Ann. return', 'Ann. vol.',
                   'Sharpe (PA)', 'Max DD (ME)'],
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15], bbox=(0, 0, 1, 1),
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor(TEXT if row == 0 else ('#EDF3F7' if row % 2 else '#F7F9FB'))
        cell.get_text().set_color('white' if row == 0 else TEXT)
        if row == 0:
            cell.get_text().set_weight('bold')
    return values


def gallery_preview(figure, *, title, detail, comparison_labels=None):
    """Rearrange four report panels without recomputing their statistics."""
    import numpy as np
    from matplotlib import dates
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    def panel(fragment):
        matches = [ax for ax in figure.axes if fragment in ax.get_title()]
        if not matches:
            raise ValueError(f'Missing factsheet panel: {fragment}')
        return matches[0]

    nav = panel('Cumulative performance')
    table = panel('RA performance table')
    drawdown = panel('Running Drawdowns')
    extra = panel(detail)
    selected = [nav, table, drawdown, extra]
    if len(set(selected)) != 4:
        raise ValueError('Preview panels must be distinct')
    line_data = [(line, np.array(line.get_xdata(), copy=True),
                  np.array(line.get_ydata(), copy=True))
                 for ax in selected for line in ax.lines]
    correlation_colours = [(text, text.get_color()) for text in extra.texts]
    for ax in list(figure.axes):
        if ax not in selected:
            figure.delaxes(ax)
    figure.set_size_inches(10, 9.5)
    _frame(
        figure, title, 'Synthetic data | 2 Jan 2018–31 Dec 2025 | selected report panels',
        'Monthly statistics: log-return volatility; compounded-return Sharpe, rf = 0.\n'
        'Running drawdowns use daily observations. Portfolio NAVs include 10 bp trading costs.',
    )
    for ax, position in zip(selected, [
        (0.09, 0.64, 0.86, 0.195), (0.09, 0.415, 0.86, 0.145),
        (0.09, 0.19, 0.39, 0.17), (0.59, 0.19, 0.36, 0.17),
    ]):
        ax.set_position(position)
    _compact_table(table, labels=comparison_labels)
    titles = ['Cumulative performance', 'Monthly performance | full sample',
              'Daily running drawdown',
              ('Monthly correlation' if 'Correlation' in detail else
               'Monthly instrument weights' if 'Exposures' in detail else
               'Rolling 12-month turnover')]
    for ax, caption in zip(selected, titles):
        ax.set_title(caption, loc='left', fontsize=12.5, color=TEXT, pad=12)
    for ax in (nav, drawdown):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    _short_legend(nav, columns=3)
    if drawdown.get_legend() is not None:
        drawdown.get_legend().remove()  # same series and colours as the NAV panel
    for ax in (nav, drawdown, extra):
        if 'Correlation' in detail and ax is extra:
            ax.xaxis.tick_bottom()
            ax.tick_params(axis='x', labelsize=10.5, rotation=45)
            ax.tick_params(axis='y', labelsize=11, rotation=0)
            for text, colour in correlation_colours:
                text.set_color(colour)
            continue
        if 'Exposures' in detail and ax is extra:
            # Area charts use observation positions, not Matplotlib date numbers.
            minor = not len(ax.get_xticklabels())
            labelled = [(tick, label.get_text()) for tick, label in zip(
                ax.get_xticks(minor=minor), ax.get_xticklabels(minor=minor))
                if label.get_text().strip()][::2]
            if not labelled:
                raise ValueError('Exposure preview has no labelled observation dates')
            ticks, labels = zip(*labelled)
            ax.set_xticks([], minor=True)
            ax.set_xticks(ticks, labels=labels)
        else:
            for patch in list(ax.patches):
                patch.remove()  # omit regime shading in the focused line-chart previews
            ax.xaxis.set_major_locator(dates.YearLocator(2))
            ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
        ax.tick_params(axis='x', labelrotation=0, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlabel('')
    if 'Correlation' not in detail:
        _short_legend(extra, below='Exposures' in detail)
    for line, x_values, y_values in line_data:
        np.testing.assert_array_equal(line.get_xdata(), x_values)
        np.testing.assert_array_equal(line.get_ydata(), y_values)
    return figure


def model_exhibit(figure, *, title, subtitle, footer):
    """Use the same frame and readable type for the existing model-layer plots."""
    figure.set_size_inches(10, 7.5)
    _frame(figure, title, subtitle, footer)
    figure.axes[0].set_position((0.10, 0.235, 0.86, 0.57))
    figure.axes[0].tick_params(axis='x', labelrotation=0)
    return figure
