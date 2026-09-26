"""``plot_top_drawdowns_paths`` highlights the ongoing episode and labels its axis truthfully.

Two defects are covered:

- ``highlight_ongoing`` compared each episode's end with the penultimate date. An unrecovered
  episode ends on the final date, so it was never highlighted, and an episode that recovered on
  the penultimate date was highlighted instead. The ongoing episode is now the one flagged
  ``is_recovered=False`` by the episode table.
- The episode table was always computed on calendar days ('D') whatever ``freq`` the paths were
  drawn on, and the x-axis was labelled 'Days in drawdown' although it counts observations of the
  plotted grid. The table now uses the same ``freq``, and the axis says 'Days in drawdown' only on
  the calendar-day grid.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# qis
from qis.plots.derived.drawdowns import plot_top_drawdowns_paths  # noqa: E402


def _legend_texts(fig: plt.Figure) -> list:
    """Legend entries of the single axis."""
    return [text.get_text() for text in fig.axes[0].get_legend().get_texts()]


def _daily(levels: list) -> pd.Series:
    """Levels on consecutive calendar days from 1 January 2024."""
    return pd.Series(levels, index=pd.date_range('2024-01-01', periods=len(levels), freq='D'),
                     name='nav')


def test_unrecovered_episode_is_highlighted() -> None:
    """The episode still under water at the last date is the ongoing one."""
    price = _daily([100.0, 110.0, 90.0, 95.0, 105.0, 120.0, 100.0, 95.0])
    fig = plot_top_drawdowns_paths(price=price, highlight_ongoing=True)
    ongoing = [text for text in _legend_texts(fig) if text.endswith('-ongoing')]
    assert len(ongoing) == 1
    assert ongoing[0].startswith('06Jan2024-08Jan2024')
    plt.close(fig)


def test_episode_recovered_on_the_penultimate_date_is_not_highlighted() -> None:
    """Recovery on the penultimate date ends an episode; nothing is ongoing."""
    price = _daily([100.0, 110.0, 90.0, 95.0, 105.0, 111.0, 112.0])
    fig = plot_top_drawdowns_paths(price=price, highlight_ongoing=True)
    assert not any(text.endswith('-ongoing') for text in _legend_texts(fig))
    plt.close(fig)


def test_native_grid_counts_observations() -> None:
    """With freq=None durations and the x-axis both count observations of the native grid."""
    dates = pd.bdate_range('2024-01-01', periods=12)  # spans two weekends
    price = pd.Series([100.0, 110.0, 99.0, 88.0, 104.5, 110.0,
                       120.0, 120.0, 108.0, 90.0, 98.4, 102.0], index=dates, name='nav')
    fig = plot_top_drawdowns_paths(price=price, freq=None)
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Observations in drawdown'
    texts = _legend_texts(fig)
    # 2 Jan to 8 Jan is four observations (calendar days would give six)
    assert any(text.startswith('02Jan2024-08Jan2024') and text.endswith('days_dd=4')
               for text in texts)
    # the drawn paths are the lines with data; legend proxies carry none
    lengths = sorted(np.count_nonzero(~np.isnan(np.asarray(line.get_ydata(), dtype=float)))
                     for line in ax.get_lines() if len(line.get_ydata()) > 0)
    assert lengths == [5, 6]  # four and five observations after the start
    plt.close(fig)


def test_calendar_day_grid_counts_days() -> None:
    """On the default calendar-day grid the axis counts days and matches days_dd."""
    dates = pd.bdate_range('2024-01-01', periods=12)
    price = pd.Series([100.0, 110.0, 99.0, 88.0, 104.5, 110.0,
                       120.0, 120.0, 108.0, 90.0, 98.4, 102.0], index=dates, name='nav')
    fig = plot_top_drawdowns_paths(price=price)
    assert fig.axes[0].get_xlabel() == 'Days in drawdown'
    assert any(text.startswith('02Jan2024-08Jan2024') and text.endswith('days_dd=6')
               for text in _legend_texts(fig))
    plt.close(fig)
