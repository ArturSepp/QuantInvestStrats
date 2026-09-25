"""The reporting-preset tables in the documentation agree with the code that sets the presets.

Three pages print the factsheet presets by hand: the reporting-frequency chapter, the packaged
reporting-frequency note and the factsheet methodology page. Each row is compared with
``qis.fetch_default_report_kwargs`` for the long and the short horizon, and the annualisation
column with ``qis.get_annualization_factor``, so a changed preset cannot leave a stale table.
"""

# packages
import re
from pathlib import Path
from typing import Dict, List

import pytest

# qis
import qis


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CHAPTER: Path = REPO_ROOT.joinpath('docs', 'frequency_convention_note.md')
NOTE: Path = REPO_ROOT.joinpath('src', 'qis', 'docs', 'reporting_frequencies.md')
FACTSHEETS: Path = REPO_ROOT.joinpath('docs', 'factsheets_and_reporting.md')
if not CHAPTER.is_file() or not FACTSHEETS.is_file():
    pytest.skip('Documentation sources are not shipped in the wheel.', allow_module_level=True)

SHORT_PERIOD = qis.TimePeriod('2021-01-01', '2023-12-31')
REGIME_NAMES = {'QE': 'quarterly', 'ME': 'monthly'}
WINDOW_PERIODS = {'B': 260, 'W-WED': 52, 'ME': 12, 'QE': 4}


def _presets(frequency: qis.ReportingFrequency) -> Dict[str, Dict]:
    """Long and short preset settings for one reporting frequency."""
    return {horizon: qis.fetch_default_report_kwargs(
        time_period=period, reporting_frequency=frequency, add_rates_data=False)
        for horizon, period in (('long', None), ('short', SHORT_PERIOD))}


def _rows(path: Path, first_cells: List[str]) -> Dict[str, List[str]]:
    """Table rows of ``path`` keyed by their first cell, for the given first cells."""
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        cells = [cell.strip() for cell in re.split(r'(?<!\\)\|', line.strip())[1:-1]]
        if cells and cells[0] in first_cells:
            rows[cells[0]] = cells
    assert set(rows) == set(first_cells), f'{path.name}: preset rows not found'
    return rows


def _pair(cell: str, separator: str) -> List[str]:
    """Split a ``long · short`` or ``long \\| short`` cell into its two values."""
    return [value.strip(' `') for value in cell.split(separator)]


@pytest.mark.parametrize('frequency', list(qis.ReportingFrequency))
def test_reporting_frequency_chapter_table(frequency: qis.ReportingFrequency) -> None:
    """Grid, windows, beta spans, regime grids and window periods in the chapter table."""
    row = _rows(CHAPTER, [f.name.capitalize() for f in qis.ReportingFrequency])[
        frequency.name.capitalize()]
    presets = _presets(frequency)
    long, short = presets['long'], presets['short']
    assert row[1].strip('`') == long['vol_freq']
    assert _pair(row[2], '·') == [str(long['vol_rolling_window']), str(short['vol_rolling_window'])]
    assert long['sharpe_rolling_window'] == long['vol_rolling_window']
    assert _pair(row[3], '·') == [str(long['factor_beta_span']), str(short['factor_beta_span'])]
    assert _pair(row[4], '·') == [REGIME_NAMES[long['freq_regime']],
                                  REGIME_NAMES[short['freq_regime']]]
    assert int(row[5]) == WINDOW_PERIODS[long['vol_freq']] == long['turnover_rolling_period']


@pytest.mark.parametrize('frequency', list(qis.ReportingFrequency))
def test_packaged_reporting_note_table(frequency: qis.ReportingFrequency) -> None:
    """The packaged note's preset table, including turnover windows and regime grids."""
    row = _rows(NOTE, [f.name for f in qis.ReportingFrequency])[frequency.name]
    presets = _presets(frequency)
    long, short = presets['long'], presets['short']
    assert row[1].strip('`') == long['vol_freq']
    assert _pair(row[2], '\\|') == [str(long['vol_rolling_window']),
                                    str(short['vol_rolling_window'])]
    assert _pair(row[3], '\\|') == [str(long['factor_beta_span']), str(short['factor_beta_span'])]
    assert int(row[4]) == long['turnover_rolling_period'] == short['turnover_rolling_period']
    assert _pair(row[5], '\\|') == [long['freq_regime'], short['freq_regime']]


@pytest.mark.parametrize('frequency', list(qis.ReportingFrequency))
def test_factsheet_page_window_and_annualisation_columns(
        frequency: qis.ReportingFrequency) -> None:
    """Window periods per year and the annualisation factor of each base grid."""
    row = _rows(FACTSHEETS, [f.name.capitalize() for f in qis.ReportingFrequency])[
        frequency.name.capitalize()]
    grid = _presets(frequency)['long']['vol_freq']
    assert f'`{grid}`' in row[1]
    assert int(row[2]) == WINDOW_PERIODS[grid]
    assert float(row[3]) == qis.get_annualization_factor(grid)
