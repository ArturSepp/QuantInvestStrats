"""
Pin the convention example's numerical results and the handbook tables that publish them.

``test_examples.py`` checks that the script runs, but execution alone would not catch a changed
draw frequency or annualised bias. The fixed seeds make every quantity deterministic, so the
tolerances below are half a unit in the last published digit, not a statistical allowance.

``docs/reproducibility.md`` retains this case study; the revised software paper no longer quotes
its measurements. Keep the numerical and handbook checks here rather than requiring removed
manuscript wording. ``src/qis/tests/test_paper_audit.py`` separately checks the measurements the
paper still quotes.
"""
# packages
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Tuple
import numpy as np
import pytest
# qis / project
import qis

REPO_ROOT: Path = Path(qis.__file__).resolve().parents[2]
EXAMPLE_PATH: Path = REPO_ROOT.joinpath('examples', 'models', 'bootstrap_convention.py')
REPRODUCIBILITY_PATH: Path = REPO_ROOT.joinpath('docs', 'reproducibility.md')
IS_REPOSITORY_CHECKOUT: bool = REPO_ROOT.joinpath('pyproject.toml').is_file()

pytestmark = pytest.mark.skipif(
    not IS_REPOSITORY_CHECKOUT,
    reason='repository convention example is absent from an installed wheel')

# half a unit in the last digit each quantity is published to
FREQUENCY_TOLERANCE: float = 0.0005     # published to three decimals
BASIS_POINT_TOLERANCE: float = 0.005    # published to two decimals, in bp
PERCENT_TOLERANCE: float = 0.005        # published to two decimals, in percent


def _load_example() -> ModuleType:
    """
    import the example module from its file.

    Root-level ``examples`` carries no ``__init__.py`` and is not part of the installed
    distribution, so it is loaded by path exactly as ``test_examples.py`` locates it.

    Returns:
        the imported module

    Raises:
        FileNotFoundError: when the example has moved
    """
    if not EXAMPLE_PATH.is_file():
        raise FileNotFoundError(f"the convention example is not at {EXAMPLE_PATH}")
    specification = importlib.util.spec_from_file_location('bootstrap_convention', EXAMPLE_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def example() -> ModuleType:
    """the example module, imported once for the file."""
    return _load_example()


@pytest.fixture(scope='module')
def indices(example: ModuleType) -> Tuple[np.ndarray, np.ndarray]:
    """
    the two index arrays the example draws, under the two conventions.

    Returns:
        the truncating draw and the circular draw, both of shape ``(index_length, num_samples)``
    """
    truncating = example.draw_truncating_indices(num_data_index=example.NUM_DATA_INDEX,
                                                 num_samples=example.NUM_SAMPLES,
                                                 index_length=example.INDEX_LENGTH,
                                                 block_size=example.BLOCK_SIZE,
                                                 seed=example.SEED)
    circular = qis.generate_bootstrapped_indices(num_data_index=example.NUM_DATA_INDEX,
                                                 bootstrap_type=qis.BootstrapType.STATIONARY,
                                                 num_samples=example.NUM_SAMPLES,
                                                 index_length=example.INDEX_LENGTH,
                                                 block_size=example.BLOCK_SIZE,
                                                 seed=example.SEED)
    return truncating, circular


def _frequency_triple(example: ModuleType,
                      draw: np.ndarray
                      ) -> Tuple[float, float, float]:
    """
    the three relative draw frequencies the example reports.

    Args:
        example: the imported example module
        draw: an index array

    Returns:
        the first observation, the mean over the first decile, and the mean over the last decile
    """
    frequency = example.relative_draw_frequency(indices=draw,
                                                num_data_index=example.NUM_DATA_INDEX)
    decile = example.NUM_DATA_INDEX // 10
    return float(frequency[0]), float(frequency[:decile].mean()), float(frequency[-decile:].mean())


@pytest.mark.parametrize('convention, first_observation, first_decile, last_decile',
                         [('truncating', 0.110, 0.526, 1.073),
                          ('circular', 0.978, 1.007, 1.020)])
def test_relative_draw_frequency(example: ModuleType,
                                 indices: Tuple[np.ndarray, np.ndarray],
                                 convention: str,
                                 first_observation: float,
                                 first_decile: float,
                                 last_decile: float,
                                 ) -> None:
    """
    each convention samples the source at the published rate.

    The truncating row is the finding: a block only ever runs forwards, so the first observation
    is reachable as a block start and almost never as a continuation, and it is drawn at roughly
    a ninth of its uniform weight.
    """
    draw = indices[0] if convention == 'truncating' else indices[1]
    measured = _frequency_triple(example=example, draw=draw)
    expected = (first_observation, first_decile, last_decile)
    for measured_value, expected_value, label in zip(measured, expected,
                                                     ('first observation', 'first decile',
                                                      'last decile')):
        assert abs(measured_value - expected_value) < FREQUENCY_TOLERANCE, (
            f"{convention} {label}: {measured_value:.4f}, published {expected_value:.3f}. "
            f"The example and docs/reproducibility.md quote the published value")


def test_source_series_mean(example: ModuleType) -> None:
    """the series the bias is measured against has the published mean."""
    source_mean_bp = float(example.make_trending_returns().mean()) * 1e4
    assert abs(source_mean_bp - 12.80) < BASIS_POINT_TOLERANCE, (
        f"the source series mean moved to {source_mean_bp:.2f} bp from the published 12.80 bp; "
        f"every bias below is measured against it")


@pytest.mark.parametrize('convention, resampled_mean_bp, bias_bp, bias_annual_percent',
                         [('truncating', 13.62, +0.83, +2.15),
                          ('circular', 12.67, -0.12, -0.32)])
def test_resampled_mean_bias(example: ModuleType,
                             indices: Tuple[np.ndarray, np.ndarray],
                             convention: str,
                             resampled_mean_bp: float,
                             bias_bp: float,
                             bias_annual_percent: float,
                             ) -> None:
    """
    the mean each convention reports, and the annualised bias that follows.

    The handbook uses the 2.47 percentage-point spread to illustrate this fixed sample's
    sensitivity to the convention; it is not a universal bootstrap adjustment.
    """
    draw = indices[0] if convention == 'truncating' else indices[1]
    returns = example.make_trending_returns()
    source_mean = float(returns.mean())
    measured_mean = float(returns[draw].mean(axis=0).mean())
    measured_bias = measured_mean - source_mean

    assert abs(measured_mean * 1e4 - resampled_mean_bp) < BASIS_POINT_TOLERANCE, (
        f"{convention} resampled mean {measured_mean * 1e4:.3f} bp, "
        f"published {resampled_mean_bp:.2f} bp")
    assert abs(measured_bias * 1e4 - bias_bp) < BASIS_POINT_TOLERANCE, (
        f"{convention} bias {measured_bias * 1e4:+.3f} bp, published {bias_bp:+.2f} bp")
    annualised_percent = measured_bias * example.PERIODS_PER_YEAR * 100.0
    assert abs(annualised_percent - bias_annual_percent) < PERCENT_TOLERANCE, (
        f"{convention} annualised bias {annualised_percent:+.3f}%, "
        f"published {bias_annual_percent:+.2f}%")


def test_conventions_differ_by_more_than_two_points(example: ModuleType,
                                                    indices: Tuple[np.ndarray, np.ndarray],
                                                    ) -> None:
    """
    the handbook's case-study consequence, asserted as a number.

    Two researchers running "a stationary bootstrap" on this series report annualised means more
    than two percentage points apart. Keep this check alongside the individually rounded biases
    so the handbook's stated difference remains supported by the full-precision calculation.
    """
    returns = example.make_trending_returns()
    source_mean = float(returns.mean())
    biases = [float(returns[draw].mean(axis=0).mean()) - source_mean for draw in indices]
    spread_percent = abs(biases[0] - biases[1]) * example.PERIODS_PER_YEAR * 100.0
    assert spread_percent > 2.0, (
        f"the two conventions now differ by {spread_percent:.2f} percentage points a year, "
        f"but the handbook's case study reports more than two")


@pytest.mark.parametrize('published', ['0.110', '0.526', '1.073', '0.978', '1.007', '1.020',
                                       '13.62 bp', '+0.83 bp', '+2.15%',
                                       '12.67 bp', '-0.12 bp', '-0.32%', '12.80 bp'])
def test_reproducibility_page_states_the_measured_values(published: str) -> None:
    """
    the documentation page still quotes what the script produces.

    The page is written by hand from the script's output, so nothing but this check couples them.
    Minus signs are normalised because the page uses a typographic minus and the script prints
    a hyphen.
    """
    if not REPRODUCIBILITY_PATH.is_file():
        pytest.skip(f"{REPRODUCIBILITY_PATH} is not in this checkout")
    page = REPRODUCIBILITY_PATH.read_text(encoding='utf-8')
    normalised = re.sub(r'[−–—]', '-', page).replace('**', '')
    assert published in normalised, (
        f"docs/reproducibility.md no longer states {published!r}. Regenerate the page from "
        f"python -m examples.models.bootstrap_convention")
