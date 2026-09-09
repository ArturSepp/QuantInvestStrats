"""Regression tests for complete IID bootstrap draws and their public data mapping.

The IID sampler fills in source-length chunks. A non-multiple output length is the smallest
boundary that proves the partial final chunk is written and that its random draws advance the
following sample columns consistently.
"""

# packages
import numpy as np
import pandas as pd

# qis / project
from qis.models.bootstrap.bootstrap_numba import (
    BootstrapOutput,
    BootstrapType,
    bootstrap_data,
    generate_bootstrapped_indices,
)


def test_generate_bootstrapped_indices_iid_fills_complete_seeded_draw() -> None:
    """IID sampling fills first, interior, and terminal rows for every sample column."""
    actual = generate_bootstrapped_indices(
        num_data_index=5,
        bootstrap_type=BootstrapType.IID,
        num_samples=3,
        index_length=6,
        seed=7,
    )
    expected = np.array(
        [
            [4, 0, 0],
            [1, 4, 0],
            [3, 0, 0],
            [3, 4, 3],
            [4, 0, 0],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(actual, expected)


def test_bootstrap_data_iid_maps_complete_draw_without_mutating_source() -> None:
    """The public data bootstrap maps the terminal IID row instead of reusing source row zero."""
    source = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], name="value")
    original = source.copy()

    actual = bootstrap_data(
        data=source,
        bootstrap_type=BootstrapType.IID,
        bootstrap_output=BootstrapOutput.SERIES_TO_DF,
        num_samples=3,
        index_length=6,
        seed=7,
    )
    expected = pd.DataFrame(
        [
            [50.0, 10.0, 10.0],
            [20.0, 50.0, 10.0],
            [40.0, 10.0, 10.0],
            [40.0, 50.0, 40.0],
            [50.0, 10.0, 10.0],
            [20.0, 40.0, 30.0],
        ],
        columns=["path_1", "path_2", "path_3"],
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_series_equal(source, original)
