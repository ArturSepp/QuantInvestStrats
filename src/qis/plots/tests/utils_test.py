import numpy as np
import pandas as pd
import warnings
from qis.plots.utils import (compute_heatmap_colors,
                            get_data_group_colors)


def test_group_colors_include_unobserved_categories_without_future_warning():
    """Categorical palettes retain the pre-Pandas-3 treatment of unused categories."""
    data = pd.DataFrame({
        'bucket': pd.Categorical(
            ['low', 'high'], categories=['low', 'mid', 'high'], ordered=True,
        ),
        'value': [1.0, 3.0],
    })

    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        actual = get_data_group_colors(df=data, x='bucket', y='value')

    # The function sorts by the observed means and leaves the unused level (NaN) last.
    expected = compute_heatmap_colors(a=np.array([1.0, 3.0, np.nan]))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
