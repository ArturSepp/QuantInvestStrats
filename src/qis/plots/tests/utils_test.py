import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from qis.plots.utils import (compute_heatmap_colors,
                             get_data_group_colors,
                             set_legend)


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


def test_set_legend_defaults_to_normal_text_weight():
    """Default legend text uses the universally available normal font weight."""
    fig, ax = plt.subplots()
    try:
        set_legend(ax=ax, labels=['Series'], colors=['blue'])
        fig.canvas.draw()
        legend_text = ax.get_legend().get_texts()[0]
        assert legend_text.get_fontweight() == 'normal'
    finally:
        plt.close(fig)
