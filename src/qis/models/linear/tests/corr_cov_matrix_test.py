import numpy as np
import pandas as pd

# qis
from qis.models.linear.corr_cov_matrix import (
    compute_masked_covar_corr,
)


def test_compute_masked_corr_uses_pairwise_complete_observations():
    """Keep ragged-history correlations bounded and equal to pandas pairwise correlation."""
    data = pd.DataFrame({
        'long': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        'short': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 1.0, 2.0, 3.0],
    })

    actual = compute_masked_covar_corr(data=data, is_covar=False)

    pd.testing.assert_frame_equal(actual, data.corr())
    assert np.nanmax(np.abs(actual.to_numpy())) <= 1.0
