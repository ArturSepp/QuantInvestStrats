"""Prediction scores retain bad forecasts and require labelled alignment."""
import numpy as np
import pandas as pd
import pytest
from qis.utils.regression import prediction_r_squared


def test_held_out_score_and_alignment():
    """Training-only benchmark makes a poor forecast negative, not clipped."""
    y = pd.Series([1., 2., 3.])
    assert prediction_r_squared(y, y) == 1.
    assert prediction_r_squared(y, -y, benchmark=0.) == -3.
    assert np.isnan(prediction_r_squared(y*0, y*0))
    with pytest.raises(ValueError, match='index'):
        prediction_r_squared(y, y.iloc[::-1])
