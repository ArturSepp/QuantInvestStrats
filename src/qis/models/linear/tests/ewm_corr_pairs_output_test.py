"""compute_ewm_corr_df pair selection and compute_ewm_corr_single, as documented."""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.corr_cov_matrix import (
    CorrMatrixOutput,
    compute_ewm_corr_df,
    compute_ewm_corr_single,
)


def _returns(num_columns: int) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(rng.standard_normal((60, num_columns)),
                        columns=[f"x{i}" for i in range(num_columns)])


def test_full_returns_pairs_below_the_diagonal() -> None:
    """FULL returns (i, j) with j < i, named '<column i> - <column j>'."""
    corr = compute_ewm_corr_df(df=_returns(3), corr_matrix_output=CorrMatrixOutput.FULL)
    assert corr.columns.tolist() == ['x1 - x0', 'x2 - x0', 'x2 - x1']


def test_sub_top_coincides_with_full() -> None:
    """SUB_TOP is documented as returning the same pairs as FULL."""
    df = _returns(4)
    full = compute_ewm_corr_df(df=df, corr_matrix_output=CorrMatrixOutput.FULL)
    sub_top = compute_ewm_corr_df(df=df, corr_matrix_output=CorrMatrixOutput.SUB_TOP)
    pd.testing.assert_frame_equal(full, sub_top)


def test_top_row_returns_pairs_of_the_first_column() -> None:
    """TOP_ROW returns the first column against every later one."""
    corr = compute_ewm_corr_df(df=_returns(3), corr_matrix_output=CorrMatrixOutput.TOP_ROW)
    assert corr.columns.tolist() == ['x0 - x1', 'x0 - x2']


def test_single_returns_the_one_pair() -> None:
    """The two-column case returns one Series equal to the FULL column."""
    df = _returns(2)
    single = compute_ewm_corr_single(returns=df, span=20)
    full = compute_ewm_corr_df(df=df, ewm_lambda=1.0 - 2.0 / 21.0)
    pd.testing.assert_series_equal(single, full.iloc[:, 0])
    assert single.name == 'x1 - x0'


def test_single_rejects_other_widths_with_column_names() -> None:
    """The error message names the columns it received."""
    with pytest.raises(ValueError, match='x2'):
        compute_ewm_corr_single(returns=_returns(3))
