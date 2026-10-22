"""CSV date defaults stay month-first on old and new pandas readers."""

import pandas as pd
import pytest

import qis


@pytest.mark.parametrize('loader', ['single', 'dictionary'])
@pytest.mark.parametrize('dayfirst', [None, False, True])
def test_csv_readers_preserve_optional_dayfirst(tmp_path, loader, dayfirst):
    frame = pd.DataFrame({'value': [1, 2]}, index=['01/02/2024', '02/03/2024'])
    if loader == 'single':
        qis.save_df_to_csv(frame, file_name='dates', local_path=str(tmp_path))
        actual = qis.load_df_from_csv(file_name='dates', local_path=str(tmp_path),
                                      dayfirst=dayfirst)
    else:
        qis.save_df_dict_to_csv({'panel': frame}, file_name='dates', local_path=str(tmp_path))
        actual = qis.load_df_dict_from_csv(['panel'], file_name='dates',
                                           local_path=str(tmp_path), dayfirst=dayfirst)['panel']
    dates = ['2024-02-01', '2024-03-02'] if dayfirst else ['2024-01-02', '2024-02-03']
    expected = pd.DataFrame({'value': [1, 2]}, index=pd.to_datetime(dates))
    pd.testing.assert_frame_equal(actual, expected, check_names=False)
