"""Private adapters preserving QIS contracts across pandas releases."""

import warnings
from typing import Optional

import pandas as pd


def infer_freq(index) -> Optional[str]:
    """Infer a frequency string without changing pandas' global inference option.

    Pandas 3.1 can return a DateOffset instead of a string. Only its warning about
    that transition is filtered; other inference warnings and errors still propagate.

    Args:
        index: Datetime-like observations accepted by pandas.infer_freq.

    Returns:
        Inferred pandas frequency string, or None for an irregular grid.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(r'A future version of pandas will return a BaseOffset object '
                     r'instead of a string from infer_freq\.'),
            category=getattr(pd.errors, 'Pandas4Warning', FutureWarning),
        )
        inferred = pd.infer_freq(index)
    return inferred.freqstr if isinstance(inferred, pd.offsets.BaseOffset) else inferred
