"""Compare portfolio NAV Sharpe ratios with a selected portfolio using paired HAC tests."""

from numbers import Integral
from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qis.perfstats.returns import to_returns
from qis.perfstats.sharpe_inference import sharpe_difference_test
from qis.plots.table import plot_df_table
from qis.utils.annualisation import get_annualization_factor
from qis.utils.dates import TimePeriod
from qis.utils.df_str import df_to_str

if TYPE_CHECKING:
    from qis.portfolio.multi_portfolio_data import MultiPortfolioData


_RESULT_COLUMNS = {
    'Base Sharpe': 'sr2_an',
    'Portfolio Sharpe': 'sr1_an',
    'Sharpe difference': 'diff_an',
    'HAC SE': 'se_an',
    'HAC z': 't_stat',
    'p-value': 'p_value',
    'Observations': 'n_obs',
    'HAC lags': 'n_lags',
}


def _format_table(table: pd.DataFrame) -> pd.DataFrame:
    return df_to_str(table, var_format='{:0.2f}',
                     var_formats={'Observations': '{:.0f}', 'HAC lags': '{:.0f}'})


def compute_sharpe_difference_table(
        multi_portfolio_data: 'MultiPortfolioData',
        base_portfolio_id: int = 0,
        time_period: Optional[TimePeriod] = None,
        freq: Optional[str] = 'ME',
        af: Optional[float] = None,
        n_lags: Optional[int] = 3,
        is_convert_to_str: bool = False,
) -> pd.DataFrame:
    """Compare every non-base portfolio with one base on a common return sample.

    NAVs are sliced before return conversion: the first sampled NAV in the requested
    window is a baseline, not a return. Use the previous period's closing NAV as the start
    to include the desired first return. Resampling uses QIS period-end conventions and
    excludes incomplete boundary periods. No stored NaN NAV is forward-filled. Returns
    are computed before complete rows across all portfolios are retained, so a missing
    NAV cannot create a return spanning multiple observations. HAC lags count retained rows.

    Args:
        multi_portfolio_data: Portfolios containing NAVs with unique portfolio names and
            a unique DatetimeIndex. External benchmark_prices are not included.
        base_portfolio_id: Zero-based position in portfolio_datas, also the NAV column order.
            The base is excluded from the output rows; differences are portfolio minus base.
        time_period: Inclusive NAV window. None uses the full available history.
        freq: Return sampling frequency; default ME uses complete calendar month ends.
            None retains the native NAV grid and requires an explicit af.
        af: Periods per year. None uses QIS get_annualization_factor(freq), including 252
            for B and 12 for ME. Pass 260 explicitly for the TF daily reporting convention.
        n_lags: Bartlett HAC truncation; defaults to 3 retained return observations.
            Explicit None uses sharpe_difference_test's automatic lag rule.
        is_convert_to_str: Format floats to two decimals and counts as integers for display.
            False preserves full numerical precision.

    Returns:
        DataFrame in original portfolio order with Base Sharpe, Portfolio Sharpe, Sharpe
        difference, HAC SE, HAC z, p-value, Observations and HAC lags. Attributes record
        the base, sampling frequency, annualisation and first/last retained return dates.
        Sharpes use arithmetic simple returns and ddof=0. Costs and fees already embedded
        in NAVs are preserved; no extra fees or risk-free adjustment are applied. P-values
        are individual two-sided asymptotic normal tests, without multiple-testing correction.

    Raises:
        ValueError: If the base, NAVs or sampling settings are invalid, fewer than 24 common
            returns remain, or any Sharpe comparison is undefined.
    """
    navs = multi_portfolio_data.get_navs()
    n_portfolios = len(multi_portfolio_data.portfolio_datas)
    if navs.shape[1] != n_portfolios or n_portfolios < 2:
        raise ValueError('at least two portfolio NAV columns in portfolio order are required')
    if (isinstance(base_portfolio_id, (bool, np.bool_))
            or not isinstance(base_portfolio_id, Integral)
            or not 0 <= base_portfolio_id < n_portfolios):
        raise ValueError('base_portfolio_id must be a valid zero-based portfolio position')
    if not navs.columns.is_unique:
        raise ValueError('portfolio NAV names must be unique')
    if (not isinstance(navs.index, pd.DatetimeIndex)
            or not navs.index.is_unique or navs.index.hasnans):
        raise ValueError('portfolio NAVs require a unique, nonmissing DatetimeIndex')
    navs = navs.sort_index()
    if time_period is not None:
        navs = time_period.locate(navs)
    if navs.empty:
        raise ValueError('the requested NAV window is empty')
    for column in navs:
        series = navs[column]
        if (not pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_complex_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)):
            raise ValueError(f'NAVs must be real numeric values: {column!r}')
    values = navs.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or np.any(values[~np.isnan(values)] <= 0):
        raise ValueError('observed NAVs must be positive and finite')
    if af is None:
        if freq is None:
            raise ValueError('af is required when freq=None')
        af = get_annualization_factor(freq)
    returns = to_returns(navs, freq=freq, is_log_returns=False, ffill_nans=False,
                         include_start_date=False, include_end_date=False,
                         drop_first=True).dropna(how='any')
    if len(returns) < 24:
        raise ValueError(f'at least 24 common returns are required, got {len(returns)}')
    base = returns.iloc[:, base_portfolio_id]
    rows, names = [], []
    for position, name in enumerate(returns.columns):
        if position == base_portfolio_id:
            continue
        try:
            result = sharpe_difference_test(returns.iloc[:, position], base, n_lags=n_lags, af=af)
        except ValueError as error:
            raise ValueError(f'Sharpe comparison {name!r} versus {base.name!r}: {error}') from error
        rows.append({column: getattr(result, field) for column, field in _RESULT_COLUMNS.items()})
        names.append(name)
    table = pd.DataFrame(rows, index=pd.Index(names, name='Portfolio'))
    table.attrs.update(base_portfolio_id=int(base_portfolio_id), base_portfolio=base.name,
                       freq=freq, af=float(af), sample_start=returns.index[0],
                       sample_end=returns.index[-1], return_type='simple',
                       sample_policy='common complete returns', cash_adjustment='none')
    return _format_table(table) if is_convert_to_str else table


def plot_sharpe_difference_table(
        multi_portfolio_data: 'MultiPortfolioData',
        base_portfolio_id: int = 0,
        time_period: Optional[TimePeriod] = None,
        freq: Optional[str] = 'ME',
        af: Optional[float] = None,
        n_lags: Optional[int] = 3,
        ax: Optional[plt.Axes] = None,
        **kwargs,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """Draw a two-decimal HAC Sharpe comparison table and return its numeric data.

    Args:
        multi_portfolio_data: Portfolio collection; only its portfolio NAVs are compared.
        base_portfolio_id: Zero-based base portfolio position, omitted from the displayed rows.
        time_period: Inclusive NAV window, with its first sampled NAV used as the baseline.
        freq: Sampling frequency; ME by default, None for native observations.
        af: Periods per year; required with freq=None, otherwise inferred from freq.
        n_lags: Bartlett truncation; defaults to 3 retained return observations.
            Explicit None uses the Sharpe test's automatic lag rule.
        ax: Optional axes for composing the table into an existing report.
        **kwargs: Display options forwarded to plot_df_table, including a custom title.

    Returns:
        (figure, numeric table). Figure is None when ax is supplied. Float cells display
        two decimal places; numeric results retain precision. Counts display as integers.
        The default title identifies the base and actual common return window.
    """
    table = compute_sharpe_difference_table(multi_portfolio_data, base_portfolio_id,
                                            time_period, freq, af, n_lags)
    start, end = table.attrs['sample_start'], table.attrs['sample_end']
    sampling = freq if freq is not None else 'native'
    options = dict(index_column_name='Portfolio', fontsize=10, column_width=1.35,
                   first_column_width=2.5, row_height=0.5,
                   title=f"Sharpe differences versus {table.attrs['base_portfolio']}\n"
                         f"{start:%d %b %Y} to {end:%d %b %Y}; {sampling} returns; "
                         f"annualisation {table.attrs['af']:0.2f}")
    options.update(kwargs)
    display = _format_table(table).rename(columns={'Base Sharpe': 'Base\nSharpe',
                                                  'Portfolio Sharpe': 'Portfolio\nSharpe',
                                                  'Sharpe difference': 'Sharpe\ndifference'})
    figure = plot_df_table(display, ax=ax, **options)
    return figure, table
