from __future__ import annotations

import warnings
import re
import pandas as pd
from typing import Union


BUS_DAYS_PER_YEAR = 252  # applied for volatility normalization
WEEK_DAYS_PER_YEAR = 260  # calendar days excluding weekends in a year
CALENDAR_DAYS_PER_YEAR = 365
CALENDAR_DAYS_IN_MONTH = 30
CALENDAR_DAYS_PER_YEAR_SHARPE = 365.25  # for total return computations for Sharpe
DEFAULT_TRADING_YEAR_DAYS = 252  # How mny trading days we assume per year, see


def get_annualization_factor(freq: str,
                             is_calendar: bool = False,
                             default_trading_days: int = BUS_DAYS_PER_YEAR
                             ) -> float:
    """
    Calculate annualization factor from pandas frequency string.
    Handles various frequency formats including multipliers and anchors.

    Args:
        freq: Pandas frequency string (e.g., 'D', 'W-FRI', '2ME', 'QE-DEC')
        is_calendar: If True, use calendar days (365); if False, use trading days (252)

    Returns:
        Annualization factor (number of periods per year)

    Examples:
        >>> get_annualization_factor('D', is_calendar=True)
        365.0
        >>> get_annualization_factor('B')
        252.0
        >>> get_annualization_factor('W-FRI')
        52.0
        >>> get_annualization_factor('2W')
        26.0
        >>> get_annualization_factor('ME')
        12.0
        >>> get_annualization_factor('QE-DEC')
        4.0
        >>> get_annualization_factor('3ME')
        4.0
    """
    an_days = 365.0 if is_calendar else default_trading_days

    # Intraday frequencies
    if freq in ['1M']:  # 1 minute
        return an_days * 24.0 * 60.0
    elif freq in ['5M']:
        return an_days * 24.0 * 12.0
    elif freq in ['15M', '15T']:
        return an_days * 24.0 * 4.0
    elif freq in ['h', 'H']:  # hourly
        return an_days * 24.0

    # Daily frequencies
    elif freq in ['D']:  # Calendar days - always 365
        return 365.0
    elif freq in ['B', 'C']:  # Business days
        return an_days

    # Weekly frequencies
    elif freq in ['W', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN', 'WE']:
        return 52.0
    elif freq in ['SM', '2W', '2W-MON', '2W-TUE', '2W-WED', '2W-THU', '2W-FRI', '2W-SAT', '2W-SUN']:
        return 26.0
    elif freq in ['3W', '3W-MON', '3W-TUE', '3W-WED', '3W-THU', '3W-FRI', '3W-SAT', '3W-SUN']:
        return 17.33
    elif freq in ['4W', '4W-MON', '4W-TUE', '4W-WED', '4W-THU', '4W-FRI', '4W-SAT', '4W-SUN']:
        return 13.0

    # Monthly frequencies
    elif freq in ['1M', 'ME', 'M', 'BM', 'MS', 'BMS']:
        return 12.0
    elif freq in ['2M', '2ME', '2BM', '2MS', '2BMS']:
        return 6.0

    # Quarterly frequencies
    elif freq in ['QE', 'Q', 'DQ', 'BQ', 'QS', 'BQS', 'QE-DEC', 'QE-JAN', 'QE-FEB', 'Q-DEC']:
        return 4.0
    elif freq in ['2Q', '2QE', '2BQ', '2QS', '2BQS']:
        return 2.0
    elif freq in ['3Q', '3QE', '3BQ', '3QS', '3BQS']:
        return 0.75  # ~4/12 of a year

    # Annual frequencies
    elif freq in ['YE', 'Y', 'A', 'BA', 'AS', 'YS', 'BAS']:
        return 1.0

    # Parse frequency string with regex for generic cases
    else:
        match = re.match(r'^(\d+)?([A-Z]+)(?:-[A-Z]+)?$', freq.upper())

        if not match:
            warnings.warn(
                f"Unknown frequency '{freq}'. Using annualization factor of 1.0.",
                UserWarning,
                stacklevel=2
            )
            return 1.0

        multiplier_str, base_freq = match.groups()
        multiplier = int(multiplier_str) if multiplier_str else 1

        # Base factors for generic parsing
        freq_to_annual = {
            'D': 365.0,
            'B': an_days,
            'W': 52.0,
            'WE': 52.0,
            'ME': 12.0,
            'M': 12.0,
            'MS': 12.0,
            'BM': 12.0,
            'BMS': 12.0,
            'QE': 4.0,
            'Q': 4.0,
            'QS': 4.0,
            'BQ': 4.0,
            'BQS': 4.0,
            'YE': 1.0,
            'Y': 1.0,
            'A': 1.0,
            'YS': 1.0,
            'AS': 1.0,
            'BA': 1.0,
            'BAS': 1.0,
            'H': an_days * 24.0,
            'T': an_days * 390.0,  # Trading minutes
            'MIN': an_days * 390.0,
        }

        if base_freq not in freq_to_annual:
            warnings.warn(
                f"Unknown base frequency '{base_freq}' (from '{freq}'). Using annualization factor of 1.0. "
                f"Supported frequencies: {', '.join(sorted(set(list(freq_to_annual.keys()) + ['SM', 'DQ'])))}",
                UserWarning,
                stacklevel=2
            )
            return 1.0

        base_factor = freq_to_annual[base_freq]
        return base_factor / multiplier


def infer_annualisation_factor_from_df(data: Union[pd.DataFrame, pd.Series]) -> float:
    """
    infer annualization factor for vol
    """
    if len(data.index) < 3:
        freq = None
    else:
        freq = pd.infer_freq(data.index)
        
    if freq is None:
        warnings.warn(
            f"in infer_annualisation_factor_from_df: cannot infer {freq} - using {BUS_DAYS_PER_YEAR}\n data.index={data.index}",
            UserWarning,
            stacklevel=2
        )
        return BUS_DAYS_PER_YEAR
    alpha_an_factor = get_annualization_factor(freq=freq)
    return alpha_an_factor


def get_annualisation_conversion_factor(from_freq: str, to_freq: str) -> float:
    """
    Get factor to convert between pandas frequencies.

    Args:
        from_freq: Source frequency
        to_freq: Target frequency

    Returns:
        Conversion factor (multiply source data by this factor)

    Examples:
        >>> get_annualisation_conversion_factor('QE', 'ME')  # Quarterly to Monthly
        0.3333333333333333
        >>> get_annualisation_conversion_factor('ME', 'QE')  # Monthly to Quarterly
        3.0
        >>> get_annualisation_conversion_factor('B', 'ME')  # Business Daily to Monthly
        21.666666666666668
    """
    from_periods = get_annualization_factor(from_freq)
    to_periods = get_annualization_factor(to_freq)

    return from_periods / to_periods