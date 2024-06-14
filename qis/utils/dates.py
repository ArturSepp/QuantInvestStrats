"""
implementation of dates analytics and frequency with TimePeriod method
construction of daate schedules and rebalancing
support for bespoke frequencies: M-FRI, Q-FRI
"""

from __future__ import annotations  # to allow class method annotations

import datetime as dt
import re
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Optional, NamedTuple, Dict
from enum import Enum

from qis.utils.struct_ops import separate_number_from_string

DATE_FORMAT = '%d%b%Y'  # 31Jan2020 - common across all reporting
DATE_FORMAT_INT = '%Y%m%d'  # 20000131

WEEKDAYS: List[str] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


BUS_DAYS_PER_YEAR = 252  # applied for volatility normalization
WEEK_DAYS_PER_YEAR = 260  # calendar days excluding weekends in a year
CALENDAR_DAYS_PER_YEAR = 365
CALENDAR_DAYS_IN_MONTH = 30
CALENDAR_DAYS_PER_YEAR_SHARPE = 365.25  # for total return computations for Sharpe


def get_current_time_with_tz(tz: Optional[str] = 'UTC',
                             days_offset: int = None,
                             normalize: bool = True,
                             hour: int = None
                             ) -> pd.Timestamp:
    t = pd.Timestamp.today(tz=tz)
    if normalize:
        t = t.normalize()  # normalize to eod date
    if days_offset is not None:
        t = t + pd.Timedelta(days=days_offset)
    if hour is not None:
        t = t.replace(hour=hour)
    return t


def get_time_to_maturity(maturity_time: pd.Timestamp,
                         value_time: pd.Timestamp,
                         is_floor_at_zero: bool = True,
                         af: float = 365
                         ) -> float:
    """
    return annualised difference between mat_date and value_time
    """
    seconds_per_year = af * 24 * 60 * 60  # days, hours, minute, seconds
    ttm = (maturity_time - value_time).total_seconds() / seconds_per_year
    if is_floor_at_zero and ttm < 0.0:
        ttm = 0.0
    return ttm


def get_period_days(freq: str = 'B',
                    is_calendar: bool = False
                    ) -> Tuple[int, float]:
    """
    for given frequency return number of days in the period
    is_calendar = True should return int for rolling and resample in pandas
    an_f will return the number of period in year
    consistent with using 252 for vol annualization
    """
    an_days = 365 if is_calendar else 252
    if freq in ['1M']:
        days = 1.0 / 24.0 / 60.0
        an_f = an_days * 24.0 * 60.0
    elif freq in ['5M']:
        days = 1.0 / 24.0 / 12.0
        an_f = an_days * 24.0 * 12.0
    elif freq in ['15M', '15T']:
        days = 1.0 / 24.0 / 12.0
        an_f = an_days * 24.0 * 4.0
    elif freq in ['h']:
        days = 1.0 / 24.0
        an_f = an_days * 24.0
    elif freq in ['D']:  # for 'D' always use 365
        days = 1
        an_f = 365
    elif freq in ['B', 'C']:
        days = 1
        an_f = an_days
    elif freq in ['W', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']:
        days = 7 if is_calendar else 5
        an_f = 52
    elif freq in ['SM', '2W', '2W-MON', '2W-TUE', '2W-WED', '2W-THU', '2W-FRI', '2W-SAT', '2W-SUN']:
        days = 14 if is_calendar else 10
        an_f = 26
    elif freq in ['3W', '3W-MON', '3W-TUE', '3W-WED', '3W-THU', '3W-FRI', '3W-SAT', '3W-SUN']:
        days = 21 if is_calendar else 15
        an_f = 17.33
    elif freq in ['4W', '4W-MON', '4W-TUE', '4W-WED', '4W-THU', '4W-FRI', '4W-SAT', '4W-SUN']:
        days = 28 if is_calendar else 20
        an_f = 13
    elif freq in ['1M', 'ME', 'BM', 'MS', 'BMS']:
        days = 30 if is_calendar else 21
        an_f = 12
    elif freq in ['2M', '2BM', '2MS', '2BMS']:
        days = 60 if is_calendar else 42
        an_f = 6
    elif freq in ['QE', 'DQ', 'BQ', 'QS', 'BQS', 'QE-DEC', 'QE-JAN', 'QE-FEB']:
        days = 91 if is_calendar else 63
        an_f = 4
    elif freq in ['2Q', '2BQ', '2QS', '2BQS']:
        days = 182 if is_calendar else 126
        an_f = 2
    elif freq in ['3Q', '3BQ', '3QS', '3BQS']:
        days = 273 if is_calendar else 189
        an_f = 0.75
    elif freq in ['YE', 'BA', 'AS', 'BAS']:
        days = an_days
        an_f = 1.0
    else:
        raise TypeError(f'freq={freq} is not impelemnted')

    return days, an_f


def infer_an_from_data(data: Union[pd.DataFrame, pd.Series], is_calendar: bool = False) -> float:
    """
    infer annualization factor for vol
    """
    if len(data.index) < 3:
        freq = None
    else:
        freq = pd.infer_freq(data.index)
    if freq is None:
        print(f"in infer_an_from_data: cannot infer {freq} - using 252")
        return 252.0
    an, an_f = get_period_days(freq, is_calendar=is_calendar)
    return an_f


class FreqData(NamedTuple):
    """
    enumerate frequencies with caption, python alias and set n_bus and calendar days
    in applications we need to set a number of calendar and business days for each frequency
    """
    cap: str
    freq_cap: str
    n_bus: Union[int, float]
    n_cal: Union[int, float]

    def print(self):
        print(f"{self.cap}, {self.freq_cap}, n_bus={self.n_bus}, n_cal={self.n_cal}")

    def to_n_bus_days(self):
        return self.n_bus

    def to_caption(self):
        return self.cap

    def to_freq_cap(self):
        return self.freq_cap


class FreqMap(FreqData, Enum):
    """
    name is linked to python aliases, value have extra data
    """
    B = FreqData('day', 'daily', 1, 1)
    D = FreqData('day', 'daily', 1, 1)
    W = FreqData('week', 'weekly', 5, 7)
    W2 = FreqData('bi-week', 'bi-weekly', 10, 14)
    M = FreqData('month', 'monthly', 21, 31)
    M2 = FreqData('bi-month', 'bi-monthly', 42, 62)
    BM = FreqData('month', 'monthly', 21, 31)
    Q = FreqData('quarter', 'quarterly', 63, 93)
    BQ = FreqData('quarter', 'quarterly', 63, 93)
    Q2 = FreqData('bi-quarter', 'bi-quarterly', 126, 180)
    BQ2 = FreqData('bi-quarter', 'bi-quarterly', 126, 180)
    A = FreqData('year', 'annual', 252, 360)

    def to_freq(self) -> str:
        if self.name == 'W2':
            freq = '2W'
        elif self.name == 'M2':
            freq = '2M'
        elif self.name == 'Q2':
            freq = '2Q'
        elif self.name == 'BQ2':
            freq = '2BQ'
        else:
            freq = self.name
        return freq

    @classmethod
    def to_value(cls, freq: str = 'B') -> FreqData:
        if freq == '2W':
            freq = 'W2'
        elif freq == '2M':
            freq = 'M2'
        elif freq == '2Q':
            freq = 'Q2'
        elif freq == '2BQ':
            freq = 'BQ2'
        return cls.map_to_value(freq)

    @classmethod
    def map_to_value(cls, name):
        """
        given name return value
        """
        for k, v in cls.__members__.items():
            if k == name:
                return v
        raise ValueError(f"nit in enum {name}")

    @classmethod
    def map_n_days(cls, n_days):
        for freq in cls:
            if freq.n_cal == n_days:
                return freq
        raise ValueError(f"cannot map {n_days}")


class TimePeriod:
    """
    TimePeriod for storing start and end dates of a schedule
    Initialized with (start_date, end_date)
    start_date, end_date with SUPPORTED_TYPES
    Allowed Strings:
    start  = '12/31/2019'  # m/d/yyyy
    start  = 31Dec2019
    start  = 20191231  # y/m/d
    """

    SUPPORTED_TYPES = Union[pd.Timestamp, str, dt.datetime, Enum, int]

    def __init__(self,
                 start: SUPPORTED_TYPES = None,
                 end: SUPPORTED_TYPES = None,
                 tz: str = None):

        # internal data
        self.start: Optional[pd.Timestamp]
        self.end: Optional[pd.Timestamp]

        if start is not None:
            if isinstance(start, str):  # most expected
                self.start = pd.Timestamp(start)
            elif isinstance(start, pd.Timestamp):  # 2nd most expected
                self.start = start
            elif isinstance(start, dt.datetime):
                self.start = pd.Timestamp(start)
            elif isinstance(start, Enum):  # enum of type string or pd.Timestamp can be used
                self.start = pd.Timestamp(start.value)
            elif isinstance(start, int):  # year is passed
                self.start = pd.Timestamp(dt.datetime(year=start, month=1, day=1))
            else:
                raise TypeError(f"unsuported type for date {start} of {type(start)}")
        else:
            self.start = None

        if end is not None:
            if isinstance(end, str):  # most expected
                self.end = pd.Timestamp(end)
            elif isinstance(end, pd.Timestamp):  # send most expected
                self.end = end
            elif isinstance(end, dt.datetime):
                self.end = pd.Timestamp(end)
            elif isinstance(end, Enum):  # enum of type string can be used
                self.end = pd.Timestamp(end.value)
            elif isinstance(end, int):  # year is passed
                self.end = pd.Timestamp(dt.datetime(year=end, month=1, day=1))
            else:
                raise TypeError(f"unsuported type for date {end} of {type(end)}")
        else:
            self.end = None

        self.tz = tz
        if tz is not None:
            self.start, self.end = tz_localize_dates(start_date=self.start, end_date=self.end, tz=tz)

    def print(self) -> None:
        print(f"start={self.start}, end={self.end}")

    def copy(self) -> TimePeriod:
        return TimePeriod(start=self.start, end=self.end)

    def tz_localize(self, tz: str = 'UTC') -> TimePeriod:
        start, end = tz_localize_dates(start_date=self.start, end_date=self.end, tz=tz)
        return TimePeriod(start, end)

    def to_str(self,
               date_separator: str = ' - ',
               is_increase_by_one_day: bool = False,
               date_format: Optional[str] = DATE_FORMAT
               ) -> str:
        if self.start is not None:
            if is_increase_by_one_day:
                start_date_str = shift_date_by_day(self.start, backward=False).strftime(date_format)
            else:
                start_date_str = self.start.strftime(date_format)
        else:
            start_date_str = ''

        if self.end is not None:
            end_date_str = self.end.strftime(date_format)
        else:
            end_date_str = ''

        label = f"{start_date_str}{date_separator}{end_date_str}"
        return label

    def start_to_str(self, format: Optional[str] = DATE_FORMAT) -> str:
        if format is None:
            start_str = str(self.start.year)
        else:
            start_str = self.start.strftime(format)
        return start_str

    def end_to_str(self, format: Optional[str] = DATE_FORMAT) -> str:
        end_date = self.end or dt.date.today()
        if format is None:
            end_str = str(end_date.year)
        else:
            end_str = end_date.strftime(format)
        return end_str

    def locate(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        truncate timeseries data to given timeperiod
        """
        if isinstance(df.index, pd.DatetimeIndex):
            tz = df.index.tz
            if tz is not None:
                time_period = self.tz_localize(tz=tz)
                start, end = time_period.start, time_period.end
            else:
                start, end = self.start, self.end
            df = df.loc[start:end]
        else:
            pass
            # print(f"df index type is {type(df.index)}")
        return df

    def fill_outside(self,
                     df: Union[pd.DataFrame, pd.Series],
                     fill_value: float = np.nan
                     ) -> Union[pd.DataFrame, pd.Series]:
        """
        fill given value outside of timeperiod
        """
        if isinstance(df.index, pd.DatetimeIndex):
            tz = df.index.tz
            if tz is not None:
                time_period = self.tz_localize(tz=tz)
                start, end = time_period.start, time_period.end
            else:
                start, end = self.start, self.end
            if start is not None:
                df.loc[:start] = fill_value
            if end is not None:
                df.loc[end:] = fill_value
        else:
            pass
            # print(f"df index type is {type(df.index)}")
        return df

    def to_pd_datetime_index(self,
                             freq: str = 'B',
                             hour_offset: Optional[int] = None,
                             tz: Optional[str] = None,
                             include_start_date: bool = False,
                             include_end_date: bool = False,
                             is_business_dates: bool = True,
                             days_shift: Optional[int] = None
                             ) -> pd.DatetimeIndex:
        """
        generate pd dt_time index
        """
        sed = self.copy()
        sed.tz = tz
        if days_shift is not None:
            sed.end = shift_date_by_day(sed.end, num_days=days_shift, backward=False)
        pd_datetime_index = generate_dates_schedule(sed, freq=freq, hour_offset=hour_offset,
                                                    include_start_date=include_start_date,
                                                    include_end_date=include_end_date,
                                                    is_business_dates=is_business_dates)
        return pd_datetime_index

    def to_period_dates_str(self,
                            freq: str = 'B',
                            tz: Optional[str] = None,
                            date_format: str = '%Y%m%d',
                            holidays: pd.DatetimeIndex = None
                            ) -> List[str]:
        """
        nb. can pass calendar
        pd.date_range(holidays=holidays)
        """
        dates = self.to_pd_datetime_index(freq=freq, tz=tz)
        date_strs = pd.Series(dates).apply(lambda x: x.strftime(date_format)).to_list()
        return date_strs

    def shift_end_date_by_days(self, backward: bool = True, num_days: int = 1) -> TimePeriod:
        return TimePeriod(start=self.start,
                          end=shift_date_by_day(self.end, backward=backward, num_days=num_days))

    def shift_start_date_by_days(self, backward: bool = False, num_days: int = 1) -> TimePeriod:
        return TimePeriod(start=shift_date_by_day(self.start, backward=backward, num_days=num_days),
                          end=self.end)

    def get_time_period_an(self) -> float:  # annualised time period
        return get_time_to_maturity(maturity_time=self.end,
                                    value_time=self.start)


def truncate_prior_to_start(df: Union[pd.DataFrame, pd.Series],
                            start: pd.Timestamp
                            ) -> Union[pd.DataFrame, pd.Series]:
    """
    truncate timeseries data from start with a data point including or prior to the star
    """
    if isinstance(df, pd.DataFrame):
        df_ = df.loc[start:, :]
        if df_.index[0] != start:
            # take last row before cutoff date
            row_before = df.loc[:start, :].iloc[-1, :].to_frame().T
            df_ = pd.concat([row_before, df_], axis=0)

    elif isinstance(df, pd.Series):
        df_ = df.loc[start:]
        if df_.index[0] != start:
            # take last row before cutoff date
            ds_before = df.loc[:start]
            row_before = pd.Series(ds_before.iloc[-1], index=[ds_before.index[-1]], name=df.name)
            df_ = row_before.append(df_)
    else:
        raise NotImplementedError(f"{type(df)}")
    return df_


def get_time_period(df: Union[pd.Series, pd.DataFrame], tz: str = None) -> TimePeriod:
    """
    get tz-aware start end dates
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"df.index must be type of pd.DatetimeIndex not of {type(df.index)}")
    if len(df.index) > 0:
        output = TimePeriod(start=df.index[0], end=df.index[-1], tz=tz or df.index.tz)
    else:
        output = TimePeriod()
    return output


def shift_time_period_by_days(time_period: TimePeriod, is_increase_by_one_day: bool = True) -> TimePeriod:
    if is_increase_by_one_day:
        start_date = shift_date_by_day(time_period.start, backward=False)
    else:
        start_date = time_period.start
    end_date = time_period.end

    return TimePeriod(start=start_date, end=end_date)


def get_time_period_label(data: pd.DataFrame,
                          date_separator: str = ':',
                          is_increase_by_one_day: bool = False,
                          date_format: str = DATE_FORMAT
                          ) -> str:
    time_period = TimePeriod(start=data.index[0], end=data.index[-1])
    time_period_label = time_period.to_str(date_format=date_format,
                                           date_separator=date_separator,
                                           is_increase_by_one_day=is_increase_by_one_day)
    return time_period_label


def get_time_period_shifted_by_years(time_period: TimePeriod,
                                     n_years: int = 1,
                                     backward: bool = True
                                     ) -> TimePeriod:
    end_date = time_period.end
    if end_date is not None:
        start_date = shift_dates_by_n_years(dates=end_date, n_years=n_years, backward=backward)
        time_period = TimePeriod(start_date, end_date)
    else:
        time_period = time_period
    return time_period


def get_ytd_time_period(year: int = 2023) -> TimePeriod:
    end = get_current_time_with_tz()
    if year is not None:
        year = end.year
    return TimePeriod(start=dt.datetime(year=year - 1, month=12, day=31), end=get_current_time_with_tz(tz=None))


def generate_dates_schedule(time_period: TimePeriod,
                            freq: str = 'ME',
                            hour_offset: Optional[int] = None,
                            include_start_date: bool = False,
                            include_end_date: bool = False,
                            is_business_dates: bool = True
                            ) -> pd.DatetimeIndex:
    """
    tz-aware rebalancing dates
    """
    if freq == 'SE':  # simple start end
        dates_schedule = pd.DatetimeIndex(data=[time_period.start, time_period.end], tz=time_period.tz)
        return dates_schedule

    end_date = time_period.end
    if end_date is None:
        raise ValueError(f"end_date must be given")
    is_24_hour_offset = False
    if freq == 'h':  # need to do offset so the end of the day will be at 22:00:00 end date
        if end_date.hour == 0:
            is_24_hour_offset = True
        end_date = time_period.end + pd.offsets.Hour(24)

    def create_range(freq_: str, tz: Optional[str] = time_period.tz) -> pd.DatetimeIndex:
        if is_business_dates:
            return pd.bdate_range(start=time_period.start,
                                  end=end_date,
                                  freq=freq_,
                                  tz=tz)
        else:
            return pd.date_range(start=time_period.start,
                                 end=end_date,
                                 freq=freq_,
                                 tz=tz)

    if freq == 'M-FRI':  # last friday of month
        # create weekly fridays
        dates_schedule1 = create_range(freq_='W-FRI', tz=time_period.tz)
        dates_schedule2 = create_range(freq_='ME', tz=time_period.tz)
        # filter last Friday per month periods
        dates_schedule = pd.Series(dates_schedule1, index=dates_schedule1).reindex(index=dates_schedule2, method='ffill').dropna()
        dates_schedule = pd.DatetimeIndex(dates_schedule.to_numpy())  # back to DatetimeIndex type
        if include_end_date is False:
            dates_schedule = dates_schedule[:-1]

    elif freq == 'Q-FRI':  # last friday of quarter
        # create weekly fridays
        dates_schedule1 = create_range(freq_='W-FRI', tz=time_period.tz)
        dates_schedule2 = create_range(freq_='QE', tz=time_period.tz)
        # filter last Friday per quarter periods
        dates_schedule = pd.Series(dates_schedule1, index=dates_schedule1).reindex(index=dates_schedule2, method='ffill').dropna()
        dates_schedule = pd.DatetimeIndex(dates_schedule.to_numpy())  # back to DatetimeIndex type
        if include_end_date is False:
            dates_schedule = dates_schedule[:-1]

    elif freq == 'Q-3FRI':  # last friday of quarter
        # create monthly 3rd fridays
        dates_schedule1 = create_range(freq_='WOM-3FRI', tz=time_period.tz)
        dates_schedule2 = create_range(freq_='QE', tz=time_period.tz)
        # filter last Friday per quarter periods
        dates_schedule = pd.Series(dates_schedule1, index=dates_schedule1).reindex(index=dates_schedule2, method='ffill').dropna()
        dates_schedule = pd.DatetimeIndex(dates_schedule.to_numpy())  # back to DatetimeIndex type
        if include_end_date is False:
            dates_schedule = dates_schedule[:-1]

    elif '_' in freq:
        # support for bespoke D_8H  # daily at 8H utc
        ss = freq.split('_')
        freq = ss[0]
        hour_offset = int(re.findall(r'\d+', ss[-1])[0])
        dates_schedule1 = create_range(freq_=freq, tz=time_period.tz)
        dates_schedule = pd.DatetimeIndex([x + pd.DateOffset(hours=hour_offset) for x in dates_schedule1])
        if include_end_date is False:
            dates_schedule = dates_schedule[:-1]

    else:
        dates_schedule = create_range(freq_=freq)

    # handle case when nothing is in
    if len(dates_schedule) == 0:
        if include_start_date and include_end_date:
            dates_schedule = pd.DatetimeIndex([time_period.start, time_period.end])
        elif include_start_date:
            dates_schedule = pd.DatetimeIndex([time_period.start])
        elif include_end_date:
            dates_schedule = pd.DatetimeIndex([time_period.end])
    elif freq == 'h':
        dates_schedule = dates_schedule[dates_schedule >= time_period.start]
        if is_24_hour_offset:
            dates_schedule = dates_schedule[:-1]  # drop the next dat at 00:00:00
        else:
            dates_schedule = dates_schedule[dates_schedule <= time_period.end]
        if include_end_date and dates_schedule[-1] < time_period.end:  # append date scedule with last elemnt
            dates_schedule = dates_schedule.append(pd.DatetimeIndex([time_period.end]))
    else:
        # hour offset should not be beyond the end period
        if hour_offset is not None:
            dates_schedule = pd.DatetimeIndex([x + pd.DateOffset(hours=hour_offset) for x in dates_schedule])

        if include_start_date:  # irrespective of hour offset
            #if hour_offset is not None:
            #    this_start = time_period.start + pd.DateOffset(hours=hour_offset)
            #else:
            this_start = time_period.start
            if dates_schedule[0] > this_start:
                # create start date and append the dates schedule
                dates_schedule = (pd.DatetimeIndex([this_start])).append(dates_schedule)

        if include_end_date:
            #if hour_offset is not None:
            #    this_end = time_period.end + pd.DateOffset(hours=hour_offset)
            #else:
            this_end = time_period.end
            if dates_schedule[-1] < this_end:  # append date scedule with last elemnt
                dates_schedule = dates_schedule.append(pd.DatetimeIndex([this_end]))

    return dates_schedule


def generate_rebalancing_indicators(df: Union[pd.DataFrame, pd.Series],
                                    freq: str = 'ME',
                                    include_start_date: bool = False,
                                    include_end_date: bool = False
                                    ) -> pd.Series:
    """
    tz awre rebalancing date indicators for rebalancing at data index
    """
    dates_schedule = generate_dates_schedule(time_period=get_time_period(df=df),
                                             freq=freq,
                                             include_start_date=include_start_date,
                                             include_end_date=include_end_date)

    all_dates_indicators = pd.Series(data=True, index=dates_schedule)  # all indicators

    # on time grid
    indicators_on_grid = all_dates_indicators.reindex(index=df.index).dropna()

    # off time grid
    indicators_off_grid = all_dates_indicators.iloc[np.in1d(all_dates_indicators.index, indicators_on_grid.index) == False]
    next_dates_off_grid = pd.Series(df.index, index=df.index).reindex(index=indicators_off_grid.index, method='bfill')
    indicators_off_grid = pd.Series(data=True, index=next_dates_off_grid.to_numpy())

    indicators_on_grid = pd.concat([indicators_on_grid, indicators_off_grid], axis=0).sort_index()

    indicators_full = pd.Series(data=np.where(np.in1d(df.index, indicators_on_grid.index), True, False), index=df.index)

    return indicators_full


def generate_sample_dates(time_period: TimePeriod,
                          freq: str = 'ME',
                          overlap_frequency: str = None,
                          include_start_date: bool = False,
                          include_end_date: bool = False
                          ) -> pd.DataFrame:
    """
    generate df with columns [start, end]
    """
    # generate end dates: DatetimeIndex
    end_dates = generate_dates_schedule(time_period=time_period,
                                        freq=freq,
                                        include_start_date=include_start_date,
                                        include_end_date=include_end_date)
    # convert to datetime.datetime
    py_end_dates = end_dates.to_pydatetime()

    # TODO: add if include first date
    # TODO: when start_N and end date_N-1 are non-overlaping
    if overlap_frequency is None:  # StartDate_N =  EndDate_{N-1}
        start_dates = py_end_dates[:-1]  # from 0 up to last-1
        end_dates = py_end_dates[1:]  # from 1 up to last
        sample_dates = pd.DataFrame({'start': start_dates, 'end': end_dates}, index=end_dates)

    elif overlap_frequency == 'YE':
        start_dates = shift_dates_by_year(end_dates, backward=True)  # from 0 up to last-1
        end_dates = py_end_dates  # from 1 up to last
        # cut dates before start date
        sample_dates = pd.DataFrame({'start': start_dates, 'end': end_dates}, index=end_dates)
        sample_dates = sample_dates[sample_dates['start'] > time_period.start]

    # frequncy of type 1A, 2A, 3A,...
    elif list(overlap_frequency)[-1] == 'YE':
        n_years = int(separate_number_from_string(overlap_frequency)[0])
        start_dates = shift_dates_by_n_years(dates=end_dates, n_years=n_years, backward=True)
        end_dates = py_end_dates  # from 1 up to last
        # cut dates before start date
        sample_dates = pd.DataFrame({'start': start_dates, 'end': end_dates}, index=end_dates)
        sample_dates = sample_dates[sample_dates['start'] > time_period.start]
    else:
        raise TypeError(f"overlap_frequency={overlap_frequency} is not implemented")

    return sample_dates


def is_leap_year(year: int) -> bool:
    """
    Check if the int given year is a leap year
    return true if leap year or false otherwise
    """
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def get_weekday(dates: Union[pd.DatetimeIndex, pd.Index]) -> List[str]:
    weekdays = [WEEKDAYS[date.weekday()] for date in dates]
    return weekdays


def get_year_quarter(dates: list, date_format: str = 'Q%d-%d') -> List[str]:
    year_quarter = [date_format % (np.ceil(date.month / 3), date.year) for date in dates]
    return year_quarter


def get_month_days(month: int, year: int) -> int:
    """
    Inputs -> month, year Booth integers
    Return the number of days of the given month
    """
    thirty_days_months: List = [4, 6, 9, 11]
    thirtyone_days_months: List = [1, 3, 5, 7, 8, 10, 12]

    if month in thirty_days_months:   # April, June, September, November
        return 30
    elif month in thirtyone_days_months:   # January, March, May, July, August, October, December
        return 31
    else:   # February
        if is_leap_year(year):
            return 29
        else:
            return 28


def shift_date_by_day(date: pd.Timestamp, backward: bool = True, num_days: int = 1) -> pd.Timestamp:
    if backward:
        date1 = date - pd.offsets.Day(num_days)
    else:
        date1 = date + pd.offsets.Day(num_days)

    return date1


def shift_dates_by_year(dates: Union[pd.Timestamp, pd.DatetimeIndex],
                        backward: bool = True
                        ) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """
    shift dates by 365 for non leap year and 366 for leap year
    """
    if isinstance(dates, pd.Timestamp):
        dates1 = [dates]
    else:
        dates1 = dates

    shifted_dates = []
    for date in dates1:
        if is_leap_year(date.year):
            delta = dt.timedelta(days=366)
        else:
            delta = dt.timedelta(days=365)

        if backward:
            delta = -delta

        shifted_dates.append(date + delta)

    if isinstance(dates, pd.Timestamp):
        shifted_dates = shifted_dates[0]

    return shifted_dates


def shift_dates_by_n_years(dates: Union[pd.Timestamp, pd.DatetimeIndex],
                           n_years: int = 1,
                           backward: bool = True
                           ) -> Union[pd.Timestamp, pd.DatetimeIndex]:

    shifted_dates = dates
    for n in range(1, n_years+1):
        shifted_dates = shift_dates_by_year(shifted_dates, backward=backward)
    return shifted_dates


def months_between(date1: dt.datetime,
                   date2: dt.datetime
                   ) -> int:
    if date1 > date2:
        date1, date2 = date2, date1
    m1 = date1.year*12+date1.month
    m2 = date2.year*12+date2.month
    months = m2 - m1
    if date1.day > date2.day:
        # months- = 1#need to account for leap
        pass
    elif date1.day == date2.day:
        seconds1 = date1.hour*3600+date1.minute+date1.second
        seconds2 = date2.hour*3600+date2.minute+date2.second
        if seconds1 > seconds2:
            months -= 1

    return months


def tz_localize_dates(start_date: pd.Timestamp,
                      end_date: pd.Timestamp,
                      tz: str = 'UTC'
                      ) -> Tuple[pd.Timestamp, pd.Timestamp]:

    if start_date is not None and start_date.tz is None:
        start = start_date.tz_localize(tz)
    else:
        start = start_date
    if end_date is not None and end_date.tz is None:
        end = end_date.tz_localize(tz)
    else:
        end = end_date
    return start, end


def split_df_by_freq(df: pd.DataFrame,
                     freq: str = 'ME',
                     overlap_frequency: str = None,
                     include_start_date: bool = True,
                     include_end_date: bool = True
                     ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    take pandas data sampled at some freq and split into lower freq
    one caveat: the correct date of he split must be last day of sample_dates
    for consistency wih different calenders
    get sample start end dates as data
    """
    time_period = TimePeriod(start=df.index[0], end=df.index[-1])

    sample_dates = generate_sample_dates(time_period=time_period,
                                         freq=freq,
                                         overlap_frequency=overlap_frequency,
                                         include_start_date=include_start_date,
                                         include_end_date=include_end_date)
    df_split = {}
    for row in sample_dates.itertuples():
        df_split[row.end] = df[row.start:row.end]
    return df_split


def get_sample_dates_idx(population_dates: pd.Index, sample_dates: pd.Index) -> List[int]:
    """
    get indixes of data sample
    """
    # use series indexs at population_dates
    pd_population_dates = pd.Series(population_dates, index=population_dates, name='population_dates')

    # sampled_population data will have last available dates in population_dates matched to the index population_dates
    sampled_population = pd_population_dates.reindex(index=sample_dates, method='ffill')

    # reset index will be idx of pd_population_dates masked with cond
    cond = np.in1d(population_dates, sampled_population, assume_unique=True)
    sample_dates_idx = pd_population_dates.reset_index().loc[cond].index.to_list()
    return sample_dates_idx


def generate_fixed_maturity_rolls(time_period: TimePeriod,
                                  freq: str = 'h',
                                  roll_freq: str = 'W-FRI',
                                  roll_hour: Optional[int] = 8,
                                  min_days_to_next_roll: int = 6,
                                  include_end_date: bool = False,
                                  future_days_offset: int = 360
                                  ) -> pd.Series:
    """
    for given time_period generate fixed maturity rolls
    rolls occur when (current_roll - value_time).days < min_days_to_next_roll
    """
    observation_times = generate_dates_schedule(time_period,
                                                freq=freq,
                                                include_start_date=True,
                                                include_end_date=include_end_date)
    # use large day shift to cover at least next quarter
    roll_days = generate_dates_schedule(time_period.shift_end_date_by_days(num_days=future_days_offset, backward=False),
                                        freq=roll_freq,
                                        hour_offset=roll_hour)

    if len(roll_days) == 1:
        roll_schedule = pd.Series(roll_days[0], index=observation_times)
    else:
        roll_schedule = {}
        starting_roll_idx = 0
        next_roll = roll_days[starting_roll_idx]
        for observed_time in observation_times:
            diff = (next_roll - observed_time).days
            if diff < min_days_to_next_roll:
                if starting_roll_idx + 1 < len(roll_days):
                    starting_roll_idx += 1
                    next_roll = roll_days[starting_roll_idx]
                else:
                    print(f"increase end date for {time_period.end} to extend to next roll, "
                          f"meanwhile using last available roll={next_roll} @ {observed_time}")
            roll_schedule[observed_time] = next_roll
        roll_schedule = pd.Series(roll_schedule)
    return roll_schedule


def min_timestamp(timestamp1: Union[str, pd.Timestamp],
                  timestamp2: Union[str, pd.Timestamp],
                  tz: str = 'UTC'
                  ) -> pd.Timestamp:
    """
    find min timespamp
    """
    if isinstance(timestamp1, str):
        timestamp1 = pd.Timestamp(timestamp1, tz=tz)
    if isinstance(timestamp2, str):
        timestamp2 = pd.Timestamp(timestamp2, tz=tz)
    min_date = timestamp1 if timestamp1 < timestamp2 else timestamp2
    return min_date


class UnitTests(Enum):
    DATES = 0
    OFFSETS = 1
    WEEK_DAY = 2
    SAMPLE_DATES_IDX = 3
    PERIOD_WITH_HOLIDAYS = 4
    EOD_FREQ_HOUR = 5
    FREQ_HOUR = 6
    FREQ_REB = 7
    FREQS = 8
    REBALANCING_INDICATORS = 9
    FIXED_MATURITY_ROLLS = 10


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.DATES:
        time_period = TimePeriod(start='31Dec2019', end='31Dec2020')
        print("time_period.to_str() = ")
        print(time_period.to_str())
        print("time_period.start_to_str() = ")
        print(time_period.start_to_str())
        print("time_period.end_to_str() = ")
        print(time_period.end_to_str())
        print("time_period.to_pd_datetime_index(freq='ME') = ")
        print(time_period.to_pd_datetime_index(freq='ME'))
        print("time_period.to_pd_datetime_index(freq='ME') = ")
        print(time_period.to_pd_datetime_index(freq='ME', tz='America/New_York'))
        print("time_period.get_period_dates_str(freq='ME', tz='America/New_York') = ")
        print(time_period.to_period_dates_str(freq='ME'))

    elif unit_test == UnitTests.OFFSETS:
        date = pd.Timestamp('28Feb2023')
        print(date-pd.offsets.Day(30))
        print(date-pd.offsets.MonthEnd(1))

        print(shift_dates_by_year(date))
        print(date-pd.offsets.Day(365))
        print(date-pd.offsets.BYearBegin(1))

    elif unit_test == UnitTests.WEEK_DAY:
        time_period = TimePeriod(start='01Jun2022', end='22Jun2022')
        sample_dates = generate_sample_dates(time_period=time_period, freq='D')
        print(get_weekday(sample_dates.index))

    elif unit_test == UnitTests.SAMPLE_DATES_IDX:
        time_period = TimePeriod(start='31Dec2004', end='31Dec2020')
        population_dates = time_period.to_pd_datetime_index(freq='ME')
        sample_dates = time_period.to_pd_datetime_index(freq='QE')
        sample_dates_idx = get_sample_dates_idx(population_dates=population_dates, sample_dates=sample_dates)

        print('population_dates')
        print(population_dates)
        print('sample_dates')
        print(sample_dates)
        print('sample_dates_idx')
        print(sample_dates_idx)

        print('sampled_population_dates')
        print(population_dates[sample_dates_idx])

    elif unit_test == UnitTests.PERIOD_WITH_HOLIDAYS:

        time_period = TimePeriod(start='01Jan2020', end='22Jan2020')
        holidays = pd.DatetimeIndex(data=['20200113', '20200120', '20200214', '20200217',
                                          '20200217', '20200219', '20200228', '20200324', '20200410'])

        dates = time_period.to_period_dates_str(freq='B', holidays=holidays)
        print(dates)

    elif unit_test == UnitTests.EOD_FREQ_HOUR:
        time_period = TimePeriod(pd.Timestamp('2022-11-10', tz='UTC'), pd.Timestamp('2023-03-03', tz='UTC'))
        rebalancing_times = generate_dates_schedule(time_period=time_period, freq='h')
        print(rebalancing_times)

    elif unit_test == UnitTests.FREQ_HOUR:

        # hour frequency with timestamp with no hour
        time_period = TimePeriod(pd.Timestamp('2022-11-10', tz='UTC'), pd.Timestamp('2022-11-16', tz='UTC'))
        rebalancing_times = generate_dates_schedule(time_period=time_period, freq='8H')
        print(rebalancing_times)

        rebalancing_times = generate_dates_schedule(time_period=time_period, freq='D_8H')
        print(rebalancing_times)

        value_time = pd.Timestamp('2023-10-04 08:00:00+00:00').normalize()
        print(value_time)
        time_period = TimePeriod(value_time, value_time)
        rebalancing_times = generate_dates_schedule(time_period=time_period, freq='h')
        print(rebalancing_times)

    elif unit_test == UnitTests.FREQ_REB:
        dates_schedule = generate_dates_schedule(time_period=TimePeriod('2023-05-01 08:00:00', '2023-05-30 10:00:00', tz='UTC'),
                                                 freq='2W-FRI', hour_offset=8,
                                                 include_start_date=True,
                                                 include_end_date=True)
        print(dates_schedule)
        dates_schedule = generate_dates_schedule(time_period=TimePeriod('2023-05-01 08:00:00', '2023-05-30 10:00:00', tz='UTC'),
                                                 freq='2W-FRI', hour_offset=8,
                                                 include_start_date=False,
                                                 include_end_date=False)
        print(dates_schedule)

    elif unit_test == UnitTests.FREQS:
        freq_map = FreqMap.BQ
        freq_map.print()
        freq = freq_map.to_freq()
        print(freq)
        n_bus_days = freq_map.to_n_bus_days()
        print(n_bus_days)

    elif unit_test == UnitTests.REBALANCING_INDICATORS:
        pd_index = pd.date_range(start='31Dec2020', end='31Dec2021', freq='W-MON')
        data = pd.DataFrame(range(len(pd_index)), index=pd_index, columns=['aaa'])
        rebalancing_schedule = generate_rebalancing_indicators(df=data, freq='M-FRI')
        print(rebalancing_schedule)
        print(rebalancing_schedule[rebalancing_schedule == True])

    elif unit_test == UnitTests.FIXED_MATURITY_ROLLS:
        time_period = TimePeriod('01Oct2022', '21Feb2023', tz='UTC')
        """
        weekly_rolls = generate_fixed_maturity_rolls(time_period=time_period, freq='h', roll_freq='W-FRI',
                                                     roll_hour=8,
                                                     min_days_to_next_roll=6)
        print(f"weekly_rolls:\n{weekly_rolls}")

        monthly_rolls = generate_fixed_maturity_rolls(time_period=time_period, freq='h', roll_freq='M-FRI',
                                                      roll_hour=8,
                                                      min_days_to_next_roll=28)  # 4 weeks before
        print(f"monthly_rolls:\n{monthly_rolls}")
        """
        quarterly_rolls = generate_fixed_maturity_rolls(time_period=time_period, freq='h', roll_freq='Q-FRI',
                                                        roll_hour=8,
                                                        min_days_to_next_roll=28)  # 4 weeks before
        print(f"quarterly_rolls:\n{quarterly_rolls}")
        quarterly_rolls = generate_fixed_maturity_rolls(time_period=time_period, freq='h', roll_freq='Q-FRI',
                                                        roll_hour=8,
                                                        min_days_to_next_roll=56)  # 8 weeks before
        print(f"quarterly_rolls:\n{quarterly_rolls}")


if __name__ == '__main__':

    unit_test = UnitTests.FREQ_HOUR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
