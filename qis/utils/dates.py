"""
implementation of core dates analytics and frequency with TimePeriod method
"""

from __future__ import annotations  # to allow class method annotations

import datetime as dt
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Optional, NamedTuple, Dict
from enum import Enum

import qis.utils.np_ops as npo
from qis.utils.struct_ops import separate_number_from_string

DATE_FORMAT = '%d%b%Y'  # 31Jan2020 - common across all reporting meta
DATE_FORMAT_INT = '%Y%m%d'  # 20000131

WEEKDAYS: List[str] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


BUS_DAYS_PER_YEAR = 252  # applied for volatility normalization
WEEK_DAYS_PER_YEAR = 260  # calendar days excluding weekends in a year
CALENDAR_DAYS_PER_YEAR = 360
CALENDAR_DAYS_IN_MONTH = 30
CALENDAR_DAYS_PER_YEAR_SHARPE = 365.25  # for total return computations for Sharpe


def get_today_str(format: str = DATE_FORMAT) -> str:
    return dt.datetime.now().date().strftime(format)


def get_today_eod_with_tz(tz: str = 'UTC', days_offset: int = None) -> pd.Timestamp:
    eod = pd.Timestamp.today(tz=tz).normalize()  # normalize to eod date
    if days_offset is not None:
        eod = eod + pd.DateOffset(days=days_offset)
    return eod


def get_current_date(day_offset: int = 0) -> pd.Timestamp:
    """
    date without time
    """
    current_date_time = dt.datetime.now()
    current_date = pd.Timestamp(dt.datetime(year=current_date_time.year,
                                            month=current_date_time.month,
                                            day=current_date_time.day))
    current_date = shift_date_by_day(date=current_date, num_days=day_offset, backward=False)
    return current_date


def find_min_time(date1: Union[str, pd.Timestamp], date2: Union[str, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(date1, str):
        date1 = pd.Timestamp(date1, tz='UTC')
    if isinstance(date2, str):
        date2 = pd.Timestamp(date2, tz='UTC')
    min_date = date1 if date1 < date2 else date2
    return min_date


def is_business_day(date: pd.Timestamp) -> bool:
    return bool(len(pd.bdate_range(date, date)))


class FreqData(NamedTuple):
    """
    enumerate frequencies with caption, python alias and set n_bus and salendar days
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
    elif freq in ['H']:
        days = 1.0 / 24.0
        an_f = an_days * 24.0
    elif freq in ['D']:
        days = 1
        an_f = an_days
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
    elif freq in ['1M', 'M', 'BM', 'MS', 'BMS']:
        days = 30 if is_calendar else 21
        an_f = 12
    elif freq in ['2M', '2BM', '2MS', '2BMS']:
        days = 60 if is_calendar else 42
        an_f = 6
    elif freq in ['Q', 'DQ', 'BQ', 'QS', 'BQS', 'Q-DEC', 'Q-JAN', 'Q-FEB']:
        days = 91 if is_calendar else 63
        an_f = 4
    elif freq in ['2Q', '2BQ', '2QS', '2BQS']:
        days = 182 if is_calendar else 126
        an_f = 2
    elif freq in ['3Q', '3BQ', '3QS', '3BQS']:
        days = 273 if is_calendar else 189
        an_f = 0.75
    elif freq in ['A', 'BA', 'AS', 'BAS']:
        days = an_days
        an_f = 1.0
    else:
        raise TypeError(f'freq={freq} is not impelemnted')

    return days, an_f


def get_an_factor(freq: str = 'B',
                  is_calendar: bool = False
                  ) -> float:
    period_days, _ = get_period_days(freq=freq, is_calendar=is_calendar)
    annual_days, _ = get_period_days(freq='A', is_calendar=is_calendar)
    an = annual_days / period_days
    return an


def infer_an_from_data(data: Union[pd.DataFrame, pd.Series], is_calendar: bool = False) -> float:
    freq = pd.infer_freq(data.index)
    if freq is None:
        print(f"in infer_an_from_data: cannot infer {freq} - using 252")
        return 252.0
    an, an_f = get_period_days(freq, is_calendar=is_calendar)
    return an_f


def get_return_an(freq: str) -> float:
    return get_an_factor(freq)


def get_vol_an(freq: str) -> float:
    return np.sqrt(get_an_factor(freq))


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
            self.end = None  # get_current_date()

        self.tz = tz
        if tz is not None:
            self.start, self.end = tz_localize_dates(start_date=self.start, end_date=self.end)

    def print(self) -> None:
        print(self.to_str())

    def copy(self) -> TimePeriod:
        return TimePeriod(start=self.start, end=self.end)

    def tz_localize(self, tz: str = 'UTC') -> TimePeriod:
        start, end = tz_localize_dates(start_date=self.start, end_date=self.end)
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

    def start_to_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    def end_to_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)

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

    def fill_outside(self, df: Union[pd.DataFrame, pd.Series], fill_value: float = np.nan) -> Union[pd.DataFrame, pd.Series]:
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

    def locate_with_fill_to_end(self,
                                df: Union[pd.DataFrame, pd.Series],
                                freq: str = 'B'
                                ) -> Union[pd.DataFrame, pd.Series]:
        loc_data = self.locate(df=df)
        if loc_data.index[-1] < self.end:
            remained_time_period = TimePeriod(loc_data.index[-1], self.end).shift_start_date_by_days(backward=False)
            remained_index = remained_time_period.to_pd_datetime_index(freq=freq)
            if isinstance(df, pd.DataFrame):
                remainded_data = pd.DataFrame(data=npo.np_array_to_df_index(df.iloc[-1, :].to_numpy(),
                                                                            n_index=len(remained_index)),
                                              index=remained_index,
                                              columns=df.columns)
            else:
                remainded_data = pd.Series(np.tile(df.iloc[-1], len(remained_index)),
                                           index=remained_index,
                                           name=df.name)
            loc_data = loc_data.append(remainded_data)
        return loc_data

    def is_before_start_date(self, date: pd.Timestamp):
        before_start_date = False
        if self.end is not None and date < self.start:
            before_start_date = True
        return before_start_date

    def get_data_index_mask(self, dates_index: pd.DatetimeIndex) -> Union[bool, np.ndarray]:
        if self.start is not None and self.end is not None:
            index_mask = np.logical_and(dates_index >= self.start, dates_index <= self.end)
        elif self.start is not None and self.end is None:
            index_mask = dates_index >= self.start
        elif self.start is None and self.end is not None:
            index_mask = dates_index <= self.end
        else:
            index_mask = np.ones_like(dates_index, dtype=bool)
        return index_mask

    def to_pd_datetime_index(self,
                             freq: str = 'B',
                             hours: Optional[int] = None,
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
        pd_datetime_index = generate_dates_schedule(sed, freq=freq, hours=hours,
                                                    include_start_date=include_start_date,
                                                    include_end_date=include_end_date,
                                                    is_business_dates=is_business_dates)
        return pd_datetime_index

    def get_period_dates_str(self,
                             freq: str = 'B',
                             tz: Optional[str] = None,
                             date_format: str = '%Y%m%d',
                             holidays: pd.DatetimeIndex = None
                             ) -> List[str]:
        """
        nb. can pass calendar
        pd.date_range(holidays=holidays)
        """
        dates = self.to_pd_datetime_index(freq=freq, tz=tz, holidays=holidays)
        date_strs = pd.Series(dates).apply(lambda x: x.strftime(date_format)).to_list()
        return date_strs

    def get_months_between(self):
        return months_between(date1=self.start, date2=self.end)

    def shift_end_date_by_days(self, backward: bool = True, num_days: int = 1) -> TimePeriod:
        return TimePeriod(start=self.start,
                          end=shift_date_by_day(self.end, backward=backward, num_days=num_days))

    def shift_start_date_by_days(self, backward: bool = False, num_days: int = 1) -> TimePeriod:
        return TimePeriod(start=shift_date_by_day(self.start, backward=backward, num_days=num_days),
                          end=self.end)


def get_time_period(df: Union[pd.Series, pd.DataFrame]) -> TimePeriod:
    """
    get tz-aware start end dates
    """
    if len(df.index) > 0:
        output = TimePeriod(start=df.index[0], end=df.index[-1], tz=df.index.tz)
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
    start_date = shift_dates_by_n_years(dates=end_date, n_years=n_years, backward=backward)
    return TimePeriod(start_date, end_date)


def get_ytd_time_period(year: int = 2020) -> TimePeriod:
    return TimePeriod(start=dt.datetime(year=year - 1, month=12, day=31))


def generate_dates_schedule(time_period: TimePeriod,
                            freq: str = 'M',
                            hours: Optional[int] = None,
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
    is_24_hour_offset = False
    if freq == 'H':  # need to do offset so the end of the day will be at 22:00:00 end date
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
        dates_schedule_ = create_range(freq_='W-FRI', tz=time_period.tz)
        w_dates_schedule_ = create_range(freq_='W', tz=time_period.tz)
        # filter last Friday per month periods
        dates_schedule = pd.Series(dates_schedule_, index=dates_schedule_).reindex(index=dates_schedule_, method='ffill')
        #dates_schedule = dates_schedule_.to_series().groupby(dates_schedule_.to_period('M')).last()
        dates_schedule = dates_schedule.to_numpy()  # back to DatetimeIndex type
        if include_end_date is False:
            dates_schedule = dates_schedule[:-1]

    elif freq == 'Q-FRI':  # last friday of quarter
        # create weekly fridays
        dates_schedule_ = create_range(freq_='W-FRI', tz=time_period.tz)
        # filter last Friday per quarter periods
        q_dates_schedule_ = create_range(freq_='Q', tz=time_period.tz)
        dates_schedule = pd.Series(dates_schedule_, index=dates_schedule_).reindex(index=q_dates_schedule_, method='ffill')
        # dates_schedule = dates_schedule_.to_series().groupby(dates_schedule_.to_period('Q')).last()
        dates_schedule = dates_schedule.to_numpy()  # back to DatetimeIndex type
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
    else:
        if include_start_date:
            if dates_schedule[0] > time_period.start:
                # create start date and append the dates schedule
                dates_schedule = (pd.DatetimeIndex([time_period.start])).append(dates_schedule)

        if include_end_date and len(dates_schedule) > 0:
            if dates_schedule[-1] < time_period.end: # append date scedule with last elemnt
                dates_schedule = dates_schedule.append(pd.DatetimeIndex([time_period.end]))

    if freq == 'H':
        dates_schedule = dates_schedule[dates_schedule >= time_period.start]
        if is_24_hour_offset:
            dates_schedule = dates_schedule[:-1]  # drop the next dat at 00:00:00
        else:
            dates_schedule = dates_schedule[dates_schedule <= time_period.end]
        if include_end_date and dates_schedule[-1] < time_period.end:  # append date scedule with last elemnt
            dates_schedule = dates_schedule.append(pd.DatetimeIndex([time_period.end]))

    if hours is not None and len(dates_schedule) > 0:
        dates_schedule = pd.DatetimeIndex([x + pd.DateOffset(hours=hours) for x in dates_schedule])

    return dates_schedule


def generate_rebalancing_indicators(df: Union[pd.DataFrame, pd.Series],
                                    freq: str = 'M',
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
                          freq: str = 'M',
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

    elif overlap_frequency == 'A':
        start_dates = shift_dates_by_year(end_dates, backward=True)  # from 0 up to last-1
        end_dates = py_end_dates  # from 1 up to last
        # cut dates before start date
        sample_dates = pd.DataFrame({'start': start_dates, 'end': end_dates}, index=end_dates)
        sample_dates = sample_dates[sample_dates['start'] > time_period.start]

    # frequncy of type 1A, 2A, 3A,...
    elif list(overlap_frequency)[-1] == 'A':
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


def get_month_days(month: int, year: int ) -> int:
    """
    Inputs -> month, year Booth integers
    Return the number of days of the given month
    """
    THIRTY_DAYS_MONTHS: List = [4, 6, 9, 11]
    THIRTYONE_DAYS_MONTHS: List = [1, 3, 5, 7, 8, 10, 12]

    if month in THIRTY_DAYS_MONTHS:   # April, June, September, November
        return 30
    elif month in THIRTYONE_DAYS_MONTHS:   # January, March, May, July, August, October, December
        return 31
    else:   # February
        if is_leap_year(year):
            return 29
        else:
            return 28


def shift_date_by_day(date: pd.Timestamp, backward: bool = True, num_days: int = 1) -> pd.Timestamp:
    if backward:
        date1 = date - dt.timedelta(days=num_days)
    else:
        date1 = date + dt.timedelta(days=num_days)

    return date1


def shift_date_by_month(date: pd.DatetimeIndex, backward: bool = True) -> pd.Timestamp:
    """
    Checks the month of the given date
    Selects the number of days it needs to add one month
    return the date with one month added
    """
    current_month_days = get_month_days(date.month, date.year)
    if backward:
        next_month_days = get_month_days(date.month - 1, date.year)
    else:
        next_month_days = get_month_days(date.month + 1, date.year)

    delta = dt.timedelta(days=current_month_days)

    if backward:
        if date.day < next_month_days:
            delta = -(delta - dt.timedelta(days=(date.day - next_month_days) + 1))
        else:
            delta = - delta
    else:
        if date.day > next_month_days:
            delta = delta - dt.timedelta(days=(date.day - next_month_days) - 1)

    return date + delta


def shift_dates_by_year(dates: Union[pd.Timestamp, pd.DatetimeIndex],
                        backward: bool = True
                        ) -> Union[pd.Timestamp, pd.DatetimeIndex]:

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
                     freq: str = 'M',
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


def rebase_model_dates(model_dates: pd.DatetimeIndex,
                       observation_dates: pd.DatetimeIndex,
                       model_dates_name: str = 'model_dates'
                       ) -> pd.Series:
    """
    generate dates in observation_dates when model was updated in model_dates
    given dates in model schedule, rebase them to observation schedule using ffill
    make a dataframe with dates = given dates in model dates
    rebased index of dates to the samples index, but the date correspond to last date in the correl dates
    """
    pd_model_dates = pd.Series(data=model_dates, index=model_dates, name=model_dates_name)
    model_dates_to_observations = pd_model_dates.reindex(index=observation_dates, method='ffill')
    return model_dates_to_observations


def get_data_at_date(data: pd.DataFrame,
                     given_date: pd.Timestamp
                     ) -> pd.Series:
    """
    get model_date <= given_date
    """
    model_dates_to_observations = rebase_model_dates(model_dates=data.index.unique(0),
                                                     observation_dates=pd.DatetimeIndex([given_date]))
    index = data.index.get_loc(model_dates_to_observations.iloc[0])
    return data.iloc[index]


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


def generate_is_weekend(df: Union[pd.DataFrame, pd.Series],
                        include_start_date: bool = False,
                        include_end_date: bool = False
                        ) -> pd.Series:
    """
    pickup calendar days
    """
    dates_schedule_c = generate_dates_schedule(time_period=get_time_period(df=df),
                                               freq='D',
                                               include_start_date=include_start_date,
                                               include_end_date=include_end_date)
    all_dates_indicators = pd.Series(data=True, index=dates_schedule_c)  # all indicators

    # on time grid
    dates_schedule_b = generate_dates_schedule(time_period=get_time_period(df=df),
                                               freq='B',
                                               include_start_date=include_start_date,
                                               include_end_date=include_end_date)
    dates_schedule_b = pd.Series(data=True, index=dates_schedule_b)  # b indicators

    # off time grid
    bday_indicators = all_dates_indicators.iloc[np.in1d(all_dates_indicators.index, dates_schedule_b.index) == False]
    indicators_full = pd.Series(data=np.where(np.in1d(df.index, bday_indicators.index), True, False), index=df.index)

    return indicators_full


def generate_fixed_maturity_rolls(time_period: TimePeriod,
                                  freq: str = 'H',
                                  roll_freq: str = 'W-FRI',
                                  roll_hour: int = 8,
                                  min_days_to_next_roll: int = 6,
                                  include_end_date: bool = False
                                  ) -> pd.Series:
    """
    for given time_period generate fixed maturity rolls
    rolls occur when (current_roll - value_time).days < min_days_to_next_roll
    """
    observed_times = generate_dates_schedule(time_period,
                                             freq=freq,
                                             include_start_date=True,
                                             include_end_date=include_end_date)
    # use large day shift to cover at least next quarter
    roll_days = generate_dates_schedule(time_period.shift_end_date_by_days(num_days=180, backward=False),
                                        freq=roll_freq,
                                        hours=roll_hour)
    roll_days_ = iter(roll_days)
    next_roll = next(roll_days_)
    roll_schedule = {}
    for observed_time in observed_times:
        diff = (next_roll - observed_time).days
        if diff < min_days_to_next_roll:
            try:
                next_roll = next(roll_days_)
            except StopIteration:
                raise ValueError(f"increase end dat for {time_period.print()}")
        roll_schedule[observed_time] = next_roll
    roll_schedule = pd.Series(roll_schedule)
    return roll_schedule


class UnitTests(Enum):
    DATES = 1
    WEEK_DAY = 2
    SAMPLE_DATES_IDX = 3
    PERIOD_WITH_HOLIDAYS = 4
    FREQ_HOUR = 5
    FREQ_REB = 6
    FREQS = 7
    REBALANCING_INDICATORS = 8
    WEEKEND_INDICATORS = 9
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
        print("time_period.to_pd_datetime_index(freq='M') = ")
        print(time_period.to_pd_datetime_index(freq='M'))
        print("time_period.to_pd_datetime_index(freq='M') = ")
        print(time_period.to_pd_datetime_index(freq='M', tz='America/New_York'))
        print("time_period.get_period_dates_str(freq='M', tz='America/New_York') = ")
        print(time_period.get_period_dates_str(freq='M'))

    elif unit_test == UnitTests.WEEK_DAY:
        time_period = TimePeriod(start='01Jun2022', end='22Jun2022')
        sample_dates = generate_sample_dates(time_period=time_period, freq='D')
        print(get_weekday(sample_dates.index))

    elif unit_test == UnitTests.SAMPLE_DATES_IDX:
        time_period = TimePeriod(start='31Dec2004', end='31Dec2020')
        population_dates = time_period.to_pd_datetime_index(freq='W-WED')
        sample_dates = time_period.to_pd_datetime_index(freq='Q')
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

        dates = time_period.get_period_dates_str(freq='B', holidays=holidays)
        print(dates)

    elif unit_test == UnitTests.FREQ_HOUR:

        # hour frequency with timestamp with no hour
        time_period = TimePeriod(pd.Timestamp('2022-11-10', tz='UTC'), pd.Timestamp('2022-11-16', tz='UTC'))
        rebalancing_times = generate_dates_schedule(time_period=time_period, freq='8H')
        print(rebalancing_times)

        #time_period = TimePeriod(pd.Timestamp('2022-11-10 8:00:00+00:00', tz='UTC'), pd.Timestamp('2022-11-16 8:00:00+00:00', tz='UTC'))
        #rebalancing_times = generate_dates_schedule(time_period=time_period, freq='H')
        #print(rebalancing_times)

    elif unit_test == UnitTests.FREQ_REB:
        dates_schedule = generate_dates_schedule(time_period=TimePeriod('2022-04-08 08:00:00', '2022-04-10 10:00:00', tz='UTC'),
                                                 freq='D', hours=8,
                                                 include_start_date=True,
                                                 include_end_date=True)
        print(dates_schedule)
        dates_schedule = generate_dates_schedule(time_period=TimePeriod('2022-04-08 08:00:00', '2022-04-10 10:00:00', tz='UTC'),
                                                 freq='D', hours=8,
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
        print(rebalancing_schedule[rebalancing_schedule==True])

    elif unit_test == UnitTests.WEEKEND_INDICATORS:
        pd_index = pd.date_range(start='31Dec2021', end='10Jan2022', freq='D')
        data = pd.DataFrame(range(len(pd_index)), index=pd_index, columns=['aaa'])
        rebalancing_schedule = generate_is_weekend(df=data)
        print(rebalancing_schedule)
        print(rebalancing_schedule[rebalancing_schedule==True])

    elif unit_test == UnitTests.FIXED_MATURITY_ROLLS:
        time_period = TimePeriod('01Oct2022', '18Jan2023', tz='UTC')
        weekly_rolls = generate_fixed_maturity_rolls(time_period=time_period, freq='H', roll_freq='W-FRI',
                                                     roll_hour=8,
                                                     min_days_to_next_roll=6)
        print(f"weekly_rolls:\n{weekly_rolls}")

        monthly_rolls = generate_fixed_maturity_rolls(time_period=time_period, freq='H', roll_freq='M-FRI',
                                                      roll_hour=8,
                                                      min_days_to_next_roll=28)  # 4 weeks before
        print(f"monthly_rolls:\n{monthly_rolls}")

        quarterly_rolls = generate_fixed_maturity_rolls(time_period=time_period, freq='H', roll_freq='Q-FRI',
                                                        roll_hour=8,
                                                        min_days_to_next_roll=56)  # 8 weeks before
        print(f"quarterly_rolls:\n{quarterly_rolls}")


if __name__ == '__main__':

    unit_test = UnitTests.FIXED_MATURITY_ROLLS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
