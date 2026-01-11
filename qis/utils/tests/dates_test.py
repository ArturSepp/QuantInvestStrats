
import pandas as pd
from enum import Enum
from qis.utils.dates import (TimePeriod,
                             shift_dates_by_year,
                             generate_sample_dates,
                             get_weekday,
                             get_sample_dates_idx,
                             generate_dates_schedule,
                             FreqMap,
                             generate_rebalancing_indicators,
                             generate_fixed_maturity_rolls,
                             find_upto_date_from_datetime_index,
                             create_rebalancing_indicators_from_freqs)

class LocalTests(Enum):
    DATES = 1
    OFFSETS = 2
    WEEK_DAY = 3
    SAMPLE_DATES_IDX = 4
    PERIOD_WITH_HOLIDAYS = 5
    EOD_FREQ_HOUR = 6
    FREQ_HOUR = 7
    FREQ_REB = 8
    FREQS = 9
    REBALANCING_INDICATORS = 10
    FIXED_MATURITY_ROLLS = 11
    TIMEPERIOD_INDEXER = 12
    FIND_UPTO_DATE = 14
    REBALANCING_FROM_FREQS = 15


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.DATES:
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

    elif local_test == LocalTests.OFFSETS:
        date = pd.Timestamp('28Feb2023')
        print(date-pd.offsets.Day(30))
        print(date-pd.offsets.MonthEnd(1))

        print(shift_dates_by_year(date))
        print(date-pd.offsets.Day(365))
        print(date-pd.offsets.BYearBegin(1))

    elif local_test == LocalTests.WEEK_DAY:
        time_period = TimePeriod(start='01Jun2022', end='22Jun2022')
        sample_dates = generate_sample_dates(time_period=time_period, freq='D')
        print(get_weekday(sample_dates.index))

    elif local_test == LocalTests.SAMPLE_DATES_IDX:
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

    elif local_test == LocalTests.PERIOD_WITH_HOLIDAYS:

        time_period = TimePeriod(start='01Jan2020', end='22Jan2020')
        holidays = pd.DatetimeIndex(data=['20200113', '20200120', '20200214', '20200217',
                                          '20200217', '20200219', '20200228', '20200324', '20200410'])

        dates = time_period.to_period_dates_str(freq='B', holidays=holidays)
        print(dates)

    elif local_test == LocalTests.EOD_FREQ_HOUR:
        time_period = TimePeriod(pd.Timestamp('2022-11-10', tz='UTC'), pd.Timestamp('2023-03-03', tz='UTC'))
        rebalancing_times = generate_dates_schedule(time_period=time_period, freq='h')
        print(rebalancing_times)

    elif local_test == LocalTests.FREQ_HOUR:

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

    elif local_test == LocalTests.FREQ_REB:
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

    elif local_test == LocalTests.FREQS:
        freq_map = FreqMap.BQ
        freq_map.print()
        freq = freq_map.to_freq()
        print(freq)
        n_bus_days = freq_map.to_n_bus_days()
        print(n_bus_days)

    elif local_test == LocalTests.REBALANCING_INDICATORS:
        pd_index = pd.date_range(start='31Dec2020', end='31Dec2021', freq='W-MON')
        data = pd.DataFrame(range(len(pd_index)), index=pd_index, columns=['aaa'])
        rebalancing_schedule = generate_rebalancing_indicators(df=data, freq='M-FRI')
        print(rebalancing_schedule)
        print(rebalancing_schedule[rebalancing_schedule == True])

    elif local_test == LocalTests.FIXED_MATURITY_ROLLS:
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

    elif local_test == LocalTests.TIMEPERIOD_INDEXER:
        time_period = TimePeriod('31Dec2023', '31Dec2024')
        times_m = generate_dates_schedule(time_period=time_period, freq='ME')
        times_q = generate_dates_schedule(time_period=time_period, freq='QE')
        print(times_m)
        print(times_q)
        idx_q_at_m = times_m.get_indexer(target=times_q, method='ffill')
        print(idx_q_at_m)
        idx_m_at_q = times_q.get_indexer(target=times_m, method='ffill')
        print(idx_m_at_q)
        indextimes_m_at_q = [times_q[idx] for idx in idx_m_at_q]
        print(indextimes_m_at_q)

    elif local_test == LocalTests.FIND_UPTO_DATE:
        time_period = TimePeriod('31Dec2023', '31Dec2024')
        times_q = generate_dates_schedule(time_period=time_period, freq='QE')
        this_dates = [pd.Timestamp('31Dec2023'), pd.Timestamp('31Jan2024'), pd.Timestamp('31Mar2024'),
                      pd.Timestamp('31Dec2024'), pd.Timestamp('31Dec2025')]
        for this_date in this_dates:
            matched_date = find_upto_date_from_datetime_index(index=times_q, date=this_date)
            print(f"given={this_date}, matched={matched_date}")

    elif local_test == LocalTests.REBALANCING_FROM_FREQS:
        time_period = TimePeriod('31Dec2023', '31Dec2024')
        rebalancing_indicators = create_rebalancing_indicators_from_freqs(
            rebalancing_freqs=pd.Series(dict(SPY='ME', TLT='QE', LQD='QE')),
            time_period=time_period)
        print(rebalancing_indicators)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.REBALANCING_FROM_FREQS)
