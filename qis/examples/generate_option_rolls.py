"""
the systematic rolls for option or futures at a given timestamp include:
selection of the roll expiry given by parameter roll_freq (roll_freq='W-FRI', 'M-FRI')
roll time set using roll_hour
when the roll references second to first available roll expiry set by min_days_to_next_roll
example on how to generate generate roll schedule: for each timestamp find the corresponding maturity of the next roll
returned pd.DateTimeIndex is the current observation time and value is the corresponding option expiry of the current roll.
"""
import pandas as pd
pd.set_option('display.max_rows', 500)
import qis
from qis import TimePeriod

# set time period to fill with the roll dates
time_period = TimePeriod('01Jun2023', '08Sep2023', tz='UTC')

# weekly on Friday
weekly_rolls = qis.generate_fixed_maturity_rolls(time_period=time_period,
                                                 freq='D',  # frequency of the returned datetime index
                                                 roll_freq='W-FRI',  # roll expiries frequency
                                                 roll_hour=8,  # hour of the roll expiry
                                                 min_days_to_next_roll=6)  # days before maturity of the first contract before we switch to next contract
print(f"weekly_rolls:\n{weekly_rolls}")

# last friday of a month
monthly_rolls = qis.generate_fixed_maturity_rolls(time_period=time_period,
                                                  freq='D', roll_freq='M-FRI',
                                                  roll_hour=8, min_days_to_next_roll=28)  # 4 weeks before
print(f"monthly rolls:\n{monthly_rolls}")

# third friday of a month
friday3_monthly_rolls = qis.generate_fixed_maturity_rolls(time_period=time_period,
                                                          freq='D', roll_freq='WOM-3FRI',
                                                          roll_hour=8, min_days_to_next_roll=28)  # 4 weeks before expiry
print(f"monthly 3rd Friday rolls:\n{friday3_monthly_rolls}")

# third wednesday of a month
wednesday3_monthly_rolls = qis.generate_fixed_maturity_rolls(time_period=time_period,
                                                          freq='D', roll_freq='WOM-3WED',
                                                          roll_hour=8, min_days_to_next_roll=1)  # 4 weeks before expiry
print(f"monthly 3rd Wednesday rolls:\n{wednesday3_monthly_rolls}")


# last Friday of quarter
quarterly_rolls = qis.generate_fixed_maturity_rolls(time_period=time_period,
                                                    freq='D', roll_freq='Q-FRI',
                                                    roll_hour=8, min_days_to_next_roll=56)  # 8 weeks before expiry
print(f"quarterly rolls:\n{quarterly_rolls}")

# third Friday of quarter
quarterly_rolls = qis.generate_fixed_maturity_rolls(time_period=time_period,
                                                    freq='D', roll_freq='Q-3FRI',
                                                    roll_hour=8, min_days_to_next_roll=56)  # 8 weeks before expiry
print(f"quarterly 3rd Friday rolls:\n{quarterly_rolls}")
