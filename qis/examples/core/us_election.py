"""
generate performances before/after US election dates and conditional on the divided/unified goverment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from dateutil.relativedelta import TU
from bbg_fetch import fetch_field_timeseries_per_tickers
from typing import Tuple, Literal, Dict, Optional
from enum import Enum

# this table is generated using data in https://stevesnotes.substack.com/p/note-1-gridlock
# year is matched to the election data
US_ELECTION_RESULTS = {1968: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1970: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1972: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1974: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1976: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Democrat', Divided_Unified='Unified'),
                       1978: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Democrat', Divided_Unified='Unified'),
                       1980: dict(Senate_Majority='Republicans', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1982: dict(Senate_Majority='Republicans', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1984: dict(Senate_Majority='Republicans', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1986: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1988: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1990: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       1992: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Democrat', Divided_Unified='Unified'),
                       1994: dict(Senate_Majority='Republicans', House_Majority='Republicans', President='Democrat', Divided_Unified='Divided'),
                       1996: dict(Senate_Majority='Republicans', House_Majority='Republicans', President='Democrat', Divided_Unified='Divided'),
                       1998: dict(Senate_Majority='Republicans', House_Majority='Republicans', President='Democrat', Divided_Unified='Divided'),
                       2000: dict(Senate_Majority='Democrats', House_Majority='Republicans', President='Republican', Divided_Unified='Divided'),
                       2002: dict(Senate_Majority='Republicans', House_Majority='Republicans', President='Republican', Divided_Unified='Unified'),
                       2004: dict(Senate_Majority='Republicans', House_Majority='Republicans', President='Republican', Divided_Unified='Unified'),
                       2006: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       2008: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Democrat', Divided_Unified='Unified'),
                       2010: dict(Senate_Majority='Democrats', House_Majority='Republicans', President='Democrat', Divided_Unified='Divided'),
                       2012: dict(Senate_Majority='Democrats', House_Majority='Republicans', President='Democrat', Divided_Unified='Divided'),
                       2014: dict(Senate_Majority='Republicans', House_Majority='Republicans', President='Democrat', Divided_Unified='Divided'),
                       2016: dict(Senate_Majority='Republicans', House_Majority='Republicans', President='Republican', Divided_Unified='Unified'),
                       2018: dict(Senate_Majority='Republicans', House_Majority='Democrats', President='Republican', Divided_Unified='Divided'),
                       2020: dict(Senate_Majority='Democrats', House_Majority='Democrats', President='Democrat', Divided_Unified='Unified'),
                       2022: dict(Senate_Majority='Democrats', House_Majority='Republicans', President='Democrat', Divided_Unified='Divided')
                       }


def generate_us_election_dates(start_date: pd.Timestamp = pd.Timestamp('01Jan1950'),
                               end_date: pd.Timestamp = pd.Timestamp('01Jan2023')
                               ) -> pd.DatetimeIndex:
    """
    see https://stackoverflow.com/questions/34708626/pandas-holiday-calendar-rule-for-us-election-day
    """
    def election_observance(dt):
        if not dt.year % 4 == 0:  # every fourth year
            return None
        else:
            return dt + pd.DateOffset(weekday=TU(1))

    class USElectionCalendar(AbstractHolidayCalendar):
        """
        Federal Presidential  and Congressional election day.
        Tuesday following the first Monday, 2 to 8 November every two even numbered years.
        Election Days can only occur from November 2nd through 8th inclusive.
        """
        rules = [
            Holiday('Election Day', month=11, day=2, observance=election_observance)
        ]

    cal = USElectionCalendar()
    dates = cal.holidays(start_date, end_date, return_name=False)
    return dates


def generate_before_after_dates(dates: pd.DatetimeIndex,
                                period: int = 150
                                ) -> Dict[pd.Timestamp, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    period is measured in business days
    """
    before_after_dates = {}
    for date in dates:
        before_after_dates[date] = (date-pd.DateOffset(days=period), date+pd.DateOffset(days=period))
    return before_after_dates


def create_before_after_performances(price: pd.Series,
                                     before_after_dates: Dict[pd.Timestamp, Tuple[pd.Timestamp, pd.Timestamp]]
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    price should be at 'B' frequency
    """
    before_prices = {}
    after_prices = {}
    for date, before_after_date in before_after_dates.items():
        before_price = price.loc[before_after_date[0]:date]
        after_price = price.loc[date:before_after_date[1]]
        # normalise
        if not before_price.empty:
            before_price = before_price / before_price.iloc[-1]
            before_price.index = np.arange(-len(before_price.index)+1, 1, 1)
            before_prices[date.year] = before_price

        if not after_price.empty:
            after_price = after_price / after_price.iloc[0]
            after_price.index = np.arange(0, len(after_price.index))
            after_prices[date.year] = after_price
    before_prices = pd.DataFrame.from_dict(before_prices, orient='columns')
    after_prices = pd.DataFrame.from_dict(after_prices, orient='columns')
    return before_prices, after_prices


def merge_before_after_prices(before_prices: pd.DataFrame,
                              after_prices: pd.DataFrame
                              ) -> pd.DataFrame:
    joint_price = pd.concat([before_prices, after_prices.iloc[1:, :]], axis=0)
    joint_price = joint_price[joint_price.columns[::-1]]
    return joint_price


def plot_before_after_prices(joint_prices: pd.DataFrame,
                             add_average: bool = True,
                             xlabel: str = 'Days before/after election date',
                             ylabel: str = 'Normalised performance',
                             title: str = None,
                             is_norm: bool = True,
                             ax: plt.Subplot = None
                             ) -> Optional[plt.Figure]:

    if add_average:
        joint_prices.insert(0, column='Average', value=joint_prices.mean(1))
    if is_norm:
        joint_prices = joint_prices - 1.0
        yvar_format='{:,.0%}'
    else:
        yvar_format='{:,.2f}',
    n = len(joint_prices.columns)
    colors = qis.get_cmap_colors(n=n, cmap='tab20')
    linestyles = ['dotted'] * n
    if add_average:
        colors[0] = 'blue'
        linestyles[0] = '-'

    if ax is None:
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None
    qis.plot_line(df=joint_prices,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  title=title,
                  colors=colors,
                  linestyles=linestyles,
                  yvar_format=yvar_format,
                  framealpha=0.9,
                  ax=ax)
    ax.axvline(0, color='lightsalmon', lw=1, alpha=0.5)  # horizontal
    return fig


def plot_unconditional_performances(price: pd.Series, title: str = None, ax: plt.Subplot = None) -> plt.Figure:
    dates = generate_us_election_dates()
    before_after_dates = generate_before_after_dates(dates)
    before_prices, after_prices = create_before_after_performances(price=price,
                                                                   before_after_dates=before_after_dates)
    joint_prices = merge_before_after_prices(before_prices=before_prices, after_prices=after_prices)
    fig = plot_before_after_prices(joint_prices=joint_prices, title=title, ax=ax)
    return fig


def compute_conditional_performances(
        price: pd.Series,
        classifier: Literal['Senate_Majority', 'House_Majority', 'President', 'Divided_Unified'] = 'Divided_Unified'
        ) -> Dict[str, pd.DataFrame]:
    """
    compute before / after performances conditional on classifier
    """
    dates = generate_us_election_dates()
    before_after_dates = generate_before_after_dates(dates)
    before_prices, after_prices = create_before_after_performances(price=price,
                                                                   before_after_dates=before_after_dates)
    joint_price = merge_before_after_prices(before_prices=before_prices, after_prices=after_prices)
    t_joint_price = joint_price.T
    # add classifiers
    election_results = pd.DataFrame.from_dict(US_ELECTION_RESULTS, orient='index').loc[t_joint_price.index]
    t_joint_price[classifier] = election_results[classifier]
    t_joint_price = t_joint_price.groupby(classifier)
    dfs = {}
    for group, prices in t_joint_price:
        dfs[str(group)] = prices.drop(classifier, axis=1).T
    return dfs


def plot_conditional_performances(dfs: Dict[str, pd.DataFrame], title: str) -> plt.Figure:
    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(1, len(dfs.keys()), figsize=(14, 8))
        qis.set_suptitle(fig, title=title)
        for idx, (group, joint_prices) in enumerate(dfs.items()):
            plot_before_after_prices(joint_prices=joint_prices, title=group, ax=axs[idx])
        qis.align_y_limits_axs(axs)
    return fig


class UnitTests(Enum):
    ELECTION_RESULTS = 1
    ELECTION_DATES = 2
    UNCONDITIONAL_PERFORMANCES = 3
    CONDITIONAL_PERFORMANCES = 4


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.ELECTION_RESULTS:
        df = pd.DataFrame.from_dict(US_ELECTION_RESULTS, orient='index')
        print(df)

    elif unit_test == UnitTests.ELECTION_DATES:
        dates = generate_us_election_dates()
        print(dates)

        before_after_dates = generate_before_after_dates(dates)
        print(before_after_dates)

    elif unit_test == UnitTests.UNCONDITIONAL_PERFORMANCES:
        tickers = {'SPX Index': 'S&P 500',
                   'DXY Curncy': 'DXY',
                   'TY1 Comdty': 'UST 10y bond future'}
        prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B').ffill()
        figs = []
        for asset in prices.columns:
            fig = plot_unconditional_performances(price=prices[asset].dropna(), title=f"{asset} performance")
            figs.append(fig)
        qis.save_figs_to_pdf(figs, file_name='election_perf_unconditional', local_path=qis.get_output_path())

    elif unit_test == UnitTests.CONDITIONAL_PERFORMANCES:
        tickers = {'SPX Index': 'S&P 500',
                   'DXY Curncy': 'DXY',
                   'TY1 Comdty': 'UST 10y bond future'}
        tickers = {'NKY Index': 'Nikkei 225',
                   'HSI Index': 'HSI',
                   'SXXP Index': 'SXXP',
                   'SPX Index': 'S&P 500',
                   'DXY Curncy': 'DXY',
                   'TY1 Comdty': 'UST 10y bond future'
                   }
        prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B',
                                                    start_date=pd.Timestamp('01Jan1968')).ffill()
        print(prices)
        figs = []
        dfss = {}
        for asset in prices.columns:
            price = prices[asset].dropna()
            dfs = compute_conditional_performances(price=price)
            fig = plot_conditional_performances(dfs=dfs, title=f"{asset} performance conditional on US Government")
            for key, df in dfs.items():
                dfss[f"{asset}-{key}"] = df - 1.0
            figs.append(fig)
        qis.save_figs_to_pdf(figs, file_name='perf_conditional_on_government', local_path=qis.get_output_path())
        qis.save_df_to_excel(data=dfss, file_name='perf_conditional_on_government', local_path=qis.get_output_path())
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CONDITIONAL_PERFORMANCES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
