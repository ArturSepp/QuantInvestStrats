"""
long volatility bank QIS universe for the JOIM convexity premium paper:
ticker definitions, Bloomberg fetch, and local CSV persistence

universe covers equity/VIX, commodity and cross-asset long volatility indices only.
rates, credit, inflation and FX long vol indices are excluded by construction.
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
# qis / project
import qis as qis
from bbg_fetch import fetch_field_timeseries_per_tickers


class VolGroup(str, Enum):
    """
    asset-class grouping used for group_data in qis reporting
    """
    EQUITY_VIX = 'Equity & VIX'
    COMMODITY = 'Commodity'
    CROSS_ASSET = 'Cross-asset'


class VolPayoff(str, Enum):
    """
    payoff family of the long volatility index
    """
    LONG_VOL = 'Long vol'  # outright long implied vol / long variance
    PUT_BUYING = 'Put buying'  # systematic long puts, delta hedged or not
    PUT_RATIO = 'Put ratio'  # long ratio structures, theta financed
    CALENDAR = 'Calendar'  # short front / long back, carry neutral by design
    VIX_CONVEXITY = 'VIX convexity'  # VIX calls, call spreads, or futures convexity
    TAIL_HEDGE = 'Tail hedge'  # explicit tail hedge or defensive overlay
    COLLAR = 'Collar'  # long put financed by short call


@dataclass(frozen=True)
class VolQisIndex:
    """
    immutable record for one bank long volatility QIS index
    """
    ticker: str  # Bloomberg ticker including the ' Index' suffix
    name: str  # short label used as the price column name
    provider: str
    group: VolGroup
    payoff: VolPayoff
    underlying: str
    is_excess_return: bool  # True for ER indices, False for TR or unverified
    description: str
    flag: Optional[str] = None  # data caveat to resolve before using the series

    def __post_init__(self) -> None:
        if not self.ticker.endswith(' Index'):
            raise ValueError(f"ticker must end with ' Index', got {self.ticker!r}")
        if len(self.name) == 0:
            raise ValueError(f"empty name for ticker {self.ticker!r}")


# core universe dict: Bloomberg ticker -> index record
# ER/TR flags follow the source documents; unverified entries carry is_excess_return=False
LONG_VOL_QIS: Dict[str, VolQisIndex] = {

    # ---------------- BNP Paribas (Equity Vol QIS Offering, all ER) ----------------
    'BNPIV1UE Index': VolQisIndex(
        ticker='BNPIV1UE Index', name='BNP US 1Y Vol', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.LONG_VOL, underlying='SPX',
        is_excess_return=True,
        description='Long 1Y ATMF put on SPX, daily delta hedged at close with skew adjustment'),
    'BNPIV1EE Index': VolQisIndex(
        ticker='BNPIV1EE Index', name='BNP EU 1Y Vol', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.LONG_VOL, underlying='SX5E',
        is_excess_return=True,
        description='Long 1Y ATMF put on SX5E, daily delta hedged at close with skew adjustment'),
    'BNPXDPPE Index': VolQisIndex(
        ticker='BNPXDPPE Index', name='BNP Dynamic POP EU', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.PUT_BUYING, underlying='SX5E',
        is_excess_return=True,
        description='Daily purchase of 1Y forward start 1Y 90% put on SX5E, notional scaled by trend'),
    'BNPXTPRU Index': VolQisIndex(
        ticker='BNPXTPRU Index', name='BNP TPR US', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.PUT_RATIO, underlying='SPX',
        is_excess_return=True,
        description='Daily strip of three 3M put ratios on SPX with theta-flat sizing, daily delta hedged'),
    'BNPXTPRE Index': VolQisIndex(
        ticker='BNPXTPRE Index', name='BNP TPR EU', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.PUT_RATIO, underlying='SX5E',
        is_excess_return=True,
        description='Daily strip of three 3M put ratios on SX5E with theta-flat sizing, daily delta hedged'),
    'BNPXTPRJ Index': VolQisIndex(
        ticker='BNPXTPRJ Index', name='BNP TPR JP', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.PUT_RATIO, underlying='NKY',
        is_excess_return=True,
        description='Daily strip of three 3M put ratios on NKY with theta-flat sizing, daily delta hedged'),
    'BNPXTEUC Index': VolQisIndex(
        ticker='BNPXTEUC Index', name='BNP TIER EU Calendar', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.CALENDAR, underlying='SX5E',
        is_excess_return=True,
        description='Sells very far OTM 1-5bd puts on SX5E and buys a delta hedged 3M put on a fixed budget'),
    'BNPXTHUE Index': VolQisIndex(
        ticker='BNPXTHUE Index', name='BNP THALIA US', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.CALENDAR, underlying='SPX',
        is_excess_return=True,
        description='Long strip of 1Y puts with advanced delta hedge, financed by short 1M OTM puts'),
    'BNPXTHEE Index': VolQisIndex(
        ticker='BNPXTHEE Index', name='BNP THALIA EU', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.CALENDAR, underlying='SX5E',
        is_excess_return=True,
        description='Long strip of 1Y puts with advanced delta hedge, financed by short 1M OTM puts'),
    'BNPXTHEN Index': VolQisIndex(
        ticker='BNPXTHEN Index', name='BNP THALIA Neutral EU', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.CALENDAR, underlying='SX5E',
        is_excess_return=True,
        description='THALIA EU variant sized for a carry neutral profile'),
    'BNPXTDEN Index': VolQisIndex(
        ticker='BNPXTDEN Index', name='BNP THALIA Dyn Neutral EU', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.CALENDAR, underlying='SX5E',
        is_excess_return=True,
        description='THALIA EU dynamic variant sized for a carry neutral profile'),
    'BNPXTHUN Index': VolQisIndex(
        ticker='BNPXTHUN Index', name='BNP THALIA Neutral US', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.CALENDAR, underlying='SPX',
        is_excess_return=True,
        description='THALIA US variant sized for a carry neutral profile',
        flag='source sheet maps this ticker to two names (THALIA Neutral US and THALIA Dynamic Neutral US)'),
    'BNPXVO3A Index': VolQisIndex(
        ticker='BNPXVO3A Index', name='BNP VOLA 3', provider='BNP Paribas',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='SPX/VIX',
        is_excess_return=True,
        description='Delta replication of 1M VIX calls'),

    # ---------------- J.P. Morgan (Flagship Systematic Strategies) ----------------
    'JPOSLVUS Index': VolQisIndex(
        ticker='JPOSLVUS Index', name='JPM US Long Variance', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.LONG_VOL, underlying='SPX',
        is_excess_return=False,
        description='Replicates a long dated down-corridor variance swap on SPX with skew-adjusted delta hedging'),
    'JPOSLVUV Index': VolQisIndex(
        ticker='JPOSLVUV Index', name='JPM US 4W Long Variance', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.LONG_VOL, underlying='SPX',
        is_excess_return=False,
        description='Daily purchase of a long dated down-corridor variance swap on SPX',
        flag='listed as Not Live in the March 2025 source, backtest only'),
    'JPOSTUDN Index': VolQisIndex(
        ticker='JPOSTUDN Index', name='JPM US Tail Hedge', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.TAIL_HEDGE, underlying='SPX',
        is_excess_return=False,
        description='Put ratio on SPX with dynamic delta hedge, targeting long vega and long gamma in stress'),
    'JPOSPRU2 Index': VolQisIndex(
        ticker='JPOSPRU2 Index', name='JPM US Put Ratio', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.PUT_RATIO, underlying='SPX',
        is_excess_return=False,
        description='Sells delta hedged SPX put ratios daily with theta neutral sizing for a defensive profile'),
    'JPUSVXCR Index': VolQisIndex(
        ticker='JPUSVXCR Index', name='JPM US VIX Call Ratio', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='VIX',
        is_excess_return=False,
        description='Rolling VIX call ratio for upside exposure to VIX with limited cost of carry'),
    'AIJPVT1U Index': VolQisIndex(
        ticker='AIJPVT1U Index', name='JPM Vol Trend Following', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='VIX',
        is_excess_return=False,
        description='Long/neutral VIX futures using a trend signal based on option delta replication'),
    'JPRC85BU Index': VolQisIndex(
        ticker='JPRC85BU Index', name='JPM 85% Collar US', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.COLLAR, underlying='SPX',
        is_excess_return=False,
        description='Rolling collar on SPX, 85% put switching to a put spread above 2.5% premium, 20% delta call'),
    'JPRC85BE Index': VolQisIndex(
        ticker='JPRC85BE Index', name='JPM 85% Collar EU', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.COLLAR, underlying='SX5E',
        is_excess_return=False,
        description='Rolling collar on SX5E, 85% put switching to a put spread above 2.5% premium, 20% delta call'),
    'JPUS2525 Index': VolQisIndex(
        ticker='JPUS2525 Index', name='JPM US Long Skew', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.TAIL_HEDGE, underlying='SPX',
        is_excess_return=False,
        description='Long skewness on SPX via 25 delta option structures',
        flag='classified Volatility RV in source, verify net long vol sign before use'),
    'JPUSSA25 Index': VolQisIndex(
        ticker='JPUSSA25 Index', name='JPM US Long Skew SA', provider='J.P. Morgan',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.TAIL_HEDGE, underlying='SPX',
        is_excess_return=False,
        description='Stress-aware variant of the SPX long skewness index',
        flag='classified Volatility RV in source, verify net long vol sign before use'),
    'JPOSPRCL Index': VolQisIndex(
        ticker='JPOSPRCL Index', name='JPM WTI Put Ratio', provider='J.P. Morgan',
        group=VolGroup.COMMODITY, payoff=VolPayoff.PUT_RATIO, underlying='WTI',
        is_excess_return=False,
        description='Put ratio structure on WTI crude for a defensive commodity profile'),
    'JMABDOR2 Index': VolQisIndex(
        ticker='JMABDOR2 Index', name='JPM Defensive Option Repl', provider='J.P. Morgan',
        group=VolGroup.CROSS_ASSET, payoff=VolPayoff.TAIL_HEDGE, underlying='FX/Rates/Commodity',
        is_excess_return=False,
        description='Replicates the delta of a long option portfolio on a defensive FX, commodity and rates basket'),

    # ---------------- Deutsche Bank (QIS weekly summary) ----------------
    'DBPPOPU1 Index': VolQisIndex(
        ticker='DBPPOPU1 Index', name='DB Equity Put Framework', provider='Deutsche Bank',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.PUT_BUYING, underlying='SPX',
        is_excess_return=False,
        description='Systematic long put hedging framework on equities',
        flag='description inferred from strategy name, source has no description field'),
    'DBUS1DDH Index': VolQisIndex(
        ticker='DBUS1DDH Index', name='DB 1DTE Put Delta Hedged', provider='Deutsche Bank',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.LONG_VOL, underlying='SPX',
        is_excess_return=False,
        description='Delta hedged 1-day-to-expiry SPX puts isolating short dated gamma',
        flag='direction inferred from name, confirm long vs short gamma sign with DB'),
    'DBCRTTUU Index': VolQisIndex(
        ticker='DBCRTTUU Index', name='DB US Put Ratio', provider='Deutsche Bank',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.PUT_RATIO, underlying='SPX',
        is_excess_return=False,
        description='Put ratio structure on US equity index',
        flag='description inferred from strategy name'),
    'DBCRUPRU Index': VolQisIndex(
        ticker='DBCRUPRU Index', name='DB SPX Calendar Put Ratio', provider='Deutsche Bank',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.CALENDAR, underlying='SPX',
        is_excess_return=False,
        description='Calendar put ratio spreads on SPX',
        flag='description inferred from strategy name'),
    'DBVIXOCR Index': VolQisIndex(
        ticker='DBVIXOCR Index', name='DB VIX OTM Call Repl', provider='Deutsche Bank',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='VIX',
        is_excess_return=False,
        description='Delta replication of OTM VIX calls',
        flag='description inferred from strategy name'),
    'DBVIXCSR Index': VolQisIndex(
        ticker='DBVIXCSR Index', name='DB VIX Call Spread Repl', provider='Deutsche Bank',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='VIX',
        is_excess_return=False,
        description='Delta replication of a VIX call spread',
        flag='description inferred from strategy name'),
    'DBCADR02 Index': VolQisIndex(
        ticker='DBCADR02 Index', name='DB XA Equity Proxy Put', provider='Deutsche Bank',
        group=VolGroup.CROSS_ASSET, payoff=VolPayoff.TAIL_HEDGE, underlying='Cross-asset',
        is_excess_return=False,
        description='Replicates an equity proxy put across asset classes for portfolio convexity',
        flag='description inferred from strategy name'),

    # ---------------- Barclays (Defensive Portfolio components) ----------------
    'BXIIEVCN Index': VolQisIndex(
        ticker='BXIIEVCN Index', name='BARC Vol Convexity', provider='Barclays',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='SPX/VIX',
        is_excess_return=False,
        description='Sells OTM SPX put spreads to finance VIX calls, targeting equity volatility convexity'),
    'BXIIDCNU Index': VolQisIndex(
        ticker='BXIIDCNU Index', name='BARC US Dynamic Convexity', provider='Barclays',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='VIX',
        is_excess_return=False,
        description='Goes long convexity in VIX futures only when the VIX futures carry is positive'),
    'BXIICUSB Index': VolQisIndex(
        ticker='BXIICUSB Index', name='BARC LS Variance Tail Hedge', provider='Barclays',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.TAIL_HEDGE, underlying='SPX',
        is_excess_return=False,
        description='Weekly short variance on SPX overlaid with a long 4-week variance swap for low carry tail hedging'),

    # ---------------- UBS (Q2 2026 QIS universe) ----------------
    'XUBSVX3H Index': VolQisIndex(
        ticker='XUBSVX3H Index', name='UBS ConVIXity Hedge', provider='UBS',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='VIX',
        is_excess_return=False,
        description='Long volatility convexity hedge implemented with VIX instruments'),
    'XUBSVXFD Index': VolQisIndex(
        ticker='XUBSVXFD Index', name='UBS VIX Futures Defensive', provider='UBS',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.VIX_CONVEXITY, underlying='VIX',
        is_excess_return=False,
        description='Long biased defensive VIX futures exposure',
        flag='live from Apr-26, short live history'),
    'XUBSDTH1 Index': VolQisIndex(
        ticker='XUBSDTH1 Index', name='UBS Defensive Tail Hedge', provider='UBS',
        group=VolGroup.EQUITY_VIX, payoff=VolPayoff.TAIL_HEDGE, underlying='SPX',
        is_excess_return=False,
        description='Financed equity tail hedge portfolio holding long convexity',
        flag='live from Nov-25, short live history'),
    'XUBSCAXE Index': VolQisIndex(
        ticker='XUBSCAXE Index', name='UBS Convexity Alpha', provider='UBS',
        group=VolGroup.COMMODITY, payoff=VolPayoff.VIX_CONVEXITY, underlying='Commodity',
        is_excess_return=False,
        description='Long convexity commodity strategy implemented as a fly on commodity options'),
}

LOCAL_FILE_NAME = 'long_vol_qis_prices'


def get_ticker_to_name(group: Optional[VolGroup] = None,
                       payoff: Optional[VolPayoff] = None,
                       tickers: Optional[List[str]] = None
                       ) -> Dict[str, str]:
    """
    map Bloomberg tickers to short column names, optionally filtered by group, payoff or explicit tickers.
    the returned dict is passed directly to bbg_fetch, which uses the values as column labels
    """
    if tickers is not None:
        missing = [x for x in tickers if x not in LONG_VOL_QIS.keys()]
        if len(missing) > 0:
            raise ValueError(f"tickers not in LONG_VOL_QIS: {missing}")
        universe = {x: LONG_VOL_QIS[x] for x in tickers}
    else:
        universe = LONG_VOL_QIS
    if group is not None:
        universe = {k: v for k, v in universe.items() if v.group == group}
    if payoff is not None:
        universe = {k: v for k, v in universe.items() if v.payoff == payoff}
    if len(universe) == 0:
        raise ValueError(f"empty universe for group={group}, payoff={payoff}")
    return {k: v.name for k, v in universe.items()}


def get_group_data(ticker_to_name: Optional[Dict[str, str]] = None) -> pd.Series:
    """
    asset-class group_data indexed by short name, for qis reporting
    """
    if ticker_to_name is None:
        ticker_to_name = get_ticker_to_name()
    group_data = {name: LONG_VOL_QIS[ticker].group.value for ticker, name in ticker_to_name.items()}
    return pd.Series(group_data, name='group')


def get_universe_table() -> pd.DataFrame:
    """
    reference table of the universe with provider, group, payoff and data flags
    """
    table = {x.name: dict(ticker=x.ticker,
                          provider=x.provider,
                          group=x.group.value,
                          payoff=x.payoff.value,
                          underlying=x.underlying,
                          is_excess_return=x.is_excess_return,
                          description=x.description,
                          flag=x.flag if x.flag is not None else '')
             for x in LONG_VOL_QIS.values()}
    return pd.DataFrame.from_dict(table, orient='index')


def fetch_long_vol_prices(ticker_to_name: Optional[Dict[str, str]] = None,
                          field: str = 'PX_LAST',
                          start_date: pd.Timestamp = pd.Timestamp('01Jan2005'),
                          end_date: Optional[pd.Timestamp] = None,  # None fetches to today
                          freq: str = 'B',  # business day grid with forward fill
                          min_obs: int = 250,  # drop series with fewer valid observations
                          ) -> pd.DataFrame:
    """
    fetch index levels for the long volatility QIS universe from Bloomberg.
    QIS indices carry no dividends or capital changes, so all Bloomberg adjustment flags are switched off
    """
    if ticker_to_name is None:
        ticker_to_name = get_ticker_to_name()
    if len(ticker_to_name) == 0:
        raise ValueError("ticker_to_name is empty")
    if min_obs < 0:
        raise ValueError(f"min_obs must be non-negative, got {min_obs!r}")

    prices = fetch_field_timeseries_per_tickers(tickers=ticker_to_name,
                                                field=field,
                                                CshAdjNormal=False,
                                                CshAdjAbnormal=False,
                                                CapChg=False,
                                                start_date=start_date,
                                                end_date=end_date,
                                                freq=freq)
    if prices is None:
        raise ValueError(f"Bloomberg returned no data for {len(ticker_to_name)} tickers, field={field!r}")

    # report tickers that came back empty rather than dropping them silently
    missing = [name for name in ticker_to_name.values() if name not in prices.columns]
    if len(missing) > 0:
        print(f"no Bloomberg data for: {missing}")

    prices = prices.asfreq(freq, method='ffill')
    num_obs = prices.notna().sum(axis=0)
    is_short = num_obs < min_obs
    if is_short.any():
        print(f"dropping series with fewer than {min_obs} observations: {num_obs[is_short].to_dict()}")
        prices = prices.loc[:, ~is_short]
    return prices


def save_long_vol_prices(prices: pd.DataFrame,
                         file_name: str = LOCAL_FILE_NAME,
                         local_path: Optional[str] = None  # None uses qis resource path
                         ) -> None:
    """
    persist fetched prices to the local resource folder as CSV
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError(f"prices must be pd.DataFrame, got {type(prices)}")
    if prices.empty:
        raise ValueError("refusing to save an empty price frame")
    if local_path is None:
        local_path = qis.local_path.get_resource_path()
    qis.save_df_to_csv(df=prices, file_name=file_name, local_path=local_path)


def load_long_vol_prices(file_name: str = LOCAL_FILE_NAME,
                         local_path: Optional[str] = None
                         ) -> pd.DataFrame:
    """
    load previously saved prices from the local resource folder
    """
    if local_path is None:
        local_path = qis.local_path.get_resource_path()
    return qis.load_df_from_csv(file_name=file_name, local_path=local_path)


class UnitTests(Enum):
    UNIVERSE_TABLE = 1
    FETCH_AND_SAVE = 2
    LOAD_AND_REPORT = 3


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.UNIVERSE_TABLE:
        table = get_universe_table()
        print(table)
        print(table['group'].value_counts())
        print(table['provider'].value_counts())
        print(f"flagged entries:\n{table.loc[table['flag'] != '', 'flag']}")

    elif unit_test == UnitTests.FETCH_AND_SAVE:
        ticker_to_name = get_ticker_to_name()
        prices = fetch_long_vol_prices(ticker_to_name=ticker_to_name)
        print(prices)
        print(prices.apply(lambda x: x.first_valid_index()).sort_values())
        save_long_vol_prices(prices=prices)

    elif unit_test == UnitTests.LOAD_AND_REPORT:
        prices = load_long_vol_prices()
        ticker_to_name = get_ticker_to_name()
        group_data = get_group_data(ticker_to_name=ticker_to_name).reindex(index=prices.columns).dropna()
        time_period = qis.TimePeriod(start=prices.index[0], end=prices.index[-1])
        # index levels are excess or total return depending on provider, treat as simple returns
        returns = qis.to_returns(prices=prices, is_log_returns=False, drop_first=True)
        print(returns.describe())
        print(group_data)
        print(time_period.to_str())


if __name__ == '__main__':

    unit_test = UnitTests.FETCH_AND_SAVE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
