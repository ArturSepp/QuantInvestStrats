"""
analytics for fx data
"""
import pandas as pd
import numpy as np
from typing import Union, Dict


def get_aligned_fx_spots(prices: pd.DataFrame,
                         asset_ccy_map: Union[pd.Series, Dict],
                         fx_prices: pd.DataFrame,
                         quote_currency: str = 'USD'
                         ) -> pd.DataFrame:
    """
    get fx currency for price_data columns = instrument ticker
    universe_local_ccy is map {instrument ticker: fx_rate_ccy}
    fx_prices is fx prices columns = fx_rate_ccy
    """
    # first backfill and the bbfill so prices will have corresponding fx spots data
    fx_prices = fx_prices.copy().ffill().bfill()
    fx_prices[quote_currency] = 1.0

    fx_spots = {}
    for asset, ccy in asset_ccy_map.items():
        fx_spots[asset] = fx_prices[ccy]
    fx_spots = pd.DataFrame.from_dict(fx_spots, orient='columns')
    fx_spots = fx_spots.where(pd.isna(prices) == False)
    return fx_spots


def compute_futures_fx_adjusted_returns(prices: pd.DataFrame,
                                        fx_spots: pd.DataFrame,
                                        periods: int = 1,
                                        is_log_returns: bool = False
                                        ) -> pd.DataFrame:
    """
    futures returns adjusted for fx rate change
    fx_return affects only the future return
    """
    price_return = prices / prices.shift(periods=periods) - 1.0
    fx_return = fx_spots / fx_spots.shift(periods=periods) - 1.0
    returns = np.log(1.0 + price_return + price_return * fx_return)
    if not is_log_returns:
        returns = np.expm1(returns)
    return returns


def compute_cash_fx_adjusted_returns(prices: pd.DataFrame,
                                     fx_spots: pd.DataFrame,
                                     periods: int = 1,
                                     is_log_returns: bool = False
                                     ) -> pd.DataFrame:
    """
    cash returns adjusted for fx rate change
    fx_return affects notional + return
    """
    price_return = prices / prices.shift(periods=periods) - 1.0
    fx_return = fx_spots / fx_spots.shift(periods=periods) - 1.0
    returns = np.log(1.0 + fx_return + price_return + price_return * fx_return)
    if not is_log_returns:
        returns = np.expm1(returns)
    return returns
