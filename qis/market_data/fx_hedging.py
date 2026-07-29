"""
how much of a foreign-currency exposure to hedge, and what the hedge costs.

Stateless: plain pandas objects in, plain pandas objects out, no container to construct. The
single-pair functions work on Series and the panel functions on frames. The container that holds
the spot and rate panels is ``FxRatesData`` in ``qis/market_data/fx_rates_data.py``, and the
report built on these numbers is ``qis/market_data/reports/fx_hedging_report.py``.

A holding in a foreign asset earns three things - the local return, the FX return on the cross,
and the forward premium given up on whatever fraction is hedged:

    r_hedged = r_local (1 + r_fx) + (1 - h) r_fx - h f

with h the hedge ratio in [0, 1] and f the per-period CIP forward premium. Both h and f enter
lagged one period, so the return realised over [t-1, t] uses the hedge decided and the forward
contracted at t-1 - the construction carries no look-ahead. The first period is set to zero so
the nav starts at 1.0 rather than inheriting the lag's NaN. The cross-product term
r_local * r_fx is kept rather than dropped, so the decomposition is exact in simple returns.

``compute_fx_optimal_hedge`` is the mean-variance choice of h. From an EWMA FX variance and the
beta of the local return on the FX return it builds

    carry ratio = (annualised f / var_fx) / (2 λ),  Optimal h = 1 - carry ratio + β_fx

alongside the two reference ratios it is read against: ``Max Carry`` = 1 - carry ratio, the pure
carry tilt, and ``Beta Hedge`` = 1 + β_fx, which removes only the exposure the beta implies. All
three are clipped to ``min_max_hedge``, (0, 1) by default; λ is ``risk_aversion_lambda``, and a
larger λ tilts less on carry.

Also here: ``compute_local_and_fx_return`` for the two-leg decomposition,
``compute_fx_vol_beta`` for the EWMA inputs, ``get_aligned_fx_spots`` to attach the right spot
series to each instrument in a panel, and ``compute_futures_fx_adjusted_returns`` /
``compute_cash_fx_adjusted_returns`` - which differ because a futures position converts only its
pnl while a cash position converts its whole notional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Tuple, Union, Optional, Dict


def compute_local_and_fx_return(asset_price_local_ccy: pd.Series,
                                local_to_reference_fx_rate: pd.Series,
                                freq: str = 'ME',
                                is_log_returns: bool = False
                                ) -> Tuple[pd.Series, pd.Series]:
    """Decompose an asset's reference-currency return into its local and FX legs.

    Resamples the local-currency asset price and the reference-per-local FX rate
    to ``freq``, takes returns of both, and returns them separately. The
    reference-currency total return of an unhedged position is recovered as
    ``local_return * (1 + fx_return) + fx_return`` inside
    ``compute_performance_of_local_ccy_asset_in_reference_ccy``.

    Args:
        asset_price_local_ccy: Asset price quoted in its local currency.
        local_to_reference_fx_rate: Units of reference currency per 1 unit of the
            local currency (e.g. ``FxRatesData.get_local_to_reference_fx_rate``).
        freq: Resampling frequency for the returns (e.g. ``'ME'``, ``'QE'``, ``'B'``).
        is_log_returns: If True compute log returns, otherwise simple returns.

    Returns:
        Tuple ``(local_return, fx_return)`` of period-return Series sampled at ``freq``.
    """
    price_data = pd.concat([asset_price_local_ccy, local_to_reference_fx_rate], axis=1).ffill()
    price_returns = qis.to_returns(prices=price_data, freq=freq, is_log_returns=is_log_returns)
    local_return = price_returns.iloc[:, 0]
    fx_return = price_returns.iloc[:, 1]
    return local_return, fx_return


def compute_performance_of_local_ccy_asset_in_reference_ccy(asset_price_local_ccy: pd.Series,
                                                            local_to_reference_fx_rate: pd.Series,
                                                            forward_rate_for_local_ccy: pd.Series,
                                                            hedge_ratio: Union[float, pd.Series],
                                                            freq: str = 'ME',
                                                            is_log_returns: bool = False
                                                            ) -> Tuple[pd.Series, pd.Series]:
    """Reference-currency NAV and return of a local-currency asset at a given hedge ratio.

    The per-period hedged return is::

        hedged_return = local_return * (1 + fx_return)
                        + (1 - h) * fx_return
                        - h * forward_premium

    where ``h`` is the hedge ratio applied to the FX exposure and
    ``forward_premium`` is the CIP forward-rate return on the local currency.
    Both ``h`` and the forward premium are lagged one period (``shift(1)``) so the
    return realised over ``[t-1, t]`` uses the hedge decided and the forward
    contracted at ``t-1`` — i.e. the construction is free of look-ahead. The first
    period is forced to 0 so the NAV starts at 1.0 rather than propagating a NaN
    from the lag.

    Args:
        asset_price_local_ccy: Asset price quoted in its local currency.
        local_to_reference_fx_rate: Units of reference currency per 1 unit of the
            local currency.
        forward_rate_for_local_ccy: Per-period CIP forward premium of the local vs
            reference currency, at the same ``freq`` and return convention as the
            asset (see ``FxRatesData.get_forward_rate_for_local_ccy``).
        hedge_ratio: FX hedge ratio in ``[0, 1]`` (``0`` = unhedged, ``1`` = fully
            hedged). Either a constant ``float`` or a time-varying ``pd.Series``.
        freq: Resampling frequency for the returns (e.g. ``'ME'``).
        is_log_returns: If True use log returns throughout, otherwise simple returns.

    Returns:
        Tuple ``(hedged_nav, hedged_return)`` — NAV levels (starting at 1.0) and the
        per-period reference-currency returns, both sampled at ``freq``.
    """
    # Convert hedge_ratio to time series
    if isinstance(hedge_ratio, float):
        hedge_ratios = pd.Series(hedge_ratio, index=asset_price_local_ccy.index)
    elif isinstance(hedge_ratio, pd.Series):
        hedge_ratios = hedge_ratio
    else:
        raise NotImplementedError(f"type={type(hedge_ratio)}")

    # Calculate local and FX return components
    local_return, fx_return = compute_local_and_fx_return(asset_price_local_ccy=asset_price_local_ccy,
                                                          local_to_reference_fx_rate=local_to_reference_fx_rate,
                                                          freq=freq,
                                                          is_log_returns=is_log_returns)

    # Align series and apply lag to hedge ratio
    h_ratio_1 = hedge_ratios.reindex(index=local_return.index).ffill().rename(asset_price_local_ccy.name).shift(1)
    forward_rate_for_local_ccy = forward_rate_for_local_ccy.reindex(index=local_return.index).ffill().rename(
        asset_price_local_ccy.name)
    fx_return = fx_return.rename(asset_price_local_ccy.name)

    # Calculate hedged return with forward cost adjustment
    hedged_return = (local_return * (1.0 + fx_return) + (1.0 - h_ratio_1) * fx_return
                     - h_ratio_1 * forward_rate_for_local_ccy.shift(1))
    # First period has NaN from the ``h_ratio_1.shift(1)`` lag (no
    # previous-period hedge to carry forward). Set to 0 for
    # consistency with ``qis.to_returns(..., is_first_zero=True)``
    # used elsewhere in this module, so the resulting NAV starts from
    # 1.0 at the first index rather than from a NaN-propagated head.
    if len(hedged_return) > 0:
        hedged_return.iloc[0] = 0.0
    hedged_nav = qis.returns_to_nav(returns=hedged_return, is_log_returns=is_log_returns)
    return hedged_nav, hedged_return


def compute_fx_vol_beta(asset_price_local_ccy: pd.Series,
                        local_to_reference_fx_rate: pd.Series,
                        freq: str = 'ME',
                        span: int = 3 * 12
                        ) -> Tuple[pd.Series, pd.Series]:
    """EWMA FX volatility and the asset's beta to the FX leg.

    Decomposes the asset into local and FX log-return legs, then estimates the
    exponentially-weighted FX volatility and the EWMA beta of the local return on
    the FX return. These feed the mean-variance hedge in
    ``compute_fx_optimal_hedge`` (the beta-hedge term is ``1 + fx_beta``).

    Args:
        asset_price_local_ccy: Asset price quoted in its local currency.
        local_to_reference_fx_rate: Units of reference currency per 1 unit of the
            local currency.
        freq: Resampling frequency for the underlying returns (e.g. ``'ME'``).
        span: EWMA span in periods of ``freq`` (default 36, i.e. 3 years monthly).

    Returns:
        Tuple ``(fx_vol, fx_beta)`` of Series at ``freq``: annualised FX volatility
        and the EWMA beta of the local return on the FX return.
    """
    local_return, fx_return = compute_local_and_fx_return(asset_price_local_ccy=asset_price_local_ccy,
                                                          local_to_reference_fx_rate=local_to_reference_fx_rate,
                                                          freq=freq,
                                                          is_log_returns=True)

    fx_beta = qis.compute_ewm_cross_xy(x_data=fx_return.to_frame(),
                                       y_data=local_return.to_frame(),
                                       span=span,
                                       cross_xy_type=qis.CrossXyType.BETA,
                                       mean_adj_type=qis.MeanAdjType.EWMA)
    fx_vol = qis.compute_ewm_vol(data=fx_return.to_frame(), span=span,
                                 mean_adj_type=qis.MeanAdjType.EWMA,
                                 init_value=0.08 ** 2 * 1.0 / 12.0,
                                 annualize=True)

    return fx_vol.iloc[:, 0], fx_beta.iloc[:, 0]


def compute_fx_optimal_hedge(asset_price_local_ccy: pd.Series,
                             local_to_reference_fx_rate: pd.Series,
                             forward_rate_for_local_ccy: pd.Series,
                             freq: str = 'ME',
                             span: int = 3 * 12,
                             risk_aversion_lambda: float = 4.0 / 3.0,
                             min_max_hedge: Optional[Tuple[float, float]] = (0.0, 1.0)
                             ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Mean-variance optimal FX hedge ratios, plus the carry and beta references.

    From the EWMA FX vol/beta and the annualised forward premium, builds three
    per-period hedge-ratio series:

      * ``carry_ratio = (annualised_forward / fx_var) / (2 * risk_aversion_lambda)``
      * ``Max Carry``    = ``1 - carry_ratio``  (pure carry tilt away from full hedge)
      * ``Beta Hedge``   = ``1 + fx_beta``      (removes the FX exposure implied by
        the local-return-on-FX beta)
      * ``Optimal``      = ``1 - carry_ratio + fx_beta`` (carry tilt plus beta hedge)

    All three are optionally clipped to ``min_max_hedge``.

    Args:
        asset_price_local_ccy: Asset price quoted in its local currency.
        local_to_reference_fx_rate: Units of reference currency per 1 unit of the
            local currency.
        forward_rate_for_local_ccy: Per-period CIP forward premium; annualised
            internally by dividing by ``dt = 1 / annualisation_factor(freq)``.
        freq: Resampling frequency (e.g. ``'ME'``); sets ``dt`` and the EWMA cadence.
        span: EWMA span in periods of ``freq`` for the vol/beta estimates.
        risk_aversion_lambda: Mean-variance risk-aversion coefficient (larger ⇒
            smaller carry tilt).
        min_max_hedge: Optional ``(min, max)`` clip applied to every hedge ratio;
            ``None`` leaves them unclipped.

    Returns:
        Tuple ``(optimal_hedge, max_carry, beta_hedged)`` of hedge-ratio Series at ``freq``.
    """
    fx_vol, fx_beta = compute_fx_vol_beta(asset_price_local_ccy=asset_price_local_ccy,
                                          local_to_reference_fx_rate=local_to_reference_fx_rate,
                                          freq=freq, span=span)
    aligned_data = pd.concat(
        [fx_vol.rename('vol'), fx_beta.rename('beta'), forward_rate_for_local_ccy.rename('forward')], axis=1)
    aligned_data = aligned_data.asfreq(freq).ffill()
    fx_var = aligned_data['vol'] ** 2

    # Calculate carry ratio using mean-variance optimization
    dt = 1.0 / qis.get_annualization_factor(freq)
    annualised_forward = aligned_data['forward'] / dt
    carry_ratio = ( annualised_forward / fx_var) / (2.0 * risk_aversion_lambda)
    fx_beta = aligned_data['beta']
    beta_hedged = 1.0 + fx_beta
    max_carry = 1.0 - carry_ratio
    optimal_hedge = 1.0 - carry_ratio + fx_beta

    # Apply hedge ratio constraints
    if min_max_hedge is not None:
        optimal_hedge = np.clip(optimal_hedge, a_min=min_max_hedge[0], a_max=min_max_hedge[1])
        max_carry = np.clip(max_carry, a_min=min_max_hedge[0], a_max=min_max_hedge[1])
        beta_hedged = np.clip(beta_hedged, a_min=min_max_hedge[0], a_max=min_max_hedge[1])
    return optimal_hedge.rename('Optimal'), max_carry.rename('Max Carry'), beta_hedged.rename('Beta Hedge')


def get_aligned_fx_spots(prices: pd.DataFrame,
                         asset_ccy_map: Union[pd.Series, Dict],
                         fx_prices: pd.DataFrame,
                         quote_currency: str = 'USD'
                         ) -> pd.DataFrame:
    """
    the FX spot series belonging to each instrument, on the index of its prices.

    An instrument panel is quoted in mixed currencies, and converting it needs a spot series per
    instrument rather than per currency. This maps each column of ``prices`` through its currency to
    the matching spot column, reindexes onto the price dates and masks where the price is missing,
    so the spots are NaN exactly where the prices are.

    Args:
        prices: instrument prices, one column per instrument
        asset_ccy_map: instrument to currency, as a Series indexed by instrument or a dict
        fx_prices: FX spots, one column per currency
        quote_currency: the numeraire, whose spot column is set to one

    Returns:
        spots in the shape of ``prices``, one column per instrument
    """
    # first backfill and the bbfill so prices will have corresponding fx spots data
    fx_prices = fx_prices.reindex(index=prices.index, method='ffill').ffill().bfill()
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
    returns of a futures position in a foreign currency, converted to the quote currency.

    For a future, only the margin flow is in the foreign currency, so the FX move applies to the
    return and not to the notional: ``r = r_local + r_local * r_fx``. Contrast
    :func:`compute_cash_fx_adjusted_returns`, where the notional is exposed as well.

    Args:
        prices: instrument prices in local currency
        fx_spots: spot per instrument, aligned with ``prices``; see :func:`get_aligned_fx_spots`
        periods: return horizon in index steps
        is_log_returns: return log returns rather than arithmetic ones

    Returns:
        quote-currency returns in the shape of ``prices``
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
    returns of a cash position in a foreign currency, converted to the quote currency.

    For a cash instrument the whole notional sits in the foreign currency, so the FX move applies to
    the notional as well as to the return: ``r = r_fx + r_local + r_local * r_fx``. The extra
    ``r_fx`` term is the difference from :func:`compute_futures_fx_adjusted_returns`, and it is the
    whole of the unhedged currency exposure.

    Args:
        prices: instrument prices in local currency
        fx_spots: spot per instrument, aligned with ``prices``; see :func:`get_aligned_fx_spots`
        periods: return horizon in index steps
        is_log_returns: return log returns rather than arithmetic ones

    Returns:
        quote-currency returns in the shape of ``prices``
    """
    price_return = prices / prices.shift(periods=periods) - 1.0
    fx_return = fx_spots / fx_spots.shift(periods=periods) - 1.0
    returns = np.log(1.0 + fx_return + price_return + price_return * fx_return)
    if not is_log_returns:
        returns = np.expm1(returns)
    return returns
