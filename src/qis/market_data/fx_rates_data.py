"""
``FxRatesData``: spot rates and domestic short rates, and everything currency conversion needs.

Two panels on one index: ``fx_spots``, quoted as USD per one unit of each currency, and
``domestic_rates``, the annualised short rate per currency. Construction sorts both source
calendars, forward fills them causally, and retains off-grid rate updates before selecting the
spot dates, so the two stored panels never drift apart.

The quote convention is the thing to get right, because it is what makes every downstream sign
correct. Each spot column is USD per 1 unit of that currency, so USD is identically one and a
cross rate is a ratio of two columns: ``get_local_to_reference_fx_rate('EUR', 'CHF')`` returns
CHF per EUR - units of the *reference* currency per one unit of the *local* one.

Cash uses opening quotes times actual calendar days / 365 on each existing reset interval.
Carry follows from matched-period covered interest parity rather than a forward panel.
With dt equal to that ACT/365F fraction, the local/reference cash-growth premium is

    simple:  (1 + dt r_loc) / (1 + dt r_ref) - 1
    log:     log( (1 + dt r_loc) / (1 + dt r_ref) )

The exact long-local forward payoff adds f/(1+f), not this premium f, to the simple
spot return. Opening quotes price each interval. The legacy premium-based currency indices
are deprecated; their retained algorithms are not exact forward settlement.

Main entry points: ``get_local_to_reference_fx_rate`` and ``get_local_rate`` for the raw
quantities, ``compute_fx_returns`` with an ``FxCurrencyReturnsRequest`` for spot/forward P&L,
``build_local_cash_nav`` and ``build_cross_fx_cash_nav`` for money-market legs, and
``compute_fx_returns`` with an ``FxAssetReturnsRequest`` to move a whole asset panel
into a reference currency. The typed result does not construct a NAV unless requested.

Loading is from the CSV pair written by the data layer, through ``FxRatesData.load`` or
``load_fx_rates_data``; building that data from a vendor is ``bbg-fetch``, not this module.
Choosing the hedge ratio rather than applying it is ``qis/market_data/fx_hedging.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import qis as qis
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Dict, Literal, ClassVar

from qis.market_data.fx_hedging import (
    compute_local_and_fx_return,
    compute_performance_of_local_ccy_asset_in_reference_ccy,
)
from qis.utils.df_ops import multiply_df_by_dt

@dataclass
class FxRatesData:
    """
    FX spot rates and domestic short rates, aligned on a common index.

    Spots are quoted as USD per one unit of the local currency, so a cross rate is a ratio of two
    columns and the quote currency's own column is identically one. Construction sorts each source
    calendar before time-directed filling and retains off-grid rate updates before reindexing rates
    onto the spot dates, so the two stored panels are causally aligned.

    Attributes:
        fx_spots: USD per 1 unit of each currency, one column per currency
        domestic_rates: annualised short rate per currency, on the same columns
        cash_day_count: Fixed ACT/365F analytical cash convention; not a stored panel.
    """
    cash_day_count: ClassVar[str] = 'ACT/365F'
    fx_spots: pd.DataFrame
    domestic_rates: pd.DataFrame

    def __post_init__(self):
        """Sort and causally align domestic rates to the forward-filled FX spot dates."""
        # Fill only after sorting so physical row order cannot carry a future spot into the past.
        self.fx_spots = self.fx_spots.sort_index(kind='stable').ffill()
        # Fill on the union calendar so rate updates between spot dates remain available as-of.
        rate_calendar = self.domestic_rates.index.union(self.fx_spots.index).sort_values()
        self.domestic_rates = (
            self.domestic_rates.sort_index(kind='stable')
            .reindex(index=rate_calendar)
            .ffill()
            .reindex(index=self.fx_spots.index)
        )

    @classmethod
    def load(cls, local_path: str, time_period: qis.TimePeriod = None) -> FxRatesData:
        """Load FX rates universe from CSV files."""
        fx_spots, domestic_rates = load_fx_rates_data(local_path=local_path)
        if time_period is not None:
            fx_spots = time_period.locate(fx_spots)
            domestic_rates = time_period.locate(domestic_rates)
        return FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)

    def get_local_to_reference_fx_rate(
            self, local_ccy: str = 'USD', reference_ccy: str = 'CHF') -> pd.Series:
        """Cross FX rate between two currencies.

        Computed as ``fx_spots[local_ccy] / fx_spots[reference_ccy]``. Because each
        ``fx_spots`` column is USD per 1 unit of that currency, the ratio is units
        of reference currency per 1 unit of the local currency (e.g.
        ``get_local_to_reference_fx_rate('EUR', 'CHF')`` returns EUR/CHF — CHF per EUR).

        Args:
            local_ccy: Base currency (numerator of the cross).
            reference_ccy: Reference currency (denominator of the cross).

        Returns:
            Series of reference-per-local FX levels, named ``f"{local_ccy}-{reference_ccy}"``.
        """
        local_per_reference = np.divide(self.fx_spots[local_ccy], self.fx_spots[reference_ccy])
        return local_per_reference.rename(f"{local_ccy}-{reference_ccy}")

    def get_local_rate(self, freq: str = 'ME', local_ccy: str = 'USD',
                       annualise: bool = False) -> pd.Series:
        """Domestic short rate for ``local_ccy`` sampled at ``freq``.

        ``domestic_rates`` are stored as annualised decimals. With
        ``annualise=False`` (default) returns earned ACT/365F cash on the
        existing grid, using the preceding grid date's as-of quote. The first
        date accrues zero. With ``annualise=True`` annual quotes are unchanged.

        Args:
            freq: Resampling frequency for the rate series (e.g. ``'ME'``, ``'B'``).
            local_ccy: Currency whose short rate to return.
            annualise: If True return the annualised rate, else the per-period rate.

        Returns:
            Series of short-rate values sampled at ``freq``.
        """
        rates = self.domestic_rates[local_ccy].asfreq(freq, method='ffill')
        if annualise:
            return rates
        return self._get_period_cash_returns(rates.index, local_ccy, freq, False)

    def get_forward_rate_for_local_ccy(
            self, local_ccy: str = 'USD', reference_ccy: str = 'CHF',
            freq: str = 'ME', verbose: bool = False, is_log_returns: bool = False,
            period_dates: Optional[pd.DatetimeIndex] = None) -> pd.Series:
        """Price dated synthetic forwards with opening-rate ACT/365 Fixed accrual.

        The premium remains local cash growth / reference cash growth - 1.
        For reference-per-local spot, F/S is its reciprocal gross factor.
        Values are labelled at the opening, for the next date in period_dates.
        Thus the existing payoff kernel applies its single opening-quote lag.

        Args:
            local_ccy: Currency delivered by the short forward.
            reference_ccy: Currency received at delivery.
            freq: Existing reset cadence when period_dates is omitted.
            verbose: Print the calculated opening premiums.
            is_log_returns: Return the log cash-growth ratio instead of ratio minus one.
            period_dates: Complete actual interval boundaries. No rate after an opening
                is read for its forward. The final boundary has no subsequent maturity,
                so its premium is missing. When omitted, use the regular reset calendar
                through the last source period and one additional known maturity; retain
                the final usable opening quote. Off-grid source quotes are aligned as-of.

        Returns:
            Opening-labelled forward premiums in the unchanged quote direction.
            The default output follows the reset grid, rather than the raw quote grid.

        Raises:
            ValueError: Boundaries are missing, repeated or unordered, or either
                cash gross growth factor is nonpositive.
        """
        default_schedule = period_dates is None
        if default_schedule:
            if self.domestic_rates.empty:
                return pd.Series(dtype=float, name=f'{local_ccy}-{reference_ccy}')
            offset = pd.tseries.frequencies.to_offset(freq)
            first = offset.rollforward(self.domestic_rates.index[0])
            last = offset.rollforward(self.domestic_rates.index[-1])
            period_dates = pd.date_range(first, last + offset, freq=freq)
        if (not isinstance(period_dates, pd.DatetimeIndex) or period_dates.hasnans
                or not period_dates.is_unique or not period_dates.is_monotonic_increasing):
            raise ValueError('period_dates must be ordered unique nonmissing timestamps')
        local_cash = self._get_period_cash_returns(
            period_dates, local_ccy, freq, False, cash_rate_lag=1)
        reference_cash = self._get_period_cash_returns(
            period_dates, reference_ccy, freq, False, cash_rate_lag=1)
        premium = (1.0 + local_cash) / (1.0 + reference_cash) - 1.0
        if is_log_returns:
            premium = np.log1p(premium)
        forward_rate = premium.shift(-1).rename(f'{local_ccy}-{reference_ccy}')
        if default_schedule:
            forward_rate = forward_rate.iloc[:-1]
        if verbose:
            print(f"forward_rate=\n{forward_rate}")
        return forward_rate

    def get_daily_carry_local_return(self, local_ccy: str = 'USD',
                                     reference_ccy: str = 'CHF',
                                     is_log_returns: bool = False
                                     ) -> Tuple[pd.Series, pd.Series]:
        """Deprecated daily spot and inverse-forward premium components of a currency pair.

        Splits the daily total FX return of holding ``local_ccy`` against
        ``reference_ccy`` into the spot return (return of the cross rate) and the
        carry return (the lagged daily CIP forward premium). Both legs honour the
        ``is_log_returns`` convention.

        Args:
            local_ccy: Local (held) currency of the pair.
            reference_ccy: Reference currency of the pair.
            is_log_returns: If True compute log returns, otherwise simple returns.

        Returns:
            Tuple ``(spot_return, carry_return)`` of daily Series; the carry leg is
            lagged one day with its first observation set to 0.
        """
        warnings.warn(
            'get_daily_carry_local_return is deprecated in QIS 5.46.0; '
            'use compute_fx_returns with FX_FORWARD and explicit FxIndexNormalization. '
            'Legacy premium-based analytics are unchanged. Removal: QIS 6.0.0.',
            DeprecationWarning, stacklevel=2)
        local_return = qis.to_returns(
            prices=self.get_local_to_reference_fx_rate(
                local_ccy=local_ccy, reference_ccy=reference_ccy),
            is_log_returns=is_log_returns)
        forward_rate_dt = self.get_forward_rate_for_local_ccy(
            local_ccy=local_ccy, reference_ccy=reference_ccy,
            freq='B', is_log_returns=is_log_returns, period_dates=local_return.index)
        carry_return = forward_rate_dt.shift(1)
        carry_return.iloc[0] = 0.0
        return local_return, carry_return

    def get_fx_total_return_nav(self,
                                local_ccy: str = 'USD',
                                reference_ccy: str = 'CHF',
                                time_period: qis.TimePeriod = None,
                                freq: Optional[str] = None
                                ) -> pd.Series:
        """Deprecated synthetic index of spot plus inverse-forward premium, not exact P&L.

        Adds the daily spot and carry legs from ``get_daily_carry_local_return``
        and compounds them into a NAV. Uses simple returns throughout.

        Args:
            local_ccy: Local (held) currency of the pair.
            reference_ccy: Reference currency of the pair.
            time_period: Optional date filter applied to the daily return before
                compounding.
            freq: Optional period-end reporting frequency. Each completed period uses its final
                available NAV; ``None`` keeps daily values.

        Returns:
            Series of NAV levels (starting at 1.0), at daily or ``freq`` cadence.
        """
        warnings.warn(
            'get_fx_total_return_nav is deprecated in QIS 5.46.0; '
            'use compute_fx_returns with FX_FORWARD and explicit FxIndexNormalization. '
            'Legacy premium-based analytics are unchanged. Removal: QIS 6.0.0.',
            DeprecationWarning, stacklevel=2)
        local_return, carry_return = self.get_daily_carry_local_return(
            local_ccy=local_ccy, reference_ccy=reference_ccy, is_log_returns=False)
        total_return = np.add(local_return, carry_return).rename(f"{local_ccy}-{reference_ccy}")
        if time_period is not None:
            total_return = time_period.locate(total_return)
        nav = qis.returns_to_nav(total_return, is_log_returns=False)
        if freq is not None:
            nav = qis.df_asfreq(df=nav, freq=freq)
        return nav

    def get_carry_fx_return_nav(self,
                                local_ccy: str = 'USD',
                                reference_ccy: str = 'CHF',
                                is_normalise_by_spot_vol: bool = True,
                                time_period: qis.TimePeriod = None,
                                freq: Optional[str] = None,
                                is_causal: bool = False,
                                ) -> pd.Series:
        """Deprecated premium-based carry diagnostic; not an exact forward payoff index.

        The ``is_normalise_by_spot_vol`` path adjusts the carry return
        stream by adding the demeaned spot return with an Itô-type
        variance correction (``- mean + 0.5 * var``). The intent is to
        make the resulting NAV's realized volatility comparable to the
        spot-return series over the same window.

        **Look-ahead warning**: when ``is_causal=False`` (default, for
        backward compatibility), the mean/variance statistics are
        computed on the **full sample** and applied uniformly across
        time. This is acceptable for ex-post reporting / plotting where
        the goal is to show a vol-comparable NAV over a fixed window,
        but is **not suitable for backtesting trading rules** or any
        use where at-time-t decisions depend on the NAV. Pass
        ``is_causal=True`` to use expanding-window statistics that only
        look backward — suitable for backtests at the cost of a
        slightly noisier head.

        Args:
            local_ccy: Local currency of the pair.
            reference_ccy: Reference currency of the pair.
            is_normalise_by_spot_vol: Apply the vol-matching adjustment.
            time_period: Optional filter applied before normalisation.
            freq: Optional period-end reporting frequency. Each completed period uses its final
                available NAV; ``None`` keeps daily values.
            is_causal: If True, use expanding-window statistics to
                avoid look-ahead (recommended for backtests).

        Returns:
            Series of NAV levels indexed by date.
        """
        warnings.warn(
            'get_carry_fx_return_nav is deprecated in QIS 5.46.0; '
            'use compute_fx_returns with FX_FORWARD and explicit FxIndexNormalization. '
            'Legacy premium-based analytics are unchanged. Removal: QIS 6.0.0.',
            DeprecationWarning, stacklevel=2)
        local_return, carry_return = self.get_daily_carry_local_return(
            local_ccy=local_ccy, reference_ccy=reference_ccy, is_log_returns=False)

        if time_period is not None:
            local_return = time_period.locate(local_return)
            carry_return = time_period.locate(carry_return)

        if is_normalise_by_spot_vol:  # vol of carry = vol of demean spot returns
            if is_causal:
                # Expanding-window mean and variance — each t uses only
                # information available up to t. Slightly noisier at
                # the start of the sample but free of look-ahead.
                expanding_mean = local_return.expanding(min_periods=1).mean()
                expanding_var = local_return.expanding(min_periods=1).var().fillna(0.0)
                local_return = local_return - expanding_mean + 0.5 * expanding_var
            else:
                # Full-sample statistics — look-ahead. Retained for
                # reporting / analytics parity with prior behaviour.
                local_return = (local_return - np.nanmean(local_return)
                                + 0.5 * float(np.nanvar(local_return)))
            carry_return = carry_return + local_return

        nav = qis.returns_to_nav(carry_return, is_log_returns=False)
        if freq is not None:
            nav = qis.df_asfreq(df=nav, freq=freq)
        return nav

    def build_local_cash_nav(self,
                             local_ccy: str,
                             freq: str = 'B') -> pd.Series:
        """Build a money-market cash NAV for ``local_ccy``.

        Compounds the local-currency short rate at the given frequency
        starting from 1.0 at the first date. Used as the building block
        for ``build_cross_fx_cash_nav`` and as a synthetic JPM-style
        cash-account series for any currency where a native cash
        index is not available.

        Returns a Series of NAV levels in the local currency.
        """
        rate_per_period = self.get_local_rate(freq=freq, local_ccy=local_ccy, annualise=False)
        # get_local_rate already uses opening quotes and the actual interval.
        period_returns = rate_per_period
        nav = qis.returns_to_nav(returns=period_returns, is_log_returns=False)
        # Unknown accrued cash cannot be repaired by treating it as zero interest.
        nav = nav.where(~period_returns.isna().cummax())
        nav.name = f'cash_{local_ccy}'
        return nav

    def build_cross_fx_cash_nav(self,
                                local_ccy: str,
                                reference_ccy: str,
                                freq: str = 'B',
                                output_freq: str = 'ME') -> pd.Series:
        """Synthetic cash-account NAV for ``local_ccy`` cash held by a
        ``reference_ccy`` investor, unhedged.

        Used to feed the isolated cross-FX MATF regression that
        produces betas for cash assets viewed in non-native reference
        currencies. The synthetic NAV combines:

        * A local-currency cash NAV (compounded at local short rate).
        * Spot translation to reference currency, no hedging applied.

        The resulting series captures the full FX exposure of the
        ``local_ccy``/``reference_ccy`` pair plus the local-currency rf
        carry. Regressed against MATF factors in isolation, this
        delivers Fx β ≈ +1 for the foreign cash leg, plus small Carry/
        Rates loadings — without contaminating the main asset-universe
        factor model.

        Args:
            local_ccy: foreign-leg currency, e.g. ``'EUR'``
            reference_ccy: reference / reporting currency, e.g. ``'USD'``
            freq: construction frequency of the underlying cash NAV. Daily is the safe
                default: the NAV is compounded at this cadence and downsampled at the end
            output_freq: reporting frequency of the returned NAV

        Returns:
            NAV levels in the reference currency, indexed at ``output_freq``. Empty when
            ``local_ccy == reference_ccy``, since the cross-FX construction is meaningful only
            for a non-native reference frame
        """
        if local_ccy == reference_ccy:
            # Native frame: no cross-FX series needed. Caller should
            # use the reference-currency cash NAV directly via
            # ``build_local_cash_nav(reference_ccy)``.
            return pd.Series(dtype=float, name=f'cash_{local_ccy}_in_{reference_ccy}')

        cash_nav_local = self.build_local_cash_nav(local_ccy=local_ccy, freq=freq)

        # Translate to reference ccy via spot, unhedged. The existing
        # FX-translation machinery handles the spot-rate composition;
        # ``hedge_ratio=0.0`` means no forward-rate adjustment is
        # applied — the position carries full FX exposure.
        from qis.market_data.fx_returns import (
            FxAssetReturnsRequest, FxReturnType, FxReturnCoordinate,
            FxIndexNormalization, compute_fx_returns,
        )
        result = compute_fx_returns(self, FxAssetReturnsRequest(
            prices=cash_nav_local.to_frame(), local_ccys=local_ccy,
            reference_ccy=reference_ccy, return_type=FxReturnType.REFERENCE_TOTAL,
            coordinate=FxReturnCoordinate.SIMPLE, hedge_ratios=0.0,
            frequency=output_freq))[output_freq]
        nav_ref = result.to_index(FxIndexNormalization(
            initial_value=1.0, notional_to_capital=1.0, notional_rule='capital_proportional',
            settlement_frequency=output_freq, reinvestment='each_interval',
            cash_treatment='as_returned', missing_returns='skip')).iloc[:, 0]
        nav_ref.name = f'cash_{local_ccy}_in_{reference_ccy}'
        return nav_ref

    def _get_period_cash_returns(self, return_index: pd.DatetimeIndex, local_ccy: str,
                                 freq: str, is_log_returns: bool,
                                 cash_rate_lag: int = 1) -> pd.Series:
        """Accrue ACT/365F cash on actual boundaries, sharing QIS funding's primitive."""
        if isinstance(cash_rate_lag, bool) or not isinstance(cash_rate_lag, (int, np.integer)) \
                or cash_rate_lag < 0:
            raise ValueError("cash_rate_lag must be a nonnegative integer")
        if return_index.empty:
            return self.domestic_rates[local_ccy].reindex(return_index)
        cash = multiply_df_by_dt(
            df=self.domestic_rates[local_ccy], dates=return_index,
            lag=int(cash_rate_lag), is_actual_calendar_dt=True,
            annualization_factor=365.0)
        if (cash <= -1.0).any():
            raise ValueError("Cash gross factors must be strictly positive")
        return np.log1p(cash) if is_log_returns else cash

    def compute_performance_of_local_ccy_asset_in_reference_ccy(
            self, asset_price_local_ccy: pd.Series,
            hedge_ratio: Union[float, pd.Series], local_ccy: str, reference_ccy: str,
            freq: str = 'ME', is_log_returns: bool = False,
            is_excess_returns: bool = False,
            cash_rate_lag: int = 1) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate hedged asset performance in reference currency.

        Returns NAV and returns adjusted for hedge ratio and forward costs.
        Valid zero total returns remain observations and accrue cash in excess
        returns. Inception without a preceding price stays missing.

        When ``is_excess_returns=True``, arithmetic excess is ``R - cash``;
        log excess is ``log1p(R) - log1p(cash)``, the log return relative to cash.
        Cash uses ACT/365F and the reference-currency short rate observed at the start of
        each actual asset-return period by default (``cash_rate_lag=1``).
        ``cash_rate_lag=0`` uses the quote on the return date for descriptive
        estimation; it does not represent cash accrued over the preceding period.
        The two excess conventions are not expm1 equivalents. A principal-only
        FX hedge leaves the local investment gain exposed to terminal FX.
        """
        warnings.warn(
            'compute_performance_of_local_ccy_asset_in_reference_ccy on FxRatesData '
            'is deprecated in QIS 5.45.0; '
            'use compute_fx_returns with FxAssetReturnsRequest. Removal: QIS 6.0.0.',
            DeprecationWarning, stacklevel=2)
        if local_ccy == reference_ccy:
            local_return = qis.to_returns(
                prices=asset_price_local_ccy, freq=freq,
                is_log_returns=is_log_returns)

        else:
            local_to_reference_fx_rate = self.get_local_to_reference_fx_rate(
                local_ccy=local_ccy, reference_ccy=reference_ccy)
            interval_returns, _ = compute_local_and_fx_return(
                asset_price_local_ccy=asset_price_local_ccy,
                local_to_reference_fx_rate=local_to_reference_fx_rate,
                freq=freq, is_log_returns=False)
            forward_rate_for_local_ccy = self.get_forward_rate_for_local_ccy(
                local_ccy=local_ccy, reference_ccy=reference_ccy, freq=freq,
                is_log_returns=is_log_returns, period_dates=interval_returns.index)

            _, local_return = compute_performance_of_local_ccy_asset_in_reference_ccy(
                hedge_ratio=hedge_ratio,
                asset_price_local_ccy=asset_price_local_ccy,
                local_to_reference_fx_rate=local_to_reference_fx_rate,
                forward_rate_for_local_ccy=forward_rate_for_local_ccy,
                freq=freq,
                is_log_returns=is_log_returns)
            # Only the kernel's forced first-index zero is synthetic; subsequent zeros
            # can be real returns. Mask before cash subtraction to avoid spurious funding.
            if not local_return.empty:
                local_return.iloc[0] = np.nan

        if is_excess_returns:
            ref_rate = self._get_period_cash_returns(
                return_index=local_return.index, local_ccy=reference_ccy,
                freq=freq, is_log_returns=is_log_returns,
                cash_rate_lag=cash_rate_lag)
            local_return = local_return - ref_rate
        local_nav = qis.returns_to_nav(returns=local_return, is_log_returns=is_log_returns)
        return local_nav, local_return

    @qis.timer
    def compute_returns_in_reference_ccy(self,
                                         asset_prices: pd.DataFrame,
                                         hedge_ratios: Union[pd.Series, pd.DataFrame],
                                         local_ccys: Union[str, pd.Series],
                                         reference_ccy: str,
                                         freq: str = 'ME',
                                         is_log_returns: bool = False,
                                         is_excess_returns: bool = False,
                                         cash_rate_lag: int = 1
                                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a multi-asset panel to a reference-currency NAV + returns.

        Dispatches each asset through
        ``compute_performance_of_local_ccy_asset_in_reference_ccy``
        and assembles the results into a pair of DataFrames. Use this
        when a single ``freq`` applies to all assets; for per-asset
        frequencies, call ``compute_fx_adjusted_returns`` which wraps
        this method in a freq-groupby.

        Args:
            asset_prices: Native-ccy price panel (columns are assets).
            hedge_ratios: Per-asset hedge ratio in [0, 1]. Series
                (one-value-per-asset) or DataFrame (time-varying
                per-asset). Constant floats must be pre-converted to a
                Series by the caller.
            local_ccys: Per-asset currency of denomination. String
                broadcasts to every column; Series resolves per asset.
            reference_ccy: Target currency for conversion.
            freq: Single frequency string (e.g. 'ME', 'QE'). The FX
                hedge rebalance happens at this cadence and returns
                are computed at the same cadence.
            is_log_returns: If True, returns are log; otherwise simple.
            is_excess_returns: If True, subtract reference-currency cash in the
                selected return convention. See
                ``compute_fx_adjusted_returns`` for the excess-return definition.
            cash_rate_lag: Return-grid periods to lag the cash quote before
                subtraction. The default 1 uses the starting-period quote;
                0 uses the quote on the return date for descriptive estimation.

        Returns:
            Tuple of (NAV DataFrame, Returns DataFrame). The NAV frame
            is reconstructed from the returns via ``returns_to_nav``
            and is provided for plotting / reporting convenience; for
            pure-returns consumers (covariance, alpha) see
            ``compute_fx_adjusted_returns``.
        """
        warnings.warn(
            'compute_returns_in_reference_ccy is deprecated in QIS 5.45.0; '
            'use compute_fx_returns with FxAssetReturnsRequest. Removal: QIS 6.0.0.',
            DeprecationWarning, stacklevel=2)
        fx_adjusted_navs, fx_adjusted_returns = {}, {}
        for asset in asset_prices.columns:
            # Extract asset-specific parameters
            if isinstance(hedge_ratios, pd.Series):
                hedge_ratio = hedge_ratios.loc[asset]
            else:
                hedge_ratio = hedge_ratios.loc[:, asset]
            if isinstance(local_ccys, pd.Series):
                local_ccy = local_ccys.loc[asset]
            else:
                local_ccy = local_ccys

            fx_adjusted_navs[asset], fx_adjusted_returns[asset] = (
                self.compute_performance_of_local_ccy_asset_in_reference_ccy(
                    asset_price_local_ccy=asset_prices[asset],
                    hedge_ratio=hedge_ratio,
                    local_ccy=local_ccy,
                    reference_ccy=reference_ccy,
                    freq=freq,
                    is_log_returns=is_log_returns,
                    is_excess_returns=is_excess_returns,
                    cash_rate_lag=cash_rate_lag)
            )
        fx_adjusted_navs = pd.DataFrame.from_dict(fx_adjusted_navs, orient='columns')
        fx_adjusted_returns = pd.DataFrame.from_dict(fx_adjusted_returns, orient='columns')
        return fx_adjusted_navs, fx_adjusted_returns

    def compute_fx_adjusted_returns(self,
                                    prices: pd.DataFrame,
                                    hedge_ratios: pd.Series,
                                    local_ccys: pd.Series,
                                    reference_ccy: Union[
                                        str, Literal['CHF', 'EUR', 'GBP', 'USD']] = 'USD',
                                    freq: Union[str, pd.Series] = 'ME',
                                    is_log_returns: bool = False,
                                    is_excess_returns: bool = False,
                                    zero_return_to_nan: bool = True,
                                    cash_rate_lag: int = 1
                                    ) -> Dict[str, pd.DataFrame]:
        """Compute per-period returns of a multi-asset panel in a reference ccy.

        Groups assets by ``freq`` when a per-asset Series is supplied,
        dispatches each group through ``compute_returns_in_reference_ccy``
        at that group's frequency, and returns a ``{freq: DataFrame}``
        mapping. This is the primary entry point consumed by the
        covariance estimator and the alpha aggregator in the PM and CMA
        pipelines.

        Returns are simple (arithmetic) by default, matching ``qis.to_returns``
        and the single-asset conversion method. Callers should specify the
        convention explicitly; use ``is_log_returns=True`` for log-return
        estimation inputs.

        **Zero-return to NaN substitution** (the default):
        step-function PE series (prices constant between quarterly
        appraisal updates) produce zero returns at intra-period dates.
        Downstream β and correlation estimators would otherwise weight
        these structural zeros as informative observations, biasing
        factor loadings toward zero for illiquid assets. Replacing them
        with NaN causes the rolling estimators to skip those periods
        entirely — mathematically equivalent to estimating on the
        sparse observation frequency, which matches the actual
        information arrival pattern.

        This is a global exact-zero replacement, so an organically zero
        return will also be converted to NaN. Set ``zero_return_to_nan=False``
        when a zero is a valid observation that must remain in the return panel.

        Args:
            prices: Native-ccy price panel (columns are assets).
            hedge_ratios: Per-asset hedge ratio in [0, 1].
            local_ccys: Per-asset currency of denomination.
            reference_ccy: Target currency for conversion.
            freq: Single frequency string or per-asset Series. A Series
                dispatches the panel into asset-frequency buckets.
            is_log_returns: If True, returns are log; otherwise simple (default).
            is_excess_returns: If True, subtract reference-currency cash in the
                selected return convention: simple cash from simple
                returns, or log1p(cash) from log returns. A principal-only FX hedge
                leaves local investment gains exposed to terminal FX.
            zero_return_to_nan: If True (default), treat exact-zero returns as
                missing observations; if False, retain them as valid returns.
            cash_rate_lag: Return-grid periods to lag the cash quote before
                excess-return subtraction. The default 1 uses the starting-period
                quote; 0 uses the quote on the return date for estimation.

        Returns:
            Mapping ``{freq_string: returns_dataframe}``. Single-freq
            input yields a single-key dict; per-asset freq input yields
            one key per unique frequency in the input.
        """
        warnings.warn(
            'compute_fx_adjusted_returns is deprecated in QIS 5.45.0; '
            'use compute_fx_returns with FxAssetReturnsRequest. Removal: QIS 6.0.0.',
            DeprecationWarning, stacklevel=2)
        if isinstance(freq, str):
            # Single frequency for all assets
            converted_prices, fx_adjusted_returns = self.compute_returns_in_reference_ccy(
                asset_prices=prices,
                hedge_ratios=hedge_ratios,
                local_ccys=local_ccys,
                reference_ccy=reference_ccy,
                freq=freq,
                is_log_returns=is_log_returns,
                is_excess_returns=is_excess_returns,
                cash_rate_lag=cash_rate_lag,
            )
            if zero_return_to_nan:
                fx_adjusted_returns = fx_adjusted_returns.replace({0.0: np.nan})
            return {freq: fx_adjusted_returns}

        else:
            # freq is pd.Series with asset-specific frequencies
            results = {}
            for frequency, assets in freq.groupby(freq):
                asset_list = assets.index.tolist()
                asset_prices = prices[asset_list]
                asset_hedge_ratios = hedge_ratios.loc[asset_list]
                asset_ccys = local_ccys.loc[asset_list]

                converted_prices, fx_adjusted_returns = self.compute_returns_in_reference_ccy(
                    asset_prices=asset_prices,
                    hedge_ratios=asset_hedge_ratios,
                    local_ccys=asset_ccys,
                    reference_ccy=reference_ccy,
                    freq=str(frequency),
                    is_log_returns=is_log_returns,
                    is_excess_returns=is_excess_returns,
                    cash_rate_lag=cash_rate_lag,
                )
                if zero_return_to_nan:
                    fx_adjusted_returns = fx_adjusted_returns.replace({0.0: np.nan})
                results[frequency] = fx_adjusted_returns
            return results

    def fetch_local_rates(self,
                          local_ccys: pd.Series,
                          freq: str = 'ME',
                          annualise: bool = False
                          ) -> Union[pd.Series, pd.DataFrame]:
        """Per-asset domestic short-rate panel.

        Builds a DataFrame of domestic short rates aligned to the assets in
        ``local_ccys`` by mapping each asset to its currency's ``get_local_rate``
        series.

        Args:
            local_ccys: Per-asset local currency (Series indexed by asset).
            freq: Resampling frequency for the rate series.
            annualise: If True use annualised rates, else per-period rates.

        Returns:
            DataFrame of short rates, columns are the assets in ``local_ccys``.
        """
        local_rate_ids = local_ccys.unique()
        local_rates_dict = {
            local_rate_id: self.get_local_rate(
                freq=freq, local_ccy=local_rate_id, annualise=annualise)
            for local_rate_id in local_rate_ids
        }
        local_rates = pd.DataFrame.from_dict(
            {asset: local_rates_dict[local_ccys.loc[asset]] for asset in local_ccys.index},
            orient='columns')
        return local_rates

    def compute_returns_adjusted_by_local_rate(self,
                                               asset_prices: pd.DataFrame,
                                               local_ccys: pd.Series,
                                               freq: str = 'ME',
                                               is_log_returns: bool = False,
                                               cash_rate_lag: int = 1
                                               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Native-currency returns and excess-over-local-rate returns for a panel.

        Computes each asset's return in its own local currency (no FX conversion)
        and the corresponding excess return over that currency's domestic short
        rate. Cash quotes are as-of aligned to the actual return grid and lagged
        ``cash_rate_lag`` periods (1 by default). Arithmetic excess subtracts
        simple cash; log excess subtracts log1p(cash). The returned annual rate
        panel remains a quote/reporting panel,
        not the lagged cash return deducted in the calculation.

        Args:
            asset_prices: Native-currency price panel (columns are assets).
            local_ccys: Per-asset local currency (Series indexed by asset).
            freq: Resampling frequency for the returns and rates.
            is_log_returns: If True compute log returns, otherwise simple returns.
            cash_rate_lag: Return-grid periods to lag the cash quote; 0 uses the
                current quote for descriptive estimation.

        Returns:
            Tuple ``(local_returns, excess_returns, local_rates)`` of DataFrames;
            ``local_rates`` holds the annualised per-asset short rates.
        """
        warnings.warn(
            'compute_returns_adjusted_by_local_rate is deprecated in QIS 5.45.0; '
            'use compute_fx_returns with FxAssetReturnsRequest. Removal: QIS 6.0.0.',
            DeprecationWarning, stacklevel=2)
        local_returns = qis.to_returns(
            prices=asset_prices, freq=freq, is_log_returns=is_log_returns)
        local_rates = self.fetch_local_rates(local_ccys=local_ccys, freq=freq, annualise=True)
        cash_by_currency = {
            ccy: self._get_period_cash_returns(
                local_returns.index, ccy, freq, is_log_returns, cash_rate_lag)
            for ccy in local_ccys.unique()
        }
        local_rates_dt = pd.DataFrame({
            asset: cash_by_currency[local_ccys.loc[asset]] for asset in local_returns.columns
        })
        excess_returns = local_returns - local_rates_dt
        return local_returns, excess_returns, local_rates


def load_fx_rates_data(local_path: str,
                       file_name: str = 'fx_hedging_data'
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    read FX spots and domestic short rates from the saved CSV pair.

    Applies one in-memory correction: historical files stored ``GBp`` as a copy of ``GBP``, where
    the correct relationship is ``GBp = 0.01 * GBP``. A file carrying the stale mapping is healed on
    load, so callers always see the corrected series; rerunning ``create_fx_rates_data`` persists
    the fix.

    Args:
        local_path: directory holding the CSV files
        file_name: base file name; the spots and rates are stored under keys of it

    Returns:
        the spots and the domestic rates, in that order
    """
    data = qis.load_df_dict_from_csv(dataset_keys=['fx_spots', 'domestic_rates'],
                                     file_name=file_name, local_path=local_path)
    fx_spots, domestic_rates = data['fx_spots'], data['domestic_rates']

    # Heal stale GBp mapping. Compare a non-NaN row; if GBp == GBP the
    # persisted CSV is pre-fix and needs in-memory correction. Use
    # ``iloc`` on the last row so we check recent data rather than any
    # early-history NaNs.
    if 'GBp' in fx_spots.columns and 'GBP' in fx_spots.columns:
        last_valid = fx_spots[['GBp', 'GBP']].dropna().tail(1)
        if not last_valid.empty:
            gbp_val = float(last_valid['GBP'].iloc[0])
            gbp_minor_val = float(last_valid['GBp'].iloc[0])
            # Exact equality → stale mapping. Ratio ≈ 0.01 → already correct.
            if gbp_val > 0 and abs(gbp_minor_val - gbp_val) < 1e-12:
                fx_spots['GBp'] = 0.01 * fx_spots['GBP']

    return fx_spots, domestic_rates
