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
spot return. Opening quotes price each interval. Legacy premium-based currency-index methods
were removed in QIS 6.0.0; use the typed FX_FORWARD request for exact forward settlement.

Main entry points: ``get_local_to_reference_fx_rate`` and ``get_local_rate`` for the raw
quantities, ``compute_fx_returns`` with an ``FxCurrencyReturnsRequest`` for spot/forward P&L,
``build_local_cash_nav`` and ``build_cross_fx_cash_nav`` for money-market legs, and
``compute_fx_returns`` with an ``FxAssetReturnsRequest`` to move a whole asset panel
into a reference currency. The typed result does not construct a NAV unless requested.

Loading is from the CSV pair written by the data layer, through ``FxRatesData.load`` or
``load_fx_rates_data``; building that data from a vendor is ``bbg-fetch``, not this module.
Choosing the hedge ratio rather than applying it is ``qis/fx/fx_hedging.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from dataclasses import dataclass
from typing import Tuple, Union, Optional, ClassVar

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
        ``get_local_to_reference_fx_rate('EUR', 'CHF')`` returns EUR/CHF â€” CHF per EUR).

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
        delivers Fx Î² â‰ˆ +1 for the foreign cash leg, plus small Carry/
        Rates loadings â€” without contaminating the main asset-universe
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
        # applied â€” the position carries full FX exposure.
        from qis.fx.fx_returns import (
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
        """Accrue ACT/365F cash on actual boundaries, sharing QIS funding's primitive.

        ``freq`` is retained for signature stability with existing callers; only
        ``return_index`` determines elapsed days and quote/reset boundaries.
        """
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
            # Exact equality â†’ stale mapping. Ratio â‰ˆ 0.01 â†’ already correct.
            if gbp_val > 0 and abs(gbp_minor_val - gbp_val) < 1e-12:
                fx_spots['GBp'] = 0.01 * fx_spots['GBP']

    return fx_spots, domestic_rates
