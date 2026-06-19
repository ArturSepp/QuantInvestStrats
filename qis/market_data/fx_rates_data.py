"""
FX rates data container.

Holds FX spot rates and domestic short rates and derives cross rates, CIP
forward premia, carry decomposition, and reference-currency conversions. The
container loads from CSV (or a SQL/Ramen source via ``from_sql``); the data is
built upstream — see the Bloomberg builder in the production layer, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Dict, Literal

from qis.market_data.fx_hedging import compute_performance_of_local_ccy_asset_in_reference_ccy


@dataclass
class FxRatesData:
    """
    Container for FX spot rates and domestic interest rates.

    Supports conversion between currencies and forward rate calculations.
    FX spots are quoted as local_ccy/USD (units of USD per 1 local currency).
    """
    fx_spots: pd.DataFrame
    domestic_rates: pd.DataFrame

    def __post_init__(self):
        """Align domestic rates to FX spot dates and forward fill missing values."""
        self.fx_spots = self.fx_spots.ffill()
        self.domestic_rates = self.domestic_rates.reindex(index=self.fx_spots.index).ffill()

    @classmethod
    def load(cls, local_path: str, time_period: qis.TimePeriod = None) -> FxRatesData:
        """Load FX rates universe from CSV files."""
        fx_spots, domestic_rates = load_fx_rates_data(local_path=local_path)
        if time_period is not None:
            fx_spots = time_period.locate(fx_spots)
            domestic_rates = time_period.locate(domestic_rates)
        return FxRatesData(fx_spots=fx_spots, domestic_rates=domestic_rates)

    def get_local_to_reference_fx_rate(self, local_ccy: str = 'USD', reference_ccy: str = 'CHF') -> pd.Series:
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

    def get_local_rate(self, freq: str = 'ME', local_ccy: str = 'USD', annualise: bool = False) -> pd.Series:
        """Domestic short rate for ``local_ccy`` sampled at ``freq``.

        ``domestic_rates`` are stored as annualised decimals. With
        ``annualise=False`` (default) the rate is scaled to a per-period rate by
        ``dt = 1 / annualisation_factor(freq)``; with ``annualise=True`` the
        annualised rate is returned unchanged.

        Args:
            freq: Resampling frequency for the rate series (e.g. ``'ME'``, ``'B'``).
            local_ccy: Currency whose short rate to return.
            annualise: If True return the annualised rate, else the per-period rate.

        Returns:
            Series of short-rate values sampled at ``freq``.
        """
        if annualise:
            dt = 1.0
        else:
            dt = 1.0 / qis.get_annualization_factor(freq)
        rate = self.domestic_rates[local_ccy].asfreq(freq, method='ffill')
        return rate*dt

    def get_forward_rate_for_local_ccy(self, local_ccy: str = 'USD', reference_ccy: str = 'CHF',
                                       freq: str = 'ME',
                                       verbose: bool = False,
                                       is_log_returns: bool = False
                                       ) -> pd.Series:
        """
        Calculate implied forward-rate return over one period of length ``dt``
        using covered interest-rate parity.

        Let r_loc, r_ref be the annualised local and reference short rates and
        ``dt = 1 / annualisation_factor(freq)``. Under CIP, the forward
        premium earned on the local currency over one period is

            simple:  (1 + dt * r_loc) / (1 + dt * r_ref) - 1
            log:     log( (1 + dt * r_loc) / (1 + dt * r_ref) )

        The previous implementation used ``np.log(a, b)``, where the second
        argument is silently treated by numpy as an ``out`` buffer rather
        than a denominator. That produced ``log(1 + dt*r_loc)`` and
        overwrote the buffer holding ``1 + dt*r_ref``. Fixed here.
        """
        dt = 1.0 / qis.get_annualization_factor(freq)
        rate_data = pd.concat([self.domestic_rates[local_ccy], self.domestic_rates[reference_ccy]], axis=1)
        numer = 1.0 + dt * rate_data.iloc[:, 0]
        denom = 1.0 + dt * rate_data.iloc[:, 1]
        if is_log_returns:
            forward_rate = np.log(numer / denom)
        else:
            forward_rate = numer / denom - 1.0
        if verbose:
            print(f"forward_rate=\n{forward_rate}")
        return forward_rate.rename(f"{local_ccy}-{reference_ccy}")

    def get_daily_carry_local_return(self, local_ccy: str = 'USD',
                                     reference_ccy: str = 'CHF',
                                     is_log_returns: bool = False
                                     ) -> Tuple[pd.Series, pd.Series]:
        """Daily spot and carry return components of a currency pair.

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
        local_return = qis.to_returns(prices=self.get_local_to_reference_fx_rate(local_ccy=local_ccy, reference_ccy=reference_ccy),
                                      is_log_returns=is_log_returns)
        forward_rate_dt = self.get_forward_rate_for_local_ccy(local_ccy=local_ccy, reference_ccy=reference_ccy,
                                                              freq='B', is_log_returns=is_log_returns)
        carry_return = forward_rate_dt.shift(1)
        carry_return.iloc[0] = 0.0
        return local_return, carry_return

    def get_fx_total_return_nav(self,
                                local_ccy: str = 'USD',
                                reference_ccy: str = 'CHF',
                                time_period: qis.TimePeriod = None,
                                freq: Optional[str] = None
                                ) -> pd.Series:
        """Total-return NAV of a currency pair (spot return plus carry).

        Adds the daily spot and carry legs from ``get_daily_carry_local_return``
        and compounds them into a NAV. Uses simple returns throughout.

        Args:
            local_ccy: Local (held) currency of the pair.
            reference_ccy: Reference currency of the pair.
            time_period: Optional date filter applied to the daily return before
                compounding.
            freq: Optional resampling of the output NAV (``None`` keeps daily).

        Returns:
            Series of NAV levels (starting at 1.0), at daily or ``freq`` cadence.
        """
        local_return, carry_return = self.get_daily_carry_local_return(local_ccy=local_ccy, reference_ccy=reference_ccy,
                                                                       is_log_returns=False)
        total_return = np.add(local_return, carry_return).rename(f"{local_ccy}-{reference_ccy}")
        if time_period is not None:
            total_return = time_period.locate(total_return)
        nav = qis.returns_to_nav(total_return, is_log_returns=False)
        if freq is not None:
            nav = nav.asfreq(freq).ffill()
        return nav

    def get_carry_fx_return_nav(self,
                                local_ccy: str = 'USD',
                                reference_ccy: str = 'CHF',
                                is_normalise_by_spot_vol: bool = True,
                                time_period: qis.TimePeriod = None,
                                freq: Optional[str] = None,
                                is_causal: bool = False,
                                ) -> pd.Series:
        """Carry-only FX return NAV for a currency pair.

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
            freq: Optional resample of the output NAV.
            is_causal: If True, use expanding-window statistics to
                avoid look-ahead (recommended for backtests).

        Returns:
            Series of NAV levels indexed by date.
        """
        local_return, carry_return = self.get_daily_carry_local_return(local_ccy=local_ccy, reference_ccy=reference_ccy,
                                                                       is_log_returns=False)

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
                local_return = local_return - np.nanmean(local_return) + 0.5 * float(np.nanvar(local_return))
            carry_return = carry_return + local_return

        nav = qis.returns_to_nav(carry_return, is_log_returns=False)
        if freq is not None:
            nav = nav.asfreq(freq).ffill()
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
        # Step-forward NAV: NAV(t) = NAV(t-1) * (1 + r_{t-1} * dt).
        # Lag the rate by one period so the t-th return uses the rate
        # known at t-1 (avoids look-ahead).
        period_returns = rate_per_period.shift(1).fillna(0.0)
        nav = qis.returns_to_nav(returns=period_returns, is_log_returns=False)
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

        Parameters
        ----------
        local_ccy : str
            Foreign-leg currency (e.g. 'EUR').
        reference_ccy : str
            Reference / reporting currency (e.g. 'USD').
        freq : str, default 'B'
            Construction frequency for the underlying cash NAV. Daily
            ('B') is the safe default; the cash NAV is rebuilt at this
            cadence and downsampled to ``output_freq`` at the end.
        output_freq : str, default 'ME'
            Reporting frequency for the returned NAV series.

        Returns
        -------
        pd.Series
            NAV levels in reference currency, indexed at ``output_freq``.
            Returns an empty series if ``local_ccy == reference_ccy``
            (the cross-FX construction is meaningful only for non-native
            reference frames).
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
        nav_ref, _ = self._compute_performance_of_local_ccy_asset_in_reference_ccy(
            asset_price_local_ccy=cash_nav_local,
            hedge_ratio=0.0,
            local_ccy=local_ccy,
            reference_ccy=reference_ccy,
            freq=output_freq,
            is_log_returns=False,
            is_excess_returns=False,
        )
        nav_ref.name = f'cash_{local_ccy}_in_{reference_ccy}'
        return nav_ref

    def _compute_performance_of_local_ccy_asset_in_reference_ccy(self,
                                                                 asset_price_local_ccy: pd.Series,
                                                                 hedge_ratio: Union[float, pd.Series],
                                                                 local_ccy: str,
                                                                 reference_ccy: str,
                                                                 freq: str = 'ME',
                                                                 is_log_returns: bool = False,
                                                                 is_excess_returns: bool = False
                                                                 ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate hedged asset performance in reference currency.

        Returns NAV and returns adjusted for hedge ratio and forward costs.

        When ``is_excess_returns=True`` the returned series is the excess over
        the REFERENCE currency risk-free rate. The forward premium already
        embedded in the hedged leg (via ``compute_performance_of_local_ccy_asset_in_reference_ccy``)
        converts the fund's local-currency short rate into the reference
        short rate under CIP, so the correct rate to subtract for excess
        is the reference-currency rf, not the local one. Subtracting the
        local rf here is a bug: for a USD-denom fund viewed in CHF, that
        double-counts the USD short rate and produces a (r_USD - r_ref)
        downward bias in the reported excess return.
        """
        if is_excess_returns:
            # Excess is relative to the REFERENCE currency rf; CIP has already
            # converted the fund's native-rf leg via the forward premium.
            ref_rate = self.get_local_rate(freq=freq, local_ccy=reference_ccy, annualise=False)
        else:
            ref_rate = 0.0

        if local_ccy == reference_ccy:
            local_return = qis.to_returns(prices=asset_price_local_ccy, freq=freq, is_log_returns=is_log_returns,
                                           is_first_zero=True)

        else:
            local_to_reference_fx_rate = self.get_local_to_reference_fx_rate(local_ccy=local_ccy,
                                                                                      reference_ccy=reference_ccy)
            forward_rate_for_local_ccy = self.get_forward_rate_for_local_ccy(local_ccy=local_ccy,
                                                                             reference_ccy=reference_ccy, freq=freq,
                                                                             is_log_returns=is_log_returns)

            _, local_return = compute_performance_of_local_ccy_asset_in_reference_ccy(
                hedge_ratio=hedge_ratio,
                asset_price_local_ccy=asset_price_local_ccy,
                local_to_reference_fx_rate=local_to_reference_fx_rate,
                forward_rate_for_local_ccy=forward_rate_for_local_ccy,
                freq=freq,
                is_log_returns=is_log_returns)

        # The first valid period(s) are a SYNTHETIC zero — both branches force
        # the inception return to 0 (same-ccy via is_first_zero=True; cross-ccy
        # via hedged_return.iloc[0] = 0.0) to anchor the NAV at 1.0. There is
        # no real price observation there. In the total path (ref_rate == 0)
        # this synthetic 0 is harmless and is dropped downstream. In the
        # excess path, ``0 - rf = -rf`` turns the synthetic head into a
        # genuine non-zero value that then survives every NaN-dropping step,
        # so the excess panel gains a spurious leading -rf observation.
        #
        # The synthetic zero sits at the FIRST VALID index of local_return,
        # which is NOT necessarily index[0]: when the asset price has leading
        # NaNs (inception later than the panel/FX grid), to_returns places the
        # forced 0 at the asset's first real date, after a NaN gap. So locate
        # the leading run of synthetic zeros starting from the first non-NaN
        # observation, and mask it to NaN in BOTH return spaces so the total
        # and excess panels carry identical observation support. The NAV still
        # anchors at 1.0 via returns_to_nav (leading NaN treated as start).
        nz = local_return.to_numpy()
        start = 0
        while start < len(nz) and np.isnan(nz[start]):  # skip leading NaN gap
            start += 1
        k = start
        while k < len(nz) and nz[k] == 0.0:             # leading run of synthetic zeros
            k += 1

        # Align the reference rf onto the asset's own return grid before
        # subtracting. Multi-week resample grids (e.g. '2W-WED') are
        # phase-dependent on the series start date: a benchmark/asset whose
        # inception falls on the opposite bi-weekly parity from the rf series
        # produces a return index that shares ZERO dates with
        # get_local_rate(freq)'s index, so a raw `local_return - ref_rate`
        # aligns on nothing and nukes the entire excess panel to NaN. As-of
        # (ffill) reindex makes the subtraction phase-invariant; the rf is a
        # smooth per-period rate so carrying the nearest prior value is exact
        # to within a sub-period of rate drift.
        if isinstance(ref_rate, pd.Series):
            ref_rate = ref_rate.reindex(local_return.index, method='ffill')
        local_return = local_return - ref_rate
        if k > start:
            local_return.iloc[start:k] = np.nan
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
                                         is_excess_returns: bool = False
                                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a multi-asset panel to a reference-currency NAV + returns.

        Dispatches each asset through
        ``_compute_performance_of_local_ccy_asset_in_reference_ccy``
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
            is_excess_returns: If True, subtract reference-ccy rf per
                period. See ``compute_fx_adjusted_returns`` for the CIP
                rationale.

        Returns:
            Tuple of (NAV DataFrame, Returns DataFrame). The NAV frame
            is reconstructed from the returns via ``returns_to_nav``
            and is provided for plotting / reporting convenience; for
            pure-returns consumers (covariance, alpha) see
            ``compute_fx_adjusted_returns``.
        """
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

            fx_adjusted_navs[asset], fx_adjusted_returns[asset] = self._compute_performance_of_local_ccy_asset_in_reference_ccy(
                asset_price_local_ccy=asset_prices[asset],
                hedge_ratio = hedge_ratio,
                local_ccy=local_ccy,
                reference_ccy=reference_ccy,
                freq=freq,
                is_log_returns=is_log_returns,
            is_excess_returns=is_excess_returns)
        fx_adjusted_navs = pd.DataFrame.from_dict(fx_adjusted_navs, orient='columns')
        fx_adjusted_returns = pd.DataFrame.from_dict(fx_adjusted_returns, orient='columns')
        return fx_adjusted_navs, fx_adjusted_returns

    def compute_fx_adjusted_returns(self,
                                    prices: pd.DataFrame,
                                    hedge_ratios: pd.Series,
                                    local_ccys: pd.Series,
                                    reference_ccy: Union[str, Literal['CHF', 'EUR', 'GBP', 'USD']] = 'USD',
                                    freq: Union[str, pd.Series] = 'ME',
                                    is_log_returns: bool = True,
                                    is_excess_returns: bool = False
                                    ) -> Dict[str, pd.DataFrame]:
        """Compute per-period returns of a multi-asset panel in a reference ccy.

        Groups assets by ``freq`` when a per-asset Series is supplied,
        dispatches each group through ``compute_returns_in_reference_ccy``
        at that group's frequency, and returns a ``{freq: DataFrame}``
        mapping. This is the primary entry point consumed by the
        covariance estimator and the alpha aggregator in the PM and CMA
        pipelines.

        **Zero-return to NaN substitution** (line ``replace({0.0: np.nan})``):
        step-function PE series (prices constant between quarterly
        appraisal updates) produce zero returns at intra-period dates.
        Downstream β and correlation estimators would otherwise weight
        these structural zeros as informative observations, biasing
        factor loadings toward zero for illiquid assets. Replacing them
        with NaN causes the rolling estimators to skip those periods
        entirely — mathematically equivalent to estimating on the
        sparse observation frequency, which matches the actual
        information arrival pattern.

        Caveat: this is a global exact-zero replacement, so an
        organically zero return (rare but possible on liquid assets
        over a single period) will also be converted to NaN. Impact is
        negligible for rolling estimators with non-trivial window size.

        Args:
            prices: Native-ccy price panel (columns are assets).
            hedge_ratios: Per-asset hedge ratio in [0, 1].
            local_ccys: Per-asset currency of denomination.
            reference_ccy: Target currency for conversion.
            freq: Single frequency string or per-asset Series. A Series
                dispatches the panel into asset-frequency buckets.
            is_log_returns: If True, returns are log; otherwise simple.
            is_excess_returns: If True, subtract reference-ccy rf per
                period. Under CIP the hedged forward premium already
                converts the asset's native rf into the reference rf,
                so this subtraction gives the correct reference-ccy
                excess return.

        Returns:
            Mapping ``{freq_string: returns_dataframe}``. Single-freq
            input yields a single-key dict; per-asset freq input yields
            one key per unique frequency in the input.
        """
        if isinstance(freq, str):
            # Single frequency for all assets
            converted_prices, fx_adjusted_returns = self.compute_returns_in_reference_ccy(
                asset_prices=prices,
                hedge_ratios=hedge_ratios,
                local_ccys=local_ccys,
                reference_ccy=reference_ccy,
                freq=freq,
                is_log_returns=is_log_returns,
                is_excess_returns=is_excess_returns
            )
            return {freq: fx_adjusted_returns.replace({0.0: np.nan})}

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
                    is_excess_returns=is_excess_returns
                )
                results[frequency] = fx_adjusted_returns.replace({0.0: np.nan})
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
        local_rates_dict = {local_rate_id: self.get_local_rate(freq=freq, local_ccy=local_rate_id, annualise=annualise) for local_rate_id in local_rate_ids}
        local_rates = pd.DataFrame.from_dict({asset: local_rates_dict[local_ccys.loc[asset]] for asset in local_ccys.index}, orient='columns')
        return local_rates

    def compute_returns_adjusted_by_local_rate(self,
                                               asset_prices: pd.DataFrame,
                                               local_ccys: pd.Series,
                                               freq: str = 'ME',
                                               is_log_returns: bool = False
                                               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Native-currency returns and excess-over-local-rate returns for a panel.

        Computes each asset's return in its own local currency (no FX conversion)
        and the corresponding excess return over that currency's domestic short
        rate. The per-period rate grid is as-of (ffill) aligned to the return grid
        before subtracting, so multi-week resample phases cannot null the panel.

        Args:
            asset_prices: Native-currency price panel (columns are assets).
            local_ccys: Per-asset local currency (Series indexed by asset).
            freq: Resampling frequency for the returns and rates.
            is_log_returns: If True compute log returns, otherwise simple returns.

        Returns:
            Tuple ``(local_returns, excess_returns, local_rates)`` of DataFrames;
            ``local_rates`` holds the annualised per-asset short rates.
        """
        local_returns = qis.to_returns(prices=asset_prices, freq=freq, is_log_returns=is_log_returns)
        # Build DataFrame with local rates aligned to assets
        local_rates_dt = self.fetch_local_rates(local_ccys=local_ccys, freq=freq, annualise=False)
        local_rates = self.fetch_local_rates(local_ccys=local_ccys, freq=freq, annualise=True)
        # Align the rate grid onto the return grid before subtracting. Multi-week
        # resample grids (e.g. '2W-WED') are phase-dependent on the series start
        # date, so a raw subtraction can align on zero shared dates and null the
        # whole excess panel (same class of bug as the reference-ccy excess path).
        local_rates_dt = local_rates_dt.reindex(index=local_returns.index, method='ffill')
        excess_returns = local_returns - local_rates_dt
        return local_returns, excess_returns, local_rates


def load_fx_rates_data(local_path: str,
                       file_name: str = 'fx_hedging_data'
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load FX spot rates and domestic short rates from saved CSV files.

    ``usd_assets`` is not loaded here — it is only needed for examples and
    reporting; use ``qis.examples.market_data.load_usd_assets`` for that.

    Applies a one-time correction for historical CSVs that stored
    ``GBp`` as a 1:1 copy of ``GBP``. The correct relationship is
    ``GBp = 0.01 * GBP`` (1 GBP = 100 pence). If the loaded data is
    detected to carry the stale mapping, it is healed in-memory so
    callers always see the corrected series. Re-running
    ``create_fx_rates_data`` will persist the fix to disk.
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
