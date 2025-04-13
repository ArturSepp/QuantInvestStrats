"""
signal data outpout for portfolio data reporting
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResults as RegModel
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from qis.utils.dates import TimePeriod
from qis.utils.df_str import date_to_str
from qis.utils.df_groups import get_group_dict
from qis.utils.df_melt import melt_df_by_columns
from qis.utils.ols import fit_ols
from qis.utils.df_freq import df_resample_at_int_index, df_resample_at_freq


@dataclass
class StrategySignalData:
    """
    data class instance applied for output of strategy backtest data
    """
    log_returns: pd.DataFrame = None
    ra_carry: pd.DataFrame = None  # risk-adjusted carry
    momentum: pd.DataFrame = None
    signal: pd.DataFrame = None  # signal output
    instrument_vols: pd.DataFrame = None  # instrument vols
    instrument_target_vols: pd.DataFrame = None  # target vols for portfolio allocation
    instrument_target_signal_vol_weights: pd.DataFrame = None  # target vols * signal
    instrument_portfolio_leverages: pd.DataFrame = None  # = portfolio weight / instrument_target_signal_vol_weights
    weights: pd.DataFrame = None  # final weights
    kwargs: Dict[str, pd.DataFrame] = None  # any other outputs

    def locate_period(self, time_period: TimePeriod) -> StrategySignalData:
        # nb: does not work for returns
        data_dict = asdict(self)
        for key, df in data_dict.items():
            if df is not None:
                data_dict[key] = time_period.locate(df)
        return StrategySignalData(**data_dict)

    def rename_data(self, names_map: Dict[str, str]) -> StrategySignalData:
        data_dict = asdict(self)
        for key, df in data_dict.items():
            if df is not None:
                data_dict[key] = df.rename(names_map, axis=1)
        return StrategySignalData(**data_dict)

    def get_current_signal_by_groups(self, group_data: pd.Series,
                                     group_order: List[str] = None
                                     ) -> Dict[str, pd.DataFrame]:
        group_dict = get_group_dict(group_data=group_data,
                                        group_order=group_order,
                                        total_column=None)
        group_signals = {}
        agg_by_group = {}
        last_date = date_to_str(self.signal.index[-21])
        current_date = date_to_str(self.signal.index[-1])
        last_signals = self.signal.iloc[-21, :]
        current_signals = self.signal.iloc[-1, :]
        for group, tickers in group_dict.items():
            last_signals_ = last_signals[tickers]
            current_signals_ = current_signals[tickers]
            group_signals[group] = pd.concat([last_signals_.rename(last_date),
                                              current_signals_.rename(current_date)
                                              ], axis=1)

            agg_by_group[group] = pd.Series({last_date: np.nanmean(last_signals_),
                                             current_date: np.nanmean(current_signals_)})
        agg_by_group = {'Total by groups': pd.DataFrame.from_dict(agg_by_group, orient='index')}
        agg_by_group.update(group_signals)
        return agg_by_group

    def asdiff(self, tickers: List[str] = None,
               freq: str = None,
               sample_size: Optional[int] = 21,  # can use rolling instead for freq
               time_period: TimePeriod = None
               ) -> StrategySignalData:

        if tickers is None:
            tickers = self.log_returns.columns.to_list()
        # nb: does not work for returns
        if time_period is not None:
            ssd = self.locate_period(time_period=time_period)
        else:
            ssd = self

        data_dict = asdict(ssd)
        if sample_size is not None:
            for key, df in data_dict.items():
                if df is not None:
                    data_dict[key] = df_resample_at_int_index(df=df[tickers], func=None, sample_size=sample_size).diff()
        else:
            for key, df in data_dict.items():
                if df is not None:
                    data_dict[key] = df_resample_at_freq(df=df[tickers], freq=freq, include_end_date=True).diff()
        return StrategySignalData(**data_dict)

    def estimate_signal_changes_joint(self,
                                      tickers: List[str] = None,
                                      freq: Optional[str] = None,
                                      sample_size: Optional[int] = 21,  # can use rolling instead for freq
                                      time_period: TimePeriod = None
                                      ) -> Tuple[pd.DataFrame, RegModel, TimePeriod]:
        if tickers is None:
            tickers = self.log_returns.columns.to_list()
        ssd = self.asdiff(tickers=tickers, sample_size=sample_size, freq=freq, time_period=time_period)
        y_var_name = 'weight_change'
        y = melt_df_by_columns(ssd.weights.iloc[:-1, :], y_var_name=y_var_name)[y_var_name]
        x_var_name1 = 'momentum_change'
        x1 = melt_df_by_columns(ssd.momentum.iloc[:-1, :], y_var_name=x_var_name1)[x_var_name1]
        x_var_name2 = 'target_vol_change'
        x2 = melt_df_by_columns(ssd.instrument_target_vols.iloc[:-1, :], y_var_name=x_var_name2)[x_var_name2]
        x_var_name3 = 'port_leverage_change'
        x3 = melt_df_by_columns(ssd.instrument_portfolio_leverages.iloc[:-1, :], y_var_name=x_var_name3)[x_var_name3]
        if self.ra_carry is not None:
            x_var_name0 = 'carry_change'
            x0 = melt_df_by_columns(ssd.ra_carry.iloc[:-1, :], y_var_name=x_var_name0)[x_var_name0]
            x = pd.concat([x0, x1, x2, x3], axis=1).dropna()
        else:
            x = pd.concat([x1, x2, x3], axis=1).dropna()
        x_names = x.columns.to_list()
        y = y.reindex(index=x.index)

        # keep last obs for prediction
        fitted_model = fit_ols(x=x.to_numpy(), y=y.to_numpy(), order=1, fit_intercept=False)
        actual_change = ssd.weights.iloc[-1, :]
        predictions = {}
        for ticker in tickers:
            if self.ra_carry is not None:
                x_ts = np.array([ssd.ra_carry[ticker].iloc[-1],
                                 ssd.momentum[ticker].iloc[-1],
                                 ssd.instrument_target_vols[ticker].iloc[-1],
                                 ssd.instrument_portfolio_leverages[ticker].iloc[-1]])
            else:
                x_ts = np.array([ssd.momentum[ticker].iloc[-1],
                                 ssd.instrument_target_vols[ticker].iloc[-1],
                                 ssd.instrument_portfolio_leverages[ticker].iloc[-1]])
            pred_t = {}
            total_pred = 0.0
            for idx, x_t in enumerate(x_ts):
                pred_x = fitted_model.params[idx] * x_t
                total_pred += pred_x
                pred_t[x_names[idx]] = pred_x
            pred_t['predicted'] = total_pred
            pred_t['actual'] = actual_change[ticker]
            pred_t['residual'] = actual_change[ticker] - total_pred
            pred_t['residual %'] = total_pred / actual_change[ticker]
            pred_t['r2'] = fitted_model.rsquared
            predictions[ticker] = pd.Series(pred_t)

        predictions = pd.DataFrame.from_dict(predictions, orient='index')
        prediction_period = TimePeriod(start=ssd.weights.index[-2], end=ssd.weights.index[-1])

        return predictions, fitted_model, prediction_period

    def estimate_signal_changes_by_groups(self,
                                          group_data: pd.Series, group_order: List[str] = None,
                                          freq: Optional[str] = None,
                                          sample_size: Optional[int] = 21,  # can use rolling instead for freq
                                          time_period: TimePeriod = None
                                          ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, RegModel], TimePeriod]:
        """
        estimate weight change for groups
        """
        group_dict = get_group_dict(group_data=group_data,
                                    group_order=group_order,
                                    total_column=None)
        predictions = {}
        fitted_models = {}
        prediction_period = None
        for group, tickers in group_dict.items():
            prediction, fitted_model, prediction_period = self.estimate_signal_changes_joint(
                tickers=tickers, freq=freq,
                sample_size=sample_size,
                time_period=time_period)
            predictions[group] = prediction
            fitted_models[group] = fitted_model
        return predictions, fitted_models, prediction_period

    def estimate_signal_changes_individual(self,
                                           tickers: List[str] = None,
                                           freq: Optional[str] = None,
                                           sample_size: Optional[int] = 21,  # can use rolling instead for freq
                                           time_period: TimePeriod = None
                                           ) -> Tuple[pd.DataFrame, Dict[str, RegModel]]:
        if tickers is None:
            tickers = self.log_returns.columns.to_list()
        ssd = self.asdiff(tickers=tickers, sample_size=sample_size, freq=freq, time_period=time_period)
        y_var_name = 'weight_change'
        y = ssd.weights
        x_var_name1 = 'momentum_change'
        x1 = ssd.momentum
        x_var_name2 = 'target_vol_change'
        x2 = ssd.instrument_target_vols
        x_var_name3 = 'port_leverage_change'
        x3 = ssd.instrument_portfolio_leverages
        if self.ra_carry is not None:
            x_var_name0 = 'carry_change'
            x0 = ssd.ra_carry
            x_names = [x_var_name0, x_var_name1, x_var_name2, x_var_name3]
        else:
            x_names = [x_var_name1, x_var_name2, x_var_name3]

        predictions = {}
        fitted_models = {}
        for ticker in tickers:
            if self.ra_carry is not None:
                x = pd.concat([x0[ticker].rename(x_var_name0),
                               x1[ticker].rename(x_var_name1),
                               x2[ticker].rename(x_var_name2),
                               x3[ticker].rename(x_var_name3)], axis=1)
            else:
                x = pd.concat([x1[ticker].rename(x_var_name1),
                               x2[ticker].rename(x_var_name2),
                               x3[ticker].rename(x_var_name3)], axis=1)

            # keep last obs for prediction
            fitted_model = fit_ols(x=x.iloc[:-1, :].to_numpy(), y=y[ticker].iloc[:-1].to_numpy(), order=1, fit_intercept=False)
            actual_change = y[ticker].iloc[-1]
            x_ts = x.iloc[-1, :].to_numpy()
            pred_t = {}
            total_pred = 0.0
            for idx, x_t in enumerate(x_ts):
                pred_x = fitted_model.params[idx] * x_t
                total_pred += pred_x
                pred_t[x_names[idx]] = pred_x
            pred_t['predicted'] = total_pred
            pred_t['actual'] = actual_change
            pred_t['residual'] = actual_change - total_pred
            pred_t['residual %'] = total_pred / actual_change
            pred_t['r2'] = fitted_model.rsquared
            predictions[ticker] = pd.Series(pred_t)
            fitted_models[ticker] = fitted_model

        predictions = pd.DataFrame.from_dict(predictions, orient='index')

        return predictions, fitted_models
