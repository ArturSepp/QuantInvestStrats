import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from bbg_fetch import fetch_field_timeseries_per_tickers

# tickers = {'CF1 Index': 'CAC', 'GX1 Index': 'DAX', 'IB1 Index': 'IBEX', 'ST1 Index': 'MIB', 'EO1 Index': 'AEX'}
tickers = {'CF1 Index': 'CAC', 'GX1 Index': 'DAX', 'IB1 Index': 'IBEX', 'ST1 Index': 'MIB'}
start_date = pd.Timestamp('1Jan1990')

futures_prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST',
                                                    start_date=start_date).ffill()
# use 3m rolling volumes
volumes = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='VOLUME',
                                             start_date=start_date).ffill().rolling(63).mean()
price_volumes = volumes.multiply(futures_prices).dropna(how='all')

# create volume-weighted portfolios
futures_weights = qis.df_to_weight_allocation_sum1(df=price_volumes.asfreq('QE').ffill())

# plot futures price data
kwargs = dict(x_date_freq='YE')
with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)
    qis.plot_prices_with_dd(prices=futures_prices, axs=axs[:, 0], **kwargs)
    qis.plot_time_series(df=volumes, title='Contact volumes', ax=axs[0, 1], **kwargs)
    qis.plot_time_series(df=price_volumes, title='Value volumes', ax=axs[1, 1], **kwargs)


# add long cash SPTR
sptr_prices = fetch_field_timeseries_per_tickers(tickers={'SPTR Index': 'SPTR'}, freq='B', field='PX_LAST',
                                                 start_date=start_date).ffill()
long_sptr_short_futures_weights = pd.concat([pd.Series(1.0, index=futures_weights.index, name='SPTR'),
                                             -1.0*futures_weights], axis=1)
print(long_sptr_short_futures_weights)

prices = pd.concat([sptr_prices, futures_prices], axis=1)

# backtest total return portfolio
time_period = qis.TimePeriod('31Dec1990', '27Sep2024')
portfolio_data = qis.backtest_model_portfolio(prices=time_period.locate(prices),
                                              weights=time_period.locate(long_sptr_short_futures_weights),
                                              rebalancing_costs=0.0005,
                                              weight_implementation_lag=1,
                                              ticker='Long SPTR / Short EU futures')

# create factcheet
figs = qis.generate_strategy_factsheet(portfolio_data=portfolio_data,
                                       benchmark_prices=sptr_prices.iloc[:, 0].rename('SPTR long').to_frame(),
                                       add_current_position_var_risk_sheet=True,
                                       time_period=time_period,
                                       **qis.fetch_default_report_kwargs(time_period=time_period))
qis.save_figs_to_pdf(figs=figs,
                     file_name=f"long_sptr_short_eu_strategy_factsheet",
                     local_path=qis.local_path.get_output_path())

plt.show()
