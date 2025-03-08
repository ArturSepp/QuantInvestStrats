
from qis.portfolio.portfolio_data import (PortfolioData,
                                          PortfolioInput,
                                          AttributionMetric,
                                          SnapshotPeriod,
                                          StrategySignalData)

from qis.portfolio.multi_portfolio_data import MultiPortfolioData

from qis.portfolio.ewm_portfolio_risk import (limit_weights_to_max_var_limit,
                                              compute_portfolio_var_np,
                                              compute_portfolio_vol,
                                              compute_portfolio_correlated_var_by_groups,
                                              compute_portfolio_independent_var_by_ac,
                                              compute_portfolio_risk_contributions)

from qis.portfolio.backtester import (backtest_model_portfolio, backtest_rebalanced_portfolio)

from qis.portfolio.reports.config import (FactsheetConfig,
                                          FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD,
                                          FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD,
                                          FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD,
                                          FACTSHEET_CONFIG_MONTHLY_DATA_SHORT_PERIOD,
                                          fetch_factsheet_config_kwargs,
                                          fetch_default_perf_params,
                                          fetch_default_report_kwargs,
                                          ReportingFrequency)

from qis.portfolio.reports.brinson_attribution import (compute_brinson_attribution_table,
                                                       plot_brinson_totals_table,
                                                       plot_brinson_attribution_table)

from qis.portfolio.reports.multi_assets_factsheet import (MultiAssetsReport, generate_multi_asset_factsheet)

from qis.portfolio.reports.strategy_factsheet import generate_strategy_factsheet

from qis.portfolio.reports.strategy_benchmark_factsheet import (generate_strategy_benchmark_factsheet_plt,
                                                                generate_strategy_benchmark_active_perf_plt,
                                                                plot_exposures_strategy_vs_benchmark_stack)

from qis.portfolio.reports.multi_strategy_factsheet import generate_multi_portfolio_factsheet

from qis.portfolio.reports.strategy_signal_factsheet import (generate_weight_change_report,
                                                             generate_current_signal_report,
                                                             generate_strategy_signal_factsheet_by_instrument)

# disable requirements for pyblogs
# from qis.portfolio.reports.multi_strategy_factseet_pybloqs import generate_multi_portfolio_factsheet_with_pyblogs
