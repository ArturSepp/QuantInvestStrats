"""Market-data containers: tradable factor prices and FX rates, and the FX hedging analytics."""
from qis.market_data.factors_data import FactorsData
from qis.market_data.fx_rates_data import FxRatesData, load_fx_rates_data
from qis.market_data.fx_hedging import (
    compute_cash_fx_adjusted_returns,
    compute_fx_optimal_hedge,
    compute_fx_vol_beta,
    compute_futures_fx_adjusted_returns,
    compute_local_and_fx_return,
    compute_performance_of_local_ccy_asset_in_reference_ccy,
    get_aligned_fx_spots,
)
from qis.market_data.reports.fx_hedging_report import (
    compute_multi_asset_fx_hedging,
    plot_multi_asset_fx_hedging_report,
    run_asset_fx_hedging_report,
)
