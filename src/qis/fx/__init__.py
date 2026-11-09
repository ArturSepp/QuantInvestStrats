"""Currency data, cash accrual, typed returns, hedging analytics and reports."""
from qis.fx.fx_rates_data import FxRatesData, load_fx_rates_data
from qis.fx.fx_hedging import (
    compute_cash_fx_adjusted_returns,
    compute_fx_optimal_hedge,
    compute_fx_vol_beta,
    compute_futures_fx_adjusted_returns,
    compute_local_and_fx_return,
    compute_performance_of_local_ccy_asset_in_reference_ccy,
    get_aligned_fx_spots,
)
from qis.fx.reports.fx_hedging_report import (
    compute_multi_asset_fx_hedging,
    plot_multi_asset_fx_hedging_report,
    run_asset_fx_hedging_report,
)

from qis.fx.fx_returns import (
    FxReturnType, FxReturnCoordinate, FxCashAccrual, FxCashPolicy, FxIndexNormalization,
    FxAssetReturnsRequest, FxCurrencyReturnsRequest, FxReturnsData, compute_fx_returns,
)

from qis.fx.cash_index import compute_monthly_cash_index
