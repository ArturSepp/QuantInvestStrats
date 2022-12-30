# built in
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Tuple, List, Dict
from enum import Enum

# qis
import qis.utils.dates as da
import qis.perfstats.returns as ret
import qis.plots.time_series as pts

# models
import qis.models.linear.ewm as ewm
import qis.models.stats.gaussian_mixture as gm
from qis.models.linear.corr_cov_matrix import compute_masked_covar_corr, matrix_regularization

# portfolio
import qis.portfolio.optimization.opt_solvers as ops
import qis.portfolio.optimization.qp_solvers as qup
from qis.portfolio.optimization.qp_solvers import PortfolioObjective, max_portfolio_sharpe_qp
import qis.portfolio.backtester as bp
from qis.portfolio.portfolio_data import PortfolioData


def solve_optimal_weights_ewm_covar(prices: pd.DataFrame,
                                     portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VAR,
                                     fixed_weights: Dict[str, float] = None,  # fixed for principal portfolios
                                     weight_mins: np.ndarray = None,
                                     weight_maxs: np.ndarray = None,
                                     target_vol: float = None,
                                     rebalancing_freq: str = 'Q',
                                     ewm_lambda: float = 0.97,
                                     is_regularize: bool = False,
                                     returns_freq: str = None,
                                     is_log_returns: bool = True,
                                     budget: np.ndarray = None
                                     ) -> pd.DataFrame:

    returns = ret.to_returns(prices=prices,
                             is_log_returns=is_log_returns,
                             freq=returns_freq,
                             ffill_nans=True,
                             include_end_date=False)

    # drift adjusted returns
    returns_np = returns.to_numpy()
    x = returns_np # - ewm.ewm_nd(a=returns_np, ewm_lambda=ewm_lambda)
    covar_tensor_txy = ewm.compute_ewm_covar_tensor(a=x, ewm_lambda=ewm_lambda)

    rebalancing_schedule = da.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)
    columns = prices.columns.to_list()
    n = len(columns)

    if fixed_weights is not None:
        if weight_mins is None:
            weight_mins = np.zeros(n)
        if weight_maxs is None:
            weight_maxs = np.ones(n)
        for asset, weight in fixed_weights.items():
            idx = columns.index(asset)
            weight_mins[idx], weight_maxs[idx] = weight, weight

    an_factor = da.infer_an_from_data(data=returns)
    weights = {}
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value:
            covar = an_factor*covar_tensor_txy[idx]
            if is_regularize:
                covar = matrix_regularization(covar=covar)

            if portfolio_objective == PortfolioObjective.EQUAL_RISK_CONTRIBUTION:
                if target_vol is None:
                    weights[date] = ops.solve_equal_risk_contribution(covar=covar,
                                                                     budget=budget,
                                                                     weight_mins=weight_mins,
                                                                     weight_maxs=weight_maxs)
                else:
                    weights[date] = ops.solve_risk_parity_constr_vol(covar=covar,
                                                                    target_vol=target_vol)

            elif portfolio_objective == PortfolioObjective.MAX_DIVERSIFICATION:
                weights[date] = ops.solve_max_diversification(covar=covar,
                                                             weight_mins=weight_mins,
                                                             weight_maxs=weight_maxs)

            elif portfolio_objective == PortfolioObjective.RISK_PARITY_ALT:
                weights[date] = ops.solve_risk_parity_alt(covar=covar)

            else:
                weights[date] = qup.maximize_portfolio_objective_qp(portfolio_objective=portfolio_objective,
                                                                    covar=covar,
                                                                    is_gross_notional_one=True,
                                                                    is_long_only=True,
                                                                    weight_mins=weight_mins,
                                                                    weight_maxs=weight_maxs)
    weights = pd.DataFrame.from_dict(weights, orient='index', columns=returns.columns)
    return weights


def run_rolling_mixure_portfolios(prices: pd.DataFrame,
                                  recalib_freq: str = 'A',
                                  roll_window: int = 5,
                                  returns_freq: str = 'M',
                                  is_log_returns: bool = True,
                                  carra: float = 0.5,
                                  n_components: int = 3,
                                  weight_mins: np.ndarray = None,
                                  weight_maxs: np.ndarray = None,
                                  ticker: str = None
                                  ) -> PortfolioData:

    rets = ret.to_returns(prices=prices, is_log_returns=is_log_returns, drop_first=True, freq=returns_freq)

    dates_schedule = da.generate_dates_schedule(time_period=da.get_time_period(df=rets),
                                                freq=recalib_freq,
                                                include_start_date=True,
                                                include_end_date=False)
    _, scaler = da.get_period_days(freq=returns_freq)
    weights = {}
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = da.TimePeriod(dates_schedule[idx - roll_window+1], end)
            # period.print()
            rets_ = period.locate(rets).to_numpy()
            params = gm.fit_gaussian_mixture(x=rets_, n_components=n_components, scaler=scaler)
            # print(params)
            weights[end] = ops.solve_cara_mixture(means=params.means,
                                                  covars=params.covars,
                                                  probs=params.probs,
                                                  carra=carra,
                                                  weight_mins=weight_mins,
                                                  weight_maxs=weight_maxs)

    weights = pd.DataFrame.from_dict(weights, orient='index', columns=prices.columns)
    portfolio_out = bp.backtest_model_portfolio(prices=prices,
                                                weights=weights,
                                                is_rebalanced_at_first_date=True,
                                                ticker=ticker,
                                                is_output_portfolio_data=True)
    return portfolio_out


def run_rolling_erc_portfolios(prices: pd.DataFrame,
                               weight_mins: np.ndarray = None,
                               weight_maxs: np.ndarray = None,
                               time_period: da.TimePeriod = None,
                               portfolio_objective: PortfolioObjective = PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
                               recalib_freq: str = 'Q',
                               ewm_lambda: float = 0.926,  #0.926,
                               returns_freq: str = 'W-WED',
                               budget: np.ndarray = None,
                               ticker: str = None
                               ) -> PortfolioData:

    weights = solve_optimal_weights_ewm_covar(prices=prices,
                                              portfolio_objective=portfolio_objective,
                                              weight_mins=weight_mins,
                                              weight_maxs=weight_maxs,
                                              rebalancing_freq=recalib_freq,
                                              returns_freq=returns_freq,
                                              ewm_lambda=ewm_lambda,
                                              budget=budget)
    if time_period is not None:
        prices = time_period.locate(prices)
        weights = time_period.locate(weights)
    portfolio_out = bp.backtest_model_portfolio(prices=prices.loc[weights.index[0]:, :],
                                                weights=weights,
                                                is_rebalanced_at_first_date=True,
                                                ticker=ticker,
                                                is_output_portfolio_data=True)
    return portfolio_out


def estimate_rolling_means_covar(prices: pd.DataFrame,
                                 returns_freq: str = 'M',
                                 recalib_freq: str = 'A',
                                 roll_window: int = 5,
                                 is_log_returns: bool = True,
                                 is_annualize: bool = True,
                                 is_regularize: bool = True,
                                 is_ewm_covar: bool = False
                                 ) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    rets = ret.to_returns(prices=prices, is_log_returns=is_log_returns, drop_first=True, freq=returns_freq)

    dates_schedule = da.generate_dates_schedule(time_period=da.get_time_period(df=rets),
                                                freq=recalib_freq,
                                                include_start_date=True,
                                                include_end_date=False)

    if is_annualize:
        _, scaler = da.get_period_days(freq=returns_freq)
    else:
        scaler = 1.0
    means = {}
    covars = []
    covar0 = np.zeros((len(prices.columns), len(prices.columns)))
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = da.TimePeriod(dates_schedule[idx - roll_window+1], end)
            # period.print()
            rets_ = period.locate(rets).to_numpy()
            means[end] = scaler*np.nanmean(rets_, axis=0)
            if is_ewm_covar:
                covar = ewm.compute_ewm_covar(a=rets_,
                                                ewm_lambda=0.846,
                                                covar0=covar0)
                covar0 = covar
            else:
                covar = compute_masked_covar_corr(returns=rets_, bias=True)

            if is_regularize:
                covar = matrix_regularization(covar=covar, cut=1e-5)

            covars.append(scaler * covar)
    means = pd.DataFrame.from_dict(means, orient="index")

    return means, covars


def run_rolling_mv_portfolios(prices: pd.DataFrame,
                              returns_freq: str = 'M',
                              recalib_freq: str = 'A',
                              roll_window: int = 5,
                              carra: float = 0.5,
                              time_period: da.TimePeriod = None,
                              is_log_returns: bool = True,
                              weight_mins: np.ndarray = None,
                              weight_maxs: np.ndarray = None,
                              ticker: str = None,
                              is_print_log: bool = True
                              ) -> PortfolioData:

    means, covars = estimate_rolling_means_covar(prices=prices,
                                                 returns_freq=returns_freq,
                                                 recalib_freq=recalib_freq,
                                                 roll_window=roll_window,
                                                 is_annualize=True,
                                                 is_log_returns=is_log_returns)
    weights = {}
    for index, covar in zip(means.index, covars):
        if np.isclose(carra, 0.0):
            weights[index] = max_portfolio_sharpe_qp(means=means.loc[index, :].to_numpy(),
                                                     covar=covar,
                                                     weight_mins=weight_mins,
                                                     weight_maxs=weight_maxs,
                                                     is_gross_notional_one=True,
                                                     is_print_log=is_print_log)
        else:
            weights[index] = ops.solve_cara(means=means.loc[index, :].to_numpy(), covar=covar,
                                            weight_mins=weight_mins,
                                            weight_maxs=weight_maxs,
                                            carra=carra)

    weights = pd.DataFrame.from_dict(weights, orient='index', columns=prices.columns)
    if time_period is not None:
        prices = time_period.locate(prices)
        weights = time_period.locate(weights)
    portfolio_out = bp.backtest_model_portfolio(prices=prices,
                                                weights=weights,
                                                is_rebalanced_at_first_date=True,
                                                ticker=ticker,
                                                is_output_portfolio_data=True)
    return portfolio_out


def estimate_rolling_mixture1(prices: pd.DataFrame,
                              returns_freq: str = 'M',
                              recalib_freq: str = 'A',
                              roll_window: int = 6,
                              n_components: int = 2,
                              is_log_returns: bool = True,
                              is_annualize: bool = True
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if len(prices.columns) > 1:
        raise ValueError
    rets = ret.to_returns(prices=prices, is_log_returns=is_log_returns, drop_first=True, freq=returns_freq)

    dates_schedule = da.generate_dates_schedule(time_period=da.get_time_period(df=rets),
                                                freq=recalib_freq,
                                                include_start_date=True,
                                                include_end_date=True)
    if is_annualize:
        _, scaler = da.get_period_days(freq=returns_freq)
    else:
        scaler = 1.0

    means, sigmas, probs = [], [], []
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = da.TimePeriod(dates_schedule[idx - roll_window+1], end)
            # period.print()
            rets_ = period.locate(rets).to_numpy()
            params = gm.fit_gaussian_mixture(x=rets_, n_components=n_components, scaler=scaler)
            mean = np.stack(params.means, axis=0).T[0]
            std = np.sqrt(np.array([params.covars[0][0], params.covars[1][0]]))
            prob = params.probs
            ranks = mean.argsort().argsort()
            means.append(pd.DataFrame(mean[ranks].reshape(1, -1), index=[end]))
            sigmas.append(pd.DataFrame(std[ranks].reshape(1, -1), index=[end]))
            probs.append(pd.DataFrame(prob[ranks].reshape(1, -1), index=[end]))

    means = pd.concat(means)
    sigmas = pd.concat(sigmas)
    probs = pd.concat(probs)

    return means, sigmas, probs


class UnitTests(Enum):
    MIN_VAR = 1
    MIN_VAR_OVERLAY = 2
    RISK_PARITY = 3
    ROLLING_MEANS_COVAR = 4
    ROLLING_PORTFOLIOS = 5
    ROLLING_MIXTURES = 6
    MIXTURE_PORTFOLIOS = 7


def run_unit_test(unit_test: UnitTests):

    # data
    from qis.data.yf_data import load_etf_data
    prices = load_etf_data()
    import qis.plots.stackplot as pst

    kwargs = dict(is_add_mean_levels=True,
                  is_yaxis_limit_01=True,
                  baseline='zero',
                  bbox_to_anchor=(0.4, 1.1),
                  legend_line_type=pst.LegendLineType.AVG_STD_LAST,
                  ncol=len(prices.columns)//3,
                  var_format='{:.0%}')

    if unit_test == UnitTests.MIN_VAR:
        weights = solve_optimal_weights_ewm_covar(prices=prices,
                                                  portfolio_objective=PortfolioObjective.MIN_VAR)
        pst.stackplot_timeseries(df=weights, **kwargs)

    elif unit_test == UnitTests.MIN_VAR_OVERLAY:
        fixed_weights = {'SPY': 0.6}
        weights = solve_optimal_weights_ewm_covar(prices=prices, fixed_weights=fixed_weights)
        pst.stackplot_timeseries(df=weights, **kwargs)

    elif unit_test == UnitTests.RISK_PARITY:
        weights = solve_optimal_weights_ewm_covar(prices=prices,
                                                  portfolio_objective=PortfolioObjective.EQUAL_RISK_CONTRIBUTION)
        pst.stackplot_timeseries(df=weights, **kwargs)

    elif unit_test == UnitTests.ROLLING_MEANS_COVAR:
        prices = prices[['SPY', 'TLT']].dropna()

        means, covars = estimate_rolling_means_covar(prices=prices, recalib_freq='A', roll_window=5)
        #  = estimate_rolling_data(prices=prices, recalib_freq='M', roll_window=60)

        vols = {}
        covs = {}
        for index, covar in zip(means.index, covars):
            vols[index] = pd.Series(np.sqrt(np.diag(covar)))
            covs[index] = pd.Series(np.extract(1 - np.eye(2), covar))
        vols = pd.DataFrame.from_dict(vols, orient='index')
        covs = pd.DataFrame.from_dict(covs, orient='index')
        print(vols)
        print(covs)

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(3, 1, figsize=(7, 12))
            pts.plot_time_series(df=means,
                                 var_format='{:.0%}',
                                 trend_line=pts.TrendLine.AVERAGE,
                                 legend_line_type=pts.LegendLineType.FIRST_AVG_LAST,
                                 ax=axs[0])
            pts.plot_time_series(df=vols,
                                 var_format='{:.0%}',
                                 trend_line=pts.TrendLine.AVERAGE,
                                 legend_line_type=pts.LegendLineType.FIRST_AVG_LAST,
                                 ax=axs[1])
            pts.plot_time_series(df=covs,
                                 var_format='{:.0%}',
                                 trend_line=pts.TrendLine.AVERAGE,
                                 legend_line_type=pts.LegendLineType.FIRST_AVG_LAST,
                                 ax=axs[2])

    elif unit_test == UnitTests.ROLLING_PORTFOLIOS:
        prices = prices.dropna()

        port_data = run_rolling_mv_portfolios(prices=prices, carra=0.0, recalib_freq='A', roll_window=5)
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))
            pts.plot_time_series(df=port_data.weights,
                                 var_format='{:.0%}',
                                 legend_line_type=pts.LegendLineType.FIRST_AVG_LAST,
                                 ax=ax)

    elif unit_test == UnitTests.ROLLING_MIXTURES:
        prices = prices['SPY'].dropna()
        means, sigmas, probs = estimate_rolling_mixture1(prices=prices)
        print(means)

    elif unit_test == UnitTests.MIXTURE_PORTFOLIOS:
        prices = prices.dropna()
        port_data = run_rolling_mixure_portfolios(prices=prices, recalib_freq='Q', n_components=4, roll_window=4 * 5, carra=0.5)
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))
            pts.plot_time_series(df=port_data.weights,
                                 var_format='{:.0%}',
                                 legend_line_type=pts.LegendLineType.FIRST_AVG_LAST,
                                 ax=ax)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.RISK_PARITY

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
