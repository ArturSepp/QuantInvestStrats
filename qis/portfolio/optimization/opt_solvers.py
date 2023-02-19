"""
Optimizers"""

from __future__ import division
import numpy as np
from scipy.optimize import minimize
from typing import List
from enum import Enum

import qis.utils.dates as da


def calculate_portfolio_var(w: np.ndarray, covar: np.ndarray) -> float:
    return w @ covar @ w.T


def calculate_risk_contribution(w: np.ndarray, covar: np.ndarray) -> np.ndarray:
    portfolio_vol = np.sqrt(calculate_portfolio_var(w, covar))
    marginal_risk_contribution = covar @ w.T
    rc = np.multiply(marginal_risk_contribution, w) / portfolio_vol
    return rc


def calculate_diversification_ratio(w: np.ndarray, covar: np.ndarray) -> float:
    avg_weighted_vol = np.sqrt(np.diag(covar)) @ w.T
    portfolio_vol = np.sqrt(calculate_portfolio_var(w, covar))
    diversification_ratio = avg_weighted_vol/portfolio_vol
    return diversification_ratio


def max_diversification_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    covar = pars[0]
    return -calculate_diversification_ratio(w=w, covar=covar)


def carra_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    means, covar, carra = pars[0], pars[1], pars[2]
    v = means.T @ w - 0.5*carra*w.T @ covar @ w
    return -v


def carra_objective_exp(w: np.ndarray, pars: List[np.ndarray]) -> float:
    means, covar, carra = pars[0], pars[1], pars[2]
    v = np.exp(-carra*means.T @ w + 0.5*carra*carra*w.T @ covar @ w)
    return v


def carra_objective_mixture(w: np.ndarray, pars: List[np.ndarray]) -> float:
    means, covars, probs, carra = pars[0], pars[1], pars[2], pars[3]
    v = 0.0
    for idx, prob in enumerate(probs):
        v = v + prob*np.exp(-carra*means[idx].T @ w + 0.5*carra*carra*w.T @ covars[idx] @ w)
    return v


def risk_budget_objective(x, pars):
    covar, budget = pars[0], pars[1]
    asset_rc = calculate_risk_contribution(x, covar)
    sig_p = np.sqrt(calculate_portfolio_var(x, covar))
    if budget is not None:
        risk_target = np.where(np.isnan(budget), asset_rc, np.multiply(sig_p, budget))  # budget can be nan f
    else:
        risk_target = np.ones_like(asset_rc) / asset_rc.shape[0]
    sse = np.nansum(np.square(asset_rc - risk_target))
    return sse


def risk_budget_objective_mod(x, pars):
    covar, budget = pars[0], pars[1]
    sig_p = np.sqrt(calculate_portfolio_var(x, covar))  # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p, budget))
    asset_RC = calculate_risk_contribution(x, covar)
    sse = np.sum(np.square(asset_RC[:-1] - risk_target.T[:-1] - np.mean(asset_RC[:-1] - risk_target.T[:-1])))
    return sse


def total_weight_constraint(x, total: float = 1.0):
    return total - np.sum(x)


def sum_of_log_weight_constraint(x, risk_budget):
    return np.log(x).dot(risk_budget)


def long_only_constraint(x):
    return x


def portfolio_volatility_min(x, covar, target_vol, freq_vol, af: float = 12.0):
    vol_dt = np.sqrt(af)
    return vol_dt * np.sqrt(calculate_portfolio_var(x, covar)) - (target_vol - 0.001)


def portfolio_volatility_max(x, V, target_vol, freq_vol, af: float = 12.0):
    vol_dt = np.sqrt(af)
    return - (vol_dt * np.sqrt(calculate_portfolio_var(x, V)) - (target_vol + 0.001))


def solve_equal_risk_contribution(covar: np.ndarray,
                                  budget: np.ndarray = None,
                                  weight_mins: np.ndarray = None,
                                  weight_maxs: np.ndarray = None,
                                  is_gross_notional_one: bool = True,
                                  disp: bool = False
                                  ) -> np.ndarray:
    n = covar.shape[0]
    x0 = np.ones(n) / n

    cons = [{'type': 'ineq', 'fun': long_only_constraint}]
    if is_gross_notional_one:
        cons.append({'type': 'eq', 'fun': total_weight_constraint})
    if weight_mins is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: x - weight_mins})
    if weight_maxs is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: weight_maxs - x})

    res = minimize(risk_budget_objective, x0, args=[covar, budget], method='SLSQP', constraints=cons,
                   options={'disp': disp, 'ftol': 1e-18, 'maxiter': 200})
    w_rb = res.x

    print(f'sigma_p = {np.sqrt(calculate_portfolio_var(w_rb, covar))}, weights: {w_rb}, '
          f'risk contrib.s: {calculate_risk_contribution(w_rb, covar).T} '
          f'sum of weights: {sum(w_rb)}')
    return w_rb


def solve_max_diversification(covar: np.ndarray,
                              weight_mins: np.ndarray = None,
                              weight_maxs: np.ndarray = None,
                              is_gross_notional_one: bool = True,
                              disp: bool = False,
                              ) -> np.ndarray:
    n = covar.shape[0]
    x0 = np.ones(n) / n

    cons = [{'type': 'ineq', 'fun': long_only_constraint}]
    if is_gross_notional_one:
        cons.append({'type': 'eq', 'fun': total_weight_constraint})
    if weight_mins is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: x - weight_mins})
    if weight_maxs is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: weight_maxs - x})

    res = minimize(max_diversification_objective, x0, args=[covar], method='SLSQP', constraints=cons,
                   options={'disp': disp, 'ftol': 1e-18, 'maxiter': 200})
    w_rb = res.x

    print(f'sigma_p = {np.sqrt(calculate_portfolio_var(w_rb, covar))}, weights: {w_rb}, '
          f'risk contrib.s: {calculate_risk_contribution(w_rb, covar).T} '
          f'sum of weights: {sum(w_rb)}')
    return w_rb


def solve_risk_parity_alt(covar: np.ndarray,
                          budget: np.ndarray = None,
                          disp: bool = False):
    n = covar.shape[0]
    if budget is None:
        budget = np.ones(n) / n
    x0 = budget
    cons = [{'type': 'ineq', 'fun': long_only_constraint},
            {'type': 'eq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun': sum_of_log_weight_constraint, 'args': (budget,)}]

    res = minimize(calculate_portfolio_var, x0, args=covar, method='SLSQP', constraints=cons,
                   options={'disp': disp, 'ftol': 1e-14})

    w_rb = res.x

    print(f'(ALT) sigma_p = {np.sqrt(calculate_portfolio_var(w_rb, covar))}, weights: {w_rb}, '
          f'risk contrib.s: {calculate_risk_contribution(w_rb, covar).T} '
          f'sum of weights: {sum(w_rb)}')
    return w_rb


def solve_risk_parity_constr_vol(covar: np.ndarray,
                                 target_vol: float = None,
                                 disp: bool = False):
    n = covar.shape[0]
    budget = np.ones(n) / n
    x0 = budget

    cons = [{'type': 'ineq', 'fun': long_only_constraint},
            {'type': 'eq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun': portfolio_volatility_min, 'args': (covar, target_vol)},
            {'type': 'ineq', 'fun': portfolio_volatility_max, 'args': (covar, target_vol)}]

    res = minimize(risk_budget_objective_mod, x0, args=[covar, budget], method='SLSQP', constraints=cons,
                   options={'disp': disp, 'ftol': 1e-14})

    w_rb = res.x

    print(f'(CON) sigma_p = {np.sqrt(calculate_portfolio_var(w_rb, covar))}, weights: {w_rb}, '
          f'risk contrib.s: {calculate_risk_contribution(w_rb, covar).T} '
          f'sum of weights: {sum(w_rb)}')
    return w_rb


def solve_cara(means: np.ndarray,
               covar: np.ndarray,
               carra: float = 0.5,
               weight_mins: np.ndarray = None,
               weight_maxs: np.ndarray = None,
               disp: bool = False,
               is_exp: bool = False
               ) -> np.ndarray:
    n = covar.shape[0]
    x0 = np.ones(n) / n
    cons = [{'type': 'ineq', 'fun': long_only_constraint},
            {'type': 'eq', 'fun': total_weight_constraint}]
    if weight_mins is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: x - weight_mins})
    if weight_maxs is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: weight_maxs - x})

    if is_exp:
        func = carra_objective_exp
    else:
        func = carra_objective
    res = minimize(func, x0, args=[means, covar, carra], method='SLSQP', constraints=cons,
                   options={'disp': disp, 'ftol': 1e-16})
    w_rb = res.x

    print(f'return_p = {w_rb@means}, '
          f'sigma_p = {np.sqrt(calculate_portfolio_var(w_rb, covar))}, weights: {w_rb}, '
          f'risk contrib.s: {calculate_risk_contribution(w_rb, covar).T} '
          f'sum of weights: {sum(w_rb)}')
    return w_rb


def solve_cara_mixture(means: List[np.ndarray],
                       covars: List[np.ndarray],
                       probs: np.ndarray,
                       carra: float = 0.5,
                       weight_mins: np.ndarray = None,
                       weight_maxs: np.ndarray = None,
                       disp: bool = False,
                       is_print_log: bool = True
                       ) -> np.ndarray:
    n = means[0].shape[0]
    x0 = np.ones(n) / n
    cons = [{'type': 'ineq', 'fun': long_only_constraint},
            {'type': 'eq', 'fun': total_weight_constraint}]
    if weight_mins is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: x - weight_mins})
    if weight_maxs is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: weight_maxs - x})

    res = minimize(carra_objective_mixture, x0, args=[means, covars, probs, carra], method='SLSQP',
                   constraints=cons,
                   options={'disp': disp, 'ftol': 1e-16})
    w_rb = res.x

    if is_print_log:
        weights_str = ' '.join([f"{w:0.3f}" for w in w_rb])
        print(weights_str)
    return w_rb


class UnitTests(Enum):
    RISK_PARITY = 1
    CARA = 2
    CARA_MIX = 3


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.RISK_PARITY:
        covar = np.array([[0.2 ** 2, 0.015, 0.0],
                          [0.015, 0.15 ** 2, 0.0],
                          [0.0, 0.0, 0.1]])
        w_rb = solve_equal_risk_contribution(covar=covar, disp=True)
        w_md = solve_max_diversification(covar=covar, disp=True)

        w_rb = solve_equal_risk_contribution(covar=covar, is_gross_notional_one=False, disp=True)
        w_md = solve_max_diversification(covar=covar, is_gross_notional_one=False, disp=True)

    elif unit_test == UnitTests.CARA:
        means = np.array([0.3, 0.1])
        covar = np.array([[0.2 ** 2, 0.01],
                          [0.01, 0.1 ** 2]])
        w_rb = solve_cara(means=means, covar=covar, carra=10, is_exp=False, disp=True)
        w_rb = solve_cara(means=means, covar=covar, carra=10, is_exp=True, disp=True)

    elif unit_test == UnitTests.CARA_MIX:
        means = [np.array([0.05, -0.1]), np.array([0.05, 2.0])]
        covars = [np.array([[0.2 ** 2, 0.01],
                          [0.01, 0.2 ** 2]]),
                 np.array([[0.2 ** 2, 0.01],
                           [0.01, 0.2 ** 2]])
                 ]
        probs = np.array([0.95, 0.05])
        w_rb = solve_cara_mixture(means=means, covars=covars, probs=probs, carra=20.0, disp=True)


if __name__ == '__main__':

    unit_test = UnitTests.RISK_PARITY

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
