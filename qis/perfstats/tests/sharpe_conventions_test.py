"""
regression guard for return and sharpe ratio conventions

pins every return convention and every sharpe convention on one deterministic
reference series so that future edits cannot silently swap conventions.

return conventions (with excess variants under rates_data):
    TOTAL_RETURN        = P_M / P_0 - 1
    PA_RETURN           = (1 + TOTAL_RETURN)^(1/T) - 1                  (compound)
    AN_LOG_RETURN       = ln(1 + PA_RETURN) = a * mean(l_m)             (log)
    AVG_LOG_RETURN      = mean(l_m) at freq_vol
    AVG_ARITH_RETURN    = mean(r_m) at freq_vol                         (arithmetic)
    AN_ARITH_RETURN     = a * mean(r_m)
    VOL                 = sqrt(a) * std(returns at freq_vol per return_type)

sharpe conventions (with excess variants under rates_data):
    SHARPE_RF0          = PA_RETURN / VOL                               (p.a., qis default)
    SHARPE_LOG_AN       = AN_LOG_RETURN / VOL = sqrt(a)*mean(l)/std(l)  (log)
    SHARPE_ARITH        = sqrt(a) * mean(r_m) / std(r_m)                (arithmetic)

identities checked:
    SR_log ~= SR_pa (exp-map twins), SR_arith - SR_pa ~= sigma_ann / 2 (volatility drag),
    all excess objects collapse to plain at rf = 0 and sit strictly below at rf > 0
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
# qis
import qis.perfstats.returns as ret
from qis.perfstats.config import PerfStat, PerfParams, ReturnTypes
from qis.perfstats.perf_stats import compute_ra_perf_table

AN_FACTOR = 12  # reference series is monthly
RATE_ANN = 0.02  # flat annualised funding rate for excess tests


def generate_reference_prices(num_periods: int = 360,  # 30y monthly
                              an_factor: int = AN_FACTOR,
                              mu_ann: float = 0.08,  # annualised log drift + 0.5*sigma^2
                              sigma_ann: float = 0.15,  # annualised vol
                              seed: int = 7,
                              ) -> pd.DataFrame:
    """generate deterministic gbm nav for convention pinning"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('1996-01-31', periods=num_periods, freq='ME')
    log_returns = rng.normal(loc=(mu_ann - 0.5 * sigma_ann ** 2) / an_factor,
                             scale=sigma_ann / np.sqrt(an_factor),
                             size=(num_periods, 1))
    prices = pd.DataFrame(np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=['ref'])
    return prices


def _check_value(label: str,
                 got: float,
                 expected: float,
                 atol: float = 1e-12,
                 ) -> None:
    """raise with the offending values when a pinned convention moves"""
    if not np.isclose(got, expected, atol=atol):
        raise ValueError(f"{label} convention changed: got {got!r}, expected {expected!r}")


def _compute_tables() -> tuple:
    """reference tables at rf=0 and with flat rates, plus the sampled return series"""
    prices = generate_reference_prices()
    rates_data = pd.Series(RATE_ANN, index=prices.index)
    ra_perf_table = compute_ra_perf_table(prices=prices, perf_params=PerfParams(freq_vol='ME'))
    ra_perf_table_ex = compute_ra_perf_table(prices=prices,
                                             perf_params=PerfParams(freq_vol='ME', rates_data=rates_data))
    simple_returns = prices.pct_change().dropna()
    log_returns = np.log(prices).diff().dropna()
    excess_simple_returns = ret.compute_excess_returns(returns=simple_returns, rates_data=rates_data)
    return prices, ra_perf_table, ra_perf_table_ex, simple_returns, log_returns, excess_simple_returns


def test_return_conventions() -> None:
    """pin every return numerator to its first-principles formula at rf=0"""
    prices, table, _, simple_returns, log_returns, _ = _compute_tables()

    def stat(perf_stat: PerfStat) -> float:
        return float(table[perf_stat.to_str()].iloc[0])

    # compound family
    total_return_manual = float(prices.iloc[-1, 0] / prices.iloc[0, 0] - 1.0)
    _check_value('TOTAL_RETURN', stat(PerfStat.TOTAL_RETURN), total_return_manual)
    pa_return_manual = (1.0 + total_return_manual) ** (1.0 / stat(PerfStat.NUM_YEARS)) - 1.0
    _check_value('PA_RETURN', stat(PerfStat.PA_RETURN), pa_return_manual)

    # log family
    _check_value('AN_LOG_RETURN', stat(PerfStat.AN_LOG_RETURN), np.log1p(stat(PerfStat.PA_RETURN)))
    _check_value('AVG_LOG_RETURN', stat(PerfStat.AVG_LOG_RETURN), float(log_returns.mean().iloc[0]))

    # arithmetic family
    avg_arith_manual = float(simple_returns.mean().iloc[0])
    _check_value('AVG_ARITH_RETURN', stat(PerfStat.AVG_ARITH_RETURN), avg_arith_manual)
    _check_value('AN_ARITH_RETURN', stat(PerfStat.AN_ARITH_RETURN), AN_FACTOR * avg_arith_manual)

    # shared vol: sqrt(a) * std of log returns under default return_type = LOG
    vol_manual = float(np.sqrt(AN_FACTOR) * log_returns.std(ddof=1).iloc[0])
    _check_value('VOL', stat(PerfStat.VOL), vol_manual, atol=1e-10)

    # rf=0: every excess numerator collapses to its plain counterpart
    for excess, plain in ((PerfStat.PA_EXCESS_RETURN, PerfStat.PA_RETURN),
                          (PerfStat.AN_LOG_EXCESS_RETURN, PerfStat.AN_LOG_RETURN),
                          (PerfStat.AVG_ARITH_EXCESS_RETURN, PerfStat.AVG_ARITH_RETURN),
                          (PerfStat.AN_ARITH_EXCESS_RETURN, PerfStat.AN_ARITH_RETURN)):
        _check_value(f"rf=0 {excess.name}", stat(excess), stat(plain))


def test_sharpe_conventions() -> None:
    """pin the three sharpe conventions and their cross-convention identities at rf=0"""
    prices, table, _, simple_returns, log_returns, _ = _compute_tables()

    sr_pa = float(table[PerfStat.SHARPE_RF0.to_str()].iloc[0])
    sr_log = float(table[PerfStat.SHARPE_LOG_AN.to_str()].iloc[0])
    sr_arith = float(table[PerfStat.SHARPE_ARITH.to_str()].iloc[0])
    vol = float(table[PerfStat.VOL.to_str()].iloc[0])

    # 1. SHARPE_ARITH is the sharpe (1994) plug-in estimator on simple returns
    sr_arith_manual = float(np.sqrt(AN_FACTOR) * simple_returns.mean().iloc[0]
                            / simple_returns.std(ddof=1).iloc[0])
    _check_value('SHARPE_ARITH', sr_arith, sr_arith_manual)

    # 2. SHARPE_LOG_AN is the plug-in estimator on log returns: a*mean(l) = ln(1+C_pa)
    sr_log_manual = float(np.sqrt(AN_FACTOR) * log_returns.mean().iloc[0]
                          / log_returns.std(ddof=1).iloc[0])
    _check_value('SHARPE_LOG_AN', sr_log, sr_log_manual, atol=1e-10)

    # 3. SHARPE_RF0 is the p.a. (compound) sharpe: C_pa / sigma_ann
    _check_value('SHARPE_RF0', sr_pa, float(table[PerfStat.PA_RETURN.to_str()].iloc[0]) / vol)

    # 4. rf=0: every excess sharpe collapses to its plain counterpart
    for excess, plain in ((PerfStat.SHARPE_EXCESS, PerfStat.SHARPE_RF0),
                          (PerfStat.SHARPE_LOG_EXCESS, PerfStat.SHARPE_LOG_AN),
                          (PerfStat.SHARPE_ARITH_EXCESS, PerfStat.SHARPE_ARITH)):
        _check_value(f"rf=0 {excess.name}",
                     float(table[excess.to_str()].iloc[0]), float(table[plain.to_str()].iloc[0]))

    # 5. identity: p.a. sharpe is numerically the log sharpe
    if not np.abs(sr_log - sr_pa) < 0.02:
        raise ValueError(f"SR_log !~ SR_pa: got {sr_log!r} vs {sr_pa!r}")

    # 6. identity: arithmetic sits above p.a. by ~ sigma_ann / 2 (volatility drag)
    if not np.abs((sr_arith - sr_pa) - 0.5 * vol) < 0.03:
        raise ValueError(f"drag wedge changed: got {sr_arith - sr_pa!r}, expected ~{0.5 * vol!r}")

    # 7. SHARPE_ARITH is invariant to the table vol convention (return_type)
    table_rel = compute_ra_perf_table(prices=prices,
                                      perf_params=PerfParams(freq_vol='ME',
                                                             return_type=ReturnTypes.RELATIVE))
    _check_value('return_type leaked into SHARPE_ARITH',
                 float(table_rel[PerfStat.SHARPE_ARITH.to_str()].iloc[0]), sr_arith)


def test_excess_sharpe_conventions() -> None:
    """pin every excess convention with a flat rf > 0 and the excess < plain ordering"""
    _, table, table_ex, _, _, excess_simple_returns = _compute_tables()

    def stat(perf_stat: PerfStat) -> float:
        return float(table_ex[perf_stat.to_str()].iloc[0])

    vol = stat(PerfStat.VOL)

    # compound excess: SHARPE_EXCESS = PA_EXCESS_RETURN / vol, numerator below plain
    if not stat(PerfStat.PA_EXCESS_RETURN) < stat(PerfStat.PA_RETURN):
        raise ValueError(f"PA_EXCESS_RETURN not below PA_RETURN: got {stat(PerfStat.PA_EXCESS_RETURN)!r}")
    _check_value('SHARPE_EXCESS', stat(PerfStat.SHARPE_EXCESS), stat(PerfStat.PA_EXCESS_RETURN) / vol)

    # log excess: AN_LOG_EXCESS_RETURN = ln(1 + PA_EXCESS_RETURN), sharpe over shared vol
    _check_value('AN_LOG_EXCESS_RETURN', stat(PerfStat.AN_LOG_EXCESS_RETURN),
                 np.log1p(stat(PerfStat.PA_EXCESS_RETURN)))
    _check_value('SHARPE_LOG_EXCESS', stat(PerfStat.SHARPE_LOG_EXCESS),
                 stat(PerfStat.AN_LOG_EXCESS_RETURN) / vol)

    # arithmetic excess: periodic excess simple returns r_m - rf_{m-1}*dt_m (lag=1),
    # numerator and denominator paired on the same excess series; the expected values
    # reuse ret.compute_excess_returns, which owns the lag and dt conventions
    avg_arith_excess_manual = float(excess_simple_returns.mean().iloc[0])
    _check_value('AVG_ARITH_EXCESS_RETURN', stat(PerfStat.AVG_ARITH_EXCESS_RETURN), avg_arith_excess_manual)
    _check_value('AN_ARITH_EXCESS_RETURN', stat(PerfStat.AN_ARITH_EXCESS_RETURN),
                 AN_FACTOR * avg_arith_excess_manual)
    sharpe_arith_excess_manual = float(np.sqrt(AN_FACTOR) * avg_arith_excess_manual
                                       / excess_simple_returns.std(ddof=1).iloc[0])
    _check_value('SHARPE_ARITH_EXCESS', stat(PerfStat.SHARPE_ARITH_EXCESS), sharpe_arith_excess_manual)

    # ordering: with rf > 0 every excess sharpe sits strictly below its plain counterpart
    for excess, plain in ((PerfStat.SHARPE_EXCESS, PerfStat.SHARPE_RF0),
                          (PerfStat.SHARPE_LOG_EXCESS, PerfStat.SHARPE_LOG_AN),
                          (PerfStat.SHARPE_ARITH_EXCESS, PerfStat.SHARPE_ARITH)):
        if not stat(excess) < stat(plain):
            raise ValueError(f"{excess.name} not below {plain.name} with rf>0: got {stat(excess)!r}")

    # plain columns are unaffected by supplying rates_data
    for perf_stat in (PerfStat.SHARPE_RF0, PerfStat.SHARPE_LOG_AN, PerfStat.SHARPE_ARITH):
        _check_value(f"rates_data leaked into {perf_stat.name}",
                     stat(perf_stat), float(table[perf_stat.to_str()].iloc[0]))


class LocalTests(Enum):
    RETURN_CONVENTIONS = 1
    SHARPE_CONVENTIONS = 2
    EXCESS_SHARPE_CONVENTIONS = 3
    ALL = 4


def run_local_test(local_test: LocalTests):
    """run local tests for development and debugging purposes"""
    if local_test in (LocalTests.RETURN_CONVENTIONS, LocalTests.ALL):
        test_return_conventions()
        print("return conventions regression guard passed")
    if local_test in (LocalTests.SHARPE_CONVENTIONS, LocalTests.ALL):
        test_sharpe_conventions()
        print("sharpe conventions regression guard passed")
    if local_test in (LocalTests.EXCESS_SHARPE_CONVENTIONS, LocalTests.ALL):
        test_excess_sharpe_conventions()
        print("excess sharpe conventions regression guard passed")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL)