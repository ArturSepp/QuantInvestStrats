# QuantInvestStrats (`qis`)

**qis - performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in
Python.**

Quantitative Investment Strategies covers time-series and cross-sectional performance,
drift-aware portfolio histories, ex-ante and ex-post risk, and reproducible reports.

[![PyPI](https://img.shields.io/pypi/v/qis?style=flat-square)](https://pypi.org/project/qis/)
[![Python](https://img.shields.io/pypi/pyversions/qis?style=flat-square)](https://pypi.org/project/qis/)
[![License](https://img.shields.io/github/license/ArturSepp/QuantInvestStrats.svg?style=flat-square)](LICENSE.txt)
[![CI](https://github.com/ArturSepp/QuantInvestStrats/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSepp/QuantInvestStrats/actions)
[![Docs](https://readthedocs.org/projects/quantinveststrats/badge/?version=latest)](https://quantinveststrats.readthedocs.io/en/latest/)
[![Downloads](https://static.pepy.tech/badge/qis)](https://pepy.tech/project/qis)
[![Monthly](https://static.pepy.tech/badge/qis/month)](https://pepy.tech/project/qis)

---

## Overview <a name="analytics"></a>
 

The package is split into 5 main modules with the 
dependency path increasing sequentially as follows.

1. ```qis.utils``` is module containing low level utilities for operations with pandas, numpy, and datetimes.

2. ```qis.perfstats``` is module for computing performance statistics and performance attribution including returns, volatilities, etc.

3. ```qis.plots``` is module for plotting and visualization apis.

4. ```qis.models``` is module containing statistical models including filtering and regressions.

5. ```qis.portfolio``` is high level module for analysis, simulation, backtesting, and reporting of quant strategies.
Function ```backtest_model_portfolio()```  in ```qis.portfolio.backtester.py``` takes instrument prices 
and simulated weights from a generic strategy and compute the total return, performance attribution, and risk analysis

Risk and tracking-error analytics are consolidated in ```qis.portfolio.risk```. The public
```qis.RiskModel``` is the point-in-time weights-and-covariance layer for ex-ante tracking
error, standalone group risk, factor exposures, benchmark beta and loadings,
systematic/residual tracking-error decomposition, and Euler marginal tracking-error
contributions. Ex-post analytics use portfolio and benchmark NAVs or return differences:
```compute_ewma_realised_tracking_error``` produces a conditional annualised series, while
```compute_te_ir_errors``` and ```compute_info_ratio_table``` produce whole-sample tracking
error and information-ratio estimates. The
```weights_tracking_error_report_by_ac_subac``` report brings these views together with
ex-ante versus realised tracking error, ex-ante versus ex-post beta, annualised ex-post alpha,
and optional factor panels. Some established API names retain the abbreviation ```tre```, but
all refer to tracking error.

Covariance-implied Euler attribution also lives here for the whole OSS stack:
```compute_portfolio_risk_contributions``` returns asset contributions in volatility units,
```compute_portfolio_risk_contribution_ratios``` returns their dimensionless shares, and
```compute_group_portfolio_risk_contribution_ratios``` aggregates those shares over clusters,
sectors, asset classes, or any other complete labelled partition.

```qis.market_data``` is an auxiliary module of market-data containers and FX analytics. ```FxRatesData``` holds FX spot and domestic short-rate panels and derives cross rates, covered-interest-parity forward premia, carry decomposition, and reference-currency / FX-hedged return translation of multi-asset panels, together with single- and multi-asset FX-hedging reports. ```FactorsData``` is a generic container for tradable-factor prices. Examples build the container from free Yahoo data or from Bloomberg via ```bbg-fetch```; see the module README at ```src/qis/market_data/README.md``` for the data contract and conventions.

The repository-root [`examples/`](examples/) directory contains runnable scripts showcasing the
analytics. It is intentionally separate from the installed `qis` package:

* ```examples/perfstats``` — performance metrics on price series: quickstart usage, Sharpe vs Sortino across return frequencies, rolling performance, bond-ETF risk/return frontier, multi-figure performance reports, miss-best-worst-days impact, infrequent-returns interpolation, and an end-to-end de-levering / unsmoothing walkthrough on a bundled BDC vs private-credit dataset.

* ```examples/models``` — numba-vs-pandas EWM kernel benchmarks, multivariate EWM linear factor models, multivariate OLS, EWM correlation tables, OHLC realised-volatility estimators, intraday/overnight return decomposition, rolling correlations, and block bootstrap of price paths.

* ```examples/regimes``` — regime-conditional analytics: bull/bear/normal Sharpe attribution, conditional return boxplots by VIX regime, calendar-month seasonality, US election regime study.

* ```examples/portfolios``` — backtests using ```backtest_model_portfolio```: balanced 60/40 with and without a BTC sleeve, constant-notional short, leveraged-ETF combinations, long/short pairs, vol-target / trend-following parameter sweeps, and separate offline ex-ante and ex-post tracking-error workflows.

* ```examples/factsheets``` — full multi-page factsheets for simulated and actual strategies, cross-sectional asset-class comparisons, multi-strategy parameter sweeps, and optional pybloqs-rendered variants.

* ```examples/plots``` — plotting primitives showcase: dual-axis figures, scatter with regression diagnostics.

* ```examples/utils``` — date schedules and rolling calendars: option / futures roll generation via ```generate_fixed_maturity_rolls```.

* ```examples/case_studies``` — cross-cutting domain studies: VIX beta to equities and bonds, VIX term-structure correlation with SPX, conditional returns on the front-month short-VIX strategy, credit-spread regression vs equity / rates.

The [`examples/README.md`](examples/README.md) index lists every script with a one-line
description; examples that need a Bloomberg terminal are flagged inline.


# Table of contents
1. [Analytics](#analytics)
2. [Installation](#installation)
3. [Offline quickstart](#offline-quickstart)
4. [Examples](#examples)
   1. [Visualization of price data](#price)
   2. [Multi assets factsheet](#multiassets)
   3. [Strategy factsheet](#strategy)
   4. [Strategy benchmark factsheet](#strategybenchmark)
   5. [Multi strategy factsheet](#multistrategy)
   6. [Runnable examples](#runnable-examples)
5. [Contributions](#contributions)
6. [Changelog](#changelog)
7. [ToDos](#todos)
8. [Disclaimer](#disclaimer)


## Installation <a name="installation"></a>
Install using
```bash
pip install qis
```
Upgrade using
```bash
pip install --upgrade qis
```

Close using
```bash
git clone https://github.com/ArturSepp/QuantInvestStrats.git
```

Core dependencies:
    python = ">=3.10",
    numba = ">=0.63.0",
    numpy = ">=2.0",
    scipy = ">=1.12.0",
    statsmodels = ">=0.14.0",
    pandas = ">=2.2.0",
    matplotlib = ">=3.8.0",
    seaborn = ">=0.13.0",
    openpyxl = ">=3.1.0",
    PyYAML = ">=6.0"

`src/qis/tests/test_documentation.py` asserts that this list is the `dependencies` table of
`pyproject.toml`, so it cannot drift from what `pip install qis` actually pulls.

Python 3.14 is supported (numba 0.63+ ships cp314 wheels).

Optional dependencies:
    yfinance = ">=0.2.40" and pandas-datareader = ">=0.10.0" (examples and tests that pull free
        price data — install with `pip install qis[data]`; never imported by library code),
    pybloqs ">=1.2.13" (for producing html and pdf factsheets — install with `pip install qis[reports]`),
    bbg-fetch ">=2.0.0" (third-party; for examples that pull data from a Bloomberg terminal)

See `pyproject.toml` for the full list of optional extras (`reports`, `visualization`, `io`, `database`, `jupyter`, `dev`, `all`).


## Offline quickstart <a name="offline-quickstart"></a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/QuantInvestStrats/blob/main/notebooks/offline_quickstart_colab.ipynb)

The authoritative first-success workflow is
[`examples/getting_started/offline_quickstart.py`](examples/getting_started/offline_quickstart.py).
It generates seeded data in-process, builds a live-universe-aware quarterly weight schedule,
backtests with explicit transaction costs, and prints performance plus benchmark-relative risk.
It needs only the core `qis` installation and writes no files.

From a repository checkout:

```bash
python examples/getting_started/offline_quickstart.py
```

With only `pip install qis`, copy the complete code from the
[hosted offline quickstart](https://quantinveststrats.readthedocs.io/en/latest/quickstart.html).
That page includes the runnable script directly, so the README, documentation, and example cannot
develop independent full-code versions.

The Colab entry point installs the latest release from public PyPI, reports its exact version and
import path, and runs that same mechanically checked source with no saved notebook outputs.


## Examples <a name="examples"></a>

### 1. Visualization of price data <a name="price"></a>

This is an optional network-backed plotting example. For the core-install first-success path, use
the offline quickstart above.

The script is located at [`examples/perfstats/quickstart.py`](examples/perfstats/quickstart.py).
Run `python -m examples.perfstats.quickstart` from the repository root to produce the figures
below; `perf1` to `perf3` are excluded from the repository by `.gitignore` on size, so only the
last is embedded here.

```python 
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import qis
from qis import PerfStat

# define tickers and fetch price data
tickers = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'GLD']
prices = yf.download(tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers].dropna()

# plotting price data with minimum usage
with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    qis.plot_prices(prices=prices, x_date_freq='YE', ax=ax)
```
```python 
# 2-axis plot with drawdowns using sns styles
with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), tight_layout=True)
    qis.plot_prices_with_dd(prices=prices, x_date_freq='YE', axs=axs)
```

```python 
# plot risk-adjusted performance table with excess Sharpe ratio
ust_3m_rate = yf.download('^IRX', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].dropna() / 100.0
# set parameters for computing performance stats including returns vols and regressions
perf_params = qis.PerfParams(freq='ME', freq_reg='QE', rates_data=ust_3m_rate)
# perf_columns is list to display different perfomance metrics from enumeration PerfStat
fig = qis.plot_ra_perf_table(prices=prices,
                             perf_columns=[PerfStat.TOTAL_RETURN, PerfStat.PA_RETURN, PerfStat.PA_EXCESS_RETURN,
                                           PerfStat.VOL, PerfStat.SHARPE_RF0,
                                           PerfStat.SHARPE_EXCESS, PerfStat.SORTINO_RATIO, PerfStat.CALMAR_RATIO,
                                           PerfStat.MAX_DD, PerfStat.MAX_DD_VOL,
                                           PerfStat.SKEWNESS, PerfStat.KURTOSIS],
                             title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')}",
                             perf_params=perf_params)
```



```python 
# add benchmark regression using excess returns for linear beta
# regression frequency is specified using perf_params.freq_reg
# regression alpha is multiplied using alpha_an_factor
fig, _ = qis.plot_ra_perf_table_benchmark(prices=prices,
                                          benchmark='SPY',
                                          perf_columns=[PerfStat.TOTAL_RETURN, PerfStat.PA_RETURN, PerfStat.PA_EXCESS_RETURN,
                                                        PerfStat.VOL, PerfStat.SHARPE_RF0,
                                                        PerfStat.SHARPE_EXCESS, PerfStat.SORTINO_RATIO, PerfStat.CALMAR_RATIO,
                                                        PerfStat.MAX_DD, PerfStat.MAX_DD_VOL,
                                                        PerfStat.SKEWNESS, PerfStat.KURTOSIS,
                                                        PerfStat.ALPHA_AN, PerfStat.BETA, PerfStat.R2],
                                          title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')} benchmarked with SPY",
                                          perf_params=perf_params)
```
![image info](examples/figures/perf4.PNG)



### 2. Multi assets factsheet <a name="multiassets"></a>
This report is adopted for reporting the risk-adjusted performance 
of several assets with the goal
of cross-sectional comparision

Run [`examples/factsheets/multi_assets.py`](examples/factsheets/multi_assets.py).

![image info](examples/figures/multiassets.PNG)


### 3. Strategy factsheet <a name="strategy"></a>
This report is adopted for report performance, risk, and trading statistics
for either backtested or actual strategy
    with strategy data passed as PortfolioData object

Run [`examples/factsheets/strategy.py`](examples/factsheets/strategy.py).

![image info](examples/figures/strategy1.PNG)
![image info](examples/figures/strategy2.PNG)
![image info](examples/figures/strategy3.PNG)

### 4. Strategy benchmark factsheet <a name="strategybenchmark"></a>
This report is adopted for report performance and marginal comparison
  of strategy vs a benchmark strategy 
(data for both are passed using individual PortfolioData object)

Run [`examples/factsheets/strategy_benchmark.py`](examples/factsheets/strategy_benchmark.py).

![image info](examples/figures/strategy_benchmark.PNG)

Brinson-Fachler performance attribution (https://en.wikipedia.org/wiki/Performance_attribution)
![image info](examples/figures/brinson_attribution.PNG)


### 5. Multi strategy factsheet <a name="multistrategy"></a>
This report is adopted to examine the sensitivity of 
backtested strategy to a parameter or set of parameters:

Run [`examples/factsheets/multi_strategy.py`](examples/factsheets/multi_strategy.py).

![image info](examples/figures/multi_strategy.PNG)


### 6. Runnable examples <a name="runnable-examples"></a>

The examples are plain scripts under
[`examples/`](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples), each
runnable top to bottom. `src/qis/tests/test_examples.py` checks them for symbols and keyword
arguments that exist, and runs the examples that need no data vendor.

The four factsheet archetypes shown above are
[`multi_assets.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/multi_assets.py),
[`strategy.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/strategy.py),
[`strategy_benchmark.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/strategy_benchmark.py)
and
[`multi_strategy.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/multi_strategy.py).

The consolidated tracking-error analytics are demonstrated offline in
[`ex_anti_tracking_error_and_risk.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_anti_tracking_error_and_risk.py)
for the covariance-based ex-ante view and
[`ex_post_tracking_error_and_risk.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/ex_post_tracking_error_and_risk.py)
for realised EWMA tracking error, whole-sample TE/IR, and EWMA beta/alpha.


## Ecosystem

This package is part of an open-source Python stack for quantitative finance — full catalogue at [github.com/ArturSepp](https://github.com/ArturSepp):

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) *(this package)* | Performance and risk analytics, factsheets, and visualisation |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor models and factor covariance estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data fetching |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Trend-following systems: closed-form theory and replication |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Private-asset return unsmoothing and capital market assumptions |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Dynamic MV allocation under regime-switching jump-diffusions |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic volatility pricing analytics |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Vectorised vanilla option pricers and implied volatility fitters |

Dependency links within the stack: `optimalportfolios` builds on `qis` and `factorlasso`; `trendfollowing` and `privateassets` build on `qis`.

## Contributions <a name="contributions"></a>
If you are interested in extending and improving QIS analytics, 
please consider contributing to the library.

I have found it is a good practice to isolate general purpose and low level analytics and visualizations, which can be outsourced and shared, while keeping 
the focus on developing high level commercial applications.

There are a number of requirements:

- The code is [Pep 8 compliant](https://peps.python.org/pep-0008/)

- Reliance on common Python data types including numpy arrays, pandas, and dataclasses.

- Transparent naming of functions and data types with enough comments. Type annotations of functions and arguments is a must.

- Each submodule has a unit test for core functions and a localised entry point to core functions.

- Avoid "super" pythonic constructions. Readability is the priority.



## Changelog <a name="changelog"></a>

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).


## ToDos <a name="todos"></a>

1. Enhanced documentation and readme examples.

2. Docstrings for key functions.

3. Reporting analytics and factsheets generation enhancing to matplotlib.



## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Disclaimer <a name="disclaimer"></a>

QIS package is distributed FREE & WITHOUT ANY WARRANTY under the MIT License.

See the [LICENSE.txt](https://github.com/ArturSepp/QuantInvestStrats/blob/main/LICENSE.txt) in the release for details.

Please report any bugs or suggestions by opening an [issue](https://github.com/ArturSepp/QuantInvestStrats/issues).


## Citation

If you use QIS in your research, please cite it as:

```bibtex
@software{sepp2026qis,
  title={qis: Implementation of visualisation and reporting analytics for Quantitative Investment Strategies},
  author={Sepp, Artur},
  year={2026},
  version={5.11.2},
  url={https://github.com/ArturSepp/QuantInvestStrats}
}
```
