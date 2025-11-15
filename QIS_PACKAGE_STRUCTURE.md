# QIS Package Structure - Comprehensive Overview

## 1. OVERALL DIRECTORY STRUCTURE

```
/home/user/QuantInvestStrats/
├── qis/                          # Main package directory
│   ├── __init__.py               # Package initialization (imports all submodules)
│   ├── file_utils.py             # File I/O utilities (CSV, Excel, Parquet, Feather)
│   ├── local_path.py             # Path management configuration
│   ├── sql_engine.py             # SQL database connectivity
│   ├── test_data.py              # Test data generation utilities
│   │
│   ├── utils/                    # Core utilities (12 modules, 3,845 LOC)
│   │   ├── __init__.py
│   │   ├── dates.py              # Date/time operations (1,212 LOC)
│   │   ├── df_agg.py             # DataFrame aggregation functions (288 LOC)
│   │   ├── df_cut.py             # DataFrame binning/cutting operations (221 LOC)
│   │   ├── df_freq.py            # DataFrame resampling/frequency ops (227 LOC)
│   │   ├── df_groups.py          # DataFrame grouping operations (296 LOC)
│   │   ├── df_melt.py            # DataFrame melting/unpivoting (181 LOC)
│   │   ├── df_ops.py             # Core DataFrame operations (674 LOC)
│   │   ├── df_str.py             # String conversions for DataFrames (292 LOC)
│   │   ├── df_to_scores.py       # Score/ranking computations (268 LOC)
│   │   ├── df_to_weights.py      # Weight allocation operations (336 LOC)
│   │   ├── generic.py            # Generic utilities & data structures (325 LOC)
│   │   ├── np_ops.py             # NumPy operations (628 LOC)
│   │   ├── regression.py         # OLS regression utilities (205 LOC)
│   │   ├── sampling.py           # Train/test splitting (133 LOC)
│   │   └── struct_ops.py         # List/dict operations (283 LOC)
│   │
│   ├── perfstats/                # Performance statistics (8 modules, 2,753 LOC)
│   │   ├── __init__.py           # Exports all perf stat functions
│   │   ├── config.py             # Performance metric enums (261 LOC)
│   │   ├── perf_stats.py         # Core performance calculations (524 LOC)
│   │   ├── returns.py            # Return computation functions (762 LOC)
│   │   ├── regime_classifier.py  # Market regime classification (441 LOC)
│   │   ├── cond_regression.py    # Conditional regression analysis (155 LOC)
│   │   ├── desc_table.py         # Descriptive statistics tables (164 LOC)
│   │   ├── timeseries_bfill.py   # Time series filling/interpolation (289 LOC)
│   │   └── fx_ops.py             # FX rate operations (62 LOC)
│   │
│   ├── models/                   # Statistical models (2 submodules)
│   │   ├── __init__.py           # Exports all model functions
│   │   │
│   │   ├── linear/               # Linear models (8 modules, 3,185 LOC)
│   │   │   ├── __init__.py
│   │   │   ├── ewm.py            # EWM filtering & covariance (1,297 LOC)
│   │   │   ├── auto_corr.py      # Autocorrelation analysis (344 LOC)
│   │   │   ├── corr_cov_matrix.py # Correlation/covariance matrices (315 LOC)
│   │   │   ├── pca.py            # Principal Component Analysis (127 LOC)
│   │   │   ├── ra_returns.py     # Risk-adjusted return transforms (364 LOC)
│   │   │   ├── ewm_convolution.py # EWM signal convolution (102 LOC)
│   │   │   ├── ewm_winsor_outliers.py # Outlier handling (370 LOC)
│   │   │   └── plot_correlations.py  # Correlation visualization (266 LOC)
│   │   │
│   │   └── stats/                # Statistical models (4 modules, 888 LOC)
│   │       ├── __init__.py
│   │       ├── bootstrap.py      # Bootstrap resampling (424 LOC)
│   │       ├── rolling_stats.py  # Rolling statistics (177 LOC)
│   │       ├── ohlc_vol.py       # OHLC volatility estimation (78 LOC)
│   │       └── test_bootstrap.py # Bootstrap tests (209 LOC)
│   │
│   ├── plots/                    # Visualization layer (2 submodules + base, ~10k LOC)
│   │   ├── __init__.py           # Main exports (173 LOC)
│   │   ├── utils.py              # Plot utilities (1,619 LOC) - axis/legend/color management
│   │   ├── bars.py               # Bar charts (555 LOC)
│   │   ├── boxplot.py            # Box plots (585 LOC)
│   │   ├── scatter.py            # Scatter plots (482 LOC)
│   │   ├── time_series.py        # Time series plots (454 LOC)
│   │   ├── table.py              # Table visualization (379 LOC)
│   │   ├── heatmap.py            # Heatmaps (130 LOC)
│   │   ├── histogram.py          # Histograms & PDF (310 LOC)
│   │   ├── lineplot.py           # Line plots (260 LOC)
│   │   ├── qqplot.py             # Q-Q plots (185 LOC)
│   │   ├── errorbar.py           # Error bars (155 LOC)
│   │   ├── stackplot.py          # Stacked area plots (210 LOC)
│   │   ├── contour.py            # Contour plots (125 LOC)
│   │   ├── pie.py                # Pie charts (80 LOC)
│   │   ├── histplot2d.py         # 2D histograms (98 LOC)
│   │   │
│   │   ├── derived/              # High-level derived plots (12 modules, 2,432 LOC)
│   │   │   ├── __init__.py
│   │   │   ├── prices.py         # Price & performance charts (395 LOC)
│   │   │   ├── perf_table.py     # Performance tables & metrics (515 LOC)
│   │   │   ├── returns_heatmap.py # Return heatmaps by period (436 LOC)
│   │   │   ├── drawdowns.py      # Drawdown analysis plots (190 LOC)
│   │   │   ├── regime_data.py    # Regime-conditional plots (287 LOC)
│   │   │   ├── regime_scatter.py # Regime scatter analysis (158 LOC)
│   │   │   ├── returns_scatter.py # Return scatter plots (131 LOC)
│   │   │   ├── regime_pdf.py     # Regime PDF plots (96 LOC)
│   │   │   ├── regime_class_table.py # Regime classification table (91 LOC)
│   │   │   ├── desc_table.py     # Descriptive statistics display (70 LOC)
│   │   │   └── data_timeseries.py # Generic data time series (63 LOC)
│   │   │
│   │   └── reports/              # Report generation (5 modules, 506 LOC)
│   │       ├── __init__.py
│   │       ├── utils.py          # Report utilities (66 LOC)
│   │       ├── price_history.py  # Price history reports (61 LOC)
│   │       ├── econ_data_single.py # Economic data reports (195 LOC)
│   │       └── gantt_data_history.py # Gantt chart reports (184 LOC)
│   │
│   ├── portfolio/                # Portfolio analysis & backtesting (4 submodules)
│   │   ├── __init__.py           # Main portfolio exports
│   │   ├── portfolio_data.py     # PortfolioData class (1,645 LOC)
│   │   ├── multi_portfolio_data.py # MultiPortfolioData class (1,121 LOC)
│   │   ├── backtester.py         # Backtesting functions (core feature)
│   │   ├── signal_data.py        # StrategySignalData class (238 LOC)
│   │   │
│   │   ├── risk/                 # Risk analysis (5 modules, 1,317 LOC)
│   │   │   ├── __init__.py
│   │   │   ├── factor_model.py   # Linear factor models (455 LOC)
│   │   │   ├── ewm_factor_model.py # EWM factor models (234 LOC)
│   │   │   ├── ewm_covar_risk.py # EWM covariance/VaR (265 LOC)
│   │   │   └── contributions.py  # Risk contributions analysis (363 LOC)
│   │   │
│   │   ├── strats/               # Strategy implementations (2 modules)
│   │   │   ├── __init__.py
│   │   │   ├── quant_strats_delta1.py # Delta-one quant strategies
│   │   │   └── seasonal_strats.py     # Seasonal strategies
│   │   │
│   │   └── reports/              # Factsheet generation (12 modules, ~300k LOC total)
│   │       ├── __init__.py
│   │       ├── config.py         # Factsheet configuration (13,148 LOC)
│   │       ├── brinson_attribution.py # Brinson decomposition (21,363 LOC)
│   │       ├── strategy_factsheet.py # Single strategy reports (41,017 LOC)
│   │       ├── strategy_benchmark_factsheet.py # Strategy vs benchmark (31,019 LOC)
│   │       ├── strategy_benchmark_tre_factsheet.py # Tracking error reports (34,987 LOC)
│   │       ├── multi_strategy_factsheet.py # Multiple strategy reports (12,887 LOC)
│   │       ├── multi_assets_factsheet.py # Asset allocation reports (29,616 LOC)
│   │       ├── strategy_signal_factsheet.py # Signal analysis (9,923 LOC)
│   │       ├── overlays_smart_diversification.py # Diversification reports (22,610 LOC)
│   │       ├── multi_strategy_factseet_pybloqs.py # PyBloqs reports (21,539 LOC)
│   │       ├── strategy_benchmark_factsheet_pybloqs.py # PyBloqs benchmark (15,532 LOC)
│   │       └── config.py         # Report configuration
│   │
│   └── examples/                 # Example scripts & notebooks (36+ example files)
│       ├── *.py                  # Individual example scripts (30+ files)
│       ├── core/                 # Core examples (4 files)
│       ├── models/               # Model examples (1 file)
│       ├── factsheets/           # Factsheet examples (5 files)
│       └── figures/              # Figure generation examples
│
├── notebooks/                    # Jupyter notebooks
├── pyproject.toml               # Modern Python packaging config
├── requirements.txt             # Core dependencies
├── README.md                    # Project documentation
├── LICENSE.txt                  # GPLv3 license
└── MANIFEST.in                  # Package manifest
```

## 2. MAIN MODULES & PURPOSES

### **qis.utils** - Foundation Layer
**Purpose**: Low-level utilities for data manipulation
**Key Responsibilities**:
- Date/time operations (rebalancing, period handling, frequency inference)
- DataFrame operations (alignment, grouping, resampling, cleaning)
- String conversion for DataFrames and series
- NumPy array operations (weighted averages, finite values, etc.)
- OLS regression and sampling utilities
- Weight allocation and scoring functions

**Dependencies**: numpy, pandas, scipy, numba
**Size**: 3,845 LOC across 13 modules

### **qis.perfstats** - Performance Analytics Layer
**Purpose**: Compute and analyze investment performance metrics
**Key Responsibilities**:
- Return calculations (simple, log, annualized, compounded)
- Volatility and risk metrics
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Performance attribution
- Regime classification (Bear/Normal/Bull markets)
- Conditional regression analysis
- Returns time series handling and interpolation

**Key Classes/Enums**:
- `PerfStat`: 100+ performance metrics (returns, volatility, Sharpe, drawdowns, etc.)
- `PerfParams`: Configuration for performance calculations
- `RegimeType`: Market regime classification enums
- `ReturnTypes`: Different return calculation methods

**Size**: 2,753 LOC across 8 modules

### **qis.models** - Statistical Modeling Layer
**Purpose**: Implement statistical models for analysis
**Size**: 4,073 LOC across 12 modules

#### **qis.models.linear** - Linear Models & Time Series
**Purpose**: Exponential weighting, correlations, PCA, regressions
- Exponentially Weighted Moving Average (EWM) - core filtering technique
- Autocorrelation and lagged correlation analysis
- Covariance/correlation matrix computation
- Principal Component Analysis (PCA)
- Risk-adjusted return transformations
- Outlier handling and winsorization

**Key Functions**:
- `compute_ewm()`, `compute_ewm_vol()`, `compute_ewm_covar()`
- `estimate_rolling_ewma_covar()`, `compute_ewm_corr_df()`
- `apply_pca()`, `compute_eigen_portfolio_weights()`

#### **qis.models.stats** - Statistical Analysis
**Purpose**: Bootstrap resampling, rolling statistics, OHLC volatility
- Bootstrap analysis with AR process support
- Rolling performance statistics
- OHLC-based volatility estimation
- High-frequency volatility estimation methods

**Key Functions**:
- `bootstrap_data()`, `bootstrap_price_data()`
- `compute_rolling_perf_stat()`
- `estimate_ohlc_var()`, `estimate_hf_ohlc_vol()`

### **qis.plots** - Visualization Layer
**Purpose**: Comprehensive financial data visualization (Matplotlib/Seaborn wrapper)
**Size**: ~10,000 LOC across 24 modules

#### **qis.plots (base)** - Fundamental Plot Types
**Purpose**: Low-level plotting functions wrapping Matplotlib
- Bar/column charts
- Box plots
- Scatter plots  
- Time series line plots
- Tables/DataFrames
- Heatmaps
- Histograms and PDFs
- Q-Q plots
- Error bars
- Stacked area plots
- Contours
- Pie charts
- 2D histograms

#### **qis.plots.utils** - Plotting Utilities (1,619 LOC)
**Purpose**: Centralized plot configuration and styling
- Color management (colormaps, legends with stats)
- Axis formatting (ticks, labels, limits)
- Trend lines and trend identification
- Legend creation and customization
- Spine manipulation
- Table sizing calculations

#### **qis.plots.derived** - High-Level Composite Plots
**Purpose**: Complex multi-component plots for financial analysis
- **prices.py**: Price charts with drawdowns, fundamentals, performance labels
- **perf_table.py**: Performance matrices, annual returns tables, scatter plots
- **returns_heatmap.py**: Calendar heatmaps, periodic returns tables
- **drawdowns.py**: Drawdown analysis, underwater plots, top drawdowns
- **regime_data.py**: Regime-conditional analysis plots
- **regime_scatter.py**: Regime-based regression scatters
- **returns_scatter.py**: Return distribution analysis
- **regime_pdf.py**: Probability distribution by regime

#### **qis.plots.reports** - Report Generation
**Purpose**: Generate complete factsheet reports
- Economic data visualizations
- Price history reports with gantt charts
- Report utilities and styling

### **qis.portfolio** - Portfolio Analysis & Backtesting (High-Level)
**Purpose**: Integrate all components for portfolio analysis and strategy backtesting
**Size**: ~5,000 LOC core + ~300k LOC in reports

#### **qis.portfolio.portfolio_data** - Core Data Structure (1,645 LOC)
**Purpose**: PortfolioData class - main output of backtesting
**Key Features**:
- Holds: prices, weights, nav, returns, performance metrics
- Enums for attribution metrics and snapshot periods
- Methods for aggregating portfolio performance
- Regime classification
- Risk contribution calculations
- Performance attribution

#### **qis.portfolio.multi_portfolio_data** - Multi-Strategy Analysis (1,121 LOC)
**Purpose**: MultiPortfolioData class for comparing multiple portfolios
- Aggregation across multiple strategies/portfolios
- Cross-portfolio performance comparison
- Peer group analysis

#### **qis.portfolio.backtester** - Backtesting Engine
**Purpose**: Core backtesting function
- `backtest_model_portfolio()`: Main function to simulate strategy
- `backtest_rebalanced_portfolio()`: Rebalancing simulation
- Inputs: prices, weights, rebalancing frequency, costs
- Outputs: PortfolioData with complete performance metrics

#### **qis.portfolio.signal_data** - Signal Analysis (238 LOC)
**Purpose**: StrategySignalData class for signal-based analysis
- Track signal changes, weights, turnover
- Analyze signal composition over time

#### **qis.portfolio.risk** - Risk Analysis (1,317 LOC)
**Purpose**: Risk metrics and factor models
- **factor_model.py (455 LOC)**: Linear factor model attribution
  - Benchmark beta attribution
  - Multi-factor regression
  - Alpha computation
- **ewm_factor_model.py (234 LOC)**: Exponentially weighted factor models
  - Time-varying betas
  - Rolling factor exposures
- **ewm_covar_risk.py (265 LOC)**: EWM-based risk metrics
  - Portfolio VaR
  - Volatility computation
  - Concentration limits
- **contributions.py (363 LOC)**: Risk contribution analysis
  - Marginal contribution to risk
  - Component risk decomposition
  - Benchmark risk attribution

#### **qis.portfolio.reports** - Factsheet Generation (12 modules, ~300k LOC)
**Purpose**: Generate comprehensive strategy performance factsheets
- **config.py (13,148 LOC)**: Factsheet configuration templates
  - FactsheetConfig: Parameterizes report generation
  - Preset configs for daily/monthly/quarterly data
  - Short/long period configurations
  - Reporting frequency enums
- **strategy_factsheet.py (41,017 LOC)**: Single strategy reports
  - Price history with performance
  - Risk metrics tables
  - Rolling Sharpe ratios
  - Drawdown analysis
  - Returns heatmaps by period
- **strategy_benchmark_factsheet.py (31,019 LOC)**: Strategy vs benchmark
  - Active returns analysis
  - Beta attribution
  - Tracking error
  - Relative performance metrics
- **strategy_benchmark_tre_factsheet.py (34,987 LOC)**: Tracking error reports
  - Tracking error decomposition
  - Position-level attribution
  - Exposure analysis
- **multi_strategy_factsheet.py (12,887 LOC)**: Compare multiple strategies
- **multi_assets_factsheet.py (29,616 LOC)**: Asset allocation analysis
- **strategy_signal_factsheet.py (9,923 LOC)**: Signal composition analysis
- **brinson_attribution.py (21,363 LOC)**: Brinson-Fachler attribution
  - Allocation effects
  - Selection effects
  - Total effect decomposition
- **overlays_smart_diversification.py (22,610 LOC)**: Diversification analysis

### **qis.examples** - Example Scripts
**Purpose**: Demonstrate QIS functionality
**Contains**: 36+ example scripts showing:
- Price visualization
- Performance analysis
- Factor models
- Volatility analysis
- Seasonality patterns
- Strategy simulation
- Factsheet generation

## 3. KEY FILES & THEIR ROLES

### Package Initialization
- **`qis/__init__.py`**: Imports and exports all major modules via `from qis.X import *`

### Core Utilities
- **`file_utils.py`**: File I/O (CSV, Excel, Parquet, Feather) and figure saving
- **`local_path.py`**: Local path management from settings.yaml
- **`sql_engine.py`**: Database connectivity (PostgreSQL via psycopg2)
- **`test_data.py`**: Generate synthetic test data

### Configuration Files
- **`pyproject.toml`**: Modern Python packaging (setuptools)
  - Dependencies (numpy, pandas, scipy, statsmodels, matplotlib, seaborn, numba)
  - Optional dependencies (pybloqs for reports, plotly, jupyter, dev tools)
  - Tool configs (black, isort, mypy, pytest, coverage)
- **`requirements.txt`**: Core dependency list
- **`perfstats/config.py`**: Performance metric configurations
- **`portfolio/reports/config.py`**: Factsheet configuration templates

## 4. PACKAGE DEPENDENCIES & IMPORTS

### Core Dependencies (Required)
```
numba >= 0.60.0          # JIT compilation for performance
numpy == 2.2.6           # Numerical arrays
scipy >= 1.15.0          # Scientific computing
statsmodels >= 0.14.0    # Statistical models
pandas >= 2.3.1          # Data manipulation
matplotlib >= 3.9.0      # Visualization
seaborn >= 0.13.0        # Statistical visualization
openpyxl >= 3.1.0        # Excel I/O
tabulate >= 0.9.0        # Table formatting
PyYAML >= 6.0            # YAML config parsing
easydev >= 0.12.0        # Development utilities
psycopg2 >= 2.9.5        # PostgreSQL adapter
pyarrow >= 18.0.0        # Arrow format support
fsspec >= 2024.12.0      # Filesystem abstraction
yfinance >= 0.2.65       # Yahoo Finance data
pandas-datareader >= 0.10.0 # Multiple data sources
setuptools >= 80.9.0     # Package distribution
```

### Optional Dependencies
- **reports**: pybloqs >= 1.2.13, jinja2 >= 3.0.0
- **visualization**: plotly >= 5.0.0
- **jupyter**: jupyter, notebook, jupyterlab, ipykernel, ipywidgets
- **dev**: pytest, pytest-cov, black, flake8, mypy, isort, pre-commit
- **performance**: memory-profiler, line-profiler, py-spy, scalene
- **database**: SQLAlchemy >= 2.0.0

### Internal Module Dependencies (Dependency Path)
```
qis.utils 
  → qis.perfstats (uses utils)
  → qis.models (uses utils)
  → qis.plots (uses utils, perfstats, models)
  → qis.portfolio (uses all above)
  → qis.examples (uses all)
```

### Key External Imports
```python
# Numerical computing
import numba  # @njit decorator for fast computation
import numpy as np
from scipy.stats import kurtosis, skew, linregress

# Data handling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics & modeling
from statsmodels.api import OLS
from sklearn (implicit via scipy)

# Utilities
import yaml  # config loading
from enum import Enum
from dataclasses import dataclass
from typing import (Union, Dict, Tuple, List, Optional, Literal)
```

## 5. TESTING STRUCTURE

### Test Files Found
```
qis/examples/test_ewm.py              # EWM filter tests
qis/examples/test_scatter.py          # Scatter plot tests
qis/models/stats/test_bootstrap.py    # Bootstrap resampling tests (209 LOC)
qis/test_data.py                      # Test data generation utilities
```

### Testing Infrastructure
- **Framework**: pytest >= 7.0.0
- **Configuration**: `pyproject.toml` [tool.pytest.ini_options]
- **Test Discovery**: Finds test_*.py and *_test.py files
- **Markers**: slow, integration, unit test categorization
- **Coverage**: pytest-cov with source="qis"
- **Parallel**: pytest-xdist support
- **Mock**: pytest-mock support

### Test Configuration
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow",
    "integration: marks as integration tests",
    "unit: marks as unit tests",
]
```

### Coverage Configuration
- Excludes: */tests/*, */test_*, setup.py, venv directories
- Exclude lines: pragma no cover, abstractmethod, __main__, etc.

### Example Test Files Overview
1. **test_bootstrap.py** (209 LOC)
   - Tests bootstrap resampling methods
   - AR process generation
   - Confidence interval computation

2. **test_ewm.py**
   - EWM filter validation
   - Covariance computation
   - Edge cases with NaN handling

3. **test_scatter.py**
   - Scatter plot generation
   - Data alignment verification
   - Label placement

## 6. CODE STATISTICS SUMMARY

### By Module Size
```
qis/portfolio/reports/      ~300k LOC  (Factsheet generation)
qis/plots/                  ~10k LOC   (Visualization layer)
qis/models/linear/          3.2k LOC   (Linear models)
qis/perfstats/              2.8k LOC   (Performance analytics)
qis/utils/                  3.8k LOC   (Foundation utilities)
qis/portfolio/ (core)       5.0k LOC   (Portfolio classes)
TOTAL: ~327k LOC
```

### By Functionality
```
Backtesting/Portfolio Analysis    7.1k LOC
Risk Analysis                     1.3k LOC
Performance Statistics            2.8k LOC
Statistical Models               4.1k LOC
Visualization                   10.2k LOC
Utilities                        3.8k LOC
Factsheet Reports              300.0k LOC
```

### Python Version Support
- Python >= 3.8 (supports 3.8-3.13)

### Code Quality Tools
- **Formatting**: Black (line-length=100)
- **Import sorting**: isort (black profile)
- **Type checking**: MyPy (partial enforcement)
- **Linting**: Flake8 (excludes E203, W503, E501)
- **Pre-commit**: Hooks available for automated checks

## 7. KEY ARCHITECTURAL PATTERNS

### Design Patterns Used
1. **Enum-based Configuration**: PerfStat, RegimeType, ReturnTypes
2. **Dataclass Configuration**: FactsheetConfig, PerfParams
3. **NamedTuple for Data**: Various config objects
4. **Factory Functions**: `backtest_model_portfolio()`, `compute_ewm()`
5. **JIT Compilation**: @njit decorator for performance-critical code
6. **Functional Decomposition**: Pure functions for calculations

### Data Flow in Backtesting
```
prices (pd.DataFrame) 
  ↓
backtester.backtest_model_portfolio(prices, weights, ...)
  ↓
PortfolioData (holds navs, weights, returns, perf metrics)
  ↓
portfolio.reports.generate_strategy_factsheet(PortfolioData)
  ↓
Factsheet (charts, tables, analysis)
```

### Computation Strategy
- **Numba JIT**: For numerical loops (EWM, covariance, regressions)
- **NumPy**: Vectorized operations where possible
- **Pandas**: DataFrames for time series and tabular data
- **Matplotlib/Seaborn**: Low-level plotting primitives

## 8. EXTENSION POINTS

### Areas for Custom Development
1. **Custom Performance Metrics**: Extend PerfStat enum and perf_stats.py
2. **New Plot Types**: Add to plots/ module
3. **Custom Models**: Extend models/linear or models/stats
4. **New Risk Metrics**: Extend portfolio/risk/
5. **Custom Factsheets**: Build on portfolio/reports/ templates
6. **Data Sources**: Extend sql_engine.py and file_utils.py

### Common Integration Points
- `PortfolioData`: Central data structure for strategy results
- `backtest_model_portfolio()`: Core backtesting entry point
- `PerfParams`, `FactsheetConfig`: Configuration customization
- Plot functions: Can be composed for custom reports
