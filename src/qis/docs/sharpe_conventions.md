# Sharpe ratio conventions

This packaged note summarises the Sharpe conventions of qis. The definitions, proofs, standard
errors, regime decomposition and a worked example are in the handbook chapter
[Sharpe ratios: conventions and inference](https://quantinveststrats.readthedocs.io/en/latest/performance_analytics_and_sharpe.html).
Symbols follow the handbook's
[Notation and conventions](https://quantinveststrats.readthedocs.io/en/latest/notation_and_conventions.html):
`AN` is the annualisation factor of the sampling grid, `r` simple returns, `l` log returns and
`s(x)` a sample standard deviation.

## The three table conventions

`compute_ra_perf_table` computes all six columns on every call. Returns are sampled on
`PerfParams.freq_vol`; `R_pa` is the compound per-annum return between the first and last
complete `freq_vol` boundaries, and `VOL = sqrt(AN) * s(v)` with `v` the
`PerfParams.return_type` returns, log returns by default.

| Convention | Columns | Formula |
|---|---|---|
| Per annum (compound) | `SHARPE_RF0`, `SHARPE_EXCESS` | `R_pa / VOL` |
| Log | `SHARPE_LOG_AN`, `SHARPE_LOG_EXCESS` | `log(1 + R_pa) / VOL` |
| Arithmetic | `SHARPE_ARITH`, `SHARPE_ARITH_EXCESS` | `sqrt(AN) * mean(r) / s(r)` on simple returns |

The excess columns replace the numerator by its excess counterpart when `rates_data` is given;
without it they equal the zero-rate columns. Reporting presets and factsheets show the per-annum
convention. The arithmetic pair must be selected by name.

## How far apart they are

The volatility drag `mean(l) ≈ mean(r) - s(r)^2 / 2` puts the per-annum and log ratios below the
arithmetic ratio by about half the annualised volatility:

    SR_log - SR_arith ≈ -sigma / 2
    SR_pa  - SR_arith ≈ -(sigma / 2) * (1 - SR_log^2)

The wedge is about 0.05 at 10% volatility and 0.10 at 20%. Rankings change only when the more
volatile asset leads on the arithmetic ratio by less than half the volatility difference. Quote
the convention with every Sharpe ratio, and compare ratios only within one convention and one
sampling grid.

## Where `SharpeConvention` applies

`PerfParams.sharpe_convention` selects the convention of regime-conditional Sharpe ratios only:
`compute_bnb_regimes_pa_perf_table`, the regime classifiers and `plot_regime_data`. It does not
change the six table columns. Under `SharpeConvention.ARITHMETIC` and `LOG` the regime
contributions `sqrt(AN) * p_g * mean_g / s` add up exactly to the Sharpe ratio on the regime
grid. Under `SharpeConvention.PA`, the default, the per-regime per-annum returns are patched so
that they add up to the per-annum return, and the residual is allocated in proportion to regime
frequencies. The derivation is in the handbook chapter
[Regime-conditional performance](https://quantinveststrats.readthedocs.io/en/latest/regime_conditional_performance.html).

## Other Sharpe-type estimators

Rolling Sharpe ratios (`compute_rolling_perf_stat` with `RollingPerfStat.SHARPE`), EWM Sharpe
ratios (`compute_ewm_sharpe`), information ratios (`compute_te_ir_errors`) and model-layer Sharpe
contributions each use their own formula and do not read `PerfParams.sharpe_convention`. The
handbook chapter tabulates every one of them.
