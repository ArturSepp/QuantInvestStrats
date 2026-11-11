# All-weather or 60/40? A portfolio backtest with qis

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-11-11](https://github.com/ArturSepp/QuantInvestStrats/commit/060c41685f8cc50ee28c1f108f575f702a0ebbf6)*

Implemented in [qis](https://github.com/ArturSepp/QuantInvestStrats). Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Reproduce the article

The article compares quarterly rebalanced all-weather and 60/40 portfolios from 31 July 2002 to 30 September 2026. The reference run used Python 3.12.14 and [qis 6.0.4](https://github.com/ArturSepp/QuantInvestStrats/tree/v6.0.4). The saved prices and Treasury bill rates fix the inputs for both the unlevered portfolios and the funded 2× comparison with SPY.

1. Save the [reference script](https://artursepp.com/wp-content/uploads/2026/11/aw6040-qis_reference_604.py_.txt) as `qis_reference_604.py`. Create a folder named `data` beside the script. Save the [ETF prices](https://raw.githubusercontent.com/ArturSepp/QuantInvestStrats/060c41685f8cc50ee28c1f108f575f702a0ebbf6/examples/all_weather_portfolio/data/prices.csv) there as `prices.csv` and the [Treasury bill rates](https://raw.githubusercontent.com/ArturSepp/QuantInvestStrats/060c41685f8cc50ee28c1f108f575f702a0ebbf6/examples/all_weather_portfolio/data/ust_3m_irx.csv) as `ust_3m_irx.csv`. The standalone HTML embeds all three downloads. In the example checkout these files are already in place.
2. With Git available, install the tagged QIS source in your Python 3.12 research environment. This commit is the one tagged `v6.0.4`.

   ```text
   python -m pip install "qis @ git+https://github.com/ArturSepp/QuantInvestStrats.git@e805adc3640d0a9ff66d0a43e5f3d808e1b8c6a8"
   ```

3. From the folder containing the script, run the following command. Choose a new or empty output directory outside the checkout. An existing `prices.csv` or `ust_3m_irx.csv` in that directory takes precedence over the copy in `data/`.

   ```text
   python qis_reference_604.py --with-leverage --output-dir <output-folder>
   ```

## Check the outputs

The script writes the article figures, CSV tables and `aw6040_factsheet.pdf`. The PDF has two pages. Page one compares the portfolios’ performance, exposures, turnover and costs. Page two reports the paired Sharpe difference and marginal alpha. The `--with-leverage` option adds Figure 7 and `table_leverage_spy.csv`, the risk-adjusted performance table with SPY as benchmark.

The full price panel has 6,306 business-day rows. Missing prices before ETF launches are retained. Drawdowns use daily NAVs. The headline Sharpe ratio divides compound annual return by monthly-return volatility annualised at 12 months per year. The funded table uses excess returns over the Treasury bill rate.

Figure 6 displays correlation and beta from 31 August 2005 after 36 monthly returns of warm-up. The downloadable `series_correlation_beta.csv` also contains earlier correlation estimates. Rolling alpha on factsheet page one starts on 30 September 2005 using the preceding month’s beta.

In the repository example, `python verify_replication.py <output-folder>` checks the inputs, portfolio accounting and inference against independent calculations. To rebuild the standalone article from the Markdown and its linked exhibits, run `python build_post.py` from this folder. The resulting `post.html` embeds the nine figures, the factsheet, the reference script, the guide and the linked CSV files.
