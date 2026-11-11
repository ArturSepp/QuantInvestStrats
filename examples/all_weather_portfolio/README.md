# All weather versus 60/40

*Author: [Artur Sepp](https://github.com/ArturSepp)*

[Read the standalone post](post.html) · [Read or edit the Markdown draft](post.md) · [QIS source](https://github.com/ArturSepp/QuantInvestStrats) · [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)

The HTML file can be copied by itself. It embeds all ten displayed figures, both factsheet PDFs, the linked data and the model scripts. The Markdown edition uses ordinary relative paths. External references to the paper and documentation open online.

## Reproduce the article

The original verified run used Python 3.12.14 and [QIS source revision 1525b6f8d5](https://github.com/ArturSepp/QuantInvestStrats/commit/1525b6f8d5fdc9be0e883e10a8139fddd66406ad). The directly used library versions are in [requirements-replication.txt](requirements-replication.txt). The reference script reads the packaged snapshots when there is no input override in the output directory. Use an empty output directory for a clean replay. All dates and windows are fixed in the example.

The corrected Figures 6 and 8 and both factsheet PDFs were generated with Python 3.12.14 and [QIS source revision 4c8c24bd37](https://github.com/ArturSepp/QuantInvestStrats/commit/4c8c24bd376d3b49d2c147ba037fff8b3ca3cb0e) plus the [factsheet warm-up correction](qis_factsheet_warmup.patch). To reproduce from that exact revision, apply the patch once from the QIS repository root with `git apply examples/all_weather_portfolio/qis_factsheet_warmup.patch` and use the resulting checkout. The verifier checks every rolling alpha from independently calculated monthly log returns and lagged betas. All 26 saved CSVs matched the corrected run byte for byte.

From an environment containing the recorded software, run the following from this folder; choose output paths outside the checkout.

These standalone replication CLIs and archived AI comparisons are retained as an article
bundle. The QIS reference runner resolves its output directory through `qis.get_output_path`:
an explicit `--output-dir` takes precedence, otherwise the adjacent QIS YAML configuration
provides `OUTPUT_PATH`. Its PDF writers receive that directory explicitly. The archived AI
callers retain their original conventions and run inside the replay's selected output folders.

```text
python qis_reference.py --with-leverage --output-dir <local-run>
python verify_replication.py <local-run>
python replay_ai.py --output-dir <local-ai-run>
python build_post.py --output <local-output.html>
```

On the maintainer's Windows host, start in the QuantInvestStrats checkout and use the external environment selected by repository governance.

```powershell
$repo = (Get-Location).Path
$ctx = . "$env:USERPROFILE\OneDrive\analytics\my_github\ArturSepp\scripts\repo_governance\Enter-AgentRepo.ps1" -RepoPath $repo -PassThru
$example = Join-Path $repo 'examples\all_weather_portfolio'
$run = Join-Path $ctx.AgentWorkRoot ('runs\all_weather_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $ctx.Python (Join-Path $example 'qis_reference.py') --with-leverage --output-dir $run
if ($LASTEXITCODE) { throw 'QIS run failed' }
& $ctx.Python (Join-Path $example 'verify_replication.py') $run
if ($LASTEXITCODE) { throw 'Numerical verification failed' }
& $ctx.Python (Join-Path $example 'replay_ai.py') --output-dir (Join-Path $run 'ai')
if ($LASTEXITCODE) { throw 'AI replay failed' }
& $ctx.Python (Join-Path $example 'build_post.py') --output (Join-Path $run 'post.html')
```

The renderer reads `post.md` and the completed exhibits in `results/`. To render newly calculated exhibits, copy the example folder to a writable local directory and use its `results/` as the QIS output directory before running `build_post.py`. The standalone HTML can then travel on its own.

## Inputs, outputs and checks

| File or directory | What it reproduces |
|---|---|
| [ETF prices](data/prices.csv) | Original 6,306-row business-day panel of adjusted USD closes for SPY, TLT, TIP, GLD and IEF, 31 July 2002 through 30 September 2026. Missing pre-launch prices are retained. |
| [Annual funding rates](data/ust_3m_irx.csv) | Yahoo `^IRX` three-month bill quote divided by 100. The 2× portfolio accrues the previous available quote on the actual cash balance using ACT/365, with no borrowing spread. |
| [Input hashes](data/SHA256SUMS) | SHA-256 of every frozen input, including the AI experiments' auxiliary data. |
| [Reference script](qis_reference.py) and [results](results/) | The unlevered accounting, all article figures and tables, monthly returns-based risk statistics and both three-page factsheets. `--with-leverage` also produces the funded comparison. |
| [Figure 4 data](results/table_attribution.csv) | Wealth-linked gross holding contributions. [Daily-sum table](results/table_attribution_daily_sum.csv) preserves the other convention for comparison. |
| [Turnover and costs](results/table_annual_turnover_costs.csv), [market correlations](results/table_spy_tlt_correlation.csv), [correlation and beta paths](results/series_correlation_beta.csv) | Each complete year's two-sided turnover and NAV-normalised costs, ordinary subperiod correlations of month-end log returns, and 36-span EW estimates. |
| [Rolling Sharpe](results/series_rolling_sharpe_difference.csv) and [marginal alpha](results/table_marginal_alpha.csv) | Month-end inference from the same QIS method as page two. Positive estimates favour all weather. Each rolling estimate uses a complete five-year window. |
| [Grouped Brinson](results/table_brinson_asset_classes.csv) | Gross linked effects for equity, nominal Treasuries (TLT and IEF), inflation-linked Treasuries and gold; allocation plus selection reconciles to the active return. In the grouped PDF the first two pages are held fixed and the third page reports the asset classes. |
| [Numerical verifier](verify_replication.py) | Independent checks of compounding, turnover, costs, EW correlation, selected holdings betas, every rolling Sharpe estimate, grouped attribution and the funded performance table. |

All 13 baseline CSV tables were reproduced byte for byte, counting the historical daily-sum attribution under its explicit filename. In the original verification, six of the seven original PNGs also reproduced byte for byte; Figure 4 has a corrected title. The alpha/beta coverage notices in [the reference log](results/stderr.txt) document estimator warm-up. Missing asset betas stay unestimated; the figures use available betas at their actual weights.

## AI experiments

The three-model question is reproduced below. All three scripts use the same five eventual ETFs. Each script's own return conventions and proxy choices are retained.

| Model and source | Treatment of the missing sleeves |
|---|---|
| [Claude Opus 5.5 High](claude_opus_5_5_high.py) | Treasury bills until the missing ETFs can be bought at quarter-end. The included `--missing renormalize` sensitivity reallocates over tradable ETFs. |
| [DeepSeek](deepseek.py) | VIPSX and a continuous gold futures series (`GC=F`), replaced when ETF returns become available. |
| [ChatGPT High](chatgpt_high.py) | VIPSX and a bullion price proxy with a 0.40% annual expense assumption; both sides of the quarter-end proxy switches are charged. The FRED request uses Requests' default user agent after the originally supplied header timed out twice. |

[Original three-model outputs](experiments/three_models/) and [the earlier four-prompt comparison](experiments/four_prompts/ai_comparison_table.csv) are archived separately. The earlier experiment includes the prompt wording, replies, scripts and original daily NAV comparison. [Offline replay](replay_ai.py) runs the supplied accounting code with frozen inputs in independent output folders. Claude's original cached adjusted closes and FRED DTB3 are in `data/ai/claude/`. The other snapshots contain the price series specified by each script: Yahoo adjusted ETF/fund closes and quoted `^IRX`, Yahoo continuous gold futures for DeepSeek, and the [historical bullion CSV](https://www.goldprice.com/gold-price-history.csv) and [FRED DGS3MO](https://fred.stlouisfed.org/series/DGS3MO) for ChatGPT. The earlier four strategies replay against common exchange-session adjusted closes. The original, separately downloaded price histories were subject to vendor revisions.

The captured DeepSeek and ChatGPT input panels for replay were retrieved on 11 November 2026 after their initial runs. Their headline metrics and win counts match. Against the original logs the largest difference in DeepSeek's printed complete-year table was 0.01 percentage point, and the maximum absolute difference in ChatGPT's annual CSV was below `9.3e-7` in decimal-return units. The original four-prompt start dates and cumulative returns reproduce to the published precision.

## Prompt used in the three chats

> I want to know when a simple all-weather portfolio did better than a classic 60/40 portfolio, and why.
>
> For the all-weather portfolio, allocate 25% each to US equities, long-term US Treasuries, US inflation-linked Treasuries and gold. For the 60/40 portfolio, allocate 60% to US equities and 40% to intermediate-term US Treasuries. Choose suitable US-listed ETFs and explain your choices.
>
> Study 31 July 2002 through 30 September 2026. Rebalance quarterly and apply a trading cost of 10 basis points to traded notional. Use publicly available data and whatever Python tools you would normally use.
>
> Please answer three questions:
>
> 1. Over the full period, what were the annualised return, volatility, Sharpe ratio and maximum drawdown of each portfolio?
> 2. In which complete calendar years did all weather beat 60/40? Show both portfolios’ returns and the difference in percentage points. Identify any partial years separately.
> 3. In the three complete years with the largest absolute return gaps, which holdings explain why one portfolio beat the other? Quantify their contributions where possible.
>
> Include one self-contained Python script that reproduces your analysis. State the first date each portfolio was invested, its holdings and weights on that date, and how you handled any unavailable price history. Define the Sharpe ratio you report.
>
> If you can execute the script, include its actual output. If you cannot, say so and provide the script so I can run it.
