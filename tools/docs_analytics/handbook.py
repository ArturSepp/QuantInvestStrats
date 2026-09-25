"""Handbook figures: one teaching exhibit per methodology chapter.

Every figure is computed with qis on the frozen synthetic universe (or on an explicitly stated
teaching construction) and each has an independent numerical check: a closed form or a direct
numpy calculation of the plotted quantity. Tables hold the plotted values at full precision so
captions can be traced to the same numbers.
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

import qis
from qis.datasets import generate_synthetic_universe
from tools.docs_analytics.style import MUTED, SERIES, handbook_exhibit


def _universe(params: dict) -> pd.DataFrame:
    """Complete synthetic price panel on the manifest sample."""
    universe = generate_synthetic_universe(start=params['start'], end=params['end'],
                                           seed=params['seed'], apply_quirks=False)
    return universe.prices


def _new_figure(nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a figure with the given axes grid."""
    import matplotlib.pyplot as plt
    return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)


def ewm_kernels(params: dict):
    """EWM weights by lag against equal-weight windows of the same length."""
    spans = params['ewm_spans']
    lags = np.arange(0, 3001)
    impulse = np.zeros(len(lags) + 1)
    impulse[1] = 1.0  # a zero seed and one unit shock: the response is the weight on each lag
    table = pd.DataFrame(index=pd.Index(lags, name='lag'))
    checks = []
    fig, ax = _new_figure()
    shown = lags <= 72
    for color, span in zip(SERIES, spans):
        lam = 1.0 - 2.0 / (span + 1.0)
        weights = qis.compute_ewm(impulse, span=span, init_type=qis.InitType.ZERO)[1:]
        window = np.where(lags < span, 1.0 / span, 0.0)
        checks.append(np.allclose(weights, (1.0 - lam) * lam ** lags, atol=1e-15))
        checks.append(abs(np.sum(lags * weights) - (span - 1) / 2) < 1e-9)
        checks.append(abs(np.sum(weights) ** 2 / np.sum(weights ** 2) - span) < 1e-6)
        table[f'ewm_span_{span}'] = weights
        table[f'window_{span}'] = window
        ax.plot(lags[shown], weights[shown], color=color, linewidth=2.0,
                label=f'EWM, span {span}')
        ax.step(lags[shown], window[shown], where='post', color=color, linewidth=2.0,
                linestyle='--', label=f'Equal window, {span} obs')
        ax.axvline((span - 1) / 2, color=color, linewidth=1.0, linestyle=':')
        ax.annotate(f'mean lag {(span - 1) / 2:g}', xy=((span - 1) / 2, 1.0 / span),
                    xytext=(6, 6), textcoords='offset points', color=MUTED, fontsize=11)
    ax.set_xlabel('Lag in observations')
    ax.set_ylabel('Weight')
    ax.set_xlim(0, 72)
    ax.legend(loc='upper right')
    handbook_exhibit(
        fig, title='Exponential weights and equal windows',
        subtitle='Weights recovered from qis.compute_ewm as the response to a unit shock',
        footer='An EWM of span N and an N-observation window share the mean lag (N-1)/2 '
               'and the effective sample size N.\nDotted lines mark the mean lags.')
    half_lives = {str(span): float(np.log(2) / -np.log(1 - 2 / (span + 1))) for span in spans}
    return fig, table.loc[lags <= 120], all(checks), {'half_lives': half_lives}


def pca_eigenvalues(params: dict):
    """Eigenvalues of the correlation matrix of monthly log returns with the noise edge."""
    prices = _universe(params)
    returns = qis.to_returns(prices, freq='ME', is_log_returns=True, drop_first=True).dropna()
    corr = returns.corr().to_numpy()
    eigenvalues, _ = qis.apply_pca(corr)
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    reference = np.sort(np.linalg.eigvalsh(corr))[::-1]
    n, t = corr.shape[0], len(returns)
    edge = (1.0 + np.sqrt(n / t)) ** 2
    check = bool(np.allclose(eigenvalues, reference, atol=1e-10)
                 and abs(eigenvalues.sum() - n) < 1e-10)
    table = pd.DataFrame({'eigenvalue': eigenvalues, 'share': eigenvalues / n},
                         index=pd.Index(np.arange(1, n + 1), name='component'))
    fig, ax = _new_figure()
    ax.bar(table.index, table['eigenvalue'], color=SERIES[0], width=0.7,
           label='Correlation eigenvalue')
    ax.axhline(edge, color=SERIES[1], linewidth=2.0, linestyle='--',
               label=f'Noise upper edge {edge:.2f}')
    for component, value in table['share'].head(3).items():
        ax.annotate(f'{value:.0%}', xy=(component, table.loc[component, 'eigenvalue']),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11,
                    zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Eigenvalue')
    ax.set_xticks(table.index)
    ax.legend(loc='upper right')
    handbook_exhibit(
        fig, title='How many independent directions?',
        subtitle=f'Synthetic universe | {n} assets | {t} monthly log returns | 2005-2025',
        footer='Bars: eigenvalues of the sample correlation matrix, labelled with their variance '
               'share.\nDashed: (1 + sqrt(n/T))^2, the largest eigenvalue expected from '
               'uncorrelated noise.')
    return fig, table, check, {'noise_edge': float(edge), 'first_share': float(table['share'].iloc[0])}


def smoothed_acf(params: dict):
    """Autocorrelation of a liquid series and of its AR(1)-smoothed version."""
    prices = _universe(params)
    liquid = qis.to_returns(prices['SEQ_US'], freq='ME', is_log_returns=True,
                            drop_first=True).dropna()
    phi = params['smoothing_phi']
    smoothed = np.empty(len(liquid))
    smoothed[0] = liquid.iloc[0]
    for i in range(1, len(liquid)):
        smoothed[i] = (1.0 - phi) * liquid.iloc[i] + phi * smoothed[i - 1]
    panel = pd.DataFrame({'Liquid (SEQ_US)': liquid.to_numpy(),
                          f'Smoothed, phi = {phi}': smoothed}, index=liquid.index)
    num_lags = params['acf_lags'] + 1
    acf = qis.compute_autocorr_df(panel, num_lags=num_lags)
    reference = pd.DataFrame({column: [1.0] + [
        np.corrcoef(panel[column].to_numpy()[k:], panel[column].to_numpy()[:-k])[0, 1]
        for k in range(1, num_lags)] for column in panel.columns}, index=acf.index)
    check = bool(np.allclose(acf.to_numpy(), reference.to_numpy(), atol=1e-12))
    band = 1.96 / np.sqrt(len(panel))
    table = acf.iloc[1:]
    fig, ax = _new_figure()
    width = 0.38
    for offset, color, column in zip((-width / 2, width / 2), SERIES, table.columns):
        ax.bar(table.index + offset, table[column], width=width, color=color, label=column)
    ax.plot(table.index, phi ** table.index.to_numpy(), color=MUTED, linewidth=1.5,
            linestyle=':', marker='o', markersize=4, label=f'AR(1) theory: {phi}^k')
    for level in (band, -band):
        ax.axhline(level, color=MUTED, linewidth=1.0, linestyle='--')
    ax.set_xlabel('Lag in months')
    ax.set_ylabel('Autocorrelation')
    ax.set_xticks(table.index)
    ax.legend(loc='upper right')
    handbook_exhibit(
        fig, title='Smoothing creates autocorrelation',
        subtitle=f'Synthetic monthly log returns | 2005-2025 | teaching AR(1) smoothing, '
                 f'phi = {phi}',
        footer=f'Dashed: approximate 95% band for white noise, +/-1.96/sqrt(T) = +/-{band:.3f}.'
               '\nThe smoothed series is (1-phi) r_t + phi times its own previous value.')
    return fig, table, check, {'band': float(band),
                               'smoothed_lag1': float(table.iloc[0, 1])}


def sharpe_wedge(params: dict):
    """Arithmetic minus p.a. Sharpe ratio against annualised volatility."""
    prices = _universe(params)
    table_all = qis.compute_ra_perf_table(
        prices=prices, perf_params=qis.PerfParams(freq='ME', return_type=qis.ReturnTypes.LOG))
    arith = table_all[qis.PerfStat.SHARPE_ARITH.to_str()]
    pa = table_all[qis.PerfStat.SHARPE_RF0.to_str()]
    vol = table_all[qis.PerfStat.VOL.to_str()]
    simple = qis.to_returns(prices, freq='ME', is_log_returns=False, drop_first=True)
    check = bool(np.allclose(arith, np.sqrt(12.0) * simple.mean() / simple.std(), atol=1e-12))
    table = pd.DataFrame({'sharpe_arith': arith, 'sharpe_pa': pa, 'vol': vol,
                          'wedge': arith - pa, 'half_vol': vol / 2.0})
    fig, ax = _new_figure()
    grid = np.linspace(0.0, float(vol.max()) * 1.08, 50)
    ax.plot(grid, grid / 2.0, color=SERIES[1], linewidth=2.0, linestyle='--',
            label='First-order drag, vol / 2')
    ax.scatter(table['vol'], table['wedge'], s=64, color=SERIES[0], zorder=3,
               edgecolor='white', linewidth=1.5, label='Asset')
    scaled = table[['vol', 'wedge']] / table[['vol', 'wedge']].max()
    for name, row in table.iterrows():
        distance = np.hypot(*(scaled.drop(index=name) - scaled.loc[name]).to_numpy().T)
        neighbour = scaled.drop(index=name).index[int(np.argmin(distance))]
        below = distance.min() < 0.1 and row['wedge'] < table.loc[neighbour, 'wedge']
        ax.annotate(name, xy=(row['vol'], row['wedge']), xytext=(6, -13 if below else 4),
                    textcoords='offset points', fontsize=10, color=MUTED)
    ax.set_xlabel('Annualised volatility of monthly log returns')
    ax.set_ylabel('Arithmetic minus p.a. Sharpe ratio')
    ax.legend(loc='upper left')
    handbook_exhibit(
        fig, title='The volatility drag between Sharpe conventions',
        subtitle='Synthetic universe | monthly returns | 2005-2025 | zero cash rate',
        footer='Points: SHARPE_ARITH minus SHARPE_RF0 from qis.compute_ra_perf_table. Dashed: '
               'the first-order\nwedge sigma/2 implied by mean(log r) = mean(r) - var(r)/2.')
    return fig, table, check, {'max_abs_gap_to_half_vol': float(
        (table['wedge'] - table['half_vol']).abs().max())}


def drawdown_path(params: dict):
    """A NAV and its running drawdown from the peak."""
    prices = _universe(params)['SEQ_US']
    drawdown = qis.compute_rolling_drawdowns(prices)
    reference = prices / prices.cummax() - 1.0
    check = bool(np.allclose(drawdown.to_numpy(), reference.to_numpy(), atol=1e-12))
    table = pd.DataFrame({'price': prices, 'drawdown': drawdown})
    fig, (top, bottom) = _new_figure(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': [1.6, 1.0]})
    top.plot(prices.index, prices, color=SERIES[0], linewidth=1.5, label='NAV')
    top.plot(prices.index, prices.cummax(), color=MUTED, linewidth=1.0, linestyle='--',
             label='Running peak')
    top.set_ylabel('NAV')
    top.legend(loc='upper left')
    bottom.fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color=SERIES[1],
                        linewidth=0.0)
    trough = drawdown.idxmin()
    bottom.annotate(f'max drawdown {drawdown.min():.1%}', xy=(trough, drawdown.min()),
                    xytext=(10, 4), textcoords='offset points', fontsize=11)
    bottom.set_ylabel('Drawdown')
    bottom.yaxis.set_major_formatter(_percent())
    handbook_exhibit(
        fig, title='Drawdown from the running peak',
        subtitle='Synthetic US equity (SEQ_US) | daily closes | 2005-2025',
        footer='D_t = P_t / max P up to t - 1, from qis.compute_rolling_drawdowns. The lower '
               'panel shows\nhow deep and how long each episode is below the previous peak.')
    return fig, table.resample('W-FRI').last(), check, {'max_drawdown': float(drawdown.min()),
                                                        'trough': str(trough.date())}


def regime_sharpe(params: dict):
    """Additive regime contributions to the arithmetic Sharpe ratio on the regime grid."""
    prices = _universe(params)[params['regime_assets']]
    benchmark = params['regime_assets'][0]
    classifier = qis.BenchmarkReturnsQuantilesRegime(freq='QE')
    perf_params = qis.PerfParams(freq='ME', sharpe_convention=qis.SharpeConvention.ARITHMETIC)
    perf_table, _ = classifier.compute_regimes_pa_perf_table(
        prices=prices, benchmark=benchmark, perf_params=perf_params)
    regimes = ['Bear', 'Normal', 'Bull']
    table = pd.DataFrame({regime: perf_table[f'{regime}-Sharpe'].astype(float)
                          for regime in regimes})
    table['Total'] = table.sum(axis=1)
    sampled = classifier.compute_sampled_returns_with_regime_id(
        prices=prices, benchmark=benchmark, include_start_date=True, include_end_date=True)
    returns = sampled[prices.columns].astype(float)
    reference = 2.0 * returns.mean() / returns.std()
    check = bool(np.allclose(table['Total'], reference[table.index], atol=1e-10))
    fig, ax = _new_figure()
    x = np.arange(len(table))
    width = 0.26
    for offset, color, regime in zip((-width, 0.0, width), SERIES, regimes):
        ax.bar(x + offset, table[regime], width=width, color=color, label=regime)
    ax.scatter(x, table['Total'], color='black', marker='D', s=40, zorder=3,
               label='Sum = Sharpe ratio')
    ax.axhline(0.0, color=MUTED, linewidth=1.0)
    ax.set_xticks(x, table.index)
    ax.set_ylabel('Contribution to the Sharpe ratio')
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.35 * (high - low))
    ax.legend(loc='upper left', ncol=4)
    handbook_exhibit(
        fig, title='Where the Sharpe ratio is earned',
        subtitle=f'Synthetic universe | quarterly returns | regimes by {benchmark} quantiles '
                 '16% / 84% | 2005-2025',
        footer='Bars: sqrt(AN) p_g m_g / s(r) for each regime under SharpeConvention.ARITHMETIC. '
               'Diamonds: their\nsum, equal to the arithmetic Sharpe ratio of the same quarterly '
               'returns.')
    return fig, table, check, {'totals': {k: float(v) for k, v in table['Total'].items()}}


def bootstrap_frequencies(params: dict):
    """Draw frequency by source position for truncating and circular stationary blocks."""
    from examples.models.bootstrap_convention import draw_truncating_indices
    n, paths, length, block, seed = (params['bootstrap_n'], params['bootstrap_paths'],
                                     params['bootstrap_n'], params['bootstrap_block'],
                                     params['bootstrap_seed'])
    legacy = draw_truncating_indices(num_data_index=n, num_samples=paths, index_length=length,
                                     block_size=block, seed=seed)
    circular = qis.generate_bootstrapped_indices(
        num_data_index=n, bootstrap_type=qis.BootstrapType.STATIONARY, num_samples=paths,
        index_length=length, block_size=block, min_block_size=1, seed=seed)
    table = pd.DataFrame({
        'Truncating (before 5.1.0)': np.bincount(legacy.ravel(), minlength=n) * n / legacy.size,
        'Circular (qis 5.1.0+)': np.bincount(circular.ravel(), minlength=n) * n / circular.size,
    }, index=pd.Index(np.arange(n), name='source_position'))
    check = bool(np.allclose(table.mean(), 1.0, atol=1e-12))
    fig, ax = _new_figure()
    for color, style, column in zip(SERIES, ('-', '--'), table.columns):
        ax.plot(table.index, table[column], color=color, linewidth=2.0, linestyle=style,
                label=column)
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=':')
    ax.set_xlabel('Source observation')
    ax.set_ylabel('Relative draw frequency')
    ax.legend(loc='lower right')
    handbook_exhibit(
        fig, title='Circular blocks sample every observation evenly',
        subtitle=f'Stationary bootstrap | n = {n} | {paths} paths | mean block {block} | '
                 f'seed {seed}',
        footer='Frequency 1 is the uniform expected count. Truncating blocks at the end of the '
               'sample\nunder-draws early observations; wrapping circularly removes the bias.')
    return fig, table, check, {
        'first_observation': {k: float(v) for k, v in table.iloc[0].items()}}


def fee_navs(params: dict):
    """Gross and net NAV under management and performance fees with a high-water mark."""
    gross = _universe(params)['SEQ_US']
    gross = gross / gross.iloc[0] * 100.0
    # the function returns a NAV that starts at one; rebase it to the gross starting level
    net = qis.compute_net_navs_ex_perf_man_fees(
        navs=gross, man_fee=params['man_fee'], perf_fee=params['perf_fee'],
        perf_fee_frequency='YE') * gross.iloc[0]
    no_fee = qis.compute_net_navs_ex_perf_man_fees(navs=gross, man_fee=0.0, perf_fee=0.0,
                                                   perf_fee_frequency='YE') * gross.iloc[0]
    check = bool(np.allclose(no_fee.to_numpy(), gross.to_numpy(), rtol=1e-12)
                 and (net.to_numpy() <= gross.to_numpy() * (1.0 + 1e-12)).all()
                 and net.iloc[-1] < gross.iloc[-1])
    table = pd.DataFrame({'gross': gross, 'net': net})
    fig, ax = _new_figure()
    ax.plot(gross.index, gross, color=SERIES[0], linewidth=1.5, label='Gross NAV')
    ax.plot(net.index, net, color=SERIES[1], linewidth=1.5, linestyle='--',
            label=f'Net of {params["man_fee"]:.0%} management and '
                  f'{params["perf_fee"]:.0%} performance fees')
    ax.set_ylabel('NAV, start = 100')
    ax.legend(loc='upper left')
    handbook_exhibit(
        fig, title='Fees compound into the NAV',
        subtitle='Synthetic US equity (SEQ_US) | daily | annual crystallisation above the '
                 'high-water mark',
        footer='Net NAV from qis.compute_net_navs_ex_perf_man_fees. Performance fees accrue only '
               'above the\nhigh-water mark, so they are paid in recovering years, not in drawdowns.')
    return fig, table.resample('W-FRI').last(), check, {
        'terminal_gross': float(gross.iloc[-1]), 'terminal_net': float(net.iloc[-1])}


def benchmark_regression(params: dict):
    """Monthly asset returns against benchmark returns with the OLS line."""
    prices = _universe(params)[[params['benchmark'], params['regression_asset']]]
    perf = qis.compute_ra_perf_table_with_benchmark(
        prices=prices, benchmark=params['benchmark'],
        perf_params=qis.PerfParams(freq='ME', freq_reg='ME'))
    row = perf.loc[params['regression_asset']]
    beta = float(row[qis.PerfStat.BETA.to_str()])
    alpha = float(row[qis.PerfStat.ALPHA.to_str()])
    r2 = float(row[qis.PerfStat.R2.to_str()])
    returns = qis.to_returns(prices, freq='ME', is_log_returns=False, drop_first=True).dropna()
    x = returns[params['benchmark']].to_numpy()
    y = returns[params['regression_asset']].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    check = bool(abs(slope - beta) < 1e-10 and abs(intercept - alpha) < 1e-10
                 and abs(np.corrcoef(x, y)[0, 1] ** 2 - r2) < 1e-10)
    table = returns.rename(columns={params['benchmark']: 'benchmark',
                                    params['regression_asset']: 'asset'})
    fig, ax = _new_figure()
    ax.scatter(x, y, s=24, color=SERIES[0], alpha=0.8, edgecolor='white', linewidth=0.5,
               label='Monthly returns')
    grid = np.linspace(x.min(), x.max(), 50)
    ax.plot(grid, alpha + beta * grid, color=SERIES[1], linewidth=2.0,
            label=f'OLS: beta {beta:.2f}, alpha {12 * alpha:.1%} p.a., $R^2$ {r2:.2f}')
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.axvline(0.0, color=MUTED, linewidth=0.8)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(_percent())
    ax.yaxis.set_major_formatter(_percent())
    ax.set_xlabel(f'Benchmark return ({params["benchmark"]})')
    ax.set_ylabel(f'Asset return ({params["regression_asset"]})')
    ax.legend(loc='upper left')
    handbook_exhibit(
        fig, title='Beta, alpha and the regression line',
        subtitle='Synthetic universe | monthly simple returns | 2005-2025',
        footer='Line: the OLS fit reported as BETA and ALPHA by '
               'qis.compute_ra_perf_table_with_benchmark\nwith freq_reg=\'ME\'. Alpha is '
               'annualised linearly as AN x alpha.')
    return fig, table, check, {'beta': beta, 'alpha_annualised': 12 * alpha, 'r2': r2}


def risk_contributions(params: dict):
    """Capital weights against Euler risk-contribution shares for a balanced allocation."""
    prices = _universe(params)[list(params['allocation'])]
    weights = pd.Series(params['allocation'], dtype=float)
    returns = qis.to_returns(prices, freq='ME', is_log_returns=True, drop_first=True).dropna()
    covar = 12.0 * returns.loc[params['risk_start']:].cov()
    contributions = qis.compute_portfolio_risk_contributions(w=weights, covar=covar)
    vol = float(np.sqrt(weights @ covar @ weights))
    marginal = covar.to_numpy() @ weights.to_numpy() / vol
    check = bool(abs(float(contributions.sum()) - vol) < 1e-12
                 and np.allclose(contributions.to_numpy(), weights.to_numpy() * marginal,
                                 atol=1e-14))
    table = pd.DataFrame({'weight': weights, 'risk_contribution': contributions,
                          'risk_share': contributions / vol})
    fig, ax = _new_figure()
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2, table['weight'], width=width, color=SERIES[0], label='Capital weight')
    ax.bar(x + width / 2, table['risk_share'], width=width, color=SERIES[1],
           label='Share of portfolio volatility')
    for position, value in zip(x + width / 2, table['risk_share']):
        ax.annotate(f'{value:.0%}', xy=(position, value), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=11)
    ax.set_xticks(x, table.index)
    ax.yaxis.set_major_formatter(_percent())
    ax.legend(loc='upper right')
    handbook_exhibit(
        fig, title='Capital weights are not risk weights',
        subtitle=f'Synthetic balanced allocation | covariance of monthly log returns '
                 f'{params["risk_start"][:4]}-2025 | portfolio vol {vol:.1%}',
        footer='Euler contributions w_i (Sigma w)_i / sigma from '
               'qis.compute_portfolio_risk_contributions sum to the\nportfolio volatility; '
               'shares are contributions divided by it.')
    return fig, table, check, {'portfolio_vol': vol,
                               'equity_risk_share': float(table['risk_share'].iloc[0])}


def vol_targeting(params: dict):
    """Rolling realised volatility of a regime-scaled asset and of its volatility-targeted version.

    The synthetic equity has nearly constant volatility, so the teaching series multiplies its
    daily returns by an explicit volatility regime (higher in the two stress windows listed in the
    manifest). ``compute_ra_returns`` compares ``vol_target`` with an unannualised EWM volatility,
    so the annual target is passed in per-period units, target / sqrt(252).
    """
    prices = _universe(params)['SEQ_US']
    base = qis.to_returns(prices, freq='B', is_log_returns=False, drop_first=True).dropna()
    multiplier = pd.Series(1.0, index=base.index)
    for start, end, level in params['vol_regimes']:
        multiplier.loc[start:end] = level
    returns = base * multiplier
    per_period_target = params['vol_target'] / np.sqrt(252.0)
    targeted, weights, _ = qis.compute_ra_returns(returns=returns.to_frame(),
                                                  span=params['vol_span'],
                                                  vol_target=per_period_target)
    targeted = targeted.iloc[:, 0].dropna()
    raw = returns.loc[targeted.index]
    window = 63
    rolling = pd.DataFrame({
        'Asset with volatility regimes': raw.rolling(window).std() * np.sqrt(252.0),
        f'Targeted to {params["vol_target"]:.0%}': targeted.rolling(window).std() * np.sqrt(252.0),
    }).dropna()
    lagged = weights.iloc[:, 0].reindex(targeted.index)
    check = bool(np.allclose(targeted, raw * lagged, atol=1e-15)
                 and rolling.iloc[:, 1].std() < 0.5 * rolling.iloc[:, 0].std()
                 and abs(rolling.iloc[:, 1].median() - params['vol_target']) < 0.02)
    fig, ax = _new_figure()
    for color, style, column in zip(SERIES, ('-', '--'), rolling.columns):
        ax.plot(rolling.index, rolling[column], color=color, linewidth=1.5, linestyle=style,
                label=column)
    ax.axhline(params['vol_target'], color=MUTED, linewidth=1.0, linestyle=':')
    ax.yaxis.set_major_formatter(_percent())
    ax.set_ylabel('Trailing three-month realised volatility')
    ax.legend(loc='upper left')
    handbook_exhibit(
        fig, title='Volatility targeting stabilises realised risk',
        subtitle=f'Synthetic US equity scaled by a teaching volatility regime | daily | '
                 f'EWM span {params["vol_span"]} | one-day weight lag',
        footer='Targeted returns from qis.compute_ra_returns: w_t = target / sigma_(t-1), with '
               'the target in per-period units.\nSpikes remain where the EWM estimate lags a '
               'sudden rise in volatility.')
    summary = {column: {'median': float(rolling[column].median()),
                        'std': float(rolling[column].std())} for column in rolling.columns}
    return fig, rolling.resample('W-FRI').last(), check, summary


def _percent():
    """Percent tick formatter."""
    from matplotlib.ticker import PercentFormatter
    return PercentFormatter(xmax=1.0, decimals=0)


FIGURES = {
    'handbook_ewm_kernels.png': ('ewm_kernels', ewm_kernels),
    'handbook_pca_eigenvalues.png': ('pca_eigenvalues', pca_eigenvalues),
    'handbook_smoothed_acf.png': ('smoothed_acf', smoothed_acf),
    'handbook_sharpe_wedge.png': ('sharpe_wedge', sharpe_wedge),
    'handbook_drawdown.png': ('drawdown_path', drawdown_path),
    'handbook_regime_sharpe.png': ('regime_sharpe', regime_sharpe),
    'handbook_bootstrap_frequencies.png': ('bootstrap_frequencies', bootstrap_frequencies),
    'handbook_fee_navs.png': ('fee_navs', fee_navs),
    'handbook_benchmark_regression.png': ('benchmark_regression', benchmark_regression),
    'handbook_risk_contributions.png': ('risk_contributions', risk_contributions),
    'handbook_vol_targeting.png': ('vol_targeting', vol_targeting),
}


def produce(spec: dict) -> dict:
    """Return all handbook figures, their supporting tables and independent checks."""
    params = spec['parameters']
    figures, tables, checks, summary = {}, {}, {}, {}
    for filename, (name, builder) in FIGURES.items():
        figure, table, check, details = builder(params)
        figures[filename] = figure
        tables[name] = table
        checks[name] = bool(check)
        summary[name] = details
    return {'figures': figures, 'tables': tables, 'checks': checks,
            'parameters': params, 'summary': summary}
