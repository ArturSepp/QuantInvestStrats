"""Execute repaired reader workflows with offline, Yahoo-shaped synthetic inputs."""
import io
import importlib
import re
import runpy
import socket
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from qis.datasets.synthetic import generate_synthetic_prices

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    not ROOT.joinpath('examples').is_dir(), reason='examples are not shipped in the wheel')


@pytest.fixture
def yahoo_panel(monkeypatch, tmp_path):
    """Replace the download boundary, retaining the single-ticker MultiIndex contract."""
    prices = generate_synthetic_prices(apply_quirks=False)

    def download(tickers, **kwargs):
        names = tickers.split() if isinstance(tickers, str) else list(tickers)
        close = pd.DataFrame({
            name: prices.iloc[:, sum(map(ord, name)) % prices.shape[1]] for name in names
        })
        if '^IRX' in close:
            close['^IRX'] = 4.0
        return pd.concat({'Open': close * 0.999, 'Close': close}, axis=1)

    def refuse(*args, **kwargs):
        raise OSError('example regression tests must run offline')

    vendor = ModuleType('yfinance')
    vendor.download = download
    monkeypatch.setitem(sys.modules, 'yfinance', vendor)
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    yield prices
    plt.close('all')


@pytest.mark.parametrize('module', [
    'case_studies.vix_beta_to_equities_bonds',
    'models.bootstrap_analysis',
    'models.overnight_intraday_returns',
    'perfstats.miss_best_worst_days_impact',
    'portfolios.constant_notional_short',
    'portfolios.leveraged_etf_strategies',
    'perfstats.quickstart',
])
def test_yahoo_workflow(module, yahoo_panel, monkeypatch, tmp_path):
    """Real calculation/rendering code accepts a one-column Yahoo Close DataFrame."""
    output = tmp_path / 'new' / 'reports'
    monkeypatch.setattr(sys, 'argv', [module, '--output-dir', str(output)])
    namespace = runpy.run_module(f'examples.{module}', run_name='__main__')
    assert plt.get_fignums(), f'{module} did not produce a figure'
    for number in plt.get_fignums():
        plt.figure(number).canvas.draw()
    if module == 'models.overnight_intraday_returns':
        overnight, intraday, daily = namespace['compute_returns']()
        np.testing.assert_allclose((1 + overnight) * (1 + intraday), 1 + daily,
                                   equal_nan=True, rtol=1e-12)
    if module == 'portfolios.leveraged_etf_strategies':
        assert isinstance(namespace['funding_rate'], pd.Series)
        np.testing.assert_allclose(namespace['funding_rate'], 0.05)
    if module in ('perfstats.quickstart', 'models.bootstrap_analysis'):
        assert list(output.glob('*')), 'the explicit output directory was not used'


def test_unsmoothing_narrative(yahoo_panel, capsys):
    """The sample numbers and range printed by the example support its revised interpretation."""
    pytest.importorskip('pyarrow')
    namespace = runpy.run_module(
        'examples.perfstats.unsmoothing_and_delevering', run_name='__main__'
    )
    output = capsys.readouterr().out
    sharpes = [float(value) for value in re.findall(r'Sharpe = ([+-]\d+\.\d+)', output)]
    np.testing.assert_allclose(sharpes[:5], [-1.25, -0.74, 2.25, 2.33, 2.84], atol=0.005)
    assert 'Raw Sharpe range: 3.50; adjusted range: 3.58' in output
    assert 'do not converge to a tighter band' in namespace['__doc__']


@pytest.mark.parametrize('days', [5, 22, 63])
def test_short_history_factsheet(yahoo_panel, days):
    """Short samples omit unavailable panels; longer samples render daily rolling statistics."""
    namespace = runpy.run_module('examples.discrete_portfolio.discrete_trend_backtest')
    daily = yahoo_panel.iloc[-days:, :2].copy()
    daily.columns = ['Momentum', 'SPY buy-and-hold']
    fig = namespace['generate_nav_factsheet'](daily)
    fig.canvas.draw()
    beta_axes = [ax for ax in fig.axes if 'rolling Beta' in ax.get_title()]
    if days < 63:
        assert not beta_axes
        assert len(fig.axes) == 3
        assert any('short-history' in text.get_text() for text in fig.texts)
    else:
        assert beta_axes and 'B-freq' in beta_axes[0].get_title()
        rolling_axes = [ax for ax in fig.axes if ax.get_title().startswith('Rolling ')
                        and ('Sharpe' in ax.get_title() or 'Vol' in ax.get_title())]
        assert len(rolling_axes) == 2
        for ax in [*beta_axes, *rolling_axes]:
            assert any(np.isfinite(line.get_ydata()).sum() > 1 for line in ax.lines)
            assert 'nan' not in ' '.join(text.get_text() for text in ax.get_legend().get_texts())


def test_cboe_public_csv(yahoo_panel, monkeypatch):
    """The documented public CSV is used without an undistributed resource file."""
    original_read_csv = pd.read_csv
    csv = yahoo_panel.iloc[:, :1].rename(columns={yahoo_panel.columns[0]: 'SVRPO'})
    csv.index.name = 'DATE'
    urls = []

    def read_csv(path, *args, **kwargs):
        if str(path).startswith('https://cdn.cboe.com/'):
            urls.append(path)
            path = io.StringIO(csv.to_csv(date_format='%m/%d/%Y'))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', read_csv)
    monkeypatch.setattr(sys, 'argv', ['cboe_vol_strats_perf'])
    runpy.run_module('examples.perfstats.cboe_vol_strats_perf', run_name='__main__')
    assert len(urls) == 1
    for number in plt.get_fignums():
        plt.figure(number).canvas.draw()


def test_fx_example_public_default(yahoo_panel, monkeypatch):
    """The default local-rate workflow needs no production CSV universe."""
    monkeypatch.setattr(sys, 'argv', ['fx_hedging_example'])
    runpy.run_module('examples.market_data.fx_hedging_example', run_name='__main__')


@pytest.mark.parametrize('missing_ticker', ['USDCAD=X', '^IRX', 'SPY'])
def test_fx_example_rejects_missing_inputs(missing_ticker, yahoo_panel, monkeypatch):
    """A partial vendor response cannot silently become an incomplete teaching panel."""
    vendor = sys.modules['yfinance']
    download = vendor.download

    def partial_download(*args, **kwargs):
        frame = download(*args, **kwargs)
        if ('Close', missing_ticker) in frame:
            frame[('Close', missing_ticker)] = np.nan
        return frame

    monkeypatch.setattr(vendor, 'download', partial_download)
    fx_module = importlib.import_module('examples.market_data.fx_rates_data_yahoo_example')
    monkeypatch.setattr(fx_module, 'yf', vendor)
    from examples.market_data.fx_hedging_example import load_example_inputs

    with pytest.raises(ValueError, match=re.escape(missing_ticker)):
        load_example_inputs()


@pytest.mark.parametrize('module', [
    'multi_assets', 'strategy', 'strategy_benchmark', 'multi_strategy',
    'multi_assets_reporting_frequencies', 'strategy_reporting_frequencies',
    'strategy_benchmark_reporting_frequencies', 'multi_strategy_reporting_frequencies',
])
def test_factsheet_output_directory(module, yahoo_panel, monkeypatch, tmp_path):
    """Exercise entry-point/save plumbing; full report layouts have separate rendering tests."""
    import qis

    def figure(**kwargs):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig

    def figures(**kwargs):
        return [figure()]

    generators = [
        ('multi_assets_factsheet', 'generate_multi_asset_factsheet', figure),
        ('strategy_factsheet', 'generate_strategy_factsheet', figures),
        ('strategy_benchmark_factsheet', 'generate_strategy_benchmark_factsheet_plt', figures),
        ('multi_strategy_factsheet', 'generate_multi_portfolio_factsheet', figures),
    ]
    for owner, name, replacement in generators:
        module_object = importlib.import_module(f'qis.portfolio.reports.{owner}')
        monkeypatch.setattr(module_object, name, replacement)
        monkeypatch.setattr(qis, name, replacement)
    output = tmp_path / 'new' / 'reports'
    monkeypatch.setattr(sys, 'argv', [module, '--output-dir', str(output)])
    runpy.run_module(f'examples.factsheets.{module}', run_name='__main__')
    pdfs = list(output.glob('*.pdf'))
    assert pdfs and all(path.stat().st_size > 100 for path in pdfs)
