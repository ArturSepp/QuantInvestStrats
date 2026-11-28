"""Attribute forward asset returns to two jointly estimated EWMA benchmark exposures.

Run offline with python -m examples.perfstats.ewma_benchmark_residuals. Prices come
from the fixed-seed QIS synthetic universe. All payoffs are sums of monthly log returns.
"""
import numpy as np
import pandas as pd

from qis import EwmLinearModel, MeanAdjType
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.forward_return_targets import (
    ForwardReturnAdjustment, compute_forward_return_targets,
)


BENCHMARKS = ['SEQ_US', 'SBD_TSY']


def monthly_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return monthly log returns for eight assets and two benchmarks on the same index."""
    prices = generate_synthetic_prices(start='2014-12-31', end='2024-12-31',
                                       seed=20260725, apply_quirks=False)
    monthly = np.log(prices.resample('ME').last()).diff().iloc[1:]
    return monthly.drop(columns=BENCHMARKS), monthly[BENCHMARKS]


def benchmark_adjustment(assets: pd.DataFrame, benchmarks: pd.DataFrame,
                         formation_dates: pd.DatetimeIndex, *,
                         ewma_span: int = 36) -> ForwardReturnAdjustment:
    """Fit both betas jointly through each close and supply dated asset-by-benchmark frames."""
    model = EwmLinearModel(x=benchmarks, y=assets)
    model.fit(span=ewma_span, is_x_correlated=True, mean_adj_type=MeanAdjType.NONE,
              warmup_period=11)
    snapshots = {date: model.get_loadings_at_date(date).T for date in formation_dates}
    return ForwardReturnAdjustment(factor_increments_by_freq={'ME': benchmarks},
                                   loadings_by_date=snapshots)


def main(ewma_span: int = 36) -> None:
    """Print the opening betas and the two benchmark contributions for the final quarter."""
    assets, benchmarks = monthly_inputs()
    decisions = assets.loc['2018-12-31':].index
    adjustment = benchmark_adjustment(assets, benchmarks, decisions[:-1], ewma_span=ewma_span)
    targets = compute_forward_return_targets(
        decision_dates=decisions, asset_increments_by_freq={'ME': assets}, horizons=(1, 3),
        adjustments={'two_benchmark': adjustment},
    )
    opening, closing = decisions[-4], decisions[-1]
    window = targets.loc[targets.horizon.eq(3) & targets.formation_date.eq(opening)].set_index(
        'asset').reindex(assets.columns)
    assert window.realisation_date.eq(closing).all() and window.two_benchmark_status.eq('ok').all()
    assert window.two_benchmark_snapshot_date.eq(opening).all()
    betas = adjustment.loadings_by_date[opening]
    holding = (benchmarks.index > opening) & (benchmarks.index <= closing)
    moves = benchmarks.loc[holding].sum()
    contributions = betas.mul(moves, axis='columns')
    np.testing.assert_allclose(window.two_benchmark_explained, contributions.sum(axis=1))
    np.testing.assert_allclose(window.two_benchmark, assets.loc[holding].sum()
                               - contributions.sum(axis=1))

    print(f'EWMA betas ({ewma_span} months), formed {opening.date()}, realised {closing.date()}')
    print('Benchmark log returns:', moves.map('{:+.2%}'.format).to_dict())
    table = pd.concat([betas.add_prefix('beta_'), contributions.add_prefix('payoff_'),
                       window[['raw', 'two_benchmark_explained', 'two_benchmark']]], axis=1)
    print(table.rename(columns={'two_benchmark_explained': 'explained',
                                'two_benchmark': 'residual'}).round(5).to_string())


if __name__ == '__main__':
    main()
