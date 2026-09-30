"""
Regime-conditional analytics: Gaussian and Student-t nulls, premia, betas and mixture moments.

The regimes are the benchmark-return buckets of a regime classifier, the one-sigma Bear, Normal
and Bull cut by default. Every function takes the classifier's sampled frame or builds it with
``create_sampled_returns_with_regime_id``, so the analytics work for any partition the
classifiers produce. The subpackage depends on ``qis.utils``, ``qis.perfstats`` and
``qis.models`` only; drawing is in ``qis.plots.derived.regime_premium``.

Imported explicitly, ``from qis.regimes import ...``; the names are not re-exported from ``qis``.

The subpackage implements the regime statistics, the Gaussian null and the mixture covariance of
Sepp, A., and Kastenholz, M. (2026), The Convexity Premium of Portfolio Overlays, Journal of
Investment Management, forthcoming. Module and function docstrings name the result of the paper
they implement; the handbook chapter maps every function to the paper's definitions, equations
and exhibits: https://quantinveststrats.readthedocs.io/en/latest/convexity_premium.html
"""
from qis.regimes.partition import (
    ONE_SIGMA_QUANTILES,
    create_sampled_returns_with_regime_id,
    get_partition_quantiles,
    get_regime_ids,
    get_regime_probabilities,
)
from qis.regimes.nulls import (
    calibrate_student_t_nu,
    compute_convexity_premium,
    compute_null_regime_contributions,
    compute_overlay_blend_frontier,
    compute_portfolio_bear_sharpe,
    compute_regime_kappa,
    compute_regime_null_loadings,
)
from qis.regimes.premium import compute_regime_premium_bootstrap, compute_regime_premium_table
from qis.regimes.betas import compute_regime_betas, compute_regime_betas_bootstrap
from qis.regimes.ewma import compute_regime_ewm_avg, compute_regime_ewm_betas
from qis.regimes.covariance import (
    compute_gaussian_regime_moments,
    compute_regime_mixture_covar,
    compute_regime_mixture_covar_from_sample,
    compute_sample_regime_moments,
)
