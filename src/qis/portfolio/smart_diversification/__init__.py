"""Smart diversification: what adding overlays to a principal portfolio buys.

The principal portfolio of this subpackage is the benchmark of ``qis.regimes``: it sets the Bear,
Normal and Bull regimes. An overlay is a smart diversifier when the principal with the overlay
has a higher Sharpe ratio and a higher Bear contribution to it than the principal alone,
Definition 4 of Sepp, A., and Kastenholz, M. (2026), The Convexity Premium of Portfolio Overlays,
Journal of Investment Management, forthcoming, which formalises the property of Sepp and
Dezeraud (2019) and Sepp (2020). ``SmartDiversificationReport`` draws candidate overlays in these
two coordinates, and ``plot_overlay_allocation_frontier`` draws the stacked portfolios and the
coverage-floor frontier of the paper's Figure 4 from precomputed statistics.
"""

from qis.portfolio.smart_diversification.overlay_curve import create_overlay_portfolio_curve
from qis.portfolio.smart_diversification.report import SmartDiversificationReport

from qis.portfolio.smart_diversification.allocation_frontier import plot_overlay_allocation_frontier
