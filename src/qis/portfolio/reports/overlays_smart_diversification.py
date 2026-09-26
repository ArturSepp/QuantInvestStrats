"""
Compatibility import path of the smart-diversification report.

The report moved to ``qis.portfolio.smart_diversification`` after qis 5.31.0. Importing from
``qis.portfolio.reports.overlays_smart_diversification`` remains valid and returns the same
objects; new code imports from ``qis`` or ``qis.portfolio.smart_diversification``.
"""
from qis.portfolio.smart_diversification.overlay_curve import create_overlay_portfolio_curve
from qis.portfolio.smart_diversification.report import (PERF_COLUMNS,
                                                        PERF_PARAMS,
                                                        SmartDiversificationReport,
                                                        safe_polyfit)

__all__ = ['PERF_COLUMNS', 'PERF_PARAMS', 'SmartDiversificationReport',
           'create_overlay_portfolio_curve', 'safe_polyfit']
