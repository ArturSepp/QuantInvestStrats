"""Development runner extracted from ``qis.perfstats.tests.sharpe_conventions_test``."""

from enum import Enum

from qis.perfstats.tests.sharpe_conventions_test import (
    test_excess_sharpe_conventions,
    test_regime_sharpe_decomposition_conventions,
    test_return_conventions,
    test_sharpe_conventions,
)

class Locals(Enum):
    RETURN_CONVENTIONS = 1
    SHARPE_CONVENTIONS = 2
    EXCESS_SHARPE_CONVENTIONS = 3
    REGIME_SHARPE_CONVENTIONS = 4
    ALL = 5

def run_local(local: Locals):
    """run local tests for development and debugging purposes"""
    if local in (Locals.RETURN_CONVENTIONS, Locals.ALL):
        test_return_conventions()
        print("return conventions regression guard passed")
    if local in (Locals.SHARPE_CONVENTIONS, Locals.ALL):
        test_sharpe_conventions()
        print("sharpe conventions regression guard passed")
    if local in (Locals.EXCESS_SHARPE_CONVENTIONS, Locals.ALL):
        test_excess_sharpe_conventions()
        print("excess sharpe conventions regression guard passed")
    if local in (Locals.REGIME_SHARPE_CONVENTIONS, Locals.ALL):
        test_regime_sharpe_decomposition_conventions()
        print("regime sharpe decomposition regression guard passed")

if __name__ == "__main__":
    run_local(local=Locals.ALL)
