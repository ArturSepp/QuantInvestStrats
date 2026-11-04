"""Compare Bartlett and AR(1)-prewhitened intercept standard errors on the synthetic panel.

The ratio of the two standard errors and the fitted rho show how much a fixed lag count
leaves to the filter: near zero rho the ratio is close to one, with the prewhitened
value slightly lower because the filter drops one observation.
"""

# packages
import pandas as pd
from enum import Enum
# qis
import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.utils import regression
from qis.utils.hac_intercept_prewhitening import estimate_prewhitened_intercepts


class Locals(Enum):
    """Mean standard errors, or alpha standard errors against the two-asset benchmark."""

    MEAN = 1
    ALPHA = 2


def run_local(local: Locals = Locals.MEAN) -> pd.DataFrame:
    """Print per-response Bartlett SE, prewhitened SE, their ratio and rho for several lags.

    Args:
        local: MEAN tests sample means; ALPHA regresses on the SEQ_US/SBD_TSY factors.

    Returns:
        A frame indexed by (hac_lags, response) with both standard errors, their ratio and rho.
    """
    prices = generate_synthetic_prices()
    returns = qis.to_returns(prices, freq='ME', is_log_returns=False, drop_first=True)
    returns = returns.dropna(axis=1).dropna(axis=0)  # complete common monthly sample only
    if local is Locals.ALPHA:
        factors = returns[['SEQ_US', 'SBD_TSY']]
        responses = returns.drop(columns=factors.columns)
    else:
        factors, responses = None, returns
    rows = []
    for hac_lags in (0, 3, 6):
        bartlett = regression.estimate_hac_regression(responses, factors, span=None,
                                                      hac_lags=hac_lags)
        filtered = estimate_prewhitened_intercepts(responses, factors, hac_lags=hac_lags)
        for name in responses.columns:
            bartlett_se = float(bartlett.hac_se.loc['Intercept', name])
            prewhitened_se = float(filtered.table.loc[name, 'standard_error'])
            rows.append(dict(hac_lags=hac_lags, response=name, bartlett_se=bartlett_se,
                             prewhitened_se=prewhitened_se, ratio=prewhitened_se / bartlett_se,
                             rho=float(filtered.table.loc[name, 'rho'])))
    table = pd.DataFrame(rows).set_index(['hac_lags', 'response'])
    print(f'{local.name}: T={len(responses)} monthly simple returns')
    print(table.to_string(float_format=lambda value: f'{value:.4g}'))
    return table


if __name__ == '__main__':
    run_local(local=Locals.MEAN)
