
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from typing import Tuple, Union, Dict
from enum import Enum


def compute_utility_weight(alpha: Union[float, np.ndarray] = 0.06,
                           vol: float = 0.16,
                           corr: float = 0.8,
                           gamma: Union[float, np.ndarray] = 1.0
                           ) -> Union[float, np.ndarray]:
    vol2 = vol * vol
    c = (alpha-gamma*vol2*(1.0-corr)) / gamma
    return c


def compute_portfolio_return_vol(leverage: Union[float, np.ndarray]  = 0.5,
                                 alpha: Union[float, np.ndarray] = 0.06,
                                 vol: float = 0.16,
                                 corr: float = 0.8
                                 ) -> Tuple[Union[float, np.ndarray], ...]:
    p_return = (1.0+leverage)*alpha + alpha
    vol2 = vol * vol
    p_var = np.square(1.0+leverage)*vol2+vol2-2.0*(1.0+leverage)*vol2*corr
    p_vol = np.sqrt(p_var)
    p_sharpe = p_return / p_vol
    return p_sharpe, p_return, p_vol


def compute_one_factor_weights(mu_market: float = 0.06,
                                   sigma_market: float = 0.15,
                                   gamma: Union[float, np.ndarray] = 1.0,
                                   beta_target: float = 1.0) -> Dict:

    sigma_market2 = sigma_market * sigma_market
    leg_betas = np.array([0.9, 1.1])
    leg_alphas = np.array([0.02, -0.02])  # long short
    leg_varthetas = np.array([0.04**2, 0.04**2])
    mus = np.array([leg_alphas[0]+leg_betas[0]*mu_market, -leg_alphas[1]-leg_betas[1]*mu_market])
    betas = np.array([leg_betas[0], -leg_betas[1]])

    sigma = np.array([
        [
            leg_betas[0] ** 2 * sigma_market2 + leg_varthetas[0],
            -leg_betas[0] * leg_betas[1] * sigma_market2,
        ],
        [
            -leg_betas[0] * leg_betas[1] * sigma_market2,
            leg_betas[1] ** 2 * sigma_market2 + leg_varthetas[1],
        ],
    ])
    sigma_inv = np.linalg.inv(sigma)
    divisor = betas.T @ sigma_inv @ betas
    lambda_star =  (betas.T @ sigma_inv @ mus-gamma*beta_target) / divisor
    weights = sigma_inv @ (mus - lambda_star * betas) / gamma
    print(f"weights={weights}, "
          f"returns={mus.T @ weights:0.2%}, "
          f"vol={np.sqrt(weights.T @ sigma @ weights):0.2%}, "
          f"betas={betas.T @ weights:0.2f}")

    output = dict(
        gamma=gamma,
        weight_long=weights[0],
        weight_short=weights[1],
        returns=mus.T @ weights,
        vol=np.sqrt(weights.T @ sigma @ weights),
        beta=betas.T @ weights,
    )

    return output


class LocalTests(Enum):
    RUN_UTILITY = 1
    RUN_PORTFOLIO_RETURN = 2
    RUN_PORTFOLIO_RETURN_UTILITY = 3
    ONE_FACTOR = 4
    ONE_FACTOR_PLOTS = 5


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    if local_test == LocalTests.RUN_UTILITY:
        gamma = np.linspace(0.01, 3.0, 21)
        leverage = compute_utility_weight(gamma=gamma)
        print(leverage)

    elif local_test == LocalTests.RUN_PORTFOLIO_RETURN:
        leverage = np.linspace(0.0, 1.0, 21)
        p_sharpe, p_return, p_vol = compute_portfolio_return_vol(leverage)
        print(p_sharpe)
        print(p_return)
        print(p_vol)

    elif local_test == LocalTests.RUN_PORTFOLIO_RETURN_UTILITY:
        gamma = np.linspace(0.02, 1.5, 21)
        leverage = compute_utility_weight(gamma=gamma)
        print(leverage)
        p_sharpe, p_return, p_vol = compute_portfolio_return_vol(leverage, corr=0.5)
        print(p_sharpe)
        print(p_return)
        print(p_vol)

    elif local_test == LocalTests.ONE_FACTOR:
        output = compute_one_factor_weights(gamma=10.0, beta_target=1.0)
        print(output)

    elif local_test == LocalTests.ONE_FACTOR_PLOTS:
        beta_target = 0.0
        mu_market = 0.06
        sigma_market = 0.15
        gammas = np.linspace(2.0, 15, 31)
        outputs = {}
        for gamma in gammas:
            outputs[f"gamma={gamma:0.2f}"] = compute_one_factor_weights(
                mu_market=mu_market,
                sigma_market=sigma_market,
                gamma=gamma,
                beta_target=beta_target,
            )
        outputs = pd.DataFrame.from_dict(outputs, orient='index')
        print(outputs)

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(2, 2, figsize=(14, 8))
            df = outputs[['gamma', 'weight_long', 'weight_short']].set_index('gamma', drop=True)
            df['gross_notional'] = df.sum(axis=1)
            qis.plot_line(df=df,
                          title='Weights',
                          xlabel='risk-aversion lambda',
                          ylabel='weights', ax=axs[0, 0])
            leverage_long_to_sort = np.divide(
                df['weight_long'], df['weight_short']).rename('Long to Short')
            qis.plot_line(df=leverage_long_to_sort,
                          title='Ratio of long to short weights',
                          xlabel='risk-aversion lambda',
                          ylabel='weights', ax=axs[1, 0])

            df = outputs[['gamma','returns', 'vol']].set_index('gamma', drop=True)
            ax = axs[0, 1]
            qis.plot_line(df=df,
                          title='Portfolio return and Vol',
                          xlabel='risk-aversion lambda',
                          yvar_format='{:,.2%}',
                          ylabel='Return and Vol', ax=ax)

            ax.axhline(y=mu_market, color='red', linestyle='--', linewidth=1.5)
            ax.annotate('Expected market return', xy=(gammas[0], mu_market), color='red')

            ax.axhline(y=sigma_market, color='orange', linestyle='-.', linewidth=1.5)
            ax.annotate('Expected market vol', xy=(gammas[0], sigma_market), color='orange')

            sharpe = np.divide(df['returns'], df['vol']).rename('Sharpe')
            ax = axs[1, 1]
            qis.plot_line(df=sharpe,
                          title='Portfolio sharpe',
                          xlabel='risk-aversion lambda',
                          ylabel='sharpe', ax=ax)

            ax.axhline(y=mu_market/sigma_market, color='red', linestyle='--', linewidth=1.5)
            ax.annotate(
                'Expected market Sharpe',
                xy=(gammas[0], mu_market/sigma_market),
                color='red',
            )

            #qis.set_suptitle(fig, f"beta-target={beta_target:.2f}")

        qis.save_fig(
            fig=fig,
            file_name=f"beta_target_{beta_target:.2f}",
            local_path=None,
        )

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ONE_FACTOR_PLOTS)
