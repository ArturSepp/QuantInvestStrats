"""
Plot regime-conditional regression analysis with scatter plots and fitted lines.

Visualizes asset returns against benchmark returns with regime-specific betas,
showing how asset sensitivities vary across different market conditions.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from matplotlib.ticker import FuncFormatter
from typing import Optional, Dict
from statsmodels import api as sm

# qis
import qis.plots.utils as put
from qis.perfstats.regime_classifier import (RegimeClassifier,
                                             BenchmarkReturnsQuantilesRegime)



class ConditionalRegressionColumns(str, Enum):
    # Column name constants
    RESID_VAR_COLUMN = 'An var resid'
    ALPHA_COLUMN = 'Alpha'
    COLOR_COLUMN = 'Color'
    PREDICTION = 'Prediction'
    VAR = 'Var'
    R2 = 'R2'


def plot_scatter_regression(prices: pd.DataFrame,
                            regime_benchmark: str,
                            regime_classifier: Optional[RegimeClassifier] = None,
                            drop_benchmark: bool = True,
                            is_add_alpha: bool = True,
                            x_var_format: str = '{:.0%}',
                            y_var_format: str = '{:.0%}',
                            beta_format: str = '{:.2f}',
                            xlabel: Optional[str] = None,
                            ylabel: Optional[str] = None,
                            title: Optional[str] = None,
                            display_estimated_betas: bool = True,
                            is_print_summary: bool = False,
                            add_last_date: bool = False,
                            framealpha: float = 0.9,
                            ax: Optional[plt.Subplot] = None,
                            **kwargs
                            ) -> Optional[plt.Figure]:
    """
    Plot scatter regression with regime-conditional betas.

    Creates a scatter plot showing asset returns vs benchmark returns, with fitted
    regression lines that have different slopes (betas) for each market regime.
    Background colors indicate regime boundaries.

    Args:
        prices: DataFrame of asset prices with benchmark included
        regime_benchmark: Column name of benchmark asset for regime classification
        regime_classifier: Regime classifier instance (default: BenchmarkReturnsQuantilesRegime with default params)
        drop_benchmark: If True, exclude benchmark from regression outputs
        x_var_format: Format string for x-axis values (benchmark returns)
        y_var_format: Format string for y-axis values (asset returns)
        beta_format: Format string for displaying beta values in legend
        xlabel: Custom x-axis label (default: "x={benchmark}")
        ylabel: Custom y-axis label (default: "Assets" or specific asset name)
        title: Plot title (default: None)
        display_estimated_betas: If True, show detailed regime betas in legend
        is_print_summary: If True, print OLS regression summaries
        framealpha: Legend frame transparency (0=transparent, 1=opaque)
        add_last_date: If True, add annotation for most recent data point
        ax: Matplotlib axis to plot on (creates new figure if None)
        **kwargs: Additional arguments passed to plotting utilities

    Returns:
        Figure object if new figure created, None if plotting on existing axis

    Examples:
        # Plot with quantile regimes (default)
        fig = plot_scatter_regression(
            prices=prices,
            regime_benchmark='SPY',
            regime_classifier=BenchmarkReturnsQuantilesRegime(freq='QE')
            ),
            is_asset_detailed=True
        )

        # Plot with positive/negative regimes
        fig = plot_scatter_regression(
            prices=prices,
            regime_benchmark='SPY',
            regime_classifier=BenchmarkReturnsPositiveNegativeRegime(freq='ME')
            ),
            is_asset_detailed=True
        )

        # Plot with default settings
        fig = plot_scatter_regression(
            prices=prices[['SPY', 'TLT']],
            regime_benchmark='SPY',
            add_last_date=True
        )
    """
    # Create new figure if axis not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Initialize regime classifier with default if not provided
    if regime_classifier is None:
        regime_classifier = BenchmarkReturnsQuantilesRegime()

    # Estimate regime-conditional regression parameters
    regmodel_out_dict = {}
    estimated_params = estimate_cond_regression(
        prices=prices,
        benchmark=regime_benchmark,
        drop_benchmark=drop_benchmark,
        regime_classifier=regime_classifier,  # Access params from classifier
        is_add_alpha=is_add_alpha,
        is_print_summary=is_print_summary,
        regmodel_out_dict=regmodel_out_dict
    )

    # Get unique markers and colors for each asset
    n_assets = len(estimated_params.index)
    markers = put.get_n_markers(n=n_assets)
    colors = put.get_n_colors(n=n_assets)

    # Build legend entries
    lines = []

    # Plot scatter and regression lines for each asset
    for (asset, pandas_out), marker, color_ in zip(regmodel_out_dict.items(), markers, colors):
        # Sort by benchmark for proper line plotting
        pandas_out = pandas_out.sort_values(regime_benchmark)

        # Determine scatter point coloring strategy
        if display_estimated_betas and len(prices.columns) == 1:
            # Single asset: use regime colors for scatter points
            scatter_hue = pandas_out[regime_classifier.REGIME_COLUMN]
            scatter_palette = list(regime_classifier.get_regime_ids_colors().values())
            scatter_hue_order = list(regime_classifier.get_regime_ids_colors().keys())
            scatter_color = None
        else:
            # Multiple assets: use asset-specific colors
            scatter_hue = None
            scatter_palette = None
            scatter_hue_order = None
            scatter_color = color_

        # Plot scatter points for actual returns
        sns.scatterplot(
            x=regime_benchmark,
            y=asset,
            hue=scatter_hue,
            data=pandas_out,
            palette=scatter_palette,
            hue_order=scatter_hue_order,
            color=scatter_color,
            marker=marker,
            legend=None,
            ax=ax
        )

        # Configure legend and line styling based on detail level
        # Configure legend and line styling based on detail level
        if display_estimated_betas:
            # Detailed view: show regime-specific betas
            legend = True
            line_marker = None
            line_hue = pandas_out[regime_classifier.REGIME_COLUMN]
            line_hue_order = list(regime_classifier.get_regime_ids_colors().keys())

            if len(prices.columns) == 1:
                # Single asset: separate legend entry per regime
                palette = list(regime_classifier.get_regime_ids_colors().values())
                for regime, color in regime_classifier.get_regime_ids_colors().items():
                    lines.append((
                        f"{regime} beta={beta_format.format(estimated_params.loc[asset, regime])}",
                        {'color': color, 'linestyle': '-', 'marker': line_marker}
                    ))
            else:
                # Multiple assets: combined legend entry with all betas
                palette = [color_] * len(regime_classifier.get_regime_ids_colors().values())
                label = f"{asset}: "
                beta_parts = [
                    f"{regime} beta={beta_format.format(estimated_params.loc[asset, regime])}"
                    for regime in regime_classifier.get_regime_ids_colors().keys()
                ]
                label += ", ".join(beta_parts)
                lines.append((label, {'color': color_, 'linestyle': '-', 'marker': marker}))
        else:
            # Simple view: one legend entry per asset, single color line
            palette = None
            legend = None
            line_marker = marker
            line_hue = None
            line_hue_order = None
            lines.append((asset, {'color': color_, 'linestyle': '-', 'marker': line_marker}))

        # Plot fitted regression lines
        sns.lineplot(
            x=regime_benchmark,
            y=ConditionalRegressionColumns.PREDICTION.value,
            hue=line_hue,
            data=pandas_out,
            hue_order=line_hue_order,
            palette=palette,
            color=color_ if not display_estimated_betas else None,  # Use direct color when no hue
            marker=line_marker,
            legend=legend,
            ax=ax
        )

    # Add colored background regions indicating market regimes
    regime_colors = list(regime_classifier.get_regime_ids_colors().values())
    regime_ids = list(regime_classifier.get_regime_ids_colors().keys())

    # Extract regime boundaries from first asset (all share same benchmark)
    for asset, pandas_out in regmodel_out_dict.items():
        # Find min/max benchmark returns for each regime
        regime_boundaries = {}
        for regime in regime_ids:
            regime_data = pandas_out[pandas_out[regime_classifier.REGIME_COLUMN] == regime]
            if not regime_data.empty:
                regime_boundaries[regime] = (
                    regime_data[regime_benchmark].min(),
                    regime_data[regime_benchmark].max()
                )

        # Save current axis limits before adding background
        axis_x_min, axis_x_max = ax.get_xlim()
        axis_y_min, axis_y_max = ax.get_ylim()

        # Sort regimes by x position for proper ordering
        sorted_regimes = sorted(regime_boundaries.items(), key=lambda x: x[1][0])

        # Draw vertical spans for each regime
        last_x_min = axis_x_min
        for idx, (regime, (x_min, x_max)) in enumerate(sorted_regimes):
            regime_idx = regime_ids.index(regime)

            # Extend first regime to axis minimum, last regime to axis maximum
            span_x_min = last_x_min # axis_x_min if idx == 0 else x_min
            span_x_max = axis_x_max if idx == len(sorted_regimes) - 1 else x_max
            last_x_min = span_x_max
            ax.axvspan(span_x_min, span_x_max, alpha=0.15, color=regime_colors[regime_idx], zorder=0)

        # Restore original axis limits (axvspan can modify them)
        ax.set_xlim(axis_x_min, axis_x_max)
        ax.set_ylim(axis_y_min, axis_y_max)

        break  # Only need to process once since all assets share same benchmark

    # Add annotation for most recent data point if requested
    if add_last_date:
        label_x_y = {}
        for asset, df in regmodel_out_dict.items():
            x = df[regime_benchmark].iloc[-1]
            y = df[asset].iloc[-1]
            label = (
                f"Last {df.index[-1].strftime('%d-%b-%Y')}: "
                f"x={x_var_format.format(x)}, y={x_var_format.format(y)}"
            )
            label_x_y[label] = (x, y)

        # Determine colors for annotation points
        if len(prices.columns) == 2:
            annotation_colors = [df[ConditionalRegressionColumns.COLOR_COLUMN].iloc[-1]]
        else:
            annotation_colors = colors

        put.add_scatter_points(ax=ax, label_x_y=label_x_y, colors=annotation_colors, **kwargs)

    # Configure plot aesthetics
    put.set_legend(ax=ax, lines=lines, framealpha=framealpha, **kwargs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: x_var_format.format(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: y_var_format.format(y)))

    # Set axis labels with defaults
    if xlabel is None:
        xlabel = f"x={regime_benchmark}"
    if ylabel is None:
        ylabel = f"y = {asset}" if len(prices.columns) == 2 else 'Assets'
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, **kwargs)

    # Set title if provided
    if title is not None:
        put.set_title(ax=ax, title=title, **kwargs)

    put.set_spines(ax=ax, **kwargs)

    return fig


def estimate_cond_regression(prices: pd.DataFrame,
                             benchmark: str,
                             regime_classifier: Optional[RegimeClassifier] = None,
                             is_print_summary: bool = False,
                             drop_benchmark: bool = True,
                             min_period: int = 0,
                             is_add_alpha: bool = False,
                             regmodel_out_dict: Optional[Dict[str, pd.DataFrame]] = None
                             ) -> pd.DataFrame:
    """
    Estimate regime-conditional regression parameters (betas) for multiple assets.

    Fits separate betas for each market regime (e.g., bear, normal, bull) by regressing
    asset returns on regime-specific benchmark returns.

    Args:
        prices: DataFrame of asset prices with benchmark as first column
        benchmark: Name of benchmark column used for regime classification
        regime_classifier: Specifications for regime classification (quantiles, frequency)
        is_print_summary: If True, print OLS regression summary for each asset
        drop_benchmark: If True, exclude benchmark from regression outputs
        min_period: Minimum number of valid returns required to estimate parameters
        is_add_alpha: If True, include intercept term in regression
        regmodel_out_dict: Optional dict to store detailed regression outputs per asset

    Returns:
        DataFrame with regime betas (and alpha if requested) as columns, assets as rows

    Raises:
        ValueError: If benchmark has all NaN returns (invalid benchmark specification)
    """
    # Initialize regime classifier with default params if not provided

    if regime_classifier is None:
        regime_classifier = BenchmarkReturnsQuantilesRegime()

    # Classify returns into market regimes based on benchmark quantiles
    sampled_returns_with_regime_id = regime_classifier.compute_sampled_returns_with_regime_id(
        prices=prices,
        benchmark=benchmark,
        **regime_classifier.to_dict()
    )

    # Remove rows where all assets have missing returns
    sampled_returns_with_regime_id = sampled_returns_with_regime_id.dropna(how='all', axis=0)

    # Group returns by regime for regime-conditional analysis
    regime_groups = sampled_returns_with_regime_id.groupby([regime_classifier.REGIME_COLUMN], observed=False)

    # Build regression matrix with regime-specific benchmark returns as columns
    regression_datas = []
    for regime in regime_classifier.get_regime_ids():
        # Extract benchmark returns for this regime and rename column by regime name
        benchmark_data = regime_groups.get_group((regime,))[benchmark].rename(regime)
        regression_datas.append(benchmark_data)

    # Concatenate regime columns and fill missing values with zeros (non-regime periods)
    regression_matrix = pd.concat(regression_datas, axis=1).fillna(0)

    # Add constant term (alpha) to regression matrix if requested
    if is_add_alpha:
        #regression_matrix_with_const = sm.add_constant(regression_matrix).rename(
        #    columns={'const': ConditionalRegressionColumns.ALPHA_COLUMN.value}
        #)
        # Add separate alpha for each regime
        for regime in regime_classifier.get_regime_ids():
            regime_dummy = (sampled_returns_with_regime_id[regime_classifier.REGIME_COLUMN] == regime).astype(float)
            regression_matrix[f'{ConditionalRegressionColumns.ALPHA_COLUMN.value}_{regime}'] = regime_dummy
        regression_matrix_with_const = regression_matrix

    else:
        regression_matrix_with_const = regression_matrix

    # Prepare asset returns for regression (drop regime column and optionally benchmark)
    columns_to_drop = [regime_classifier.REGIME_COLUMN]
    if drop_benchmark:
        columns_to_drop.append(benchmark)
    asset_returns = sampled_returns_with_regime_id.drop(columns=columns_to_drop)

    # Create regime color mapping for visualization outputs
    regime_colors = regime_classifier.class_data_to_colors(
        sampled_returns_with_regime_id[regime_classifier.REGIME_COLUMN]
    ).rename(ConditionalRegressionColumns.COLOR_COLUMN.value)

    # Fit regression model for each asset
    model_params = []
    for asset in asset_returns.columns:
        y = asset_returns[asset]
        nan_mask = y.isna()

        # Check if all returns are missing
        if nan_mask.all():
            if len(model_params) == 0:
                # First asset must be benchmark - should never have all NaNs
                raise ValueError(f"Benchmark '{benchmark}' has all NaN returns - invalid specification")
            else:
                # For other assets, record NaN parameters matching structure of previous assets
                estimated_model_params = pd.Series(np.nan, index=model_params[0].index, name=asset)
        # Check if insufficient valid observations for regression
        elif (~nan_mask).sum() < min_period:
            estimated_model_params = pd.Series(np.nan, index=model_params[0].index, name=asset)
        else:
            # Filter to valid observations only
            if nan_mask.any():
                y_clean = y[~nan_mask]
            else:
                y_clean = y

            # Align regression matrix with valid y observations
            X = regression_matrix_with_const.loc[y_clean.index, :]

            # Fit OLS regression model
            model = sm.OLS(y_clean, X)
            estimated_model = model.fit()

            # Print regression summary if requested
            if is_print_summary:
                print(f"\n{'=' * 80}")
                print(f"Regression Summary for {asset}")
                print('=' * 80)
                print(estimated_model.summary())

            # Store detailed regression outputs if dict provided
            if regmodel_out_dict is not None:
                prediction = estimated_model.predict(regression_matrix_with_const)
                pandas_out = pd.concat([
                    sampled_returns_with_regime_id[regime_classifier.REGIME_COLUMN],
                    sampled_returns_with_regime_id[benchmark],
                    regression_matrix,
                    y,
                    prediction.rename(ConditionalRegressionColumns.PREDICTION.value),
                    regime_colors
                ], axis=1)
                regmodel_out_dict[asset] = pandas_out

            # Extract parameter estimates and residual variance
            estimated_model_params = estimated_model.params.copy()
            estimated_model_params[ConditionalRegressionColumns.RESID_VAR_COLUMN.value] = estimated_model.mse_resid
            estimated_model_params[ConditionalRegressionColumns.R2.value] = estimated_model.rsquared_adj
            estimated_model_params.name = asset

        # Append parameters for this asset
        model_params.append(estimated_model_params)

    # Combine all asset parameters into single DataFrame (assets as rows)
    model_params_df = pd.concat(model_params, axis=1).T

    return model_params_df


def get_regime_regression_params(prices: pd.DataFrame,
                                 regime_classifier: RegimeClassifier,
                                 benchmark: str,
                                 is_print_summary: bool = True,
                                 drop_benchmark: bool = False,
                                 min_period: int = 0,
                                 is_add_alpha: bool = False
                                 ) -> pd.DataFrame:
    """
    Convenience wrapper for estimate_cond_regression with common parameters.

    Args:
        prices: DataFrame of asset prices
        regime_classifier: Regime classification specifications
        benchmark: Benchmark asset name
        is_print_summary: Whether to print regression summaries
        drop_benchmark: Whether to exclude benchmark from results
        min_period: Minimum observations required per asset
        is_add_alpha: Whether to include regression intercept

    Returns:
        DataFrame of estimated regime betas per asset
    """
    estimated_params = estimate_cond_regression(
        prices=prices,
        benchmark=benchmark,
        regime_classifier=regime_classifier,
        is_print_summary=is_print_summary,
        drop_benchmark=drop_benchmark,
        min_period=min_period,
        is_add_alpha=is_add_alpha
    )
    return estimated_params
