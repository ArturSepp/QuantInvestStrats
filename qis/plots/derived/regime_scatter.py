"""
Plot regime-conditional regression analysis with scatter plots and fitted lines.

Visualizes asset returns against benchmark returns with regime-specific betas,
showing how asset sensitivities vary across different market conditions.
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from matplotlib.ticker import FuncFormatter
from typing import Optional

# qis
import qis.plots.utils as put
import qis.perfstats.cond_regression as cre
from qis.perfstats.regime_classifier import (RegimeClassifier,
                                             BenchmarkReturnsQuantilesRegime)


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
    estimated_params = cre.estimate_cond_regression(
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
            y=cre.ConditionalRegressionColumns.PREDICTION.value,
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
            annotation_colors = [df[cre.COLOR_COLUMN].iloc[-1]]
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
