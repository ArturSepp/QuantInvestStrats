
import numpy as np
import pandas as pd
from qis.models.linear.ra_returns import map_signal_to_weight, SignalMapType
import matplotlib.pyplot as plt


def signal_mapping_illustration():
    """
    Illustrate signal-to-weight mapping functions across different types.
    Shows how raw signals (x-axis) are transformed to portfolio weights (y-axis).
    """
    # Create signal range from -4 to 4
    x_values = np.linspace(-4, 4, 201)
    signals_df = pd.DataFrame({'signal': x_values})

    # Test different mapping types
    mapping_configs = [
        {
            'name': 'Normal CDF (σ=1.0)',
            'signal_map_type': SignalMapType.NormalCDF,
            'params': {'loc': 0.0, 'scale': 1.0}
        },
        {
            'name': 'Normal CDF (σ=0.5)',
            'signal_map_type': SignalMapType.NormalCDF,
            'params': {'loc': 0.0, 'scale': 0.5}
        },
        {
            'name': 'Laplace CDF (σ=1.0)',
            'signal_map_type': SignalMapType.LaplaceCDF,
            'params': {'loc': 0.0, 'scale': 1.0}
        },
        {
            'name': 'Exp CDF (symmetric)',
            'signal_map_type': SignalMapType.ExpCDF,
            'params': {'loc': 0.0, 'scale': 1.0, 'tail_level': 1.0,
                       'slope_right': 0.5, 'slope_left': 0.5}
        },
        {
            'name': 'Exp CDF (asymmetric)',
            'signal_map_type': SignalMapType.ExpCDF,
            'params': {'loc': 0.0, 'scale': 1.0, 'tail_level': 1.0,
                       'slope_right': 0.7, 'slope_left': 0.3}
        },
        {
            'name': 'Exp CDF with tail decay',
            'signal_map_type': SignalMapType.ExpCDF,
            'params': {'loc': 0.0, 'scale': 1.0, 'tail_level': 1.0,
                       'slope_right': 0.5, 'slope_left': 0.5,
                       'tail_decay_right': 1.0, 'tail_decay_left': 1.0}
        }
    ]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, config in enumerate(mapping_configs):
        ax = axes[idx]

        # Compute weights
        weights = map_signal_to_weight(
            signals=signals_df,
            signal_map_type=config['signal_map_type'],
            **config['params']
        )

        # Plot
        ax.plot(x_values, weights['signal'].values, linewidth=2, color='darkblue')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)

        # Formatting
        ax.set_xlabel('Signal (x)', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(config['name'], fontsize=12, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.1, 1.1)

        # Add reference lines at ±1
        ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(y=-1.0, color='red', linestyle=':', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.suptitle('Signal-to-Weight Mapping Functions',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.96)

    return fig


def signal_mapping_comparison():
    """
    Compare all mapping types on a single plot for easy comparison.
    """
    import matplotlib.pyplot as plt

    # Create signal range
    x_values = np.linspace(-4, 4, 201)
    signals_df = pd.DataFrame({'signal': x_values})

    fig, ax = plt.subplots(figsize=(12, 7))

    # Normal CDF
    weights_normal = map_signal_to_weight(
        signals=signals_df,
        signal_map_type=SignalMapType.NormalCDF,
        loc=0.0, scale=1.0
    )
    ax.plot(x_values, weights_normal['signal'], label='Normal CDF (σ=1.0)',
            linewidth=2, alpha=0.8)

    # Laplace CDF
    weights_laplace = map_signal_to_weight(
        signals=signals_df,
        signal_map_type=SignalMapType.LaplaceCDF,
        loc=0.0, scale=1.0
    )
    ax.plot(x_values, weights_laplace['signal'], label='Laplace CDF (σ=1.0)',
            linewidth=2, alpha=0.8)

    # Exp CDF
    weights_exp = map_signal_to_weight(
        signals=signals_df,
        signal_map_type=SignalMapType.ExpCDF,
        loc=0.0, scale=1.0, tail_level=1.0,
        slope_right=0.5, slope_left=0.5
    )
    ax.plot(x_values, weights_exp['signal'], label='Exp CDF (symmetric)',
            linewidth=2, alpha=0.8)

    # Exp CDF with tail decay
    weights_exp_decay = map_signal_to_weight(
        signals=signals_df,
        signal_map_type=SignalMapType.ExpCDF,
        loc=0.0, scale=1.0, tail_level=1.0,
        slope_right=0.5, slope_left=0.5,
        tail_decay_right=1.0, tail_decay_left=1.0
    )
    ax.plot(x_values, weights_exp_decay['signal'], label='Exp CDF (tail decay)',
            linewidth=2, alpha=0.8, linestyle='--')

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=-1.0, color='red', linestyle=':', alpha=0.3, linewidth=1)

    # Formatting
    ax.set_xlabel('Signal (x)', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Comparison of Signal-to-Weight Mapping Functions',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    return fig


def signal_mapping_properties():
    """
    Show key properties: slope at origin, saturation points, asymmetry.
    """
    import matplotlib.pyplot as plt

    x_values = np.linspace(-4, 4, 201)
    signals_df = pd.DataFrame({'signal': x_values})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Different scales (Normal CDF)
    for scale in [0.5, 1.0, 2.0]:
        weights = map_signal_to_weight(
            signals=signals_df,
            signal_map_type=SignalMapType.NormalCDF,
            loc=0.0, scale=scale
        )
        axes[0].plot(x_values, weights['signal'],
                     label=f'σ={scale}', linewidth=2, alpha=0.8)

    axes[0].set_title('Normal CDF: Effect of Scale', fontweight='bold')
    axes[0].set_xlabel('Signal (x)')
    axes[0].set_ylabel('Weight')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Asymmetric slopes (Exp CDF)
    for slope_r, slope_l in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]:
        weights = map_signal_to_weight(
            signals=signals_df,
            signal_map_type=SignalMapType.ExpCDF,
            loc=0.0, scale=1.0, tail_level=1.0,
            slope_right=slope_r, slope_left=slope_l
        )
        axes[1].plot(x_values, weights['signal'],
                     label=f'slope_R={slope_r}, slope_L={slope_l}',
                     linewidth=2, alpha=0.8)

    axes[1].set_title('Exp CDF: Asymmetric Slopes', fontweight='bold')
    axes[1].set_xlabel('Signal (x)')
    axes[1].set_ylabel('Weight')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Tail decay effect
    weights_no_decay = map_signal_to_weight(
        signals=signals_df,
        signal_map_type=SignalMapType.ExpCDF,
        loc=0.0, scale=1.0, tail_level=1.0,
        slope_right=0.5, slope_left=0.5
    )
    axes[2].plot(x_values, weights_no_decay['signal'],
                 label='No tail decay', linewidth=2, alpha=0.8)

    for decay in [0.5, 1.0, 2.0]:
        weights_decay = map_signal_to_weight(
            signals=signals_df,
            signal_map_type=SignalMapType.ExpCDF,
            loc=0.0, scale=1.0, tail_level=1.0,
            slope_right=0.5, slope_left=0.5,
            tail_decay_right=decay, tail_decay_left=decay
        )
        axes[2].plot(x_values, weights_decay['signal'],
                     label=f'Tail decay={decay}', linewidth=2, alpha=0.8)

    axes[2].set_title('Exp CDF: Tail Decay Effect', fontweight='bold')
    axes[2].set_xlabel('Signal (x)')
    axes[2].set_ylabel('Weight')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    return fig


def signal_mapping_properties2():
    """
    Show key properties: slope at origin, saturation points, asymmetry.
    """
    import matplotlib.pyplot as plt

    x_values = np.linspace(-4, 4, 201)
    signals_df = pd.DataFrame({'signal': x_values})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Different scales (Normal CDF)
    for slope_right in [0.0, 0.5, 0.75]:
        weights = map_signal_to_weight(
            signals=signals_df,
            signal_map_type=SignalMapType.ExpCDF,
            slope_right=slope_right,
            slope_left=0.5,
            loc=0.0, scale=1.0
        )
        axes[0].plot(x_values, weights['signal'],
                     label=f'slope={slope_right}', linewidth=2, alpha=0.8)

    axes[0].set_title('Normal CDF: Effect of slope', fontweight='bold')
    axes[0].set_xlabel('Signal (x)')
    axes[0].set_ylabel('Weight')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Asymmetric slopes (Exp CDF)
    for slope_r, slope_l in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]:
        weights = map_signal_to_weight(
            signals=signals_df,
            signal_map_type=SignalMapType.ExpCDF,
            loc=0.0, scale=1.0, tail_level=1.0,
            slope_right=slope_r, slope_left=slope_l
        )
        axes[1].plot(x_values, weights['signal'],
                     label=f'slope_R={slope_r}, slope_L={slope_l}',
                     linewidth=2, alpha=0.8)

    axes[1].set_title('Exp CDF: Asymmetric Slopes', fontweight='bold')
    axes[1].set_xlabel('Signal (x)')
    axes[1].set_ylabel('Weight')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Tail decay effect
    weights_no_decay = map_signal_to_weight(
        signals=signals_df,
        signal_map_type=SignalMapType.ExpCDF,
        loc=0.0, scale=1.0, tail_level=1.0,
        slope_right=0.5, slope_left=0.5
    )
    axes[2].plot(x_values, weights_no_decay['signal'],
                 label='No tail decay', linewidth=2, alpha=0.8)

    for decay in [0.5, 1.0, 2.0]:
        weights_decay = map_signal_to_weight(
            signals=signals_df,
            signal_map_type=SignalMapType.ExpCDF,
            loc=0.0, scale=1.0, tail_level=1.0,
            slope_right=0.5, slope_left=0.5,
            tail_decay_right=decay, tail_decay_left=decay
        )
        axes[2].plot(x_values, weights_decay['signal'],
                     label=f'Tail decay={decay}', linewidth=2, alpha=0.8)

    axes[2].set_title('Exp CDF: Tail Decay Effect', fontweight='bold')
    axes[2].set_xlabel('Signal (x)')
    axes[2].set_ylabel('Weight')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    return fig


if __name__ == '__main__':
    # Run tests
    #fig1 = signal_mapping_illustration()
    #fig2 = signal_mapping_comparison()
    #fig3 = signal_mapping_properties()
    fig4 = signal_mapping_properties2()
    plt.show()