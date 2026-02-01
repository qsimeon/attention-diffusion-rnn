#!/usr/bin/env python3
"""
Attention-Diffusion RNN Demo with Real-World Benchmarks

This demo showcases the attention-diffusion reservoir network on classic
time series prediction benchmarks:

1. Mackey-Glass Chaotic Time Series - A standard reservoir computing benchmark
2. NARMA-10 Nonlinear System - Tests memory and nonlinear processing
3. Lorenz Attractor Prediction - Chaotic dynamics forecasting

The architecture combines:
- Echo State Network (chaotic reservoir with frozen weights)
- Attention: reservoir state → input gating (dynamic feature selection)
- Diffusion: output → state correction (self-stabilizing feedback)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Import from the lib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
from core import AttentionDiffusionRNN
from utils import train_readout_ridge, collect_states, evaluate_network, analyze_attention

# ============================================================================
# STYLING
# ============================================================================

# Modern color palette
COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Raspberry
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Vermillion
    'dark': '#1B1B1E',         # Almost black
    'light': '#F5F5F5',        # Off white
    'grid': '#E0E0E0',         # Light gray
}

def setup_plot_style():
    """Configure matplotlib for modern, clean plots."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['dark'],
        'axes.labelcolor': COLORS['dark'],
        'axes.titleweight': 'bold',
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.color': COLORS['dark'],
        'ytick.color': COLORS['dark'],
        'text.color': COLORS['dark'],
        'font.family': 'sans-serif',
        'font.size': 10,
        'legend.framealpha': 0.9,
        'legend.edgecolor': COLORS['grid'],
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })

setup_plot_style()

# ============================================================================
# REAL-WORLD DATASET GENERATORS
# ============================================================================

def generate_mackey_glass(n_samples: int, tau: int = 17, delta_t: float = 1.0,
                          history_len: int = 1000, seed: int = None) -> np.ndarray:
    """
    Generate the Mackey-Glass chaotic time series.
    
    The Mackey-Glass system is a classic benchmark for time series prediction,
    exhibiting chaotic behavior for tau > 17.
    
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    
    Args:
        n_samples: Number of samples to generate
        tau: Time delay (tau=17 gives mild chaos, tau=30 gives strong chaos)
        delta_t: Time step for discretization
        history_len: Initial transient to discard
        seed: Random seed
        
    Returns:
        Time series of shape (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    beta = 0.2
    gamma = 0.1
    n_exp = 10
    
    # Total length including history
    total_len = n_samples + history_len
    
    # Initialize with random values for the delay
    x = np.zeros(total_len + tau)
    x[:tau] = 0.9 + 0.2 * (np.random.rand(tau) - 0.5)
    
    # Runge-Kutta 4th order integration
    for t in range(tau, total_len + tau - 1):
        x_tau = x[t - tau]
        x_t = x[t]
        
        # RK4 steps
        k1 = delta_t * (beta * x_tau / (1 + x_tau**n_exp) - gamma * x_t)
        k2 = delta_t * (beta * x_tau / (1 + x_tau**n_exp) - gamma * (x_t + k1/2))
        k3 = delta_t * (beta * x_tau / (1 + x_tau**n_exp) - gamma * (x_t + k2/2))
        k4 = delta_t * (beta * x_tau / (1 + x_tau**n_exp) - gamma * (x_t + k3))
        
        x[t + 1] = x_t + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # Discard transient and return
    return x[tau + history_len:]


def generate_narma(n_samples: int, order: int = 10, seed: int = None) -> tuple:
    """
    Generate the NARMA (Nonlinear Auto-Regressive Moving Average) system.
    
    NARMA-10 is a standard benchmark testing memory capacity and nonlinear
    processing. The output depends on inputs up to 10 steps in the past.
    
    Args:
        n_samples: Number of samples to generate
        order: Order of the NARMA system (default 10)
        seed: Random seed
        
    Returns:
        Tuple of (inputs, targets) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Input: uniform random in [0, 0.5]
    u = np.random.uniform(0, 0.5, n_samples + order)
    y = np.zeros(n_samples + order)
    
    # NARMA-n recurrence relation
    alpha = 0.3
    beta = 0.05
    gamma = 1.5
    delta = 0.1
    
    for t in range(order, n_samples + order):
        y[t] = (alpha * y[t-1] + 
                beta * y[t-1] * np.sum(y[t-order:t]) + 
                gamma * u[t-order] * u[t-1] + 
                delta)
        # Clip to prevent explosion
        y[t] = np.clip(y[t], -1, 1)
    
    return u[order:], y[order:]


def generate_lorenz(n_samples: int, dt: float = 0.01, 
                    transient: int = 1000, seed: int = None) -> np.ndarray:
    """
    Generate the Lorenz attractor trajectory.
    
    The Lorenz system is the prototypical chaotic dynamical system,
    often used to test prediction of chaotic dynamics.
    
    Args:
        n_samples: Number of samples to generate
        dt: Time step
        transient: Initial transient to discard
        seed: Random seed for initial conditions
        
    Returns:
        Trajectory of shape (n_samples, 3) for (x, y, z)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Lorenz parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    total_len = n_samples + transient
    trajectory = np.zeros((total_len, 3))
    
    # Random initial conditions near the attractor
    trajectory[0] = [1.0 + np.random.randn() * 0.1,
                     1.0 + np.random.randn() * 0.1,
                     1.0 + np.random.randn() * 0.1]
    
    # Integrate using RK4
    for t in range(total_len - 1):
        x, y, z = trajectory[t]
        
        # Derivatives
        def lorenz_deriv(state):
            x, y, z = state
            return np.array([
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z
            ])
        
        k1 = dt * lorenz_deriv(trajectory[t])
        k2 = dt * lorenz_deriv(trajectory[t] + k1/2)
        k3 = dt * lorenz_deriv(trajectory[t] + k2/2)
        k4 = dt * lorenz_deriv(trajectory[t] + k3)
        
        trajectory[t + 1] = trajectory[t] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return trajectory[transient:]


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_mackey_glass_results(t, true_signal, predictions, train_end, 
                              metrics, save_path=None):
    """Create a comprehensive visualization for Mackey-Glass prediction."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Main prediction plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t[:train_end], true_signal[:train_end], 
             color=COLORS['primary'], alpha=0.6, label='Training Data', linewidth=1)
    ax1.plot(t[train_end:], true_signal[train_end:], 
             color=COLORS['primary'], label='True Signal', linewidth=1.5)
    ax1.plot(t[train_end:], predictions, 
             color=COLORS['accent'], linestyle='--', label='Prediction', linewidth=1.5)
    ax1.axvline(x=t[train_end], color=COLORS['secondary'], linestyle=':', 
                alpha=0.7, label='Train/Test Split')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('x(t)')
    ax1.set_title('Mackey-Glass Chaotic Time Series Prediction', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Zoomed prediction
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_start = 0
    zoom_len = min(200, len(predictions))
    ax2.plot(true_signal[train_end:train_end+zoom_len], 
             color=COLORS['primary'], label='True', linewidth=1.5)
    ax2.plot(predictions[:zoom_len], 
             color=COLORS['accent'], linestyle='--', label='Predicted', linewidth=1.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('x(t)')
    ax2.set_title('Prediction Detail (First 200 Test Steps)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = fig.add_subplot(gs[1, 1])
    errors = predictions - true_signal[train_end:train_end+len(predictions)]
    ax3.hist(errors, bins=50, color=COLORS['secondary'], alpha=0.7, edgecolor='white')
    ax3.axvline(x=0, color=COLORS['dark'], linestyle='--', linewidth=1)
    ax3.axvline(x=np.mean(errors), color=COLORS['accent'], linestyle='-', 
                linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(true_signal[train_end:train_end+len(predictions)], predictions, 
                alpha=0.3, s=10, color=COLORS['primary'])
    lims = [min(ax4.get_xlim()[0], ax4.get_ylim()[0]),
            max(ax4.get_xlim()[1], ax4.get_ylim()[1])]
    ax4.plot(lims, lims, color=COLORS['success'], linestyle='--', linewidth=2, label='Perfect Fit')
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.set_xlabel('True Value')
    ax4.set_ylabel('Predicted Value')
    ax4.set_title(f'Prediction Accuracy (R² = {metrics["correlation"]**2:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Metrics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    metrics_text = f"""
    ╔══════════════════════════════════════╗
    ║       PERFORMANCE METRICS            ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║   MSE:          {metrics['mse']:.6f}           ║
    ║   NMSE:         {metrics['nmse']:.6f}           ║
    ║   Correlation:  {metrics['correlation']:.6f}           ║
    ║   R²:           {metrics['correlation']**2:.6f}           ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """
    ax5.text(0.1, 0.5, metrics_text, transform=ax5.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  → Saved: {save_path}")
    
    plt.show()


def plot_attention_analysis(attention_data, inputs, save_path=None):
    """Visualize how attention weights evolve and correlate with inputs."""
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Attention heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    im = ax1.imshow(attention_data.T, aspect='auto', cmap='YlOrRd', 
                    interpolation='bilinear')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Input Dimension')
    ax1.set_title('Attention Weights Over Time', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Attention Weight')
    
    # Attention statistics
    ax2 = fig.add_subplot(gs[0, 2])
    mean_att = np.mean(attention_data, axis=0)
    std_att = np.std(attention_data, axis=0)
    dims = np.arange(len(mean_att))
    ax2.barh(dims, mean_att, xerr=std_att, color=COLORS['accent'], 
             alpha=0.8, capsize=3, edgecolor='white')
    ax2.set_ylabel('Input Dimension')
    ax2.set_xlabel('Mean Attention')
    ax2.set_title('Attention Distribution')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Attention dynamics (3 dimensions)
    ax3 = fig.add_subplot(gs[1, 0])
    n_show = min(3, attention_data.shape[1])
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    for i in range(n_show):
        ax3.plot(attention_data[:300, i], color=colors[i], 
                 label=f'Dim {i}', alpha=0.8, linewidth=1.2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Attention Weight')
    ax3.set_title('Attention Dynamics (First 300 Steps)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Attention variance
    ax4 = fig.add_subplot(gs[1, 1])
    att_var = np.var(attention_data, axis=1)
    ax4.fill_between(range(len(att_var)), att_var, alpha=0.5, color=COLORS['secondary'])
    ax4.plot(att_var, color=COLORS['secondary'], linewidth=0.5)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Attention Variance')
    ax4.set_title('Attention Selectivity Over Time')
    ax4.grid(True, alpha=0.3)
    
    # Attention entropy
    ax5 = fig.add_subplot(gs[1, 2])
    # Normalize attention to probabilities
    att_prob = attention_data / (attention_data.sum(axis=1, keepdims=True) + 1e-10)
    entropy = -np.sum(att_prob * np.log(att_prob + 1e-10), axis=1)
    ax5.plot(entropy, color=COLORS['primary'], linewidth=1)
    ax5.axhline(y=np.log(attention_data.shape[1]), color=COLORS['success'], 
                linestyle='--', label='Max Entropy')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Entropy')
    ax5.set_title('Attention Entropy (Concentration)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  → Saved: {save_path}")
    
    plt.show()


def plot_comparison(results, save_path=None):
    """Visualize the comparison between different configurations."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    configs = [r['config'] for r in results]
    short_names = ['Baseline', 'Attention', 'Diffusion', 'Full Model']
    
    mse = [r['metrics']['mse'] for r in results]
    nmse = [r['metrics']['nmse'] for r in results]
    corr = [r['metrics']['correlation'] for r in results]
    
    colors = [COLORS['dark'], COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    
    # MSE comparison
    bars1 = axes[0].bar(short_names, mse, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('MSE Comparison (Lower is Better)', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)
    
    # NMSE comparison
    bars2 = axes[1].bar(short_names, nmse, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Normalized MSE')
    axes[1].set_title('NMSE Comparison (Lower is Better)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)
    
    # Correlation comparison
    bars3 = axes[2].bar(short_names, corr, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    axes[2].set_ylabel('Correlation')
    axes[2].set_title('Correlation (Higher is Better)', fontweight='bold')
    axes[2].set_ylim([min(0.9, min(corr) - 0.05), 1.0])
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  → Saved: {save_path}")
    
    plt.show()


def plot_state_dynamics(states, save_path=None):
    """Visualize reservoir state dynamics."""
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 3D state trajectory
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    # Use PCA for dimensionality reduction
    from numpy.linalg import svd
    states_centered = states - states.mean(axis=0)
    U, S, Vt = svd(states_centered, full_matrices=False)
    states_3d = U[:, :3] * S[:3]
    
    ax1.plot(states_3d[:, 0], states_3d[:, 1], states_3d[:, 2], 
             color=COLORS['primary'], alpha=0.6, linewidth=0.5)
    ax1.scatter(states_3d[0, 0], states_3d[0, 1], states_3d[0, 2], 
                color='green', s=100, marker='o', label='Start')
    ax1.scatter(states_3d[-1, 0], states_3d[-1, 1], states_3d[-1, 2], 
                color='red', s=100, marker='x', label='End')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('State Trajectory (PCA Projection)', fontweight='bold')
    ax1.legend()
    
    # State activation distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(states.flatten(), bins=100, color=COLORS['primary'], 
             alpha=0.7, edgecolor='white', density=True)
    ax2.set_xlabel('Activation Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Reservoir Activation Distribution')
    ax2.grid(True, alpha=0.3)
    
    # State norms over time
    ax3 = fig.add_subplot(gs[1, 0])
    norms = np.linalg.norm(states, axis=1)
    ax3.plot(norms, color=COLORS['secondary'], linewidth=1)
    ax3.axhline(y=np.mean(norms), color=COLORS['accent'], linestyle='--', 
                label=f'Mean: {np.mean(norms):.2f}')
    ax3.fill_between(range(len(norms)), norms, alpha=0.3, color=COLORS['secondary'])
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('State Norm ||h(t)||')
    ax3.set_title('Reservoir Activity Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Principal component variance
    ax4 = fig.add_subplot(gs[1, 1])
    variance_explained = (S ** 2) / np.sum(S ** 2)
    cumulative = np.cumsum(variance_explained)
    ax4.bar(range(min(20, len(variance_explained))), 
            variance_explained[:20] * 100, 
            color=COLORS['primary'], alpha=0.7, label='Individual')
    ax4.plot(range(min(20, len(cumulative))), 
             cumulative[:20] * 100, 
             color=COLORS['accent'], marker='o', markersize=4, label='Cumulative')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Variance Explained (%)')
    ax4.set_title('Reservoir Dimensionality (PCA)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  → Saved: {save_path}")
    
    plt.show()


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_mackey_glass():
    """
    Demonstrate the network on Mackey-Glass chaotic time series prediction.
    
    This is the classic reservoir computing benchmark - predicting the next
    value of a chaotic system from its delayed embedding.
    """
    print("\n" + "="*70)
    print("  MACKEY-GLASS CHAOTIC TIME SERIES PREDICTION")
    print("="*70)
    
    # Generate data
    print("\n[1] Generating Mackey-Glass time series (τ=17)...")
    n_total = 5000
    tau = 17
    mg_series = generate_mackey_glass(n_total, tau=tau, seed=42)
    
    # Create delayed embedding as input features
    embedding_dim = 10
    delay = 1
    
    # Prepare input/output pairs
    X = np.zeros((n_total - embedding_dim * delay, embedding_dim))
    for i in range(embedding_dim):
        X[:, i] = mg_series[i * delay : n_total - (embedding_dim - i) * delay]
    
    # Target: predict next value
    y = mg_series[embedding_dim * delay:]
    
    # Normalize
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std
    
    # Train/test split
    train_size = 3000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"  → Training samples: {train_size}")
    print(f"  → Test samples: {len(X_test)}")
    print(f"  → Input dimension: {embedding_dim}")
    
    # Create network
    print("\n[2] Initializing Attention-Diffusion RNN...")
    network = AttentionDiffusionRNN(
        input_dim=embedding_dim,
        recurrent_dim=500,
        output_dim=1,
        spectral_radius=1.2,
        input_scaling=0.5,
        attention_scaling=0.3,
        diffusion_scaling=0.2,
        leak_rate=0.3,
        seed=42
    )
    print(f"  → Reservoir size: 500")
    print(f"  → Spectral radius: 1.2")
    print(f"  → Attention + Diffusion: ENABLED")
    
    # Collect states
    print("\n[3] Collecting reservoir states...")
    warmup = 100
    
    network.reset_state()
    train_states = []
    train_attentions = []
    
    for t in range(len(X_train)):
        _, state, attention = network.step(X_train[t], 
                                           apply_attention=True, 
                                           apply_diffusion=True)
        if t >= warmup:
            train_states.append(state)
            train_attentions.append(attention)
    
    train_states = np.array(train_states)
    train_attentions = np.array(train_attentions)
    train_targets = y_train[warmup:]
    
    print(f"  → State matrix shape: {train_states.shape}")
    
    # Train readout
    print("\n[4] Training readout (Ridge Regression)...")
    W_out, b_out = train_readout_ridge(train_states, train_targets.reshape(-1, 1), 
                                        ridge_param=1e-4)
    network.W_out = W_out
    network.b_out = b_out
    print(f"  → Readout weights: {W_out.shape}")
    
    # Test
    print("\n[5] Evaluating on test set...")
    network.reset_state()
    
    # Warmup on end of training data
    for t in range(warmup):
        network.step(X_train[-(warmup-t)], apply_attention=True, apply_diffusion=True)
    
    predictions = []
    test_states = []
    
    for t in range(len(X_test)):
        output, state, _ = network.step(X_test[t], 
                                        apply_attention=True, 
                                        apply_diffusion=True)
        predictions.append(output[0])
        test_states.append(state)
    
    predictions = np.array(predictions)
    test_states = np.array(test_states)
    
    # Calculate metrics
    from utils import compute_mse, compute_nmse, compute_correlation
    metrics = {
        'mse': compute_mse(predictions, y_test),
        'nmse': compute_nmse(predictions, y_test),
        'correlation': compute_correlation(predictions, y_test)
    }
    
    print(f"\n  ╔════════════════════════════════╗")
    print(f"  ║     TEST SET PERFORMANCE       ║")
    print(f"  ╠════════════════════════════════╣")
    print(f"  ║  MSE:         {metrics['mse']:.6f}        ║")
    print(f"  ║  NMSE:        {metrics['nmse']:.6f}        ║")
    print(f"  ║  Correlation: {metrics['correlation']:.6f}        ║")
    print(f"  ╚════════════════════════════════╝")
    
    # Visualizations
    print("\n[6] Generating visualizations...")
    
    t = np.arange(len(mg_series))
    plot_mackey_glass_results(
        t[embedding_dim * delay:], 
        y * y_std + y_mean,  # Denormalize for plotting
        predictions * y_std + y_mean,
        train_size,
        metrics,
        save_path='results_mackey_glass.png'
    )
    
    plot_attention_analysis(
        train_attentions,
        X_train[warmup:],
        save_path='results_attention_analysis.png'
    )
    
    plot_state_dynamics(
        test_states,
        save_path='results_state_dynamics.png'
    )
    
    return metrics


def demo_configuration_comparison():
    """
    Compare performance with/without attention and diffusion.
    """
    print("\n" + "="*70)
    print("  ABLATION STUDY: ATTENTION & DIFFUSION IMPACT")
    print("="*70)
    
    # Generate Mackey-Glass data
    print("\n[1] Preparing benchmark data...")
    n_total = 3000
    mg_series = generate_mackey_glass(n_total, tau=17, seed=42)
    
    embedding_dim = 8
    X = np.zeros((n_total - embedding_dim, embedding_dim))
    for i in range(embedding_dim):
        X[:, i] = mg_series[i : n_total - (embedding_dim - i)]
    y = mg_series[embedding_dim:]
    
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()
    
    train_size = 2000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"  → Train/Test: {train_size}/{len(X_test)}")
    
    configurations = [
        {'name': 'Baseline (No Attention, No Diffusion)', 
         'attention': False, 'diffusion': False},
        {'name': 'Attention Only', 
         'attention': True, 'diffusion': False},
        {'name': 'Diffusion Only', 
         'attention': False, 'diffusion': True},
        {'name': 'Full Model (Attention + Diffusion)', 
         'attention': True, 'diffusion': True},
    ]
    
    results = []
    warmup = 100
    
    print("\n[2] Running experiments...")
    
    for config in configurations:
        print(f"\n  Testing: {config['name']}")
        
        network = AttentionDiffusionRNN(
            input_dim=embedding_dim,
            recurrent_dim=300,
            output_dim=1,
            spectral_radius=1.2,
            input_scaling=0.5,
            attention_scaling=0.3,
            diffusion_scaling=0.2,
            leak_rate=0.3,
            seed=42
        )
        
        # Collect training states
        network.reset_state()
        train_states = []
        
        for t in range(len(X_train)):
            _, state, _ = network.step(X_train[t], 
                                       apply_attention=config['attention'], 
                                       apply_diffusion=config['diffusion'])
            if t >= warmup:
                train_states.append(state)
        
        train_states = np.array(train_states)
        train_targets = y_train[warmup:]
        
        # Train
        W_out, b_out = train_readout_ridge(train_states, train_targets.reshape(-1, 1), 
                                            ridge_param=1e-4)
        network.W_out = W_out
        network.b_out = b_out
        
        # Test
        network.reset_state()
        for t in range(warmup):
            network.step(X_train[-(warmup-t)], 
                        apply_attention=config['attention'], 
                        apply_diffusion=config['diffusion'])
        
        predictions = []
        for t in range(len(X_test)):
            output, _, _ = network.step(X_test[t], 
                                        apply_attention=config['attention'], 
                                        apply_diffusion=config['diffusion'])
            predictions.append(output[0])
        
        predictions = np.array(predictions)
        
        from utils import compute_mse, compute_nmse, compute_correlation
        metrics = {
            'mse': compute_mse(predictions, y_test),
            'nmse': compute_nmse(predictions, y_test),
            'correlation': compute_correlation(predictions, y_test)
        }
        
        print(f"    MSE: {metrics['mse']:.6f}  |  Corr: {metrics['correlation']:.4f}")
        
        results.append({'config': config['name'], 'metrics': metrics})
    
    # Print summary
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"\n  {'Configuration':<40} {'MSE':<12} {'NMSE':<12} {'Corr':<10}")
    print("  " + "-"*70)
    for r in results:
        print(f"  {r['config']:<40} {r['metrics']['mse']:<12.6f} "
              f"{r['metrics']['nmse']:<12.6f} {r['metrics']['correlation']:<10.4f}")
    
    # Plot comparison
    print("\n[3] Generating comparison plot...")
    plot_comparison(results, save_path='results_comparison.png')
    
    return results


def demo_narma():
    """
    Test on NARMA-10, a challenging memory and nonlinearity benchmark.
    """
    print("\n" + "="*70)
    print("  NARMA-10 NONLINEAR SYSTEM IDENTIFICATION")
    print("="*70)
    
    print("\n[1] Generating NARMA-10 data...")
    n_total = 4000
    u, y = generate_narma(n_total, order=10, seed=42)
    
    # Normalize
    u = (u - u.mean()) / u.std()
    y = (y - y.mean()) / y.std()
    
    train_size = 3000
    u_train, u_test = u[:train_size], u[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"  → Train/Test: {train_size}/{len(u_test)}")
    
    print("\n[2] Training Attention-Diffusion RNN...")
    
    network = AttentionDiffusionRNN(
        input_dim=1,
        recurrent_dim=400,
        output_dim=1,
        spectral_radius=0.95,  # NARMA needs more stable dynamics
        input_scaling=0.8,
        attention_scaling=0.2,
        diffusion_scaling=0.15,
        leak_rate=0.2,
        seed=42
    )
    
    warmup = 100
    
    # Collect states
    network.reset_state()
    train_states = []
    
    for t in range(len(u_train)):
        _, state, _ = network.step(np.array([u_train[t]]), 
                                   apply_attention=True, 
                                   apply_diffusion=True)
        if t >= warmup:
            train_states.append(state)
    
    train_states = np.array(train_states)
    train_targets = y_train[warmup:]
    
    # Train
    W_out, b_out = train_readout_ridge(train_states, train_targets.reshape(-1, 1), 
                                        ridge_param=1e-3)
    network.W_out = W_out
    network.b_out = b_out
    
    # Test
    network.reset_state()
    for t in range(warmup):
        network.step(np.array([u_train[-(warmup-t)]]), 
                    apply_attention=True, apply_diffusion=True)
    
    predictions = []
    for t in range(len(u_test)):
        output, _, _ = network.step(np.array([u_test[t]]), 
                                    apply_attention=True, 
                                    apply_diffusion=True)
        predictions.append(output[0])
    
    predictions = np.array(predictions)
    
    from utils import compute_mse, compute_nmse, compute_correlation
    metrics = {
        'mse': compute_mse(predictions, y_test),
        'nmse': compute_nmse(predictions, y_test),
        'correlation': compute_correlation(predictions, y_test)
    }
    
    print(f"\n  ╔════════════════════════════════╗")
    print(f"  ║   NARMA-10 TEST PERFORMANCE    ║")
    print(f"  ╠════════════════════════════════╣")
    print(f"  ║  MSE:         {metrics['mse']:.6f}        ║")
    print(f"  ║  NMSE:        {metrics['nmse']:.6f}        ║")
    print(f"  ║  Correlation: {metrics['correlation']:.6f}        ║")
    print(f"  ╚════════════════════════════════╝")
    
    # Quick plot
    print("\n[3] Generating visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(y_test[:500], color=COLORS['primary'], label='True', linewidth=1.5)
    axes[0].plot(predictions[:500], color=COLORS['accent'], linestyle='--', 
                 label='Predicted', linewidth=1.5)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Output')
    axes[0].set_title('NARMA-10 Prediction (First 500 Test Steps)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test, predictions, alpha=0.3, s=10, color=COLORS['primary'])
    lims = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
            max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(lims, lims, color=COLORS['success'], linestyle='--', linewidth=2)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel('True Value')
    axes[1].set_ylabel('Predicted Value')
    axes[1].set_title(f'Prediction Accuracy (R² = {metrics["correlation"]**2:.4f})')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_narma10.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  → Saved: results_narma10.png")
    plt.show()
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demonstrations."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "   ATTENTION-DIFFUSION RECURRENT NEURAL NETWORK".center(68) + "█")
    print("█" + "   Real-World Benchmark Demonstration".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("""
    This demo tests the Attention-Diffusion RNN architecture on classic
    time series benchmarks used in reservoir computing research:
    
    • Mackey-Glass: Chaotic time series prediction (τ=17)
    • NARMA-10: Nonlinear system with 10-step memory
    • Ablation: Impact of attention and diffusion mechanisms
    
    Architecture highlights:
    ┌─────────────────────────────────────────────────────────┐
    │  FROZEN: Input weights (W_in), Reservoir (W_rec)        │
    │  TRAINED: Output weights (W_out) via Ridge Regression   │
    │  ATTENTION: h(t) → gates on input features              │
    │  DIFFUSION: y(t) → state denoising correction           │
    └─────────────────────────────────────────────────────────┘
    """)
    
    try:
        # Run demonstrations
        print("\n" + "="*70)
        mg_metrics = demo_mackey_glass()
        
        print("\n" + "="*70)
        comparison_results = demo_configuration_comparison()
        
        print("\n" + "="*70)
        narma_metrics = demo_narma()
        
        # Final summary
        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "   ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)
        
        print("\n  Generated files:")
        print("    • results_mackey_glass.png     - Prediction visualization")
        print("    • results_attention_analysis.png - Attention mechanism analysis")
        print("    • results_state_dynamics.png   - Reservoir state analysis")
        print("    • results_comparison.png       - Ablation study results")
        print("    • results_narma10.png          - NARMA-10 results")
        print()
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
