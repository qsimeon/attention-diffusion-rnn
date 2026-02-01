#!/usr/bin/env python3
"""
Demo script for Attention-Diffusion Recurrent Network

This script demonstrates a unifying mechanism for attention and diffusion in a recurrent network:
- The recurrent network serves as a reservoir with chaotic dynamics
- W_att_rec converts recurrent state to an attention vector that multiplies the input
- The projection from input to recurrent is untrained (random)
- Readout weights from recurrent to output are trained
- W_out_rec converts output to a "diffusion" vector that denoises the recurrent state

The demo includes:
1. Synthetic task generation (temporal pattern recognition)
2. Network initialization with attention and diffusion mechanisms
3. Training with ridge regression
4. Performance evaluation and comparison
5. Visualization of attention patterns and state dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

# Import from the lib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
from core import AttentionDiffusionRNN
from utils import (
    train_readout_ridge,
    train_readout_online,
    collect_states,
    compute_mse,
    compute_nmse,
    compute_correlation,
    evaluate_network,
    analyze_attention,
    analyze_state_dynamics,
    generate_synthetic_task
)


def plot_results(results: Dict, save_path: str = None):
    """
    Plot comprehensive results including predictions, attention, and dynamics.
    
    Args:
        results: Dictionary containing predictions, targets, attention, and states
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Predictions vs Targets
    ax1 = plt.subplot(3, 2, 1)
    time_steps = np.arange(len(results['predictions']))
    ax1.plot(time_steps, results['targets'][:, 0], 'b-', label='Target', alpha=0.7, linewidth=2)
    ax1.plot(time_steps, results['predictions'][:, 0], 'r--', label='Prediction', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Output Value')
    ax1.set_title('Predictions vs Targets (Dimension 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Error
    ax2 = plt.subplot(3, 2, 2)
    errors = np.abs(results['predictions'] - results['targets'])
    ax2.plot(time_steps, errors[:, 0], 'g-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Prediction Error Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Attention Patterns
    ax3 = plt.subplot(3, 2, 3)
    if 'attention' in results and results['attention'] is not None:
        attention_data = results['attention']
        im = ax3.imshow(attention_data.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Input Dimension')
        ax3.set_title('Attention Patterns Over Time')
        plt.colorbar(im, ax=ax3, label='Attention Weight')
    else:
        ax3.text(0.5, 0.5, 'No attention data', ha='center', va='center')
        ax3.set_title('Attention Patterns (N/A)')
    
    # Plot 4: State Dynamics (first 5 dimensions)
    ax4 = plt.subplot(3, 2, 4)
    if 'states' in results and results['states'] is not None:
        states_data = results['states']
        n_dims_to_plot = min(5, states_data.shape[1])
        for i in range(n_dims_to_plot):
            ax4.plot(time_steps, states_data[:, i], label=f'Dim {i}', alpha=0.7)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('State Value')
        ax4.set_title('Recurrent State Dynamics (First 5 Dims)')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No state data', ha='center', va='center')
        ax4.set_title('State Dynamics (N/A)')
    
    # Plot 5: Attention Statistics
    ax5 = plt.subplot(3, 2, 5)
    if 'attention' in results and results['attention'] is not None:
        attention_stats = analyze_attention(results['attention'])
        mean_att = attention_stats['mean_attention']
        std_att = attention_stats['std_attention']
        dims = np.arange(len(mean_att))
        ax5.bar(dims, mean_att, yerr=std_att, alpha=0.7, color='orange', capsize=5)
        ax5.set_xlabel('Input Dimension')
        ax5.set_ylabel('Mean Attention Weight')
        ax5.set_title('Average Attention per Input Dimension')
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No attention data', ha='center', va='center')
        ax5.set_title('Attention Statistics (N/A)')
    
    # Plot 6: State Space Trajectory (2D projection)
    ax6 = plt.subplot(3, 2, 6)
    if 'states' in results and results['states'] is not None:
        states_data = results['states']
        # Use first two dimensions for visualization
        ax6.plot(states_data[:, 0], states_data[:, 1], 'b-', alpha=0.5, linewidth=1)
        ax6.scatter(states_data[0, 0], states_data[0, 1], c='green', s=100, 
                   marker='o', label='Start', zorder=5)
        ax6.scatter(states_data[-1, 0], states_data[-1, 1], c='red', s=100, 
                   marker='x', label='End', zorder=5)
        ax6.set_xlabel('State Dimension 0')
        ax6.set_ylabel('State Dimension 1')
        ax6.set_title('State Space Trajectory (2D Projection)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No state data', ha='center', va='center')
        ax6.set_title('State Space (N/A)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def compare_configurations(task_type: str = 'temporal_pattern'):
    """
    Compare different network configurations (with/without attention and diffusion).
    
    Args:
        task_type: Type of synthetic task to generate
    """
    print("\n" + "="*80)
    print("COMPARING NETWORK CONFIGURATIONS")
    print("="*80)
    
    # Generate synthetic task
    n_samples = 500
    input_dim = 10
    output_dim = 3
    seq_length = 100
    noise_level = 0.05
    
    print(f"\nGenerating synthetic task: {task_type}")
    print(f"  Samples: {n_samples}, Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"  Sequence length: {seq_length}, Noise level: {noise_level}")
    
    inputs, targets = generate_synthetic_task(
        task_type=task_type,
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_length=seq_length,
        noise_level=noise_level,
        seed=42
    )
    
    # Split into train and test
    train_size = int(0.7 * n_samples)
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]
    
    print(f"  Train samples: {train_size}, Test samples: {n_samples - train_size}")
    
    # Network configuration
    reservoir_size = 200
    spectral_radius = 1.2
    input_scaling = 0.5
    leak_rate = 0.3
    
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
    
    results_comparison = []
    
    for config in configurations:
        print(f"\n{'-'*80}")
        print(f"Configuration: {config['name']}")
        print(f"{'-'*80}")
        
        # Create network
        network = AttentionDiffusionRNN(
            input_dim=input_dim,
            reservoir_size=reservoir_size,
            output_dim=output_dim,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            use_attention=config['attention'],
            use_diffusion=config['diffusion'],
            seed=42
        )
        
        # Collect training states
        print("Collecting training states...")
        train_data = collect_states(
            network, 
            train_inputs, 
            apply_attention=config['attention'],
            apply_diffusion=config['diffusion'],
            warmup_steps=50
        )
        
        # Train readout with ridge regression
        print("Training readout weights with ridge regression...")
        W_out, bias = train_readout_ridge(
            train_data['states'], 
            train_targets,
            ridge_param=1e-4
        )
        
        # Set trained weights
        network.W_out = W_out
        network.bias_out = bias
        
        # Evaluate on test set
        print("Evaluating on test set...")
        metrics = evaluate_network(
            network,
            test_inputs,
            test_targets,
            apply_attention=config['attention'],
            apply_diffusion=config['diffusion'],
            warmup_steps=50
        )
        
        print(f"\nTest Performance:")
        print(f"  MSE:         {metrics['mse']:.6f}")
        print(f"  NMSE:        {metrics['nmse']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.6f}")
        
        results_comparison.append({
            'config': config['name'],
            'metrics': metrics
        })
    
    # Print comparison summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Configuration':<45} {'MSE':<12} {'NMSE':<12} {'Correlation':<12}")
    print("-"*80)
    for result in results_comparison:
        print(f"{result['config']:<45} "
              f"{result['metrics']['mse']:<12.6f} "
              f"{result['metrics']['nmse']:<12.6f} "
              f"{result['metrics']['correlation']:<12.6f}")
    print("="*80)


def detailed_demo():
    """
    Detailed demonstration of the attention-diffusion RNN with visualization.
    """
    print("\n" + "="*80)
    print("DETAILED ATTENTION-DIFFUSION RNN DEMONSTRATION")
    print("="*80)
    
    # Task parameters
    task_type = 'memory'
    n_samples = 300
    input_dim = 8
    output_dim = 2
    seq_length = 150
    noise_level = 0.1
    
    print(f"\nTask Configuration:")
    print(f"  Type: {task_type}")
    print(f"  Samples: {n_samples}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Noise level: {noise_level}")
    
    # Generate data
    print("\nGenerating synthetic data...")
    inputs, targets = generate_synthetic_task(
        task_type=task_type,
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_length=seq_length,
        noise_level=noise_level,
        seed=123
    )
    
    # Split data
    train_size = int(0.8 * n_samples)
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]
    
    print(f"  Training samples: {train_size}")
    print(f"  Test samples: {n_samples - train_size}")
    
    # Create network with full attention and diffusion
    print("\nInitializing Attention-Diffusion RNN...")
    network = AttentionDiffusionRNN(
        input_dim=input_dim,
        reservoir_size=300,
        output_dim=output_dim,
        spectral_radius=1.3,
        input_scaling=0.6,
        leak_rate=0.4,
        use_attention=True,
        use_diffusion=True,
        attention_strength=0.5,
        diffusion_strength=0.3,
        seed=123
    )
    
    config = network.get_config()
    print(f"  Reservoir size: {config['reservoir_size']}")
    print(f"  Spectral radius: {config['spectral_radius']}")
    print(f"  Input scaling: {config['input_scaling']}")
    print(f"  Leak rate: {config['leak_rate']}")
    print(f"  Attention enabled: {config['use_attention']}")
    print(f"  Diffusion enabled: {config['use_diffusion']}")
    print(f"  Attention strength: {config['attention_strength']}")
    print(f"  Diffusion strength: {config['diffusion_strength']}")
    
    # Collect training states
    print("\nCollecting training states...")
    train_data = collect_states(
        network,
        train_inputs,
        apply_attention=True,
        apply_diffusion=True,
        warmup_steps=50
    )
    
    print(f"  Collected states shape: {train_data['states'].shape}")
    print(f"  Collected attention shape: {train_data['attention'].shape}")
    
    # Analyze state dynamics
    print("\nAnalyzing state dynamics...")
    dynamics = analyze_state_dynamics(train_data['states'])
    print(f"  Mean state norm: {dynamics['mean_norm']:.4f}")
    print(f"  Std state norm: {dynamics['std_norm']:.4f}")
    print(f"  Max state value: {dynamics['max_value']:.4f}")
    print(f"  Min state value: {dynamics['min_value']:.4f}")
    
    # Train readout
    print("\nTraining readout weights...")
    W_out, bias = train_readout_ridge(
        train_data['states'],
        train_targets,
        ridge_param=1e-3
    )
    network.W_out = W_out
    network.bias_out = bias
    print(f"  Readout weights shape: {W_out.shape}")
    print(f"  Bias shape: {bias.shape}")
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate_network(
        network,
        train_inputs,
        train_targets,
        apply_attention=True,
        apply_diffusion=True,
        warmup_steps=50
    )
    print(f"  Training MSE: {train_metrics['mse']:.6f}")
    print(f"  Training NMSE: {train_metrics['nmse']:.6f}")
    print(f"  Training Correlation: {train_metrics['correlation']:.6f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_network(
        network,
        test_inputs,
        test_targets,
        apply_attention=True,
        apply_diffusion=True,
        warmup_steps=50
    )
    print(f"  Test MSE: {test_metrics['mse']:.6f}")
    print(f"  Test NMSE: {test_metrics['nmse']:.6f}")
    print(f"  Test Correlation: {test_metrics['correlation']:.6f}")
    
    # Collect detailed results for visualization
    print("\nCollecting detailed results for visualization...")
    test_sample_idx = 0
    network.reset_state()
    
    predictions = []
    states_list = []
    attention_list = []
    
    for t in range(seq_length):
        output, state, attention = network.step(
            test_inputs[test_sample_idx, t],
            apply_attention=True,
            apply_diffusion=True
        )
        predictions.append(output)
        states_list.append(state)
        if attention is not None:
            attention_list.append(attention)
    
    predictions = np.array(predictions)
    states_array = np.array(states_list)
    attention_array = np.array(attention_list) if attention_list else None
    
    # Prepare results dictionary
    results = {
        'predictions': predictions,
        'targets': test_targets[test_sample_idx],
        'states': states_array,
        'attention': attention_array
    }
    
    # Plot results
    print("\nGenerating visualization...")
    plot_results(results, save_path='attention_diffusion_results.png')
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


def online_training_demo():
    """
    Demonstrate online training of readout weights.
    """
    print("\n" + "="*80)
    print("ONLINE TRAINING DEMONSTRATION")
    print("="*80)
    
    # Generate simple task
    print("\nGenerating synthetic task...")
    inputs, targets = generate_synthetic_task(
        task_type='sine_wave',
        n_samples=200,
        input_dim=5,
        output_dim=2,
        seq_length=100,
        noise_level=0.05,
        seed=456
    )
    
    # Create network
    print("\nInitializing network...")
    network = AttentionDiffusionRNN(
        input_dim=5,
        reservoir_size=150,
        output_dim=2,
        spectral_radius=1.1,
        input_scaling=0.5,
        leak_rate=0.3,
        use_attention=True,
        use_diffusion=True,
        seed=456
    )
    
    # Train online
    print("\nTraining with online gradient descent...")
    losses = train_readout_online(
        network,
        inputs,
        targets,
        learning_rate=0.001,
        n_epochs=20,
        apply_attention=True,
        apply_diffusion=True
    )
    
    print(f"\nTraining complete!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Online Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('online_training_curve.png', dpi=150, bbox_inches='tight')
    print("\nTraining curve saved to 'online_training_curve.png'")
    plt.show()


def main():
    """
    Main demonstration function.
    """
    print("\n" + "="*80)
    print("ATTENTION-DIFFUSION RECURRENT NETWORK DEMO")
    print("="*80)
    print("\nThis demo showcases a unifying mechanism for attention and diffusion")
    print("in a recurrent network with the following key features:")
    print("  1. Chaotic reservoir dynamics (untrained recurrent weights)")
    print("  2. Attention mechanism: W_att_rec converts state to attention vector")
    print("  3. Untrained input projection (random W_in)")
    print("  4. Trained readout weights (W_out)")
    print("  5. Diffusion mechanism: W_out_rec denoises recurrent state")
    print("\n" + "="*80)
    
    try:
        # Run detailed demo with visualization
        detailed_demo()
        
        # Compare different configurations
        compare_configurations(task_type='temporal_pattern')
        
        # Demonstrate online training
        online_training_demo()
        
        print("\n" + "="*80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  - attention_diffusion_results.png")
        print("  - online_training_curve.png")
        print("\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
