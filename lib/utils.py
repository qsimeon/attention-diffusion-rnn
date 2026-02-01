"""Utility functions for attention-diffusion recurrent network.

This module provides helper functions for training, evaluation, and analysis
of the attention-diffusion RNN, including:
- Training algorithms (ridge regression, online learning)
- Evaluation metrics
- Visualization helpers
- State collection and analysis
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Callable
import warnings


def train_readout_ridge(
    states: np.ndarray,
    targets: np.ndarray,
    ridge_param: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Train readout weights using ridge regression.
    
    Args:
        states: Recurrent states of shape (n_samples, recurrent_dim).
        targets: Target outputs of shape (n_samples, output_dim).
        ridge_param: Ridge regularization parameter.
        
    Returns:
        Tuple of (W_out, b_out) where:
            - W_out: Output weights of shape (output_dim, recurrent_dim).
            - b_out: Output bias of shape (output_dim,).
    """
    n_samples, recurrent_dim = states.shape
    output_dim = targets.shape[1] if targets.ndim > 1 else 1
    
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    
    # Add bias term to states
    states_with_bias = np.hstack([states, np.ones((n_samples, 1))])
    
    # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
    XtX = states_with_bias.T @ states_with_bias
    XtX += ridge_param * np.eye(recurrent_dim + 1)
    XtY = states_with_bias.T @ targets
    
    try:
        weights = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        warnings.warn("Ridge regression failed, using pseudoinverse")
        weights = np.linalg.pinv(XtX) @ XtY
    
    W_out = weights[:-1, :].T  # Shape: (output_dim, recurrent_dim)
    b_out = weights[-1, :]      # Shape: (output_dim,)
    
    return W_out, b_out


def train_readout_online(
    network,
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float = 0.01,
    n_epochs: int = 10,
    apply_attention: bool = True,
    apply_diffusion: bool = True
) -> List[float]:
    """Train readout weights using online gradient descent.
    
    Args:
        network: AttentionDiffusionRNN instance.
        inputs: Input sequence of shape (time_steps, input_dim).
        targets: Target sequence of shape (time_steps, output_dim).
        learning_rate: Learning rate for gradient descent.
        n_epochs: Number of training epochs.
        apply_attention: Whether to apply attention mechanism.
        apply_diffusion: Whether to apply diffusion mechanism.
        
    Returns:
        List of mean squared errors per epoch.
    """
    time_steps = inputs.shape[0]
    losses = []
    
    for epoch in range(n_epochs):
        network.reset_state()
        epoch_loss = 0.0
        
        for t in range(time_steps):
            # Forward pass
            output, state, _ = network.step(
                inputs[t],
                apply_attention=apply_attention,
                apply_diffusion=apply_diffusion
            )
            
            # Compute error
            target = targets[t]
            error = output - target
            epoch_loss += np.mean(error ** 2)
            
            # Gradient descent update
            network.W_out -= learning_rate * np.outer(error, state)
            network.b_out -= learning_rate * error
        
        losses.append(epoch_loss / time_steps)
    
    return losses


def collect_states(
    network,
    inputs: np.ndarray,
    apply_attention: bool = True,
    apply_diffusion: bool = True,
    warmup_steps: int = 0
) -> Dict[str, np.ndarray]:
    """Collect recurrent states for a sequence of inputs.
    
    Args:
        network: AttentionDiffusionRNN instance.
        inputs: Input sequence of shape (time_steps, input_dim).
        apply_attention: Whether to apply attention mechanism.
        apply_diffusion: Whether to apply diffusion mechanism.
        warmup_steps: Number of initial steps to discard (transient dynamics).
        
    Returns:
        Dictionary containing collected states, outputs, and attentions.
    """
    network.reset_state()
    
    # Run warmup
    if warmup_steps > 0:
        for t in range(warmup_steps):
            network.step(
                inputs[t],
                apply_attention=apply_attention,
                apply_diffusion=apply_diffusion
            )
    
    # Collect states
    result = network.forward(
        inputs[warmup_steps:],
        apply_attention=apply_attention,
        apply_diffusion=apply_diffusion,
        return_states=True
    )
    
    return result


def compute_mse(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """Compute mean squared error.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        
    Returns:
        Mean squared error.
    """
    return np.mean((predictions - targets) ** 2)


def compute_nmse(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """Compute normalized mean squared error.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        
    Returns:
        Normalized mean squared error.
    """
    mse = compute_mse(predictions, targets)
    variance = np.var(targets)
    return mse / variance if variance > 0 else float('inf')


def compute_correlation(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """Compute Pearson correlation coefficient.
    
    Args:
        predictions: Predicted values.
        targets: Target values.
        
    Returns:
        Pearson correlation coefficient.
    """
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    if len(predictions_flat) < 2:
        return 0.0
    
    correlation = np.corrcoef(predictions_flat, targets_flat)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def evaluate_network(
    network,
    inputs: np.ndarray,
    targets: np.ndarray,
    apply_attention: bool = True,
    apply_diffusion: bool = True,
    warmup_steps: int = 0
) -> Dict[str, float]:
    """Evaluate network performance on a dataset.
    
    Args:
        network: AttentionDiffusionRNN instance.
        inputs: Input sequence of shape (time_steps, input_dim).
        targets: Target sequence of shape (time_steps, output_dim).
        apply_attention: Whether to apply attention mechanism.
        apply_diffusion: Whether to apply diffusion mechanism.
        warmup_steps: Number of initial steps to discard.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    network.reset_state()
    
    # Run warmup
    if warmup_steps > 0:
        for t in range(warmup_steps):
            network.step(
                inputs[t],
                apply_attention=apply_attention,
                apply_diffusion=apply_diffusion
            )
    
    # Collect predictions
    result = network.forward(
        inputs[warmup_steps:],
        apply_attention=apply_attention,
        apply_diffusion=apply_diffusion
    )
    
    predictions = result['outputs']
    targets_eval = targets[warmup_steps:]
    
    # Compute metrics
    metrics = {
        'mse': compute_mse(predictions, targets_eval),
        'nmse': compute_nmse(predictions, targets_eval),
        'correlation': compute_correlation(predictions, targets_eval)
    }
    
    return metrics


def analyze_attention(
    attentions: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """Analyze attention patterns.
    
    Args:
        attentions: Attention vectors of shape (time_steps, input_dim).
        threshold: Threshold for considering an input dimension as "attended".
        
    Returns:
        Dictionary containing attention statistics.
    """
    mean_attention = np.mean(attentions, axis=0)
    std_attention = np.std(attentions, axis=0)
    max_attention = np.max(attentions, axis=0)
    min_attention = np.min(attentions, axis=0)
    
    # Compute attention sparsity (fraction of time each dimension is attended)
    attended = (attentions > threshold).astype(float)
    sparsity = np.mean(attended, axis=0)
    
    return {
        'mean': mean_attention,
        'std': std_attention,
        'max': max_attention,
        'min': min_attention,
        'sparsity': sparsity
    }


def analyze_state_dynamics(
    states: np.ndarray
) -> Dict[str, float]:
    """Analyze recurrent state dynamics.
    
    Args:
        states: Recurrent states of shape (time_steps, recurrent_dim).
        
    Returns:
        Dictionary containing dynamics statistics.
    """
    # Compute activity statistics
    mean_activity = np.mean(np.abs(states))
    max_activity = np.max(np.abs(states))
    
    # Compute temporal correlation (autocorrelation at lag 1)
    if len(states) > 1:
        autocorr = np.mean([
            np.corrcoef(states[:-1, i], states[1:, i])[0, 1]
            for i in range(states.shape[1])
            if np.std(states[:, i]) > 1e-10
        ])
        autocorr = autocorr if not np.isnan(autocorr) else 0.0
    else:
        autocorr = 0.0
    
    # Compute effective dimensionality (participation ratio)
    state_cov = np.cov(states.T)
    eigenvalues = np.linalg.eigvalsh(state_cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) > 0:
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    else:
        participation_ratio = 0.0
    
    return {
        'mean_activity': mean_activity,
        'max_activity': max_activity,
        'autocorrelation': autocorr,
        'participation_ratio': participation_ratio
    }


def generate_synthetic_task(
    task_type: str,
    n_samples: int,
    input_dim: int,
    output_dim: int,
    seq_length: int,
    noise_level: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic task data for testing.
    
    Args:
        task_type: Type of task ('memory', 'pattern', 'regression').
        n_samples: Number of samples.
        input_dim: Input dimension.
        output_dim: Output dimension.
        seq_length: Sequence length.
        noise_level: Noise level to add to inputs.
        seed: Random seed.
        
    Returns:
        Tuple of (inputs, targets).
    """
    if seed is not None:
        np.random.seed(seed)
    
    if task_type == 'memory':
        # Memory task: recall input after delay
        inputs = np.random.randn(n_samples, seq_length, input_dim)
        targets = np.roll(inputs[:, :, :output_dim], seq_length // 2, axis=1)
    
    elif task_type == 'pattern':
        # Pattern recognition: detect specific patterns
        inputs = np.random.randn(n_samples, seq_length, input_dim)
        pattern = np.random.randn(input_dim)
        targets = np.zeros((n_samples, seq_length, output_dim))
        
        for i in range(n_samples):
            similarity = inputs[i] @ pattern
            targets[i, :, 0] = similarity / np.linalg.norm(pattern)
    
    elif task_type == 'regression':
        # Nonlinear regression task
        inputs = np.random.randn(n_samples, seq_length, input_dim)
        targets = np.zeros((n_samples, seq_length, output_dim))
        
        for i in range(n_samples):
            for t in range(seq_length):
                targets[i, t] = np.tanh(inputs[i, max(0, t-5):t+1].sum(axis=0)[:output_dim])
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Add noise
    if noise_level > 0:
        inputs += noise_level * np.random.randn(*inputs.shape)
    
    return inputs, targets
