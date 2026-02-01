"""Core module for attention-diffusion recurrent network.

This module implements a unifying mechanism for attention and diffusion in a
recurrent network with chaotic reservoir dynamics. The network features:
- A reservoir with chaotic dynamics (untrained recurrent weights)
- Attention mechanism via feedback from recurrent state to input modulation
- Diffusion mechanism via feedback from output to recurrent state denoising
- Trainable readout weights from recurrent to output
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class AttentionDiffusionRNN:
    """Recurrent network with attention and diffusion mechanisms.
    
    The network architecture consists of:
    - Input layer (modulated by attention)
    - Recurrent reservoir with chaotic dynamics (untrained)
    - Output layer with trainable readout weights
    - Attention feedback: recurrent state -> attention vector -> input modulation
    - Diffusion feedback: output -> diffusion vector -> recurrent state denoising
    
    Attributes:
        input_dim (int): Dimension of input vectors.
        recurrent_dim (int): Dimension of recurrent state (reservoir size).
        output_dim (int): Dimension of output vectors.
        W_in (np.ndarray): Input-to-recurrent weights (untrained).
        W_rec (np.ndarray): Recurrent-to-recurrent weights (untrained, chaotic).
        W_out (np.ndarray): Recurrent-to-output weights (trainable).
        W_att_rec (np.ndarray): Recurrent-to-attention weights (feedback).
        W_out_rec (np.ndarray): Output-to-diffusion weights (feedback).
        state (np.ndarray): Current recurrent state.
    """
    
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        output_dim: int,
        spectral_radius: float = 1.5,
        input_scaling: float = 1.0,
        attention_scaling: float = 0.1,
        diffusion_scaling: float = 0.1,
        leak_rate: float = 0.3,
        seed: Optional[int] = None
    ):
        """Initialize the attention-diffusion RNN.
        
        Args:
            input_dim: Dimension of input vectors.
            recurrent_dim: Dimension of recurrent state (reservoir size).
            output_dim: Dimension of output vectors.
            spectral_radius: Spectral radius for recurrent weights (>1 for chaos).
            input_scaling: Scaling factor for input weights.
            attention_scaling: Scaling factor for attention feedback.
            diffusion_scaling: Scaling factor for diffusion feedback.
            leak_rate: Leak rate for recurrent dynamics (0-1).
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.input_dim = input_dim
        self.recurrent_dim = recurrent_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.attention_scaling = attention_scaling
        self.diffusion_scaling = diffusion_scaling
        self.leak_rate = leak_rate
        
        # Initialize untrained weights
        self._initialize_reservoir_weights()
        
        # Initialize trainable readout weights (small random values)
        self.W_out = np.random.randn(output_dim, recurrent_dim) * 0.01
        self.b_out = np.zeros(output_dim)
        
        # Initialize recurrent state
        self.state = np.zeros(recurrent_dim)
        
    def _initialize_reservoir_weights(self) -> None:
        """Initialize untrained reservoir weights with chaotic dynamics."""
        # Input-to-recurrent weights (sparse, random)
        self.W_in = np.random.randn(self.recurrent_dim, self.input_dim)
        self.W_in *= self.input_scaling / np.sqrt(self.input_dim)
        
        # Recurrent weights (sparse, scaled to spectral radius for chaos)
        density = min(10.0 / self.recurrent_dim, 1.0)
        self.W_rec = np.random.randn(self.recurrent_dim, self.recurrent_dim)
        mask = np.random.rand(self.recurrent_dim, self.recurrent_dim) > density
        self.W_rec[mask] = 0
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.W_rec)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            self.W_rec *= self.spectral_radius / current_radius
        
        # Attention feedback weights (recurrent -> attention vector)
        self.W_att_rec = np.random.randn(self.input_dim, self.recurrent_dim)
        self.W_att_rec *= self.attention_scaling / np.sqrt(self.recurrent_dim)
        
        # Diffusion feedback weights (output -> diffusion vector)
        self.W_out_rec = np.random.randn(self.recurrent_dim, self.output_dim)
        self.W_out_rec *= self.diffusion_scaling / np.sqrt(self.output_dim)
    
    def compute_attention(self, state: np.ndarray) -> np.ndarray:
        """Compute attention vector from recurrent state.
        
        Args:
            state: Current recurrent state.
            
        Returns:
            Attention vector (same dimension as input).
        """
        # Linear projection with sigmoid activation for gating
        attention_logits = self.W_att_rec @ state
        attention = 1.0 / (1.0 + np.exp(-attention_logits))  # Sigmoid
        return attention
    
    def compute_diffusion(self, output: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Compute diffusion correction for recurrent state.
        
        The diffusion mechanism acts as a denoising process that refines
        the recurrent state based on the current output.
        
        Args:
            output: Current output vector.
            state: Current recurrent state.
            
        Returns:
            Diffusion correction vector (same dimension as recurrent state).
        """
        # Project output to recurrent space
        diffusion_signal = self.W_out_rec @ output
        
        # Compute correction as a denoising step (pushing state toward lower energy)
        correction = -state + np.tanh(diffusion_signal)
        return correction
    
    def step(
        self,
        input_vec: np.ndarray,
        apply_attention: bool = True,
        apply_diffusion: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform one step of recurrent dynamics.
        
        Args:
            input_vec: Input vector at current time step.
            apply_attention: Whether to apply attention mechanism.
            apply_diffusion: Whether to apply diffusion mechanism.
            
        Returns:
            Tuple of (output, new_state, attention_vector).
        """
        # Compute attention and modulate input
        if apply_attention:
            attention = self.compute_attention(self.state)
            modulated_input = input_vec * attention
        else:
            attention = np.ones(self.input_dim)
            modulated_input = input_vec
        
        # Compute recurrent update (leaky integration)
        input_drive = self.W_in @ modulated_input
        recurrent_drive = self.W_rec @ self.state
        state_update = np.tanh(input_drive + recurrent_drive)
        
        # Leaky integration
        new_state = (1 - self.leak_rate) * self.state + self.leak_rate * state_update
        
        # Compute output (before diffusion)
        output = self.W_out @ new_state + self.b_out
        
        # Apply diffusion correction to state
        if apply_diffusion:
            diffusion_correction = self.compute_diffusion(output, new_state)
            new_state = new_state + self.diffusion_scaling * diffusion_correction
        
        # Update state
        self.state = new_state
        
        return output, new_state, attention
    
    def forward(
        self,
        inputs: np.ndarray,
        apply_attention: bool = True,
        apply_diffusion: bool = True,
        return_states: bool = False
    ) -> Dict[str, np.ndarray]:
        """Process a sequence of inputs.
        
        Args:
            inputs: Input sequence of shape (time_steps, input_dim).
            apply_attention: Whether to apply attention mechanism.
            apply_diffusion: Whether to apply diffusion mechanism.
            return_states: Whether to return recurrent states.
            
        Returns:
            Dictionary containing:
                - 'outputs': Output sequence of shape (time_steps, output_dim).
                - 'attentions': Attention vectors of shape (time_steps, input_dim).
                - 'states': (optional) Recurrent states of shape (time_steps, recurrent_dim).
        """
        time_steps = inputs.shape[0]
        outputs = np.zeros((time_steps, self.output_dim))
        attentions = np.zeros((time_steps, self.input_dim))
        
        if return_states:
            states = np.zeros((time_steps, self.recurrent_dim))
        
        for t in range(time_steps):
            output, state, attention = self.step(
                inputs[t],
                apply_attention=apply_attention,
                apply_diffusion=apply_diffusion
            )
            outputs[t] = output
            attentions[t] = attention
            
            if return_states:
                states[t] = state
        
        result = {
            'outputs': outputs,
            'attentions': attentions
        }
        
        if return_states:
            result['states'] = states
        
        return result
    
    def reset_state(self, state: Optional[np.ndarray] = None) -> None:
        """Reset the recurrent state.
        
        Args:
            state: Optional initial state. If None, resets to zeros.
        """
        if state is None:
            self.state = np.zeros(self.recurrent_dim)
        else:
            assert state.shape == (self.recurrent_dim,), "Invalid state shape"
            self.state = state.copy()
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters.
        
        Returns:
            Dictionary of trainable parameters.
        """
        return {
            'W_out': self.W_out,
            'b_out': self.b_out
        }
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters.
        
        Args:
            parameters: Dictionary of parameters to set.
        """
        if 'W_out' in parameters:
            self.W_out = parameters['W_out'].copy()
        if 'b_out' in parameters:
            self.b_out = parameters['b_out'].copy()
    
    def get_config(self) -> Dict[str, Any]:
        """Get network configuration.
        
        Returns:
            Dictionary of configuration parameters.
        """
        return {
            'input_dim': self.input_dim,
            'recurrent_dim': self.recurrent_dim,
            'output_dim': self.output_dim,
            'spectral_radius': self.spectral_radius,
            'input_scaling': self.input_scaling,
            'attention_scaling': self.attention_scaling,
            'diffusion_scaling': self.diffusion_scaling,
            'leak_rate': self.leak_rate
        }
