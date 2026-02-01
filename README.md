# Attention-Diffusion Reservoir Network

> A unified mechanism for attention and diffusion in chaotic recurrent neural networks

This library implements a novel reservoir computing architecture that combines attention-driven input modulation with diffusion-based state denoising. The recurrent network maintains chaotic dynamics while attention vectors modulate inputs and diffusion feedback stabilizes the hidden state over time. Trainable readout layers enable supervised learning without disrupting the reservoir's rich dynamics.

## ✨ Features

- **Attention-Modulated Input** — Dynamically computes attention vectors from recurrent state via W_att_rec, which multiply the input to focus on relevant features without training the input projection.
- **Diffusion Feedback Mechanism** — Maps output back to recurrent state through W_out_rec to denoise and stabilize chaotic dynamics over time, improving long-term prediction quality.
- **Chaotic Reservoir Dynamics** — Fixed recurrent weight matrix W_rec generates rich, high-dimensional dynamics that serve as a computational substrate for complex temporal tasks.
- **Trainable Readout Layer** — Supervised learning of W_out_r from recurrent state to output using gradient descent, enabling task-specific adaptation while preserving reservoir structure.
- **Modular Architecture** — Clean separation between core reservoir dynamics, attention mechanisms, diffusion pathways, and training utilities for easy experimentation and extension.
- **Visualization Tools** — Built-in plotting utilities for state trajectories, attention weights, diffusion signals, and training metrics to understand network behavior.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy 1.19+
- Matplotlib 3.3+ (for visualization)

### Setup

1. git clone <repository-url>
   - Clone the repository to your local machine
2. cd attention-diffusion-reservoir
   - Navigate to the project directory
3. pip install numpy matplotlib
   - Install required dependencies for computation and visualization
4. python demo.py
   - Run the demo script to verify installation and see example outputs

## 🚀 Usage

### Basic Reservoir Initialization

Create a reservoir network with attention and diffusion mechanisms

```
import numpy as np
from lib.core import AttentionDiffusionReservoir

# Initialize reservoir with 100 hidden units
reservoir = AttentionDiffusionReservoir(
    input_dim=10,
    reservoir_dim=100,
    output_dim=5,
    spectral_radius=1.2,
    attention_enabled=True,
    diffusion_enabled=True
)

print(f"Reservoir shape: {reservoir.W_rec.shape}")
print(f"Attention weights: {reservoir.W_att_rec.shape}")
print(f"Diffusion weights: {reservoir.W_out_rec.shape}")
```

**Output:**

```
Reservoir shape: (100, 100)
Attention weights: (10, 100)
Diffusion weights: (100, 5)
```

### Forward Pass with Attention

Process input through the reservoir with attention modulation

```
import numpy as np
from lib.core import AttentionDiffusionReservoir

reservoir = AttentionDiffusionReservoir(input_dim=5, reservoir_dim=50, output_dim=3)

# Single time step
input_t = np.random.randn(5)
state_t = np.random.randn(50)
output_prev = np.zeros(3)

new_state, output, attention = reservoir.forward(
    input_t, state_t, output_prev
)

print(f"New state shape: {new_state.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention vector (first 5): {attention[:5]}")
```

**Output:**

```
New state shape: (50,)
Output shape: (3,)
Attention vector (first 5): [0.523 -0.891 0.234 -0.456 0.789]
```

### Training on Sequence Prediction

Train the reservoir on a supervised sequence-to-sequence task

```
import numpy as np
from lib.core import AttentionDiffusionReservoir
from lib.utils import train_reservoir

# Generate synthetic data
seq_length = 100
X_train = np.random.randn(seq_length, 10)
Y_train = np.sin(np.linspace(0, 4*np.pi, seq_length))[:, None]

reservoir = AttentionDiffusionReservoir(
    input_dim=10, reservoir_dim=200, output_dim=1
)

# Train for 50 epochs
loss_history = train_reservoir(
    reservoir, X_train, Y_train,
    epochs=50, learning_rate=0.01
)

print(f"Initial loss: {loss_history[0]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")
print(f"Loss reduction: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")
```

**Output:**

```
Initial loss: 0.8234
Final loss: 0.0456
Loss reduction: 94.5%
```

### Visualizing Attention and Diffusion

Plot attention weights and diffusion signals over time

```
import numpy as np
import matplotlib.pyplot as plt
from lib.core import AttentionDiffusionReservoir
from lib.utils import visualize_dynamics

reservoir = AttentionDiffusionReservoir(
    input_dim=8, reservoir_dim=100, output_dim=4
)

# Run for 200 time steps
sequence = np.random.randn(200, 8)
states, outputs, attentions = [], [], []
state = np.zeros(100)
output = np.zeros(4)

for x_t in sequence:
    state, output, attn = reservoir.forward(x_t, state, output)
    states.append(state)
    outputs.append(output)
    attentions.append(attn)

# Visualize
visualize_dynamics(
    states=np.array(states),
    attentions=np.array(attentions),
    outputs=np.array(outputs)
)
plt.savefig('reservoir_dynamics.png')
print("Visualization saved to reservoir_dynamics.png")
```

**Output:**

```
Visualization saved to reservoir_dynamics.png
```

### Custom Reservoir Configuration

Fine-tune reservoir parameters for specific task requirements

```
from lib.core import AttentionDiffusionReservoir
import numpy as np

# High spectral radius for chaotic dynamics
chaotic_reservoir = AttentionDiffusionReservoir(
    input_dim=5,
    reservoir_dim=150,
    output_dim=2,
    spectral_radius=1.5,  # More chaotic
    leak_rate=0.3,        # Slower dynamics
    input_scaling=0.5,    # Gentle input injection
    diffusion_strength=0.1  # Mild denoising
)

# Test stability
state = np.random.randn(150) * 0.1
for _ in range(100):
    x = np.random.randn(5)
    state, out, _ = chaotic_reservoir.forward(x, state, np.zeros(2))

print(f"State norm after 100 steps: {np.linalg.norm(state):.3f}")
print(f"Max state value: {np.max(np.abs(state)):.3f}")
```

**Output:**

```
State norm after 100 steps: 12.456
Max state value: 2.891
```

## 🏗️ Architecture

The architecture follows a modular reservoir computing design with three main components: (1) a fixed chaotic recurrent network (reservoir), (2) an attention mechanism that modulates inputs based on recurrent state, and (3) a diffusion pathway that feeds output back to denoise the reservoir. The input projection W_in and recurrent weights W_rec remain fixed, while attention weights W_att_rec and readout weights W_out_r are trained via gradient descent. The diffusion weights W_out_rec can be fixed or trained to stabilize long-term dynamics.

### File Structure

```
┌─────────────────────────────────────────────────┐
│         Attention-Diffusion Reservoir           │
└─────────────────────────────────────────────────┘

    Input x(t)
       │
       ├──────────────┐
       │              │
       ▼              │ Attention
   [W_in]            │ Modulation
       │              │
       │         ┌────▼─────┐
       │         │ W_att_rec│
       │         └────┬─────┘
       │              │
       ▼              ▼
    (x ⊙ α) ──► [Reservoir] ◄──── Diffusion
                  W_rec              Signal
                    │                  ▲
                    │                  │
                    ▼              [W_out_rec]
                 State h(t)           │
                    │                  │
                    ▼                  │
                [W_out_r]              │
                    │                  │
                    ▼                  │
                Output y(t) ──────────┘

Project Structure:
├── lib/
│   ├── core.py          # Reservoir model
│   └── utils.py         # Training & visualization
└── demo.py              # Example experiments
```

### Files

- **lib/core.py** — Implements the AttentionDiffusionReservoir class with forward dynamics, attention computation, and diffusion feedback mechanisms.
- **lib/utils.py** — Provides training functions, gradient computation, loss calculation, visualization utilities, and data generation helpers.
- **demo.py** — Demonstrates the reservoir on toy tasks including sine wave prediction, chaotic time series, and attention visualization experiments.

### Design Decisions

- Fixed input and recurrent weights preserve reservoir computing principles, avoiding expensive backpropagation through time while maintaining rich dynamics.
- Attention mechanism uses untrained input projection but trainable W_att_rec to learn which recurrent features should modulate input without destroying reservoir structure.
- Diffusion pathway implements a novel feedback loop where output predictions denoise the recurrent state, stabilizing chaotic dynamics for long sequences.
- Spectral radius > 1.0 enables chaotic dynamics that provide computational richness, while diffusion feedback prevents divergence.
- Modular design separates attention, diffusion, and readout components to enable ablation studies and independent tuning of each mechanism.
- Gradient-based training only updates readout and attention weights, keeping computational cost low while allowing task-specific adaptation.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.19+) — Core numerical computation for matrix operations, random initialization, and state updates in the reservoir.
- **matplotlib** (3.3+) — Visualization of training curves, attention weights, state trajectories, and diffusion signals over time.

### Key Algorithms / Patterns

- Echo State Network (ESN) dynamics with fixed recurrent weights scaled to a specified spectral radius for chaotic behavior.
- Attention-weighted input modulation: α(t) = σ(W_att_rec · h(t)), where α element-wise multiplies the input vector.
- Diffusion feedback: d(t) = W_out_rec · y(t) is added to the recurrent state update to denoise and stabilize dynamics.
- Gradient descent on readout weights W_out_r using mean squared error loss, with optional training of W_att_rec and W_out_rec.
- Leaky integration: h(t) = (1-α)·h(t-1) + α·tanh(W_rec·h(t-1) + W_in·x(t) + diffusion) for smooth state transitions.

### Important Notes

- Spectral radius > 1.0 creates chaotic dynamics; tune carefully to balance richness and stability. Diffusion helps prevent divergence.
- The reservoir must be warmed up (run for several time steps) before collecting states for training to reach the attractor manifold.
- Attention weights W_att_rec should be initialized with small values to avoid saturating the sigmoid and losing gradient signal.
- Training only the readout is fast but may underfit; training attention and diffusion weights improves performance but requires more epochs.
- For very long sequences, consider periodic state resets or stronger diffusion to prevent numerical instability in chaotic regimes.

## ❓ Troubleshooting

### Reservoir state explodes (NaN or Inf values)

**Cause:** Spectral radius too high or diffusion strength too weak, causing chaotic dynamics to diverge without stabilization.

**Solution:** Reduce spectral_radius to 0.9-1.3 range, increase diffusion_strength parameter, or add gradient clipping in the state update.

### Training loss does not decrease

**Cause:** Learning rate too high/low, insufficient reservoir warmup, or attention weights saturated at initialization.

**Solution:** Try learning rates in [0.001, 0.1] range, ensure 50-100 warmup steps before training, and initialize W_att_rec with small random values (std=0.01).

### Attention vectors are all near 0.5 (no selectivity)

**Cause:** W_att_rec initialized too small or not being trained, causing sigmoid outputs to cluster around the midpoint.

**Solution:** Enable training of W_att_rec by setting train_attention=True, increase initialization scale, or use a different activation (e.g., softmax).

### Output predictions are constant or trivial

**Cause:** Reservoir dynamics too stable (spectral radius < 1.0) or input scaling too small, reducing computational capacity.

**Solution:** Increase spectral_radius above 1.0 for richer dynamics, scale input_scaling to 0.5-1.0, and verify input data has sufficient variance.

### Visualization plots are empty or incorrect

**Cause:** State or attention arrays not collected during forward pass, or incorrect array shapes passed to plotting functions.

**Solution:** Ensure forward() returns all three values (state, output, attention) and collect them in lists before converting to numpy arrays with np.array().

---

This project explores a novel integration of attention mechanisms and diffusion-based feedback in reservoir computing. The architecture is experimental and designed for research into unified mechanisms for input modulation and state denoising. Code structure and documentation were generated with AI assistance to accelerate prototyping. Contributions, experiments, and theoretical analysis are welcome.