# Attention-Diffusion RNN: Architecture Deep Dive

## What Is This?

This project implements a **Reservoir Computing** architecture enhanced with two novel feedback mechanisms:

1. **Attention**: The reservoir's internal state dynamically gates which input features matter
2. **Diffusion**: The output feeds back to "denoise" and stabilize the chaotic reservoir dynamics

Think of it as an Echo State Network (ESN) that can focus on relevant inputs and self-correct its internal state.

---

## The Core Idea

Traditional reservoir computing uses a fixed, chaotic recurrent network as a "computational substrate" - you only train the output layer. This project adds two feedback loops that let the network:

- **Attend**: Learn which inputs are important at each moment (without training the input weights)
- **Diffuse**: Use its own predictions to stabilize chaotic dynamics (like a denoising autoencoder for the hidden state)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION-DIFFUSION RESERVOIR NETWORK                     │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Input x(t)     │
                              │  [input_dim]     │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    │                  ▼                  │
                    │         ┌────────────────┐          │
                    │         │   ATTENTION    │          │
                    │         │   MECHANISM    │          │
                    │         └────────┬───────┘          │
                    │                  │                  │
                    │    α(t) = σ(W_att_rec · h(t-1))    │
                    │         [input_dim]                 │
                    │                  │                  │
                    │                  ▼                  │
                    │         ┌────────────────┐          │
                    │         │  x(t) ⊙ α(t)   │◄─────────┘
                    │         │ (element-wise) │   Attention gates input
                    │         └────────┬───────┘
                    │                  │
                    │                  ▼
                    │         ┌────────────────┐
                    │         │     W_in       │  FROZEN (random)
                    │         │ [rec × input]  │
                    │         └────────┬───────┘
                    │                  │
                    │                  ▼
┌───────────────────┴──────────────────────────────────────────────────────────┐
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                      CHAOTIC RESERVOIR                               │    │
│   │                                                                      │    │
│   │    h(t) = (1-λ)·h(t-1) + λ·tanh(W_rec·h(t-1) + W_in·[x⊙α])         │    │
│   │                                                                      │    │
│   │    W_rec: FROZEN (scaled to spectral radius > 1 for chaos)          │    │
│   │    λ: leak rate (controls memory vs. reactivity)                    │    │
│   │    [recurrent_dim neurons]                                          │    │
│   │                                                                      │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   State h(t) [recurrent_dim]                                                 │
│                                                                               │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │     W_out      │  ◄── TRAINED (ridge regression)
                          │ [out × rec]    │
                          └────────┬───────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │   Output y(t)  │
                          │  [output_dim]  │
                          └────────┬───────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
           ┌────────────────┐            ┌────────────────┐
           │   Prediction   │            │   DIFFUSION    │
           │    (final)     │            │   FEEDBACK     │
           └────────────────┘            └────────┬───────┘
                                                  │
                                    d(t) = W_out_rec · y(t)
                                         [recurrent_dim]
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │  State correction:       │
                                    │  h(t) += γ·(-h + tanh(d)) │
                                    │                          │
                                    │  (pushes state toward    │
                                    │   lower energy config)   │
                                    └─────────────────────────┘
```

---

## Component Breakdown

### 1. Input Layer (FROZEN)
```
W_in: [recurrent_dim × input_dim]
```
- Random initialization, never trained
- Scaled by `input_scaling / √input_dim`
- Projects input into high-dimensional reservoir space

### 2. Attention Mechanism
```
α(t) = sigmoid(W_att_rec · h(t-1))
modulated_input = x(t) ⊙ α(t)
```
- **W_att_rec**: `[input_dim × recurrent_dim]` - random, frozen
- Reservoir state determines which input dimensions to amplify/suppress
- Sigmoid ensures values in [0, 1] for gating

### 3. Chaotic Reservoir (FROZEN)
```
W_rec: [recurrent_dim × recurrent_dim]
```
- Sparse random matrix scaled to `spectral_radius > 1`
- Creates rich, chaotic dynamics (echo state property at the edge of chaos)
- Leaky integration: `h(t) = (1-λ)h(t-1) + λ·tanh(...)`

### 4. Output Layer (TRAINED)
```
W_out: [output_dim × recurrent_dim]
b_out: [output_dim]
```
- **Only trainable weights in the network**
- Trained via ridge regression (closed-form solution)
- Maps reservoir state to predictions

### 5. Diffusion Mechanism
```
d(t) = W_out_rec · y(t)
correction = -h(t) + tanh(d(t))
h(t) += diffusion_scaling · correction
```
- **W_out_rec**: `[recurrent_dim × output_dim]` - random, frozen
- Output prediction feeds back to "denoise" the reservoir state
- Acts like a score function in diffusion models: pushes state toward lower energy

---

## Why This Architecture?

### The Reservoir Computing Advantage
- **Fast training**: Only train output layer (ridge regression = one matrix solve)
- **Rich dynamics**: Chaotic reservoir provides complex, nonlinear feature extraction
- **Temporal memory**: Recurrent connections capture dependencies across time

### Why Add Attention?
- Standard ESNs treat all inputs equally at all times
- Attention lets the network **dynamically focus** on relevant input features
- The reservoir state "knows" what it needs and gates inputs accordingly

### Why Add Diffusion?
- Chaotic reservoirs can become unstable or noisy
- Diffusion feedback uses the output as a **self-correction signal**
- Similar to denoising: refines the reservoir state based on predictions
- Stabilizes long-term dynamics without losing computational richness

---

## Data Flow Example

```
Time t=0: x=[0.5, -0.3, 0.8, 0.1]  (4D input)
          ↓
          Attention: α=[0.9, 0.2, 0.7, 0.5] (from h(t-1))
          ↓
          Modulated: [0.45, -0.06, 0.56, 0.05]
          ↓
          Reservoir update → h(t) [200D state]
          ↓
          Output: y(t) = W_out · h(t) + b  [2D prediction]
          ↓
          Diffusion: d = W_out_rec · y(t) [200D correction]
          ↓
          State refinement: h(t) adjusted by diffusion
```

---

## Training Pipeline

```
1. WARMUP PHASE (discard first N steps)
   └── Run network to reach attractor manifold
   
2. STATE COLLECTION
   └── Record h(t) for all training sequences (after warmup)
   
3. RIDGE REGRESSION (closed-form)
   └── W_out = (H'H + λI)^(-1) H'Y
   └── Single matrix solve, no backpropagation needed
   
4. EVALUATION
   └── Run on test sequences, measure MSE/correlation
```

---

## Hyperparameters Guide

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `recurrent_dim` | 100-1000 | Network capacity. Larger = more expressive but slower |
| `spectral_radius` | 0.9-1.5 | <1: stable, fading memory. >1: chaotic, long memory |
| `leak_rate` | 0.1-0.9 | Higher = faster dynamics, less memory |
| `input_scaling` | 0.1-1.0 | How strongly inputs drive the reservoir |
| `attention_scaling` | 0.05-0.5 | Strength of attention gating |
| `diffusion_scaling` | 0.05-0.3 | Strength of state correction |

---

## Comparison with Related Methods

| Method | Input Weights | Recurrent | Output | Attention | Diffusion |
|--------|---------------|-----------|--------|-----------|-----------|
| **RNN/LSTM** | Trained | Trained | Trained | ✗ | ✗ |
| **Transformer** | Trained | ✗ | Trained | ✓ | ✗ |
| **ESN** | Frozen | Frozen | Trained | ✗ | ✗ |
| **This Work** | Frozen | Frozen | Trained | ✓ | ✓ |

---

## Key Equations Summary

**Attention:**
$$\alpha(t) = \sigma(W_{att} \cdot h(t-1))$$

**State Update:**
$$h(t) = (1-\lambda) h(t-1) + \lambda \cdot \tanh(W_{rec} h(t-1) + W_{in}(x(t) \odot \alpha(t)))$$

**Output:**
$$y(t) = W_{out} \cdot h(t) + b$$

**Diffusion Correction:**
$$h(t) \leftarrow h(t) + \gamma \cdot (-h(t) + \tanh(W_{out\_rec} \cdot y(t)))$$

---

## File Structure

```
attention-diffusion-rnn/
├── lib/
│   ├── core.py      # AttentionDiffusionRNN class
│   └── utils.py     # Training, evaluation, data generation
├── demo.py          # Example experiments with real datasets
├── ARCHITECTURE.md  # This document
├── README.md        # Quick start guide
└── requirements.txt # Dependencies
```
