# freq-nav-sim2real: High-Level System Architecture

## Overview

This system studies visual frequency requirements for sim-to-real navigation by:
1. Applying Fourier Domain Adaptation (FDA) to manipulate visual frequencies in synthetic images
2. Training PPO navigation policies on frequency-modified images in Habitat
3. Evaluating how different frequency components affect navigation performance

---

## (1) Data Flow

### Input Data Sources
- **Synthetic Images** (`data/synthetic/*.jpg`): Source domain images from simulation
- **Real-World Images** (`data/real/*.jpg`): Target domain images for FDA adaptation
- **Habitat Scene Datasets**: 3D environments (Matterport3D, Gibson, etc.) for navigation training

### Data Processing Pipeline

```
┌─────────────────┐
│ Synthetic Images│ (data/synthetic/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│   FDA Module    │◄────│ Real Images  │ (data/real/)
│ (fda.py)        │     └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FDA-Adapted     │ (data/fda_output/)
│ Images          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SyntheticDataset│ (synthetic_loader.py)
│ Loader          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Habitat Env     │ (env_wrapper.py)
│ RGB Observations│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visual Encoder  │ (encoder.py)
│ Feature Extract │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Policy Network  │ (policy.py)
│ Actor + Critic  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Actions         │ (forward/left/right)
└─────────────────┘
```

### Key Data Flow Components

1. **Preprocessing Stage** (`src/data/fda.py`):
   - Reads synthetic images from `data/synthetic/`
   - Reads real-world images from `data/real/`
   - Applies FDA transformation (frequency swapping)
   - Writes adapted images to `data/fda_output/`

2. **Data Loading** (`src/data/synthetic_loader.py`):
   - `SyntheticDataset` class loads images from directory
   - Returns images as numpy arrays (via OpenCV)

3. **Environment Interface** (`src/habitat_env/env_wrapper.py`):
   - `HabitatWrapper` wraps Habitat-Lab environment
   - Provides `reset()` → returns RGB observation
   - Provides `step(action)` → returns (RGB, reward, done)
   - RGB observations flow to the policy network

4. **Model Forward Pass**:
   - RGB images → `VisualEncoder` → 512-dim feature vector
   - Feature vector → `Policy` → (actor_logits, critic_value)
   - Actor outputs 3 action logits (forward/left/right)

### Data Transformations
- **Image Resizing** (`src/utils/transforms.py`): `resize()` function for image preprocessing
- **FDA Frequency Manipulation**: Core transformation in `fourier_swap()` function

---

## (2) Visual Frequency Manipulation

### Location: `src/data/fda.py`

The **Fourier Domain Adaptation (FDA)** algorithm is implemented in the `fourier_swap()` function. This is where all visual frequency manipulation occurs.

### Frequency Manipulation Process

1. **Input**: 
   - Source image (synthetic, from `data/synthetic/`)
   - Target image (real-world, from `data/real/`)

2. **Frequency Domain Transformation**:
   ```python
   # Convert to frequency domain via 2D FFT
   src_fft = np.fft.fftshift(np.fft.fft2(src, axes=(0, 1)))
   tgt_fft = np.fft.fftshift(np.fft.fft2(tgt, axes=(0, 1)))
   ```

3. **Low-Frequency Region Extraction**:
   - Calculates frequency cutoff radius: `b = int(min(h, w) * beta)`
   - `beta` parameter (default 0.01) controls the size of the low-frequency region
   - Extracts a square region centered at the DC component (image center in frequency domain)

4. **Frequency Swapping**:
   - Takes low-frequency amplitudes from target (real) image
   - Replaces low-frequency amplitudes in source (synthetic) image
   - Preserves high-frequency components (geometry/details) from source
   - Preserves low-frequency components (style/appearance) from target

5. **Inverse Transformation**:
   ```python
   mixed = np.fft.ifft2(np.fft.ifftshift(mixed_fft), axes=(0, 1))
   mixed = np.real(mixed)  # Take real part
   mixed = np.clip(mixed, 0, 255).astype(np.uint8)  # Clamp to valid range
   ```

### Frequency Manipulation Characteristics

- **Low-Frequency (LF)**: Captures style, color, lighting, texture patterns
- **High-Frequency (HF)**: Captures edges, geometry, fine details, object boundaries
- **Beta Parameter**: Controls the frequency cutoff boundary
  - Smaller beta (e.g., 0.01) = smaller LF region = more HF preserved
  - Larger beta = larger LF region = more style transfer

### Research Purpose

The system uses FDA to create frequency-controlled variants for ablation studies:
- **HF-only**: High-frequency geometry from synthetic, minimal LF style
- **LF-only**: Low-frequency style from real, minimal HF geometry  
- **Mixed**: Various beta values to test frequency sensitivity

### Configuration
- Controlled via `configs/fda.yaml`:
  - `beta: 0.01` - Frequency cutoff parameter
  - `image_size: 224` - Target image dimensions
  - Input/output directories

---

## (3) RL Policy Learning

### Location: Intended in `src/train/train_nav.py` (currently placeholder)

### Policy Architecture

The policy network is defined in `src/models/policy.py`:

1. **Visual Encoder** (`src/models/encoder.py`):
   - Input: RGB images (3 channels, 224x224)
   - Architecture: Simple CNN with 2 convolutional layers
     - Conv2d(3→16, stride=2) → ReLU
     - Conv2d(16→32, stride=2) → ReLU
     - Flatten → Linear(32×54×54 → 512)
   - Output: 512-dimensional feature vector
   - Note: Code comments indicate this should be replaced with ResNet18 or ViT

2. **Policy Network** (`src/models/policy.py`):
   - Input: 512-dim feature vector from encoder
   - Actor Head: `Linear(512 → 3)` - outputs action logits
     - 3 discrete actions: forward, left, right
   - Critic Head: `Linear(512 → 1)` - outputs value estimate
   - Forward pass: `(actor_logits, critic_value) = policy(encoder(obs))`

### Training Configuration

From `configs/training.yaml`:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Training Steps**: 500,000
- **Batch Size**: 64
- **Learning Rate**: 3e-4
- **Discount Factor (gamma)**: 0.99
- **DD-PPO**: Flag exists but not implemented (`use_ddppo: false`)

### Environment Interface

`src/habitat_env/env_wrapper.py` provides the RL environment:
- **Task**: PointNav (PointGoal Navigation)
- **Observations**: RGB images from Habitat-Sim
- **Actions**: Discrete navigation actions (forward/left/right)
- **Reward**: Navigation reward signal from Habitat
- **Episode Length**: Max 500 steps (from `habitat_pointnav.yaml`)
- **Success Criteria**: Distance to goal < 0.2m

### Training Loop (Intended Flow)

The training script `src/train/train_nav.py` is currently a placeholder. The intended flow would be:

1. **Initialize**:
   - Load Habitat environment with scene dataset
   - Initialize Policy network (encoder + actor/critic)
   - Load FDA-adapted images or apply FDA on-the-fly

2. **PPO Training Loop**:
   - Collect rollouts: agent interacts with environment, stores (obs, action, reward, done)
   - Compute advantages using GAE (Generalized Advantage Estimation)
   - Update policy using PPO clipped objective
   - Update value function (critic) using TD error
   - Repeat for 500,000 steps

3. **Checkpointing** (not implemented):
   - Save model checkpoints periodically
   - Save training logs/metrics

### Current Status

- **Policy Architecture**: ✅ Defined (encoder + policy)
- **Environment Wrapper**: ✅ Implemented (HabitatWrapper)
- **Training Loop**: ❌ Placeholder only
- **DD-PPO Support**: ❌ Not implemented
- **Checkpointing**: ❌ Not implemented

---

## (4) Evaluation

### Location: Intended in `src/eval/eval_nav.py` (currently missing)

### Evaluation Metrics

**Success Rate (SR)**:
- Binary metric: Did agent reach goal within success distance (0.2m)?
- Computed as: `num_successful_episodes / total_episodes`

**Success weighted by Path Length (SPL)**:
- Implemented in `src/utils/metrics.py`:
  ```python
  def compute_spl(success, path_len, shortest_path):
      if not success:
          return 0.0
      return (shortest_path / max(path_len, shortest_path))
  ```
- Measures efficiency: successful episodes get higher SPL if they take shorter paths
- Range: [0, 1], where 1.0 = optimal path

### Evaluation Script

`scripts/eval_realworld.sh` references `src/eval/eval_nav.py`, but this file does not exist.

**Intended Evaluation Flow** (inferred from script):
1. Load trained checkpoint
2. Initialize policy network with checkpoint weights
3. Run episodes in Habitat environment (or real-world if supported)
4. Collect metrics: success rate, SPL, path lengths
5. Report results

### Evaluation Scenarios

Based on README research goals, evaluation should test:
- **Frequency Ablations**: 
  - HF-only images (high-frequency geometry preserved)
  - LF-only images (low-frequency style preserved)
  - Mixed frequencies (various beta values)
- **Sim-to-Real Transfer**:
  - Performance on synthetic images
  - Performance on FDA-adapted images
  - Performance on real-world images (if real-world evaluation is supported)

### Current Status

- **Metrics Implementation**: ✅ SPL computation exists
- **Evaluation Script**: ❌ Missing (`src/eval/eval_nav.py`)
- **Evaluation Infrastructure**: ❌ Not implemented
- **Real-World Evaluation**: ❌ Not implemented (script name suggests intent)

---

## System Architecture Summary

### Component Status

| Component | Status | Location |
|-----------|--------|----------|
| FDA Frequency Manipulation | ✅ Implemented | `src/data/fda.py` |
| Data Loading | ✅ Implemented | `src/data/synthetic_loader.py` |
| Visual Encoder | ✅ Implemented | `src/models/encoder.py` |
| Policy Network | ✅ Implemented | `src/models/policy.py` |
| Habitat Wrapper | ✅ Implemented | `src/habitat_env/env_wrapper.py` |
| PPO Training Loop | ❌ Placeholder | `src/train/train_nav.py` |
| Evaluation Script | ❌ Missing | `src/eval/eval_nav.py` |
| Metrics (SPL) | ✅ Implemented | `src/utils/metrics.py` |

### Data Flow Summary

1. **Preprocessing**: Synthetic + Real images → FDA → Adapted images
2. **Training**: Adapted images → Habitat Env → Encoder → Policy → Actions
3. **Evaluation**: Trained Policy → Habitat Env → Metrics (SR, SPL)

### Research Pipeline

The system is designed to answer: "Which visual frequencies do navigation agents rely on?"

1. Generate frequency-controlled image variants via FDA (different beta values)
2. Train policies on each variant
3. Evaluate performance to identify frequency dependencies
4. Compare sim-to-real transfer performance across frequency conditions
