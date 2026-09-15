# freq-nav-sim2real

Understanding **Visual Frequency Requirements** for **Sim-to-Real Embodied Navigation**

This project studies which visual frequencies PPO navigation agents rely on when transferring from simulation to the real world. Using **Fourier Domain Adaptation (FDA)**, the repository generates frequency-controlled image variants and runs ablation studies in Habitat to evaluate how low-frequency style and high-frequency geometry affect navigation performance.

---

## Repository Includes

- Frequency-domain sim-to-real adaptation (**FDA**)
- Habitat **PointNav** environment wrapper + mock environment for local testing
- Visual encoder + **PPO** actor-critic policy
- **Frequency ablation** experiments (HF-only, LF-only, mixed)
- Navigation metrics (**SR**, **SPL**)
- **C++ extensions** (pybind11) for GAE, FDA, frequency perturbation, and mock environment

---

## Research Goal

Understand:

- Which **visual frequencies** RL agents depend on
- How **low-frequency style** affects sim-to-real transfer
- Why high-frequency geometry is critical for navigation
- Whether GAN-style methods introduce harmful distortions
- How different **frequency cutoffs** change policy robustness

---

## Structure

```
src/
  data/               # FDA + synthetic/real loaders
  habitat_env/        # Habitat + mock env wrappers, sensors
  models/             # Visual encoder + PPO policy
  train/              # PPO training, GAE, domain/FDA utilities
  eval/               # Evaluation + ablations
  utils/              # Metrics (SPL) + transforms

csrc/                 # C++ extensions (pybind11 / PyTorch C++ API)
  gae.cpp             # Generalized Advantage Estimation
  fda.cpp             # Fourier Domain Adaptation (frequency swap)
  freq_adapt.cpp      # Frequency-domain perturbation (LF preserve, HF noise)
  mock_env.cpp        # MockPointNavEnv (2D navigation simulator)
  build_ext.py        # Build script

configs/              # YAML configs
scripts/              # Launch scripts
experiments/          # Results + logs
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash setup.sh
```

### Building C++ Extensions (optional)

The C++ extensions accelerate GAE, FDA, frequency perturbation, and mock environment stepping. Python fallbacks are used automatically when they are not built.

| Extension | Source | Accelerates |
|---|---|---|
| `gae_cpp` | `gae.cpp` | GAE advantage computation (`ppo_utils.py`) |
| `fda_cpp` | `fda.cpp` | Fourier domain frequency swap (`fda.py`) |
| `freq_adapt_cpp` | `freq_adapt.cpp` | Online frequency perturbation (`frequency_adapt.py`) |
| `mock_env_cpp` | `mock_env.cpp` | Mock PointNav environment (`mock_env.py`) |

```bash
pip install pybind11
cd csrc && python build_ext.py build_ext --inplace && cd ..
```

To use them, add `csrc/` to your Python path:

```bash
export PYTHONPATH="csrc:$PYTHONPATH"
```

---

## Quickstart

### 1. Frequency-domain preprocessing (FDA)

Apply FDA to synthetic → real images using `configs/fda.yaml`:

```bash
bash scripts/run_fda.sh
```

By default this reads from `data/synthetic/`, `data/real/` and writes adapted images into `data/fda_output/`.

### 2. Domain analysis (FDA statistics)

Run a lightweight domain adaptation / FDA statistics pass:

```bash
python src/train/train_domain.py --config configs/fda.yaml --epochs 1 --batch-size 8 --save-images
```

This writes summary metrics to `experiments/domain/domain_summary.json` and sample FDA images under `experiments/domain/samples/`.

### 3. Navigation training (PPO)

Train a PPO navigation policy with frequency adaptation disabled (baseline):

```bash
bash scripts/run_nav.sh
```

For local smoke tests without Habitat scenes installed, you can force the mock PointNav environment:

```bash
python src/train/train_nav.py --config configs/training.yaml --mock-env --num-steps 2000
```

### 4. Evaluation

Given a checkpoint produced by training (e.g. `experiments/checkpoints/ppo_step_2000.pt`):

```bash
python src/eval/eval_nav.py --checkpoint experiments/checkpoints/ppo_step_2000.pt --mock-env
```

Or using the helper script:

```bash
bash scripts/eval_realworld.sh experiments/checkpoints/ppo_step_2000.pt
```

---

## Stacks | Frameworks

- **PyTorch** — models, training, C++ extension API
- **Habitat-Sim** — 3D navigation environments
- **C++ / pybind11** — performance-critical extensions (GAE, FDA, freq perturbation, mock env)
- **NumPy** — data processing, FFT
- **OpenCV** — image I/O and preprocessing

---

## Status

Research in progress. Focused on FDA, ablations, and navigation modules.
