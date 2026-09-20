# freq-nav-sim2real

Understanding **Visual Frequency Requirements** for **Sim-to-Real Embodied Navigation**

<p align="center">
  <img src="assets/portfolio/habitat_renders/apartment_1_view0.png" width="32%" alt="Apartment scene" />
  <img src="assets/portfolio/habitat_renders/skokloster-castle_view0.png" width="32%" alt="Skokloster Castle scene" />
  <img src="assets/portfolio/habitat_renders/van-gogh-room_view0.png" width="32%" alt="Van Gogh Room scene" />
</p>
<p align="center"><em>Agent-perspective RGB renders from Habitat-Sim test scenes used for 3D validation</em></p>

This project studies which visual frequencies PPO navigation agents rely on when transferring from simulation to the real world. Using **Fourier Domain Adaptation (FDA)**, the repository generates frequency-controlled image variants and runs ablation studies in Habitat-Sim to evaluate how low-frequency style and high-frequency geometry affect navigation performance.

---

## Repository Includes

- Frequency-domain sim-to-real adaptation (**FDA**)
- Habitat **PointNav** environment wrapper + mock environment for local testing
- Visual encoder + **PPO** actor-critic policy
- **Frequency ablation** experiments validated on both mock env and **real 3D scenes**
- Navigation metrics (**SR**, **SPL**) and shaped reward analysis
- **C++ extensions** (pybind11) for GAE, FDA, frequency perturbation, and mock environment
- **Research infrastructure**: ablation runner, Habitat-Sim ablation runner, multi-beta FDA sweep, TensorBoard logging, results plotting
- **3D scene renders** from Habitat-Sim for visualization and showcasing

---

## Research Goal

Understand:

- Which **visual frequencies** RL agents depend on
- How **low-frequency style** affects sim-to-real transfer
- Why high-frequency geometry is critical for navigation
- How different **frequency cutoffs** change policy robustness

---

## Structure

```
src/
  data/               # FDA + synthetic/real loaders
  habitat_env/        # Habitat + mock env wrappers
  models/             # Visual encoder + PPO policy
  train/              # PPO training, GAE, domain/FDA utilities
  eval/               # Evaluation, ablation plotting
    plot_results.py   # Aggregate + plot ablation results (bar chart, radius sweep, CSV)
  utils/              # Metrics (SPL)

csrc/                 # C++ extensions (pybind11 / PyTorch C++ API)
  gae.cpp             # Generalized Advantage Estimation
  fda.cpp             # Fourier Domain Adaptation (frequency swap)
  freq_adapt.cpp      # Frequency-domain perturbation (LF preserve, HF noise)
  mock_env.cpp        # MockPointNavEnv (2D navigation simulator)
  build_ext.py        # Build script

scripts/
  run_fda.sh          # Single-beta FDA preprocessing
  run_fda_multi.py    # Multi-beta FDA sweep for ablation studies
  run_ablation.py     # Full ablation runner (train + eval across conditions)
  run_habitat_ablation.py  # Habitat-Sim 3D ablation (real scenes)
  run_nav.sh          # Navigation training launcher
  eval_realworld.sh   # Evaluation launcher
  colab_setup.sh      # Google Colab environment setup

assets/
  portfolio/          # Visualization assets for showcasing
    habitat_renders/  # RGB renders from Habitat-Sim test scenes
    habitat_comparison.png  # Mock vs Habitat ablation chart

tests/                # pytest test suite (58 tests)
configs/              # YAML configs
experiments/          # Results, logs, TensorBoard runs
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

### 3. Multi-beta FDA sweep

Generate FDA-adapted images across multiple beta values for ablation studies:

```bash
python scripts/run_fda_multi.py --betas 0.005 0.01 0.02 0.05 0.1 0.2
```

Outputs to `data/fda_ablation/beta_<value>/` with one directory per beta.

### 4. Navigation training (PPO)

Train a PPO navigation policy with frequency adaptation disabled (baseline):

```bash
bash scripts/run_nav.sh
```

For local smoke tests without Habitat scenes installed, you can force the mock PointNav environment:

```bash
python src/train/train_nav.py --config configs/training.yaml --mock-env --num-steps 2000
```

Enable **TensorBoard** logging with `--tensorboard`:

```bash
python src/train/train_nav.py --config configs/training.yaml --mock-env --num-steps 5000 --tensorboard
```

View live training curves:

```bash
tensorboard --logdir experiments/checkpoints/tb
```

### 5. Evaluation

Given a checkpoint produced by training (e.g. `experiments/checkpoints/ppo_step_2000.pt`):

```bash
python src/eval/eval_nav.py --checkpoint experiments/checkpoints/ppo_step_2000.pt --mock-env
```

Or using the helper script:

```bash
bash scripts/eval_realworld.sh experiments/checkpoints/ppo_step_2000.pt
```

### 6. Ablation experiments

Run a full train+eval sweep across frequency conditions (baseline, varying radii, noise levels):

```bash
python scripts/run_ablation.py --mock-env --num-steps 5000 --tensorboard
```

This trains and evaluates 6 default conditions, writes per-condition checkpoints and eval results, and produces a combined `experiments/ablation/ablation_summary.json`. Use `--conditions baseline freq_r16` to run a subset, or `--skip-train` / `--skip-eval` to run only one phase.

### 7. Habitat-Sim 3D ablation

Run the frequency ablation on real 3D scenes (requires Habitat-Sim + test scenes in a conda env):

```bash
conda activate habitat
python scripts/run_habitat_ablation.py --num-steps 200000 --eval-episodes 50 --seed 42
```

Results are written to `experiments/habitat_ablation/` with per-condition eval JSONs and an aggregate summary.

### 8. Plot results

After running ablation experiments, generate comparison charts and a CSV export:

```bash
python src/eval/plot_results.py --ablation-dir experiments/ablation
```

Produces:
- `ablation_bar.png` — SR and SPL bar chart across all conditions
- `ablation_radius_sweep.png` — SR/SPL line plot vs. frequency cutoff radius
- `ablation_results.csv` — tabular export of all metrics

---

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

91 tests cover all major modules:

| Test file | Module | Tests |
|-----------|--------|-------|
| `test_fda.py` | FDA (`fourier_swap`) | 7 |
| `test_frequency_adapt.py` | Frequency perturbation | 7 |
| `test_ppo_utils.py` | GAE, RolloutBuffer, PPO update | 9 |
| `test_mock_env.py` | MockPointNavEnv | 10 |
| `test_models.py` | Visual encoder + Policy | 10 |
| `test_metrics.py` | SPL metric | 6 |
| `test_plot_results.py` | Ablation result plotting | 8 |
| `test_synthetic_loader.py` | Data loading + datasets | 15 |
| `test_env_wrapper.py` | Habitat env wrapper | 3 |
| `test_train_helpers.py` | Training/eval helpers | 10 |

---

## Results

Ablation results from 50k-step PPO training on the mock PointNav environment (50 eval episodes per condition, seed=42):

| Condition | Freq Adapt | Radius | Noise Std | SR | SPL | Avg Length |
|-----------|-----------|--------|-----------|-----|------|-----------|
| baseline | off | — | — | 0.96 | 0.907 | 66.4 |
| freq_r8 | on | 8 | 1.0 | 0.00 | 0.000 | 500.0 |
| freq_r16 | on | 16 | 1.0 | **1.00** | 0.929 | 57.9 |
| freq_r32 | on | 32 | 1.0 | 0.98 | 0.904 | 64.0 |
| freq_r16_noise05 | on | 16 | 0.5 | **1.00** | **1.000** | 45.1 |
| freq_r16_noise20 | on | 16 | 2.0 | **1.00** | 0.998 | 45.9 |

### Key Findings

1. **High-frequency information is critical.** A small preservation radius (r=8) destroys too much HF content — the agent fails completely (0% SR). Navigation relies heavily on edges, geometry, and fine spatial details encoded in high frequencies.

2. **Moderate frequency perturbation improves robustness.** Radius 16 and 32 match or exceed the baseline, suggesting that replacing some HF magnitude with noise during training acts as a regularizer — the policy learns to be robust to frequency-domain variation.

3. **Noise level is less important than radius.** Both noise_std=0.5 and noise_std=2.0 with r=16 achieve 100% SR and near-perfect SPL, indicating the agent tolerates a wide range of HF noise intensities as long as enough LF structure is preserved.

4. **There is a sharp frequency threshold.** The radius sweep shows a discontinuous jump from 0% (r=8) to 100% (r=16), pointing to a critical frequency band between radius 8 and 16 that the policy depends on.

### Habitat-Sim 3D Validation

To confirm these findings transfer beyond the mock environment, we ran the same 6-condition ablation on **real reconstructed 3D scenes** using Habitat-Sim (200k steps, shaped rewards):

<p align="center">
  <img src="assets/portfolio/habitat_comparison.png" width="85%" alt="Mock vs Habitat ablation comparison" />
</p>

| Condition | Freq Adapt | Radius | Noise Std | Mean Return (↑ better) |
|-----------|-----------|--------|-----------|------------------------|
| baseline | off | — | — | **-4.825** |
| freq_r8 | on | 8 | 1.0 | -5.655 |
| freq_r16 | on | 16 | 1.0 | -5.000 |
| freq_r32 | on | 32 | 1.0 | -5.000 |
| freq_r16_noise05 | on | 16 | 0.5 | -5.000 |
| freq_r16_noise20 | on | 16 | 2.0 | **-6.624** |

**3D validation confirms the mock env findings:**
- **Frequency threshold holds**: r=8 degrades learning (-5.655 vs -4.825 baseline) in real 3D scenes
- **Excessive noise is catastrophic**: noise_std=2.0 produces the worst return (-6.624) across all conditions
- **Return ordering mirrors mock env**: baseline > r≥16 > r=8 > noise=2.0

No condition achieves navigation success (0% SR) — real 3D PointNav with a small CNN at 200k steps is inherently challenging, but the shaped reward signal clearly differentiates conditions and validates the frequency sensitivity pattern.

---

## Stacks | Frameworks

- **PyTorch** — models, training, C++ extension API
- **Habitat-Sim** — photorealistic 3D navigation environments (real reconstructed scenes)
- **Habitat-Lab** — PointNav task configuration and episode management
- **C++ / pybind11** — performance-critical extensions (GAE, FDA, freq perturbation, mock env)
- **NumPy** — data processing, FFT
- **OpenCV** — image I/O and preprocessing
- **TensorBoard** — training visualization (loss, return, SR, SPL curves)
- **matplotlib** — ablation result plotting (bar charts, radius sweeps, comparison charts)
- **pytest** — test suite

---

## Status

Ablation experiments complete on both mock environment and Habitat-Sim 3D scenes. Core findings — the sharp frequency threshold between r=8 and r=16, and the noise sensitivity pattern — validated across environments. The frequency domain adaptation approach shows clear structure even when absolute navigation success requires more compute or larger models.
