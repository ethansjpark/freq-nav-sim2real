# freq-nav-sim2real

Understanding **Visual Frequency Requirements** for **Sim-to-Real Embodied Navigation**

This project studies which visual frequencies PPO navigation agents rely on when transferring from simulation to the real world. Using **Fourier Domain Adaptation (FDA)**, the repository generates frequency-controlled image variants and runs ablation studies in Habitat to evaluate how low-frequency style and high-frequency geometry affect navigation performance.

---

## Repository Includes

- Frequency-domain sim-to-real adaptation (**FDA**)
- Habitat **PointNav** environment wrapper
- Visual encoder + **PPO/DD-PPO** policy
- **Frequency ablation** experiments (HF-only, LF-only, mixed)
- Navigation metrics (**SR**, **SPL**)

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
  train/              # PPO training, domain/FDA utilities
  eval/               # Evaluation + ablations
  utils/              # Metrics (SPL) + transforms

configs/              # YAML configs
scripts/              # Launch scripts
experiments/          # Results + logs
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or `.\.venv\Scripts\activate` on Windows (CPU-only)
pip install -r requirements.txt
bash setup.sh
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

- PyTorch
- Habitat-Sim
- NumPy
- Matplotlib
- OpenCV

---

## Status

Research in progress. Focused on FDA, ablations, and navigation modules
