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
  fda/                # FDA + Fourier utilities
  habitat_env/        # Habitat wrappers
  data/               # Synthetic + adapted loaders
  models/             # Encoder + policy
  train/              # Training scripts
  eval/               # Evaluation + ablations
  utils/              # Metrics + transforms

configs/              # YAML configs
scripts/              # Launch scripts
experiments/          # Results + logs
```

---

## Setup

```bash
pip install -r requirements.txt
bash setup.sh
```

---

## Quickstart

Apply FDA to source → target:

`bash scripts/run_fda.sh`

Train PPO/DD-PPO navigation:
`bash scripts/run_nav.sh`

---

## Stacks | Frameworks

- PyTorch
- Habitat-Sim
- NumPy
- Matplotlib
- OpenCV

---

## Status

Research in progress. Focused on fDA, ablations, and navigation modules
