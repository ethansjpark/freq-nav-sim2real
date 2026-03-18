# freq-nav-sim2real: End-to-End Runtime Analysis

## Dependency List

### Explicit Dependencies (from `requirements.txt`)
- **numpy** - Numerical operations for FDA and data processing
- **opencv-python** - Image I/O and preprocessing (FDA, transforms)
- **torch** - PyTorch for neural network models (encoder, policy)
- **habitat-sim** - 3D simulation engine for navigation environments
- **habitat-lab** - High-level RL API and task definitions
- **pyyaml** - Configuration file parsing

### Implicit/Inferred Dependencies
- **Python 3.7+** - Required for Habitat-Sim compatibility
- **CMake** - Build dependency for Habitat-Sim (if building from source)
- **Magnum** - Graphics engine used by Habitat-Sim
- **Bullet Physics** - Physics engine for Habitat-Sim
- **Assimp** - 3D model loading (via Habitat-Sim)
- **CUDA Toolkit 10.2+** - Required for GPU-accelerated Habitat-Sim rendering
- **cuDNN** - Deep learning primitives (if using GPU training)
- **Pillow** - Image processing (likely used by Habitat-Lab)
- **gym** or **gymnasium** - RL environment interface (used by Habitat-Lab)

### Missing from requirements.txt (should be added)
- **matplotlib** - Mentioned in README but not in requirements
- **tensorboard** or **wandb** - Typically needed for training logs (not present but likely needed)
- **tqdm** - Progress bars (common in training scripts)

### Data Dependencies
- **Synthetic images** - Expected in `data/synthetic/*.jpg` (source domain)
- **Real-world images** - Expected in `data/real/*.jpg` (target domain for FDA)
- **Habitat scene datasets** - 3D meshes/scenes for navigation (not explicitly configured, but required for Habitat-Sim to run)

---

## Runtime Assumptions

### Operating System
- **Primary**: Linux (Ubuntu 18.04+ recommended for Habitat-Sim)
- **Secondary**: macOS (may work but Habitat-Sim GPU support is limited)
- **Not supported**: Windows (Habitat-Sim has limited/no Windows support)

### GPU Requirements
- **Required for Habitat-Sim**: 
  - NVIDIA GPU with CUDA compute capability 6.0+ (Pascal or newer)
  - Minimum 4GB VRAM (6GB+ recommended for complex scenes)
  - CUDA 10.2, 11.0, or 11.1 (Habitat-Sim compatibility)
- **Required for Training**:
  - GPU strongly recommended for PPO training (PyTorch)
  - 8GB+ VRAM recommended for batch training with visual encoders
- **Optional for FDA preprocessing**: CPU is sufficient, but GPU can accelerate if using PyTorch-based FFT

### CUDA/CuDNN
- **CUDA Toolkit**: 10.2, 11.0, or 11.1 (must match Habitat-Sim build)
- **cuDNN**: Version compatible with CUDA version (7.6+ for CUDA 10.2, 8.0+ for CUDA 11.x)
- **Note**: Habitat-Sim binaries are pre-built for specific CUDA versions. Must install matching CUDA or build from source.

### System Resources
- **RAM**: 16GB+ recommended (8GB minimum)
- **Disk**: 10GB+ for Habitat scene datasets, additional space for experiments/logs
- **CPU**: Multi-core recommended for data loading and preprocessing

### Python Environment
- **Python**: 3.7, 3.8, or 3.9 (Habitat-Sim compatibility)
- **Virtual environment**: Recommended (venv/conda)

---

## Minimal Smoke Test Plan

### Phase 1: Environment Setup Verification
1. **Python & Package Installation**
   ```bash
   python --version  # Should be 3.7-3.9
   pip install -r requirements.txt
   python -c "import torch; import habitat_sim; import habitat; print('OK')"
   ```

2. **CUDA/GPU Verification**
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
   # For Habitat-Sim GPU:
   python -c "import habitat_sim; print(habitat_sim.cuda_enabled)"
   ```

3. **Directory Structure**
   ```bash
   bash setup.sh
   ls -la data/synthetic/ data/real/ data/fda_output/ experiments/
   ```

### Phase 2: FDA Preprocessing Pipeline
4. **FDA Module Test** (requires sample images)
   ```bash
   # Create minimal test images if none exist
   python -c "import cv2, numpy as np; cv2.imwrite('data/synthetic/test.jpg', np.random.randint(0,255,(224,224,3), dtype=np.uint8)); cv2.imwrite('data/real/test.jpg', np.random.randint(0,255,(224,224,3), dtype=np.uint8))"
   
   # Run FDA
   python src/data/fda.py --config configs/fda.yaml
   
   # Verify output
   ls data/fda_output/*.jpg
   ```

### Phase 3: Habitat Environment Test
5. **Habitat Configuration Validation**
   ```bash
   python -c "from src.habitat_env.env_wrapper import HabitatWrapper; print('Import OK')"
   # Note: Full test requires Habitat scene dataset (not present in repo)
   ```

6. **Model Architecture Test**
   ```bash
   python -c "
   from src.models.encoder import VisualEncoder
   from src.models.policy import Policy
   import torch
   enc = VisualEncoder()
   pol = Policy(enc)
   x = torch.randn(1, 3, 224, 224)
   a, v = pol(x)
   print(f'Actor output shape: {a.shape}, Critic output shape: {v.shape}')
   "
   ```

### Phase 4: Integration Smoke Test
7. **End-to-End Data Flow** (mock, since training is placeholder)
   ```bash
   # Test config loading
   python -c "import yaml; cfg = yaml.safe_load(open('configs/training.yaml')); print(cfg)"
   python -c "import yaml; cfg = yaml.safe_load(open('configs/fda.yaml')); print(cfg)"
   python -c "import yaml; cfg = yaml.safe_load(open('configs/habitat_pointnav.yaml')); print(cfg)"
   
   # Test script execution (will print placeholder messages)
   bash scripts/run_fda.sh  # Should work if test images exist
   bash scripts/run_nav.sh   # Will print placeholder message
   ```

### Phase 5: Known Gaps (Not Testable Yet)
8. **Missing Implementations** (identified from code review):
   - `src/train/train_nav.py` - Placeholder only, needs PPO implementation
   - `src/train/train_domain.py` - Placeholder only
   - `src/habitat_env/sensors.py` - RGBSensor is placeholder
   - `scripts/eval_realworld.sh` - References non-existent `src/eval/eval_nav.py`
   - Habitat scene datasets - Not included, must be downloaded separately
   - PPO/DD-PPO training loop - Not implemented

### Expected Smoke Test Results
- ✅ **Phase 1-2**: Should pass if dependencies installed correctly
- ✅ **Phase 3**: Model tests should pass; Habitat wrapper import OK but full env test requires scene data
- ⚠️ **Phase 4**: Scripts run but training is placeholder
- ❌ **Phase 5**: Full pipeline not yet implemented

---

## Critical Missing Components for Full Pipeline

1. **PPO Training Loop** - `train_nav.py` is placeholder
2. **Habitat Scene Datasets** - Must download separately (e.g., Matterport3D, Gibson)
3. **Evaluation Script** - `src/eval/eval_nav.py` referenced but doesn't exist
4. **DD-PPO Support** - Config flag exists but implementation missing
5. **Checkpoint Saving/Loading** - Not implemented
6. **Logging/Metrics Tracking** - Not implemented
7. **Habitat Config Completeness** - Current config is minimal, needs full simulator settings

---

## Recommendations

1. **Add to requirements.txt**: matplotlib, tensorboard (or wandb), tqdm
2. **Document Habitat dataset download** - Add instructions for obtaining scene datasets
3. **Implement training loop** - Complete PPO/DD-PPO implementation
4. **Add version pinning** - Habitat-Sim is sensitive to CUDA/PyTorch versions
5. **Create minimal test dataset** - Include 2-3 sample images for FDA smoke test
6. **Add environment validation script** - Automated check for CUDA, datasets, etc.
