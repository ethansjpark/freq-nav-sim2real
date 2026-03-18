#!/bin/bash
set -e

pip -q install --upgrade pip

# Core
pip -q install numpy opencv-python pyyaml tqdm matplotlib

# Torch (Colab usually has it, but this ensures correct)
pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121
