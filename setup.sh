#!/bin/bash
set -euo pipefail

echo "Setting up project directories..."

# Data roots used by FDA + domain pipelines.
mkdir -p data/synthetic
mkdir -p data/real
mkdir -p data/fda_output

# Experiment outputs used by train/eval/domain scripts.
mkdir -p experiments
mkdir -p experiments/checkpoints
mkdir -p experiments/domain
mkdir -p experiments/eval

echo "Done. Directories are ready."
