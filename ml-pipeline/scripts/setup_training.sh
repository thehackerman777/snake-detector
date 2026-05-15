#!/bin/bash
# Snake Detector - Training Environment Setup
# Run this on a GPU instance (AWS EC2 G4dn, Colab, etc.)

set -e

echo "=== Snake Detector Training Setup ==="

# Install system deps
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Download dataset
python3 scripts/train.py --download

echo ""
echo "=== Training environment ready ==="
echo "Run: python3 scripts/train.py --train --epochs 100 --device cuda"
