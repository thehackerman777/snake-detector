#!/bin/bash
# Run this on an AWS GPU instance to train the snake detector
# Usage: bash scripts/run_gpu_training.sh

set -e

echo "=== Snake Detector GPU Training ==="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'Unknown')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'None')"

# Setup
cd /home/ubuntu
git clone https://github.com/thehackerman777/snake-detector.git
cd snake-detector/ml-pipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train
python3 scripts/train.py --train --epochs 100 --device cuda

# Export to OpenVINO
python3 -c "
from scripts.train import export_model
import glob
best_pt = glob.glob('runs/detect/train*/weights/best.pt')[-1]
export_model(best_pt, format='openvino')
print('Model exported to OpenVINO format!')
"

echo "=== Training complete ==="
echo "Model files in: runs/detect/"
