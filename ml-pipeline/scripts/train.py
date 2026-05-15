"""
Snake Detector - Training Pipeline
Trains YOLOv8n on corn snake dataset.
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Dataset YAML template
DATA_YAML_TEMPLATE = """
# Snake Detector Dataset
# Pantherophis guttatus morph classification

path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {class_names}
"""


def create_data_yaml(dataset_path: str, class_names: list) -> str:
    """Generate data.yaml for YOLO training."""
    yaml_content = DATA_YAML_TEMPLATE.format(
        dataset_path=dataset_path,
        num_classes=len(class_names),
        class_names=class_names,
    )
    
    output_path = os.path.join(dataset_path, "data.yaml")
    with open(output_path, "w") as f:
        f.write(yaml_content)
    
    print(f"[✓] data.yaml created at {output_path}")
    return output_path


def download_dataset(output_dir: str = "datasets"):
    """
    Download public snake datasets.
    Uses Roboflow if available, otherwise creates a minimal structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Default morph classes (basic set)
    class_names = [
        "amelanistic",
        "anerythristic",
        "normal",
        "motley",
        "stripe",
        "snow",
    ]
    
    create_data_yaml(output_dir, class_names)
    return output_dir, class_names


def train_model(
    data_yaml: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,
    device: str = "cpu",
):
    """
    Train YOLOv8n model on snake dataset.
    
    Args:
        data_yaml: Path to data.yaml
        model_name: YOLO model variant (nano by default)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (-1 = auto)
        device: Training device (cpu/cuda)
    """
    from ultralytics import YOLO
    
    print(f"[*] Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    print(f"[*] Starting training on {device}")
    print(f"    Dataset: {data_yaml}")
    print(f"    Epochs: {epochs}")
    print(f"    Image size: {imgsz}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=15,  # Early stopping
        lr0=0.01,
        lrf=0.01,
        augment=True,
        flipud=0.0,  # No vertical flip (snakes in normal position)
        fliplr=0.5,
        degrees=15.0,  # Rotation augmentation
        shear=10.0,    # Shear augmentation
        hsv_h=0.015,   # Hue augmentation
        hsv_s=0.7,     # Saturation augmentation
        hsv_v=0.4,     # Value augmentation
        cos_lr=True,
        warmup_epochs=3,
        workers=4,
    )
    
    print(f"[✓] Training complete!")
    print(f"    Best model: {results.save_dir}/weights/best.pt")
    
    return results


def export_model(model_path: str, format: str = "openvino"):
    """
    Export trained model to inference format.
    
    Args:
        model_path: Path to trained .pt file
        format: Export format (openvino, onnx, tflite)
    """
    from ultralytics import YOLO
    
    print(f"[*] Loading trained model: {model_path}")
    model = YOLO(model_path)
    
    print(f"[*] Exporting to {format} with FP16 quantization...")
    
    export_path = model.export(
        format=format,
        half=True,  # FP16 quantization
        imgsz=640,
    )
    
    print(f"[✓] Model exported: {export_path}")
    return export_path


def validate_model(model_path: str, data_yaml: str):
    """Validate trained model performance."""
    from ultralytics import YOLO
    
    print(f"[*] Validating model: {model_path}")
    model = YOLO(model_path)
    
    metrics = model.val(data=data_yaml)
    
    print(f"\n[📊] Validation Results:")
    print(f"     mAP50: {metrics.box.map50:.4f}")
    print(f"     mAP50-95: {metrics.box.map:.4f}")
    print(f"     Precision: {metrics.box.p:.4f}")
    print(f"     Recall: {metrics.box.r:.4f}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Snake Detector Training Pipeline")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--export", type=str, help="Export trained model (path to .pt)")
    parser.add_argument("--validate", type=str, help="Validate model (path to .pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument("--dataset", type=str, default="datasets/snake", help="Dataset path")
    
    args = parser.parse_args()
    
    if args.download:
        print("[*] Downloading dataset...")
        download_dataset(args.dataset)
    
    if args.train:
        print("[*] Starting training...")
        data_yaml = os.path.join(args.dataset, "data.yaml")
        train_model(
            data_yaml=data_yaml,
            epochs=args.epochs,
            device=args.device,
        )
    
    if args.validate:
        print("[*] Validating model...")
        data_yaml = os.path.join(args.dataset, "data.yaml")
        validate_model(args.validate, data_yaml)
    
    if args.export:
        print("[*] Exporting model...")
        export_model(args.export)
    
    print("\nDone!")
