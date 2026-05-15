"""
Minimal dataset preparation for initial training.
Downloads public corn snake images and creates YOLO annotations.
"""

import os
import json
import random
import urllib.request
from pathlib import Path

# Initial morph classes (will expand later)
CLASSES = [
    "amelanistic",     # 0
    "anerythristic",   # 1  
    "normal",          # 2
    "motley",          # 3
    "stripe",          # 4
    "snow",            # 5
]

# URLs for public domain corn snake images (Wikimedia Commons)
# These are CC-licensed images
SAMPLE_IMAGES = {
    "normal": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Pantherophis_guttatus_%28corn_snake%29.jpg/640px-Pantherophis_guttatus_%28corn_snake%29.jpg",
    ],
}

def download_image(url, output_path):
    """Download an image from URL."""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def create_yolo_label(image_width, image_height, class_id):
    """
    Create a simple YOLO annotation.
    For demo purposes, assumes the snake occupies the center ~70% of the image.
    """
    center_x = 0.5
    center_y = 0.5
    width = 0.7
    height = 0.7
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def create_dataset(output_dir="datasets/snake"):
    """Create a minimal dataset structure."""
    base = Path(output_dir)
    
    for split in ["train", "val", "test"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Download sample images  
    print("Downloading sample images...")
    for morph, urls in SAMPLE_IMAGES.items():
        class_id = CLASSES.index(morph) if morph in CLASSES else 2
        for i, url in enumerate(urls):
            img_name = f"{morph}_{i}.jpg"
            img_path = base / "images" / "train" / img_name
            if download_image(url, str(img_path)):
                # Create annotation
                label = create_yolo_label(640, 480, class_id)
                label_path = base / "labels" / "train" / f"{Path(img_name).stem}.txt"
                with open(label_path, "w") as f:
                    f.write(label + "\n")
                print(f"  ✓ {morph}_{i}.jpg")
    
    # Create data.yaml
    yaml_content = f"""
# Snake Detector Dataset
# Pantherophis guttatus morph classification

path: {base.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(base / "data.yaml", "w") as f:
        f.write(yaml_content.strip())
    
    print(f"\nDataset created at {base.absolute()}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("\nNOTE: This is a minimal dataset. Add more images for proper training.")
    return str(base / "data.yaml")


if __name__ == "__main__":
    create_dataset()
