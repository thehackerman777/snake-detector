"""
Corn Snake Image Scraper
Downloads images from web sources for dataset creation.
"""

import os
import json
import time
import requests
from pathlib import Path


MORPH_SEARCHES = {
    "amelanistic": [
        "amelanistic corn snake",
        "albino corn snake",
        "reverse okeetee corn snake",
    ],
    "anerythristic": [
        "anerythristic corn snake",
        "charcoal corn snake", 
        "cinder corn snake",
    ],
    "normal": [
        "normal corn snake",
        "wild type corn snake",
        "okeetee corn snake",
    ],
    "motley": [
        "motley corn snake",
        "bloodred corn snake",
        "diffused corn snake",
    ],
    "stripe": [
        "stripe corn snake",
        "tessera corn snake",
        "masque corn snake",
    ],
    "snow": [
        "snow corn snake",
        "blizzard corn snake",
        "butter corn snake",
        "avalanche corn snake",
    ],
}


def search_image_urls(query, max_images=200):
    """
    Search for images using DuckDuckGo or other free API.
    Returns list of image URLs.
    """
    # Using DuckDuckGo instant answer API
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "iax": "images",
        "ia": "images",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("RelatedTopics", [])
            image_urls = []
            for r in results:
                if isinstance(r, dict) and "Icon" in r:
                    icon_url = r["Icon"].get("URL")
                    if icon_url and icon_url.startswith("http"):
                        image_urls.append(icon_url)
            return image_urls[:max_images]
    except Exception as e:
        print(f"  Error searching '{query}': {e}")
    
    return []


def download_and_save(url, output_path):
    """Download a single image."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
            if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"
            output_path = output_path.with_suffix(ext)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return str(output_path)
    except Exception as e:
        print(f"    Download error: {e}")
    return None


def build_dataset(output_dir="datasets/snake"):
    """Scrape and build the dataset."""
    base = Path(output_dir)
    
    for split in ["train", "val", "test"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    class_names = list(MORPH_SEARCHES.keys())
    total_images = 0
    
    for class_id, (morph, queries) in enumerate(MORPH_SEARCHES.items()):
        print(f"\n[{class_id}] Scraping '{morph}'...")
        
        morph_urls = set()
        for query in queries:
            urls = search_image_urls(query, max_images=200)
            morph_urls.update(urls)
            time.sleep(1)  # Be polite
        
        print(f"  Found {len(morph_urls)} URLs")
        
        # Download first N images per class
        max_per_class = 100
        downloaded = 0
        
        for url in list(morph_urls)[:max_per_class * 2]:  # Try more than needed
            if downloaded >= max_per_class:
                break
            
            # Split: 80% train, 10% val, 10% test
            r = hash(url) % 100
            if r < 80:
                split_dir = "train"
            elif r < 90:
                split_dir = "val"
            else:
                split_dir = "test"
            
            img_name = f"{morph}_{downloaded}"
            img_path = base / "images" / split_dir / img_name
            
            result = download_and_save(url, img_path)
            if result:
                # Create placeholder YOLO label (will need proper annotation)
                label_path = base / "labels" / split_dir / f"{img_name}.txt"
                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")
                downloaded += 1
                total_images += 1
        
        print(f"  Downloaded: {downloaded} / {min(max_per_class, len(morph_urls))}")
    
    # Create data.yaml
    yaml_content = f"""
# Snake Detector Dataset
path: {base.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    with open(base / "data.yaml", "w") as f:
        f.write(yaml_content.strip())
    
    print(f"\n{'='*50}")
    print(f"Dataset created: {base.absolute()}")
    print(f"Total images: {total_images}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"{'='*50}")
    print("\n⚠️  NOTE: This is an automated scrape. Annotations are approximate ")
    print("   bounding boxes. You should review and fix labels before training!")
    return str(base / "data.yaml")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/snake")
    args = parser.parse_args()
    build_dataset(args.output)
