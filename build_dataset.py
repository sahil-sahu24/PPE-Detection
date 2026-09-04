import os
import shutil
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

# Set seed for reproducibility
random.seed(42)

# Configuration
SOURCE_IMG_DIR = r"D:\PPE-Detection\archive (1)\images"
SOURCE_XML_DIR = r"D:\PPE-Detection\archive (1)\annotations"
DEST_DATASET_DIR = r"D:\PPE-Detection\dataset_helmet"

CLASS_MAP = {
    "helmet": 0,
    "head": 1
}

# Directories to create
dirs = {
    "train_images": os.path.join(DEST_DATASET_DIR, "images", "train"),
    "val_images": os.path.join(DEST_DATASET_DIR, "images", "val"),
    "train_labels": os.path.join(DEST_DATASET_DIR, "labels", "train"),
    "val_labels": os.path.join(DEST_DATASET_DIR, "labels", "val")
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Collect all matching image/xml pairs
all_xmls = sorted([f for f in os.listdir(SOURCE_XML_DIR) if f.endswith(".xml")])
samples = []

for xml_file in all_xmls:
    base = os.path.splitext(xml_file)[0]
    img_file = base + ".png"
    img_path = os.path.join(SOURCE_IMG_DIR, img_file)
    xml_path = os.path.join(SOURCE_XML_DIR, xml_file)
    if os.path.exists(img_path):
        samples.append((base, xml_path, img_path))
    else:
        print(f"Warning: Image missing for {xml_file}")

print(f"Total verified samples: {len(samples)}")

# Shuffle deterministically
random.shuffle(samples)

# 80/20 train/val split
split_idx = int(0.8 * len(samples))
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

print(f"Train split: {len(train_samples)} samples")
print(f"Val split:   {len(val_samples)} samples")

def process_sample(item, split):
    base, xml_path, img_path = item
    img_dest_dir = dirs[f"{split}_images"]
    lbl_dest_dir = dirs[f"{split}_labels"]

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    yolo_lines = []
    cls_counts = {0: 0, 1: 0}

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASS_MAP:
            continue

        cls_id = CLASS_MAP[cls_name]
        cls_counts[cls_id] += 1
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Normalize YOLO coordinates
        xc = ((xmin + xmax) / 2.0) / img_w
        yc = ((ymin + ymax) / 2.0) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        # Clamp between 0.0 and 1.0
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # Write label file
    lbl_path = os.path.join(lbl_dest_dir, base + ".txt")
    with open(lbl_path, "w") as f:
        f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    # Copy image file
    img_dest = os.path.join(img_dest_dir, base + ".png")
    shutil.copy2(img_path, img_dest)

    return cls_counts

print("Copying images and writing YOLO annotations (using multithreading)...")

train_stats = {0: 0, 1: 0}
val_stats = {0: 0, 1: 0}

with ThreadPoolExecutor(max_workers=8) as executor:
    train_results = list(executor.map(lambda item: process_sample(item, "train"), train_samples))

with ThreadPoolExecutor(max_workers=8) as executor:
    val_results = list(executor.map(lambda item: process_sample(item, "val"), val_samples))

for res in train_results:
    train_stats[0] += res[0]
    train_stats[1] += res[1]

for res in val_results:
    val_stats[0] += res[0]
    val_stats[1] += res[1]

# Create data.yaml
dest_path_forward = DEST_DATASET_DIR.replace("\\", "/")
yaml_content = f"""path: {dest_path_forward}
train: images/train
val: images/val

nc: 2
names:
  0: helmet
  1: head
"""

yaml_path = os.path.join(DEST_DATASET_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("Dataset creation complete!")
print("--- SUMMARY ---")
print(f"Train: {len(train_samples)} images | Helmet boxes: {train_stats[0]} | Head boxes: {train_stats[1]}")
print(f"Val:   {len(val_samples)} images | Helmet boxes: {val_stats[0]} | Head boxes: {val_stats[1]}")
print(f"Total: {len(samples)} images | Helmet boxes: {train_stats[0] + val_stats[0]} | Head boxes: {train_stats[1] + val_stats[1]}")
print(f"Dataset YAML written to: {yaml_path}")
