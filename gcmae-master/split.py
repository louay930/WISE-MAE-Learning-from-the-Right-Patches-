import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ✅ Corrected path to where your images actually are:
original_path = Path("/user/louay.hamdi/u13592/.project/dir.project/NSCLC_pretraining_dataset/images")
split_root = Path("/user/louay.hamdi/u13592/.project/dir.project/NSCLC_pretraining_split")
train_path = split_root / "train" / "dummy"
val_path = split_root / "val" / "dummy"

# Create output dirs
train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)

# List image files (handles .png and .PNG just in case)
images = list(original_path.glob("*.png")) + list(original_path.glob("*.PNG"))
print(f"📸 Found {len(images)} total images.")

# Shuffle and split 90/10
random.seed(42)
random.shuffle(images)
split_idx = int(0.9 * len(images))
train_images = images[:split_idx]
val_images = images[split_idx:]

# Copy function
def copy_files(file_list, target_dir):
    for f in tqdm(file_list, desc=f"Copying to {target_dir.name}"):
        shutil.copy(f, target_dir / f.name)

copy_files(train_images, train_path)
copy_files(val_images, val_path)

print(f"✅ Done: {len(train_images)} train, {len(val_images)} val")
print(f"📂 Output dir: {split_root}")
