# =============================================================================
# patches.py — Slice images + masks into 128x128 training patches
# =============================================================================
# WHAT THIS DOES:
#   Takes each full Sentinel-2 image and its matching fault mask and slices
#   them into thousands of small 128x128 pixel patches.
#
#   Each patch is a pair:
#     - image patch:  [6, 128, 128] — 6 bands, 128x128 pixels
#     - label patch:  [1, 128, 128] — 1 = fault, 0 = no fault
#
# WHY 128x128:
#   This is what the proposal specifies and what Prithvi-EO 2.0 expects.
#   At 10m resolution, 128x128 pixels = 1.28km x 1.28km of real landscape.
#   Big enough to see fault context, small enough to train efficiently.
#
# CLASS IMBALANCE HANDLING:
#   Only ~1% of pixels are fault pixels. If we kept all patches, 99% would
#   have no faults and the model would just learn to predict "no fault"
#   everywhere. So we only keep patches that contain at least some fault pixels.
#
# OUTPUT:
#   data/patches/images/  — image patches as .npy files
#   data/patches/labels/  — matching label patches as .npy files
#   data/patches/splits/  — train/val/test split lists
# =============================================================================

import numpy as np
import rasterio
import os
import glob
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Force correct working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.makedirs("data/patches/images", exist_ok=True)
os.makedirs("data/patches/labels", exist_ok=True)
os.makedirs("data/patches/splits", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =============================================================================
# SETTINGS
# =============================================================================
PATCH_SIZE     = 128    # 128x128 pixels per patch
STRIDE         = 64     # 50% overlap between patches — gives more training data
                        # and helps model learn fault features at patch boundaries
MIN_FAULT_FRAC = 0.005  # patch must have at least 0.5% fault pixels to be kept
                        # filters out patches with no fault content

# Train/val/test split (70% train, 15% val, 15% test)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15 (remainder)

# =============================================================================
# REGION FILE PAIRS
# Each region: one image file + one matching mask file
# =============================================================================
REGIONS = {
    "carrizo": {
        "image": "data/imagery/sentinel2_carrizo_10m.tif",
        "mask":  "data/masks/sentinel2_carrizo_10m_mask.tif",
    },
    "mojave": {
        "image": "data/imagery/sentinel2_mojave_10m_merged.tif",
        "mask":  "data/masks/sentinel2_mojave_10m_mask.tif",
    },
    "bay_area": {
        "image": "data/imagery/sentinel2_bay_area_10m.tif",
        "mask":  "data/masks/sentinel2_bay_area_10m_mask.tif",
    },
}

# =============================================================================
# PATCH EXTRACTION FUNCTION
# =============================================================================
def extract_patches(region_name, image_path, mask_path):
    print(f"\nProcessing: {region_name}")

    # --- Load image ---
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # shape: [6, H, W]

    # --- Load mask ---
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)    # shape: [H, W]

    num_bands, H, W = image.shape
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape:  {mask.shape}")

    # --- Normalize each band to 0-1 range ---
    # Sentinel-2 values can range from 0-10000
    # Normalizing helps the model train faster and more stably
    for b in range(num_bands):
        band = image[b]
        valid = band[band > 0]  # ignore NoData pixels (value=0)
        if len(valid) == 0:
            continue
        p2  = np.percentile(valid, 2)
        p98 = np.percentile(valid, 98)
        image[b] = np.clip((band - p2) / (p98 - p2 + 1e-10), 0, 1)

    # --- Slice into patches ---
    patches_kept    = 0
    patches_skipped = 0
    patch_names     = []

    for row in range(0, H - PATCH_SIZE, STRIDE):
        for col in range(0, W - PATCH_SIZE, STRIDE):

            # Extract image patch [6, 128, 128]
            img_patch = image[:, row:row+PATCH_SIZE, col:col+PATCH_SIZE]

            # Extract matching label patch [128, 128]
            lbl_patch = mask[row:row+PATCH_SIZE, col:col+PATCH_SIZE]

            # Skip patches that are mostly NoData (black corners of image)
            valid_pixels = (img_patch[0] > 0).sum()
            if valid_pixels < (PATCH_SIZE * PATCH_SIZE * 0.5):
                patches_skipped += 1
                continue

            # Skip patches with too few fault pixels
            fault_frac = lbl_patch.sum() / (PATCH_SIZE * PATCH_SIZE)
            if fault_frac < MIN_FAULT_FRAC:
                patches_skipped += 1
                continue

            # Save the patch pair
            patch_name = f"{region_name}_r{row:05d}_c{col:05d}"
            np.save(f"data/patches/images/{patch_name}.npy", img_patch)
            np.save(f"data/patches/labels/{patch_name}.npy", lbl_patch)
            patch_names.append(patch_name)
            patches_kept += 1

    print(f"  Patches kept:    {patches_kept:,}")
    print(f"  Patches skipped: {patches_skipped:,}")
    print(f"  Fault frac filter: >{MIN_FAULT_FRAC*100:.1f}% fault pixels required")

    return patch_names

# =============================================================================
# RUN ALL REGIONS
# =============================================================================
print("=" * 60)
print("GENERATING TRAINING PATCHES")
print("=" * 60)
print(f"Patch size:     {PATCH_SIZE}x{PATCH_SIZE} pixels")
print(f"Stride:         {STRIDE} pixels (50% overlap)")
print(f"Min fault frac: {MIN_FAULT_FRAC*100:.1f}%")

all_patches = []
for region_name, paths in REGIONS.items():
    patches = extract_patches(region_name, paths["image"], paths["mask"])
    all_patches.extend(patches)

print(f"\nTotal patches across all regions: {len(all_patches):,}")

# =============================================================================
# TRAIN / VAL / TEST SPLIT — stratified by region
# We split within each region first to guarantee every split has representation
# from all 3 regions. A pure random shuffle risks one region ending up entirely
# in one split (especially Bay Area which has fewer fault patches).
# =============================================================================
print("\nSplitting into train/val/test (stratified by region)...")

np.random.seed(42)

from collections import defaultdict

# Group patches by region using prefix matching
by_region = defaultdict(list)
for name in all_patches:
    region = next(r for r in REGIONS if name.startswith(r))
    by_region[region].append(name)

train_patches, val_patches, test_patches = [], [], []

for region, patches in by_region.items():
    np.random.shuffle(patches)
    n = len(patches)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    train_patches.extend(patches[:n_train])
    val_patches.extend(patches[n_train:n_train + n_val])
    test_patches.extend(patches[n_train + n_val:])
    print(f"  {region}: {n_train} train / {n_val} val / {n - n_train - n_val} test")

# Shuffle within each split so regions are interleaved during training
np.random.shuffle(train_patches)
np.random.shuffle(val_patches)
np.random.shuffle(test_patches)

print(f"  ── totals ──────────────────────────")
print(f"  Train: {len(train_patches):,} patches")
print(f"  Val:   {len(val_patches):,} patches")
print(f"  Test:  {len(test_patches):,} patches")

# Save split lists
with open("data/patches/splits/train.txt", "w") as f:
    f.write("\n".join(train_patches))
with open("data/patches/splits/val.txt", "w") as f:
    f.write("\n".join(val_patches))
with open("data/patches/splits/test.txt", "w") as f:
    f.write("\n".join(test_patches))

print("  Split lists saved to data/patches/splits/")

# =============================================================================
# SAVE DATASET INFO
# =============================================================================
info = {
    "total_patches":  len(all_patches),
    "train_patches":  len(train_patches),
    "val_patches":    len(val_patches),
    "test_patches":   len(test_patches),
    "patch_size":     PATCH_SIZE,
    "stride":         STRIDE,
    "min_fault_frac": MIN_FAULT_FRAC,
    "num_bands":      6,
    "bands":          ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    "regions":        list(REGIONS.keys()),
}
with open("data/patches/dataset_info.json", "w") as f:
    json.dump(info, f, indent=2)
print("  Dataset info saved to data/patches/dataset_info.json")

# =============================================================================
# VISUALIZE SAMPLE PATCHES
# Show a few example patch pairs so we can verify they look correct
# =============================================================================
print("\nGenerating sample patch visualization...")

sample_names = train_patches[:6]
fig, axes = plt.subplots(2, 6, figsize=(18, 6))

for i, name in enumerate(sample_names):
    img = np.load(f"data/patches/images/{name}.npy")
    lbl = np.load(f"data/patches/labels/{name}.npy")

    # Show RGB (bands 2, 1, 0 = R, G, B)
    rgb = np.dstack([img[2], img[1], img[0]])
    rgb = np.clip(rgb, 0, 1)

    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f"Image\n{name.split('_')[0]}", fontsize=7)
    axes[0, i].axis("off")

    axes[1, i].imshow(lbl, cmap="Reds", vmin=0, vmax=1)
    axes[1, i].set_title(f"Label\nfault%: {lbl.mean()*100:.1f}%", fontsize=7)
    axes[1, i].axis("off")

plt.suptitle("Sample Training Patches (top=image, bottom=fault label)", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/sample_patches.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Sample visualization saved to outputs/sample_patches.png")

print("\n" + "=" * 60)
print("DONE — DATASET IS READY")
print("=" * 60)
print(f"""
Your training dataset is complete:
  {len(train_patches):,} training patches
  {len(val_patches):,}  validation patches
  {len(test_patches):,}  test patches

Each patch pair:
  Image: [6, 128, 128] float32 (normalized 0-1)
  Label: [128, 128]    uint8   (0=no fault, 1=fault)

NEXT STEP: train.py — fine-tune Prithvi-EO 2.0 on these patches
""")