# =============================================================================
# labels.py — Convert fault lines into pixel masks matching Sentinel-2 images
# =============================================================================
# WHAT THIS DOES:
#   1. Loads each Sentinel-2 GeoTIFF from data/imagery/
#   2. Loads the filtered California fault shapefile
#   3. Burns the fault lines onto a matching pixel grid
#   4. Saves a binary mask (1=fault, 0=no fault) for each region
#
# OUTPUT:
#   data/masks/sentinel2_carrizo_10m_mask.tif
#   data/masks/sentinel2_mojave_10m_mask.tif
#   data/masks/sentinel2_bay_area_10m_mask.tif
#
# AFTER THIS:
#   Run patches.py to slice images + masks into 128x128 training pairs
# =============================================================================

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

os.makedirs("data/masks", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =============================================================================
# STEP 1: LOAD THE FAULT SHAPEFILE
# This is the filtered California faults we saved at the end of main.py
# =============================================================================
print("Loading fault shapefile...")
faults = gpd.read_file("data/ca_active_faults_utm.shp")
print(f"  Loaded {len(faults)} fault segments")

# =============================================================================
# STEP 2: BUFFER THE FAULT LINES
# Fault lines are 1D — on a 10m pixel grid they'd be 1 pixel wide which is
# too thin for the model to learn from. We fatten them to 50m wide (5 pixels).
# This also accounts for slight misalignment between the shapefile and imagery.
# =============================================================================
BUFFER_METERS = 50  # 50 meters = 5 pixels at 10m resolution

print(f"Buffering fault lines by {BUFFER_METERS}m...")
faults["geometry"] = faults.geometry.buffer(BUFFER_METERS)
print("  Done")

# =============================================================================
# STEP 3: DEFINE WHICH FILES BELONG TO WHICH REGION
# Mojave came in 2 pieces so we merge those first
# =============================================================================
REGIONS = {
    "carrizo": {
        "files": ["data/imagery/sentinel2_carrizo_10m.tif"],
        "merge": False
    },
    "mojave": {
        "files": sorted(glob.glob("data/imagery/sentinel2_mojave_10m*.tif")),
        "merge": True   # multiple files need merging
    },
    "bay_area": {
        "files": sorted(glob.glob("data/imagery/sentinel2_bay_area_10m*.tif")),
        "merge": len(glob.glob("data/imagery/sentinel2_bay_area_10m*.tif")) > 1
    },
}

# =============================================================================
# STEP 4: PROCESS EACH REGION
# For each region:
#   - Load (and merge if needed) the GeoTIFF
#   - Reproject faults to match the image CRS
#   - Burn fault polygons onto a pixel grid
#   - Save the mask
# =============================================================================

def create_mask(region_name, region_info):
    print(f"\nProcessing: {region_name}")
    files = region_info["files"]

    if len(files) == 0:
        print(f"  WARNING: No files found for {region_name} — skipping")
        return

    print(f"  Found {len(files)} file(s): {[os.path.basename(f) for f in files]}")

    # --- Merge tiles if region came in multiple pieces ---
    if region_info["merge"] and len(files) > 1:
        print(f"  Merging {len(files)} tiles...")
        src_files = [rasterio.open(f) for f in files]
        mosaic, mosaic_transform = merge(src_files)
        meta = src_files[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width":  mosaic.shape[2],
            "transform": mosaic_transform
        })
        merged_path = f"data/imagery/sentinel2_{region_name}_10m_merged.tif"
        with rasterio.open(merged_path, "w", **meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        image_path = merged_path
        print(f"  Merged → {merged_path}")
    else:
        image_path = files[0]

    # --- Load the image to get its pixel grid info ---
    with rasterio.open(image_path) as src:
        img_crs       = src.crs        # coordinate system
        img_transform = src.transform  # maps pixel coords to real-world coords
        img_height    = src.height     # number of rows
        img_width     = src.width      # number of columns
        img_bounds    = src.bounds     # geographic extent

    print(f"  Image size: {img_width} x {img_height} pixels")
    print(f"  CRS: {img_crs}")

    # --- Reproject faults to match the image CRS ---
    faults_reprojected = faults.to_crs(img_crs)

    # --- Clip faults to just this region's bounds (speeds things up) ---
    from shapely.geometry import box
    region_box = box(img_bounds.left, img_bounds.bottom,
                     img_bounds.right, img_bounds.top)
    faults_clipped = faults_reprojected[
        faults_reprojected.geometry.intersects(region_box)
    ]
    print(f"  Faults in this region: {len(faults_clipped)}")

    if len(faults_clipped) == 0:
        print(f"  WARNING: No faults found in this region's bounds!")
        return

    # --- Burn fault polygons onto pixel grid ---
    # Each fault polygon becomes 1s, everything else stays 0
    shapes = [(geom, 1) for geom in faults_clipped.geometry if geom is not None]

    mask = rasterize(
        shapes,
        out_shape   = (img_height, img_width),
        transform   = img_transform,
        fill        = 0,     # no fault
        dtype       = np.uint8
    )

    fault_pixels = mask.sum()
    total_pixels = img_height * img_width
    print(f"  Fault pixels: {fault_pixels:,} ({100*fault_pixels/total_pixels:.2f}% of image)")

    # --- Save the mask as a GeoTIFF ---
    mask_path = f"data/masks/sentinel2_{region_name}_10m_mask.tif"
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()

    meta.update({
        "count": 1,         # single band (fault/no fault)
        "dtype": "uint8"
    })

    with rasterio.open(mask_path, "w", **meta) as dest:
        dest.write(mask, 1)

    print(f"  Mask saved → {mask_path}")

    # --- Quick visualization: overlay mask on true color image ---
    with rasterio.open(image_path) as src:
        # Read R, G, B bands (bands 3, 2, 1 = indices 2, 1, 0)
        r = src.read(3).astype(float)
        g = src.read(2).astype(float)
        b = src.read(1).astype(float)

    # Normalize for display
    def normalize(band):
        p2, p98 = np.percentile(band[band > 0], (2, 98))
        return np.clip((band - p2) / (p98 - p2), 0, 1)

    rgb = np.dstack([normalize(r), normalize(g), normalize(b)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(rgb)
    axes[0].set_title(f"{region_name} — Sentinel-2 True Color")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].imshow(mask, alpha=0.4, cmap="Reds")  # fault pixels in red overlay
    axes[1].set_title(f"{region_name} — Fault Mask Overlay (red = fault)")
    axes[1].axis("off")

    plt.tight_layout()
    preview_path = f"outputs/{region_name}_mask_preview.png"
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Preview saved → {preview_path}")

# =============================================================================
# RUN ALL REGIONS
# =============================================================================
print("=" * 60)
print("GENERATING FAULT MASKS")
print("=" * 60)

for region_name, region_info in REGIONS.items():
    create_mask(region_name, region_info)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("""
CHECK:
  Look at outputs/*_mask_preview.png for each region
  You should see red overlay where fault lines are
  The San Andreas should be clearly visible in carrizo + mojave

NEXT STEP:
  Run patches.py to slice into 128x128 training patches
""")