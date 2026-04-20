# =============================================================================
# dataset.py — Download Sentinel-2 imagery from Google Earth Engine
# =============================================================================
# WHAT THIS DOES:
#   Connects to GEE, pulls cloud-free Sentinel-2 imagery over 3 training
#   regions in California, and exports them to your Google Drive.
#
# AFTER THIS RUNS:
#   Go to Google Drive → fault_detection_data folder → download the 3 files
#   Place them in your data/imagery/ folder
#   Then run labels.py
#
# RUN TIME: Export tasks take 5-30 minutes each in GEE background
# =============================================================================

import ee
import os

# Connect to GEE
ee.Initialize(project='geo-project-486919')
print("GEE connected")

os.makedirs("data/imagery", exist_ok=True)

# =============================================================================
# TRAINING REGIONS
# 3 regions covering diverse California environments (per proposal section 1.2.2)
# Format: [min_longitude, min_latitude, max_longitude, max_latitude]
# =============================================================================
REGIONS = {

    # Arid desert — clearest fault features, good starting region
    # Southern San Andreas + Mojave faults
    "mojave": {
        "bbox": [-116.5, 33.8, -115.5, 34.5],
        "description": "Mojave Desert — Southern San Andreas"
    },

    # Mediterranean shrubland — Carrizo Plain
    # One of the most visually clear sections of the San Andreas from above
    "carrizo": {
        "bbox": [-120.2, 35.0, -119.5, 35.6],
        "description": "Carrizo Plain — Central San Andreas"
    },

    # Northern CA — more vegetation, harder for model, tests generalization
    # Hayward + Calaveras faults
    "bay_area": {
        "bbox": [-122.5, 37.5, -121.5, 38.2],
        "description": "Bay Area — Hayward + Calaveras faults"
    },
}

# =============================================================================
# SENTINEL-2 SETTINGS
# These 6 bands match exactly what Prithvi-EO 2.0 expects (from proposal)
# Summer months = less cloud cover in California + dry vegetation = better
# rock/soil contrast for fault visibility
# =============================================================================
BANDS      = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
DATE_START = "2022-06-01"
DATE_END   = "2023-09-30"
MAX_CLOUD  = 10   # only use images with less than 10% cloud cover
SCALE      = 10   # 10 meters per pixel

# =============================================================================
# CLOUD MASKING
# Sentinel-2 includes a Scene Classification Layer (SCL) that labels each pixel
# We remove clouds (8,9,10) and cloud shadows (3) so they don't confuse the model
# =============================================================================
def mask_clouds(image):
    scl = image.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(clear)

# =============================================================================
# DOWNLOAD EACH REGION
# =============================================================================
def export_region(name, region):
    print(f"\nProcessing: {name} — {region['description']}")

    bbox = region["bbox"]
    aoi  = ee.Geometry.Rectangle(bbox)

    # Pull all Sentinel-2 images for this area + date range
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .map(mask_clouds)
        .select(BANDS)
    )

    count = collection.size().getInfo()
    print(f"  Found {count} cloud-free images")

    if count == 0:
        print(f"  WARNING: No images found — skipping this region")
        return None

    # Take the median across all images
    # This gives one clean image per region with no clouds or noise
    composite = collection.median().clip(aoi)

    # Export to Google Drive
    filename = f"sentinel2_{name}_10m"
    task = ee.batch.Export.image.toDrive(
        image          = composite,
        description    = filename,
        folder         = "fault_detection_data",  # folder in your Google Drive
        fileNamePrefix = filename,
        region         = aoi,
        scale          = SCALE,
        crs            = "EPSG:32610",            # UTM Zone 10N, standard for CA
        maxPixels      = 1e13,
        fileFormat     = "GeoTIFF"
    )
    task.start()
    print(f"  Export started → Google Drive/fault_detection_data/{filename}.tif")
    print(f"  Monitor at: https://code.earthengine.google.com/tasks")
    return task

# =============================================================================
# RUN
# =============================================================================
print("=" * 60)
print("EXPORTING SENTINEL-2 IMAGERY")
print("=" * 60)
print(f"Bands:      {BANDS}")
print(f"Dates:      {DATE_START} to {DATE_END}")
print(f"Max cloud:  {MAX_CLOUD}%")
print(f"Resolution: {SCALE}m per pixel")
print(f"CRS:        EPSG:32610 (UTM Zone 10N)")

for region_name, region_info in REGIONS.items():
    export_region(region_name, region_info)

print("\n" + "=" * 60)
print("ALL TASKS SUBMITTED")
print("=" * 60)
print("""
NEXT STEPS:
  1. Go to https://code.earthengine.google.com/tasks
  2. You should see 3 running tasks (one per region)
  3. Wait for them to complete (green checkmark) — usually 5-30 min
  4. Open Google Drive → fault_detection_data folder
  5. Download all 3 .tif files
  6. Place them in your data/imagery/ folder
  7. Run labels.py
""")