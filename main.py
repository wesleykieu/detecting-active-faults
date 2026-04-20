import geopandas as gpd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# =============================================================================
# LOAD
# =============================================================================
faults = gpd.read_file("Qfaults_GIS/SHP/Qfaults_US_Database.shp")
print(f"Total faults loaded: {len(faults)}")

# =============================================================================
# FILTER 1: Active faults only (skip undifferentiated, class B, unspecified)
# =============================================================================
active_ages = [
    'historic',
    'latest Quaternary',
    'late Quaternary',
    'Late Quaternary',   # same thing, different capitalization in the data
]

active = faults[faults['age'].isin(active_ages)].copy()
print(f"After age filter: {len(active)} faults")

# =============================================================================
# FILTER 2: California only (by bounding box since there's no state column)
# California lat/lon bounds: roughly -124.5 to -114.1 lon, 32.5 to 42.0 lat
# =============================================================================
ca_bounds = (-124.5, 32.5, -114.1, 42.0)

ca_faults = active.cx[
    ca_bounds[0]:ca_bounds[2],  # longitude range
    ca_bounds[1]:ca_bounds[3]   # latitude range
].copy()

print(f"After California filter: {len(ca_faults)} faults")

# =============================================================================
# REPROJECT to UTM Zone 10N (meters)
# We need meters, not degrees, for rasterization later
# EPSG:32610 = UTM Zone 10N, covers most of California
# =============================================================================
ca_faults_utm = ca_faults.to_crs("EPSG:32610")
print(f"Reprojected to: {ca_faults_utm.crs}")

# =============================================================================
# QUICK SANITY CHECK — print a few fault names to confirm it looks right
# =============================================================================
print("\nSample faults in our filtered dataset:")
print(ca_faults[['fault_name', 'age', 'slip_rate']].head(10).to_string())

# =============================================================================
# PLOT — color by age category
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 14))

colors = {
    'historic':          'red',
    'latest Quaternary': 'orange',
    'late Quaternary':   'green',
    'Late Quaternary':   'green',
}

for age_label, color in colors.items():
    subset = ca_faults[ca_faults['age'] == age_label]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, linewidth=0.6, label=f"{age_label} ({len(subset)})")

ax.legend(title="Fault Age", loc="lower right", fontsize=9)
ax.set_title("California Active Faults — Filtered Dataset", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("outputs/california_faults_filtered.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nMap saved to outputs/california_faults_filtered.png")

# =============================================================================
# SAVE the filtered + reprojected shapefile for use in later steps
# labels.py will load this instead of the full 112k row file
# =============================================================================
ca_faults_utm.to_file("data/ca_active_faults_utm.shp")
print("Filtered shapefile saved to data/ca_active_faults_utm.shp")

print(f"\nFinal dataset summary:")
print(f"  Total fault segments: {len(ca_faults_utm)}")
print(f"  Total fault length:   {ca_faults_utm.geometry.length.sum() / 1000:.0f} km")
print(f"  CRS:                  {ca_faults_utm.crs}")
print(f"\nNEXT STEP: dataset.py — download Sentinel-2 imagery from Google Earth Engine")