# Detecting Active Faults from Satellite Imagery

A deep learning pipeline for detecting active geological faults in California using Sentinel-2 satellite imagery and the Prithvi-EO 2.0 geospatial foundation model.

## Overview

Fault lines are identified from the USGS Quaternary Fault and Fold Database and used to create pixel-level labels on multispectral Sentinel-2 imagery. The model is trained to segment fault zones directly from satellite data across three geologically diverse regions of California.

**Training regions:**
- **Mojave Desert** — Southern San Andreas fault, arid environment
- **Carrizo Plain** — Central San Andreas fault, one of the most visually distinct sections
- **Bay Area** — Hayward and Calaveras faults, denser vegetation

## Pipeline

```
main.py → dataset.py → labels.py → patches.py → train_colab.ipynb
```

### 1. `main.py` — Fault data preparation
Loads the USGS fault database, filters to active California faults (historic, late/latest Quaternary), reprojects to UTM Zone 10N (EPSG:32610), and saves a filtered shapefile to `data/ca_active_faults_utm.shp`.

### 2. `dataset.py` — Imagery download
Connects to Google Earth Engine and exports cloud-free Sentinel-2 median composites (2022–2023) for each training region to Google Drive as GeoTIFFs.

**Bands used:** B2, B3, B4, B8, B11, B12 (Blue, Green, Red, NIR, SWIR1, SWIR2) — 10m resolution

### 3. `labels.py` — Mask generation
Reprojects and rasterizes fault lines onto the Sentinel-2 pixel grids, producing binary masks (1 = fault, 0 = background). Fault lines are buffered to 50m (5 pixels) to account for misalignment and improve learnability.

Output: `data/masks/sentinel2_{region}_10m_mask.tif`

### 4. `patches.py` — Patch extraction
Slices each image/mask pair into 128×128 pixel patches (1.28 km × 1.28 km at 10m resolution). Retains only patches containing fault pixels to address class imbalance (~1% of pixels are faults). Generates train/val/test splits.

Output: `data/patches/images/`, `data/patches/labels/`, `data/patches/splits/`

### 5. `train_colab.ipynb` — Model training
Trains the Prithvi-EO 2.0 geospatial foundation model on the extracted patches for fault segmentation. Designed to run on Google Colab with GPU.

## Setup

### Requirements
```bash
pip install geopandas rasterio numpy matplotlib earthengine-api
```

### Google Earth Engine
Authenticate with GEE before running `dataset.py`:
```bash
earthengine authenticate
```

### Running the pipeline
```bash
python main.py       # filter and reproject fault data
python dataset.py    # export Sentinel-2 imagery to Google Drive
# download .tif files from Google Drive → data/imagery/
python labels.py     # generate fault masks
python patches.py    # create training patches
# open train_colab.ipynb in Google Colab
```

## Data

Large data files are not tracked in this repo (see `.gitignore`). You will need to:

1. Download the [USGS Quaternary Fault and Fold Database](https://www.usgs.gov/programs/earthquake-hazards/faults) GIS files → place in `data/Qfaults_GIS/`
2. Run `dataset.py` to export Sentinel-2 imagery via GEE → place `.tif` files in `data/imagery/`
3. Run `labels.py` and `patches.py` to regenerate masks and patches locally

## Project Structure

```
├── main.py               # fault data filtering and reprojection
├── dataset.py            # Sentinel-2 imagery download via GEE
├── labels.py             # fault mask generation
├── patches.py            # image/mask patch extraction
├── train_colab.ipynb     # model training notebook
├── data/
│   ├── Qfaults_GIS/      # USGS fault database (not tracked)
│   ├── ca_active_faults_utm.shp  # filtered fault shapefile
│   ├── imagery/          # Sentinel-2 GeoTIFFs (not tracked)
│   ├── masks/            # binary fault masks (not tracked)
│   └── patches/          # 128x128 training patches (not tracked)
└── outputs/              # visualizations and previews (not tracked)
```
