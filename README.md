# Assignment 4: Bengaluru Urban Cooling Island (UCI) Analysis

This repository contains the complete implementation for **Assignment 4: Urban Cooling**, applying **Support Vector Regression (SVR) with Kernels** and **Neural-Network-KKT Fuzzy Regression** using **Google AlphaEarth Satellite Embeddings (V1)** to predict urban cooling potential classes across Bengaluru.

---

## Project Structure

```
.
├── data/
│   ├── BBMP.geojson                     # Official Bengaluru BBMP boundary (EPSG:4326)
│   └── raw/
│       └── bengaluru_cooling_samples_2024.csv # Sample dataset with A00-A63 & LST
├── gee/
│   ├── extract_urban_cooling.js         # Google Earth Engine Code Editor script (1-click export to Drive)
│   └── extract_urban_cooling.py         # Earth Engine Python API extraction script
├── src/
│   ├── __init__.py
│   ├── dataset.py                       # Data loading, 3-class tertiles, spatial stratified split (70/15/15)
│   ├── models_svr_fuzzy.py              # SVR (RBF, Poly, Linear) & Neural-Network-KKT Fuzzy Regression
│   ├── evaluate.py                      # Plots (kernel comparison, KKT intervals, confusion matrices)
│   └── export_geotiff.py                # 30 m GeoTIFF generation in EPSG:32643 & Presentation Map layout
├── outputs/
│   ├── maps/
│   │   ├── bengaluru_cooling_potential_30m.tif  # Required GeoTIFF (EPSG:32643, 30 m resolution)
│   │   ├── bengaluru_cooling_map_presentation.png # High-res cartographic map with legend & scale
│   │   └── bengaluru_cooling_map_presentation.pdf # PDF format for report submission
│   ├── plots/
│   │   ├── svr_kernel_comparison.png    # Validation balanced accuracy across kernels & C
│   │   ├── fuzzy_kkt_bounds.png         # Fuzzy inclusion interval [y_lower, y_upper] & true ΔT
│   │   ├── confusion_matrix_svr.png     # SVR 3-class test confusion matrix
│   │   ├── confusion_matrix_fuzzy_nn.png# NN-KKT 3-class test confusion matrix
│   │   └── class_distribution.png       # Stratified split class distribution
│   ├── models/
│   │   └── champion_model.joblib        # Serialized trained model & scaler
│   └── report_summary.md                # Quantitative metrics table & conclusions
├── run_pipeline.py                      # Master pipeline script
└── requirements.txt
```

---

## How to Run Locally

### 1. Run the Full ML Pipeline
To train both models, evaluate all metrics, generate charts, and export the 30 m GeoTIFF:

```bash
/home/soumya/myenv/bin/python3 run_pipeline.py
```

### 2. Live Data Extraction via Google Earth Engine (Optional)
If you wish to pull new multi-temporal composite imagery from Earth Engine:
1. Open the **[Google Earth Engine Code Editor](https://code.earthengine.google.com/)**.
2. Copy and paste the script from `gee/extract_urban_cooling.js`.
3. Click **Run**, then go to the **Tasks** tab on the right to export:
   * `bengaluru_cooling_samples_2024.csv` to Google Drive.
4. Download the CSV into `data/raw/` and rerun `run_pipeline.py`.

---

## Technical Highlights

1. **Support Vector Regression (SVR) with Kernels:**
   - Evaluated RBF, Polynomial, and Linear kernels across regularization strengths ($C$) and kernel coefficients ($\gamma$).
   - Predicts continuous cooling intensity ($\Delta T$ in °C), thresholded into 3 equal-frequency tertile classes (**Low**, **Moderate**, **High**) derived from the training set.

2. **Neural-Network-KKT Fuzzy Regression:**
   - A dual-head PyTorch neural architecture predicting the center cooling intensity ($\hat{y}$) and non-negative fuzzy spread ($s \ge 0$).
   - Trained using Karush-Kuhn-Tucker (KKT) inequality constraint penalties ensuring each observed cooling sample satisfies fuzzy inclusion ($y_{\text{lower}} \le y \le y_{\text{upper}}$) while minimizing the total fuzziness/spread.

3. **Geospatial Integrity & Spatial Leakage Prevention:**
   - **Spatially Stratified Split:** Uses geographic coordinate clustering to divide Bengaluru into discrete geographic blocks (70% train, 15% validation, 15% test).
   - **GeoTIFF Raster Output:** Projected in `EPSG:32643` (UTM Zone 43N) at $30\text{ m} \times 30\text{ m}$ native resolution with documented `NoData = 255`.
