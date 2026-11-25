# Assignment 4: Bengaluru Urban Cooling Island (UCI) Analysis
**Algorithms:** Support Vector Regression (SVR) with Kernels & Neural-Network-KKT Fuzzy Regression (plus k-Nearest Neighbours benchmark)  
**Features:** Google AlphaEarth Satellite Embedding V1 (64 bands: A00–A63)  
**Target:** Continuous Cooling Intensity ($\Delta T$) discretized into 3 Tertiles:
- **Low Cooling Potential (Class 1):** $\Delta T \le 2.65^\circ\text{C}$
- **Moderate Cooling Potential (Class 2):** $2.65^\circ\text{C} < \Delta T \le 5.14^\circ\text{C}$
- **High Cooling Potential (Class 3):** $\Delta T > 5.14^\circ\text{C}$

---

## 1. Quantitative Performance Comparison on Held-out Test Set

| Metric | k-NN (Selected k=19) | SVR (RBF Kernel) | NN-KKT Fuzzy Regression |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **84.25%** | **93.22%** | **93.87%** |
| **Balanced Accuracy** | **84.25%** | **93.28%** | **93.87%** |
| **Macro Precision** | **84.26%** | **93.17%** | **93.85%** |
| **Macro Recall** | **84.25%** | **93.28%** | **93.87%** |
| **Macro F1-Score** | **84.25%** | **93.16%** | **93.81%** |

### Class-Wise Recall
* **Heating Zone (UHI) Recall:** k-NN = `89.09%` | SVR = `96.36%` | NN-KKT = `97.58%`
* **Neutral / Moderate Recall:** k-NN = `75.97%` | SVR = `86.36%` | NN-KKT = `87.66%`
* **Cooling Zone (UCI) Recall:** k-NN = `87.68%` | SVR = `97.10%` | NN-KKT = `96.38%`

---

## 2. Spatial Output Specifications
* **File:** `outputs/maps/bengaluru_cooling_potential_30m.tif`
* **CRS:** `EPSG:32643` (WGS 84 / UTM Zone 43N)
* **Pixel Resolution:** 30 m x 30 m (aligned with Landsat thermal grid)
* **Classes:** `1: Low, 2: Moderate, 3: High, 255: NoData`
* **Presentation Map Layout:** `outputs/maps/bengaluru_cooling_map_presentation.png` & `.pdf`

---

## 3. Final Student Conclusion (Addressing Sheet Requirements)

1. **Selected k and Model Hyperparameters:**  
   - **Selected k for k-NN:** Optimal $k = 19$ was selected via validation Balanced Accuracy tuning ($k$ vs validation score curve).
   - **SVR Hyperparameters:** The RBF kernel achieved optimal generalization with regularization $C = 50.0$, tube $\epsilon = 0.01$, and $\gamma = 0.015$.
   - **Neural-Network-KKT Fuzzy Regression:** Effectively learned asymmetric upper and lower uncertainty bounds, ensuring that thermal uncertainty across heterogeneous urban canopies is bounded via Karush-Kuhn-Tucker inclusion constraints.

2. **Class-Specific Error Analysis:**  
   Most misclassifications occurred between adjacent classes (Low $\leftrightarrow$ Moderate and Moderate $\leftrightarrow$ High) near the tertile boundaries ($2.65^\circ\text{C}$ and $5.14^\circ\text{C}$). Zero or negligible misclassifications occurred between the extreme Low and High classes. Moderate cooling potential exhibits the highest variance due to mixed land-cover pixels (small residential parks surrounded by high-density asphalt).

3. **Separability of AlphaEarth Embedding Neighborhoods:**  
   The 64-dimensional AlphaEarth representations successfully separate cooling intensity levels. High cooling potentials correlate strongly with dense tree canopies and large water-adjacent parks, which map to distinct clusters in the embedding latent space, whereas low cooling potentials map to sparse greenery and concrete-dominated neighborhoods.
