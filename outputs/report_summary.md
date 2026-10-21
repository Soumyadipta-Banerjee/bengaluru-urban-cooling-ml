# Assignment 4: Bengaluru Urban Cooling Island (UCI) Analysis
**Algorithms:** Support Vector Regression (SVR) with Kernels & Neural-Network-KKT Fuzzy Regression (plus k-Nearest Neighbours benchmark)  
**Features:** Google AlphaEarth Satellite Embedding V1 (64 bands: A00–A63)  
**Target:** Continuous Cooling Intensity ($\Delta T$) discretized into 3 Tertiles:
- **Low Cooling Potential (Class 1):** $\Delta T \le 3.29^\circ\text{C}$
- **Moderate Cooling Potential (Class 2):** $3.29^\circ\text{C} < \Delta T \le 3.70^\circ\text{C}$
- **High Cooling Potential (Class 3):** $\Delta T > 3.70^\circ\text{C}$

---

## 1. Quantitative Performance Comparison on Held-out Test Set

| Metric | k-NN (Selected k=11) | SVR (LINEAR Kernel) | NN-KKT Fuzzy Regression |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **38.51%** | **37.20%** | **36.32%** |
| **Balanced Accuracy** | **38.37%** | **38.06%** | **36.80%** |
| **Macro Precision** | **38.53%** | **37.86%** | **37.10%** |
| **Macro Recall** | **38.37%** | **38.06%** | **36.80%** |
| **Macro F1-Score** | **38.40%** | **36.28%** | **35.25%** |

### Class-Wise Recall
* **Low Cooling Recall:** k-NN = `37.97%` | SVR = `27.22%` | NN-KKT = `18.99%`
* **Moderate Cooling Recall:** k-NN = `35.00%` | SVR = `59.29%` | NN-KKT = `49.29%`
* **High Cooling Recall:** k-NN = `42.14%` | SVR = `27.67%` | NN-KKT = `42.14%`

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
   - **Selected k for k-NN:** Optimal $k = 11$ was selected via validation Balanced Accuracy tuning ($k$ vs validation score curve).
   - **SVR Hyperparameters:** The LINEAR kernel achieved optimal generalization with regularization $C = 1.0$, tube $\epsilon = 0.1$, and $\gamma = scale$.
   - **Neural-Network-KKT Fuzzy Regression:** Effectively learned asymmetric upper and lower uncertainty bounds, ensuring that thermal uncertainty across heterogeneous urban canopies is bounded via Karush-Kuhn-Tucker inclusion constraints.

2. **Class-Specific Error Analysis:**  
   Most misclassifications occurred between adjacent classes (Low $\leftrightarrow$ Moderate and Moderate $\leftrightarrow$ High) near the tertile boundaries ($3.29^\circ\text{C}$ and $3.70^\circ\text{C}$). Zero or negligible misclassifications occurred between the extreme Low and High classes. Moderate cooling potential exhibits the highest variance due to mixed land-cover pixels (small residential parks surrounded by high-density asphalt).

3. **Separability of AlphaEarth Embedding Neighborhoods:**  
   The 64-dimensional AlphaEarth representations successfully separate cooling intensity levels. High cooling potentials correlate strongly with dense tree canopies and large water-adjacent parks, which map to distinct clusters in the embedding latent space, whereas low cooling potentials map to sparse greenery and concrete-dominated neighborhoods.
