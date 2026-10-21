# Assignment 4: Bengaluru Urban Cooling Island (UCI) Analysis
**Algorithms:** Support Vector Regression (SVR) with Kernels & Neural-Network-KKT Fuzzy Regression (plus k-Nearest Neighbours benchmark)  
**Features:** Google AlphaEarth Satellite Embedding V1 (64 bands: A00–A63)  
**Target:** Continuous Cooling Intensity ($\Delta T$) discretized into 3 Tertiles:
- **Low Cooling Potential (Class 1):** $\Delta T \le 2.84^\circ\text{C}$
- **Moderate Cooling Potential (Class 2):** $2.84^\circ\text{C} < \Delta T \le 4.45^\circ\text{C}$
- **High Cooling Potential (Class 3):** $\Delta T > 4.45^\circ\text{C}$

---

## 1. Quantitative Performance Comparison on Held-out Test Set

| Metric | k-NN (Selected k=23) | SVR (RBF Kernel) | NN-KKT Fuzzy Regression |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **80.53%** | **83.81%** | **83.59%** |
| **Balanced Accuracy** | **80.88%** | **83.76%** | **83.79%** |
| **Macro Precision** | **80.60%** | **84.16%** | **83.39%** |
| **Macro Recall** | **80.88%** | **83.76%** | **83.79%** |
| **Macro F1-Score** | **79.75%** | **83.92%** | **83.25%** |

### Class-Wise Recall
* **Low Cooling Recall:** k-NN = `91.25%` | SVR = `88.75%` | NN-KKT = `91.88%`
* **Moderate Cooling Recall:** k-NN = `57.05%` | SVR = `78.85%` | NN-KKT = `67.31%`
* **High Cooling Recall:** k-NN = `94.33%` | SVR = `83.69%` | NN-KKT = `92.20%`

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
   - **Selected k for k-NN:** Optimal $k = 23$ was selected via validation Balanced Accuracy tuning ($k$ vs validation score curve).
   - **SVR Hyperparameters:** The RBF kernel achieved optimal generalization with regularization $C = 10.0$, tube $\epsilon = 0.1$, and $\gamma = 0.05$.
   - **Neural-Network-KKT Fuzzy Regression:** Effectively learned asymmetric upper and lower uncertainty bounds, ensuring that thermal uncertainty across heterogeneous urban canopies is bounded via Karush-Kuhn-Tucker inclusion constraints.

2. **Class-Specific Error Analysis:**  
   Most misclassifications occurred between adjacent classes (Low $\leftrightarrow$ Moderate and Moderate $\leftrightarrow$ High) near the tertile boundaries ($2.84^\circ\text{C}$ and $4.45^\circ\text{C}$). Zero or negligible misclassifications occurred between the extreme Low and High classes. Moderate cooling potential exhibits the highest variance due to mixed land-cover pixels (small residential parks surrounded by high-density asphalt).

3. **Separability of AlphaEarth Embedding Neighborhoods:**  
   The 64-dimensional AlphaEarth representations successfully separate cooling intensity levels. High cooling potentials correlate strongly with dense tree canopies and large water-adjacent parks, which map to distinct clusters in the embedding latent space, whereas low cooling potentials map to sparse greenery and concrete-dominated neighborhoods.
