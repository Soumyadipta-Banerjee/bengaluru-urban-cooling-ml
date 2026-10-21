"""
Dataset processing module for Bengaluru Urban Cooling Island analysis.
Handles data loading, tertile class labeling, spatial stratified splitting, and feature standardization.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [f"A{i:02d}" for i in range(64)]
TARGET_COL = "cooling_intensity"
CLASS_COL = "cooling_class"
CLASS_NAMES = ["Low", "Moderate", "High"]
CLASS_CODES = {1: "Low", 2: "Moderate", 3: "High"}

def generate_benchmark_dataset(bbmp_geojson_path, output_csv_path, num_samples=3000, random_state=42):
    """
    Generates a realistic benchmark dataset aligned with Bengaluru's geographic boundary
    and AlphaEarth 64-band embedding properties when raw satellite data is being acquired.
    """
    np.random.seed(random_state)
    gdf = gpd.read_file(bbmp_geojson_path)
    bounds = gdf.total_bounds # minx, miny, maxx, maxy
    
    # Generate points within BBMP bounds
    lons = np.random.uniform(bounds[0], bounds[2], num_samples * 2)
    lats = np.random.uniform(bounds[1], bounds[3], num_samples * 2)
    pts = gpd.points_from_xy(lons, lats, crs=gdf.crs)
    pts_gdf = gpd.GeoDataFrame(geometry=pts, crs=gdf.crs)
    
    # Spatial join to keep only points inside BBMP
    inside = gpd.sjoin(pts_gdf, gdf, predicate='within')
    sampled = inside.head(num_samples).copy()
    
    final_lons = sampled.geometry.x.values
    final_lats = sampled.geometry.y.values
    n = len(final_lons)
    
    # Simulate AlphaEarth 64 embeddings with realistic biophysical latent manifold
    # Foundation satellite embeddings project environmental signals across orthogonal latent components
    n_latents = 8
    # Latent 0: Canopy density / NDVI
    # Latent 1: Impervious built surface
    # Latent 2: Water / moisture proximity
    # Latent 3: Grass / open soil
    # Latent 4: Building height / roughness
    # Latent 5: Albedo / solar reflectance
    # Latent 6: Elevation / thermal lapse
    # Latent 7: Bare dry land
    latent = np.random.normal(0, 1.0, size=(n, n_latents))
    
    # Orthogonal projection into 64 AlphaEarth bands
    q, _ = np.linalg.qr(np.random.normal(0, 1.0, size=(64, n_latents)))
    projection = q.T # (8, 64)
    embeddings = latent @ projection + np.random.normal(0, 0.12, size=(n, 64))
    # Normalize embeddings to unit sphere (typical for satellite embedding representations)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Continuous cooling intensity with realistic urban thermal physics
    # Cooling is strongly positive with canopy & water, negative with dense impervious built surfaces
    cooling_intensity = 3.6 + 1.45 * latent[:, 0] - 1.05 * latent[:, 1] + 0.60 * latent[:, 2] - 0.35 * latent[:, 7] + np.random.normal(0, 0.25, n)
    cooling_intensity = np.clip(cooling_intensity, 1.2, 7.8)
    
    green_lst = 32.5 - 0.75 * cooling_intensity + np.random.normal(0, 0.4, n)
    ring_lst = green_lst + cooling_intensity
    
    df_dict = {
        'longitude': final_lons,
        'latitude': final_lats,
        'LST_green': green_lst,
        'LST_ring_median': ring_lst,
        'cooling_intensity': cooling_intensity
    }
    for i, col in enumerate(FEATURE_COLS):
        df_dict[col] = embeddings[:, i]
        
    df = pd.DataFrame(df_dict)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Benchmark dataset saved to: {output_csv_path} ({n} samples with realistic AlphaEarth manifold)")
    return df

def load_and_preprocess_dataset(csv_path):
    """
    Loads raw CSV samples, removes NaNs and duplicates, and verifies features.
    """
    df = pd.read_csv(csv_path)
    initial_len = len(df)
    
    # Drop NoData and duplicates
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df.drop_duplicates(subset=['longitude', 'latitude'])
    
    print(f"Loaded dataset: {len(df)} samples (removed {initial_len - len(df)} duplicates/NoData).")
    return df

def spatial_stratified_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, n_clusters=20, random_state=42):
    """
    Performs spatially stratified splitting using geographic spatial blocking (K-Means on coordinates).
    Prevents spatial autocorrelation / spatial leakage between training, validation, and testing sets.
    """
    coords = df[['longitude', 'latitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['spatial_cluster'] = kmeans.fit_predict(coords)
    
    cluster_ids = np.unique(df['spatial_cluster'])
    np.random.seed(random_state)
    np.random.shuffle(cluster_ids)
    
    # Assign entire geographic clusters to train, val, or test
    cluster_counts = df['spatial_cluster'].value_counts()
    total_samples = len(df)
    
    train_target = int(train_frac * total_samples)
    val_target = int(val_frac * total_samples)
    
    train_clusters = []
    val_clusters = []
    test_clusters = []
    
    curr_train = 0
    curr_val = 0
    
    for cid in cluster_ids:
        c_count = cluster_counts[cid]
        if curr_train + c_count <= train_target:
            train_clusters.append(cid)
            curr_train += c_count
        elif curr_val + c_count <= val_target:
            val_clusters.append(cid)
            curr_val += c_count
        else:
            test_clusters.append(cid)
            
    df_train = df[df['spatial_cluster'].isin(train_clusters)].copy()
    df_val = df[df['spatial_cluster'].isin(val_clusters)].copy()
    df_test = df[df['spatial_cluster'].isin(test_clusters)].copy()
    
    print(f"Spatial Split -> Train: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%), "
          f"Val: {len(df_val)} ({len(df_val)/len(df)*100:.1f}%), "
          f"Test: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    
    # Compute tertiles ONLY on training data to prevent label leakage
    t_low, t_high = np.percentile(df_train[TARGET_COL], [33.333, 66.667])
    print(f"Training set Cooling Intensity tertile thresholds: Low <= {t_low:.2f}°C, "
          f"Moderate = ({t_low:.2f}°C, {t_high:.2f}°C], High > {t_high:.2f}°C")
    
    def assign_class(val):
        if val <= t_low:
            return 1 # Low
        elif val <= t_high:
            return 2 # Moderate
        else:
            return 3 # High
            
    df_train[CLASS_COL] = df_train[TARGET_COL].apply(assign_class)
    df_val[CLASS_COL] = df_val[TARGET_COL].apply(assign_class)
    df_test[CLASS_COL] = df_test[TARGET_COL].apply(assign_class)
    
    # Standardize AlphaEarth features using statistics from TRAINING DATA ONLY
    scaler = StandardScaler()
    scaler.fit(df_train[FEATURE_COLS])
    
    X_train = scaler.transform(df_train[FEATURE_COLS])
    X_val = scaler.transform(df_val[FEATURE_COLS])
    X_test = scaler.transform(df_test[FEATURE_COLS])
    
    y_train = df_train[CLASS_COL].values
    y_val = df_val[CLASS_COL].values
    y_test = df_test[CLASS_COL].values
    
    y_cont_train = df_train[TARGET_COL].values
    y_cont_val = df_val[TARGET_COL].values
    y_cont_test = df_test[TARGET_COL].values
    
    return {
        'X_train': X_train, 'y_train': y_train, 'y_cont_train': y_cont_train, 'df_train': df_train,
        'X_val': X_val, 'y_val': y_val, 'y_cont_val': y_cont_val, 'df_val': df_val,
        'X_test': X_test, 'y_test': y_test, 'y_cont_test': y_cont_test, 'df_test': df_test,
        'scaler': scaler,
        'tertiles': (t_low, t_high)
    }

